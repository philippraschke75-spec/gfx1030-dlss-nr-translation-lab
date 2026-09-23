"""Host-only: chain the REAL stage-1 output through the now-identified fused pool path (block4 writes a second,
channel-doubled/halved output to kernarg +0x38 when flags=4) into block5 (stage 2, C=64), on real captured pixels.
"""
import sys, struct, time
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

IMPORT_SYM = '_Z8k_import12ImportParams'
PRE_SYM, PRE_LDS = D.SYMS['32_1']
ENC32_SYM, ENC32_LDS = D.SYMS['32_0']; ENC32_LDS = ENC32_LDS or D.group_size(ENC32_SYM)
ENC64_SYM, ENC64_LDS = D.SYMS['64_0']; ENC64_LDS = ENC64_LDS or D.group_size(ENC64_SYM)

color_path = '../build/rebuild/20260921-221209-819c53031b104a728bd75305ddd41727.log.color_rgba16f.bin'
full = np.fromfile(color_path, np.uint8).reshape(960, 1707, 8)
tx, ty, N = 500, 300, 64
tile = np.ascontiguousarray(full[ty:ty + N, tx:tx + N])
IN_SLOT = V.PTR_FIELDS.index(0x40)
POOL_SLOT = V.PTR_FIELDS.index(0x38)
WDIR = D.ROOT / 'build' / 'weights'


def run_kernel(sym, lds, H, W, flags, ox, oy, patch, label, pre=False):
    grid = ((W + 7) // 8, (H + 7) // 8)
    if sym == IMPORT_SYM:
        pitch = W * 8
        ka = bytearray(0x130)
        struct.pack_into('<Q', ka, 0x00, V.ARENA)
        struct.pack_into('<iiiiii', ka, 0x08, pitch, 0, H, W, H, W)
        struct.pack_into('<Q', ka, 0x20, V.ARENA + V.SLOT)
        struct.pack_into('<if', ka, 0x28, 0, 1.0)
        struct.pack_into('<III', ka, 0x30, grid[0], grid[1], 1)
        struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
        grid = ((W + 255) // 256, H)
    elif pre:
        ka = V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)
        struct.pack_into('<Q', ka, 0x00, 0)
        # +0x68 is the pre-block's RNG seed, not a pointer. PTR_FIELDS lists it, so make_kernarg wrote a
        # whole pointer; zeroing only its low dword left arena bits in +0x6c, the GPU runner rebased the
        # qword, and the kernel ran seeded with the device arena address while the emulator used 0. That
        # one field made the pre-block differ by 67%% of its output. Same for the other scalars here.
        for off in (0x50, 0x58, 0x60, 0x68): struct.pack_into('<Q', ka, off, 0)
        for off in (0x54, 0x5c, 0x64): struct.pack_into('<I', ka, off, 0)
        struct.pack_into('<f', ka, 0x50, 1.0)
        for off in range(0x70, 0xa0, 8): struct.pack_into('<Q', ka, off, 0)
    else:
        ka = V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)
    prog = E.load_program(R.DIS, {sym})
    g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    patch(a)
    init = a.copy()
    t0 = time.time()
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=500_000)
    print('  [%s] done %.1fs grid=%s' % (label, time.time() - t0, grid))
    return a, init


r, _ = run_kernel(IMPORT_SYM, D.group_size(IMPORT_SYM), N, N, 0, 0, 0, lambda a: a.__setitem__(slice(0, tile.size), tile.reshape(-1)), 'import')
rgb = r[V.SLOT:V.SLOT + N * N * 12].view('<f4').copy()

w0 = np.frombuffer((WDIR / 'block0.bin').read_bytes(), np.uint8)
def patch_pb(a):
    a[2 * V.SLOT:2 * V.SLOT + len(w0)] = w0
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + rgb.nbytes] = rgb.view(np.uint8)
r, _ = run_kernel(PRE_SYM, PRE_LDS, N, N, 0x14, 0, 0, patch_pb, 'preblock', pre=True)
cur = r[1 * V.SLOT:2 * V.SLOT].copy()

SCHEDULE = [('block1.bin', 1, 0, 0), ('block2.bin', 0, -4, -4), ('block3.bin', 0, -4, 0), ('block4.bin', 4, 0, -4)]
pooled = None
for i, (wf, flags, ox, oy) in enumerate(SCHEDULE, start=1):
    w = np.frombuffer((WDIR / wf).read_bytes(), np.uint8)
    cur_local = cur
    def patch_b(a, w=w, cur_local=cur_local):
        a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
        a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
    r, init = run_kernel(ENC32_SYM, ENC32_LDS, N, N, flags, ox, oy, patch_b, 'block%d' % i)
    cur = r[1 * V.SLOT:2 * V.SLOT].copy()
    if flags & 4:
        pool_seg = r[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT]; pool_init = init[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT]
        changed = np.nonzero(pool_seg != pool_init)[0]
        print('  block%d pooled-output (+0x38): %d bytes changed, range [%d,%d)' % (i, len(changed), changed.min() if len(changed) else -1, changed.max() + 1 if len(changed) else -1))
        pooled = pool_seg.copy()

print('\n=== feeding block4 real +0x38 pooled output into block5 (C=64, H=W=%d) ===\n' % (N // 2))
w5 = np.frombuffer((WDIR / 'block5.bin').read_bytes(), np.uint8)
def patch5(a):
    a[0 * V.SLOT:0 * V.SLOT + len(pooled)] = pooled
    a[2 * V.SLOT:2 * V.SLOT + len(w5)] = w5
r, init = run_kernel(ENC64_SYM, ENC64_LDS, N // 2, N // 2, 1, 0, 0, patch5, 'block5')
seg = r[1 * V.SLOT:2 * V.SLOT]; iseg = init[1 * V.SLOT:2 * V.SLOT]
changed = np.nonzero(seg != iseg)[0]
vals = seg[changed]
def e4(b):
    s = -1. if b & 128 else 1.; e = (b >> 3) & 15; m = b & 7
    return np.nan if (e == 15 and m == 7) else s * (m / 8 * 2**-6 if e == 0 else (1 + m / 8) * 2**(e - 7))
L = np.array([e4(i) for i in range(256)])
dec = L[vals]
print('block5 output: %d bytes written, %d distinct values | as e4m3: nan %.4f, |v| median %.4g max %.4g, zeros %.3f' % (
    len(changed), len(np.unique(vals)), np.isnan(dec).mean(),
    np.nanmedian(np.abs(dec)) if np.isfinite(dec).any() else float('nan'),
    np.nanmax(np.abs(dec)) if np.isfinite(dec).any() else float('nan'), (vals == 0).mean()))
