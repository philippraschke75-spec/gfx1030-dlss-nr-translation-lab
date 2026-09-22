"""Host-only (no GPU) probe: compute real block4 output (stage-1 end) on a real captured tile, then test hypotheses
for how block5 (stage 2, C=64) should read it - same H,W vs halved H,W, same buffer vs no change - by running
k_swin_var<64,false> against it and checking: does it terminate, does it stay in-bounds, is the output finite/plausible.
This is a structural probe, not a correctness proof (no GPU/emulator cross-check here, no real block5 weights needed
for the "does it terminate & read in-bounds" part, though we use real weights anyway since we have them).
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
WDIR = D.ROOT / 'build' / 'weights'


def run_kernel(sym, lds, H, W, flags, ox, oy, patch, C_label):
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
    elif flags & 0x14 == 0x14:   # pre-block style
        ka = V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)
        struct.pack_into('<Q', ka, 0x00, 0)
        for off in (0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68): struct.pack_into('<I', ka, off, 0)
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
    try:
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=500_000)
    except Exception as ex:
        print('  [%s] FAULT after %.1fs: %s' % (C_label, time.time() - t0, ex))
        return None
    print('  [%s] terminated cleanly in %.1fs, grid=%s' % (C_label, time.time() - t0, grid))
    return a, init


# ---- import ----
r = run_kernel(IMPORT_SYM, D.group_size(IMPORT_SYM), N, N, 0, 0, 0, lambda a: a.__setitem__(slice(0, tile.size), tile.reshape(-1)), 'import')
a0, init0 = r
rgb = a0[V.SLOT:V.SLOT + N * N * 12].view('<f4').copy()
print('import out finite %.3f range %.4g..%.4g' % (np.isfinite(rgb).mean(), rgb[np.isfinite(rgb)].min(), rgb[np.isfinite(rgb)].max()))

# ---- pre-block ----
w0 = np.frombuffer((WDIR / 'block0.bin').read_bytes(), np.uint8)
def patch_pb(a):
    a[2 * V.SLOT:2 * V.SLOT + len(w0)] = w0
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + rgb.nbytes] = rgb.view(np.uint8)
r = run_kernel(PRE_SYM, PRE_LDS, N, N, 0x14, 0, 0, patch_pb, 'preblock')
a_pb, init_pb = r
cur = a_pb[1 * V.SLOT:2 * V.SLOT].copy()

# ---- blocks 1-4 (stage 1, C=32) ----
SCHEDULE = [('block1.bin', 1, 0, 0), ('block2.bin', 0, -4, -4), ('block3.bin', 0, -4, 0), ('block4.bin', 4, 0, -4)]
for i, (wf, flags, ox, oy) in enumerate(SCHEDULE, start=1):
    w = np.frombuffer((WDIR / wf).read_bytes(), np.uint8)
    cur_local = cur
    def patch_b(a, w=w, cur_local=cur_local):
        a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
        a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
    r = run_kernel(ENC32_SYM, ENC32_LDS, N, N, flags, ox, oy, patch_b, 'block%d' % i)
    a_b, init_b = r
    cur = a_b[1 * V.SLOT:2 * V.SLOT].copy()
    changed = np.nonzero(a_b[1*V.SLOT:2*V.SLOT] != init_b[1*V.SLOT:2*V.SLOT])[0]
    print('  block%d output slot: %d bytes changed' % (i, len(changed)))

real_block4_out = cur   # C=32, H=W=64 real, non-degenerate encoder output
print('\n=== stage-1 output ready; probing stage-2 (block5, C=64) hypotheses ===\n')

w5 = np.frombuffer((WDIR / 'block5.bin').read_bytes(), np.uint8)


def try_hypothesis(label, H, W, flags, ox, oy):
    def patch5(a):
        a[0 * V.SLOT:0 * V.SLOT + len(real_block4_out)] = real_block4_out
        a[2 * V.SLOT:2 * V.SLOT + len(w5)] = w5
    print('hypothesis: %s (H=%d W=%d flags=%d origin=(%d,%d))' % (label, H, W, flags, ox, oy))
    r = run_kernel(ENC64_SYM, ENC64_LDS, H, W, flags, ox, oy, patch5, 'block5:' + label)
    if r is None:
        return
    a_b, init_b = r
    seg = a_b[1 * V.SLOT:2 * V.SLOT]; iseg = init_b[1 * V.SLOT:2 * V.SLOT]
    changed = np.nonzero(seg != iseg)[0]
    if len(changed) == 0:
        print('  -> NO bytes written (degenerate / grid empty)')
        return
    vals = seg[changed]
    def e4(b):
        s = -1. if b & 128 else 1.; e = (b >> 3) & 15; m = b & 7
        return np.nan if (e == 15 and m == 7) else s * (m / 8 * 2**-6 if e == 0 else (1 + m / 8) * 2**(e - 7))
    L = np.array([e4(i) for i in range(256)])
    dec = L[vals]
    print('  -> %d bytes written, %d distinct values | as e4m3: nan %.3f, |v| median %.4g max %.4g, zeros %.3f' % (
        len(changed), len(np.unique(vals)), np.isnan(dec).mean(),
        np.nanmedian(np.abs(dec)) if np.isfinite(dec).any() else float('nan'),
        np.nanmax(np.abs(dec)) if np.isfinite(dec).any() else float('nan'), (vals == 0).mean()))


try_hypothesis('same H,W as stage1 (no change)', N, N, 1, 0, 0)
try_hypothesis('halved H,W (32x32)', N // 2, N // 2, 1, 0, 0)
try_hypothesis('halved H,W, offset -4,-4 (mode1)', N // 2, N // 2, 0, -4, -4)
