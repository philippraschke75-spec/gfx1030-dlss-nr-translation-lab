"""Compute real pooled data (import->preblock->block1-4, host-only emulator) then GPU-verify block5 consuming it -
the specific new piece not yet checked on hardware. Stages 1-4 are not re-dispatched to the GPU here because they
were already independently GPU-verified (RESULTS_REAL_CONFIG.md); using their emulator output as the real value is
equivalent, since GPU==emulator was already established for those kernels.
"""
import sys, struct, time, re, hashlib, json
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

IMPORT_SYM = '_Z8k_import12ImportParams'
PRE_SYM, PRE_LDS = D.SYMS['32_1']
ENC32_SYM, ENC32_LDS = D.SYMS['32_0']; ENC32_LDS = ENC32_LDS or D.group_size(ENC32_SYM)
ENC64_SYM, ENC64_LDS = D.SYMS['64_0']; ENC64_LDS = ENC64_LDS or D.group_size(ENC64_SYM)
POOL_SLOT = V.PTR_FIELDS.index(0x38)
color_path = 'rdna2/build/real-input/color.bin'
full = np.fromfile(color_path, np.uint8).reshape(960, 1707, 8)
tx, ty, N = 500, 300, 64
tile = np.ascontiguousarray(full[ty:ty + N, tx:tx + N])
IN_SLOT = V.PTR_FIELDS.index(0x40)
WDIR = D.ROOT / 'build' / 'weights'
t_start = time.time()

def run_local(sym, lds, H, W, flags, ox, oy, patch, label, pre=False):
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
        for off in (0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68): struct.pack_into('<I', ka, off, 0)
        struct.pack_into('<f', ka, 0x50, 1.0)
        for off in range(0x70, 0xa0, 8): struct.pack_into('<Q', ka, off, 0)
    else:
        ka = V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)
    prog = E.load_program(R.DIS, {sym})
    g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    patch(a)
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=500_000)
    print('  [%s] emulator done, %.0fs since start' % (label, time.time() - t_start), flush=True)
    return a

r = run_local(IMPORT_SYM, D.group_size(IMPORT_SYM), N, N, 0, 0, 0, lambda a: a.__setitem__(slice(0, tile.size), tile.reshape(-1)), 'import')
rgb = r[V.SLOT:V.SLOT + N * N * 12].view('<f4').copy()

w0 = np.frombuffer((WDIR / 'block0.bin').read_bytes(), np.uint8)
def patch_pb(a):
    a[2 * V.SLOT:2 * V.SLOT + len(w0)] = w0
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + rgb.nbytes] = rgb.view(np.uint8)
r = run_local(PRE_SYM, PRE_LDS, N, N, 0x14, 0, 0, patch_pb, 'preblock', pre=True)
cur = r[1 * V.SLOT:2 * V.SLOT].copy()

SCHEDULE = [('block1.bin', 1, 0, 0), ('block2.bin', 0, -4, -4), ('block3.bin', 0, -4, 0), ('block4.bin', 4, 0, -4)]
pooled = None
for i, (wf, flags, ox, oy) in enumerate(SCHEDULE, start=1):
    w = np.frombuffer((WDIR / wf).read_bytes(), np.uint8)
    cur_local = cur
    def patch_b(a, w=w, cur_local=cur_local):
        a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
        a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
    r = run_local(ENC32_SYM, ENC32_LDS, N, N, flags, ox, oy, patch_b, 'block%d' % i)
    cur = r[1 * V.SLOT:2 * V.SLOT].copy()
    if flags & 4:
        pooled = r[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT].copy()

print('real pooled data ready, %.0fs; now GPU-verifying block5 against it' % (time.time() - t_start), flush=True)

# ---- GPU-verify block5 with this real pooled input ----
w5 = np.frombuffer((WDIR / 'block5.bin').read_bytes(), np.uint8)
HALF = N // 2
grid5 = ((HALF + 7) // 8, (HALF + 7) // 8)

def kernarg5():
    return V.make_kernarg(H=HALF, W=HALF, offy=0, offx=0, flags=1, grid=grid5)

def emulate5():
    ka = kernarg5()
    prog = E.load_program(R.DIS, {ENC64_SYM}); g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    a[0 * V.SLOT:0 * V.SLOT + len(pooled)] = pooled
    a[2 * V.SLOT:2 * V.SLOT + len(w5)] = w5
    init = a.copy()
    for wy in range(grid5[1]):
        for wx in range(grid5[0]):
            E.run_workgroup(prog, g, ENC64_LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=500_000)
    return bytes(ka), init, a.copy()

kab, init, ref = emulate5()
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (ENC64_SYM + '.co')
rc, msg, out = D.gpu(module, ENC64_SYM, 'block5_real_pool_verify', kab, init, grid5)
if out is not None:
    V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
    kab, init, ref = emulate5()
if out is None:
    print(json.dumps(dict(status='GPU_FAIL', gpu=msg)))
else:
    diff = np.nonzero(out != ref)[0]
    print(json.dumps(dict(status='PASS' if len(diff) == 0 else 'FAIL', mismatches=int(len(diff)), gpu=msg,
                           bytes_written=int((ref != init).sum()))))
