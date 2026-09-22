"""Chain two real stages on REAL captured Cyberpunk pixels: k_import (format=0/RGBA16F -> RGB32F) feeding directly
into the pre-block (k_swin_var<32,true>, flags=0x14, real block0 weights). Two separate bounded GPU dispatches in one
sandbox run; each stage individually checked against the emulator at the device's actual base before being chained.
usage: chain_import_preblock_real.py <color.bin> <src_w> <src_h> <tile_x> <tile_y> <tile_n>  (tile_n x tile_n, multiple of 8)
env DLSSNR_WEIGHT_BLOB=<block0 record path>"""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

IMPORT_SYM = '_Z8k_import12ImportParams'
PRE_SYM, PRE_LDS = D.SYMS['32_1']
color_path, SRC_W, SRC_H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tx, ty, N = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
assert N % 8 == 0
full = np.fromfile(color_path, np.uint8).reshape(SRC_H, SRC_W, 8)
tile = np.ascontiguousarray(full[ty:ty + N, tx:tx + N])
results = {}

# ---- stage 1: k_import ----
pitch = N * 8; grid1 = ((N + 255) // 256, N)

def kernarg1():
    ka1 = bytearray(0x130)
    struct.pack_into('<Q', ka1, 0x00, V.ARENA)
    struct.pack_into('<iiiiii', ka1, 0x08, pitch, 0, N, N, N, N)
    struct.pack_into('<Q', ka1, 0x20, V.ARENA + V.SLOT)
    struct.pack_into('<if', ka1, 0x28, 0, 1.0)
    struct.pack_into('<III', ka1, 0x30, grid1[0], grid1[1], 1)
    struct.pack_into('<HHH', ka1, 0x3c, 256, 1, 1)
    return ka1

def emulate1():
    ka1 = kernarg1()
    prog = E.load_program(R.DIS, {IMPORT_SYM}); g, KA = V.build(1, ka1, len(V.PTR_FIELDS))
    a = g.regions[1].arr; a[:tile.size] = tile.reshape(-1)
    init = a.copy(); steps = 0; lds = D.group_size(IMPORT_SYM)
    for wy in range(grid1[1]):
        for wx in range(grid1[0]):
            steps += E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=2_000_000)['steps']
    return bytes(ka1), init, a.copy(), steps

ka1b, init1, ref1, steps1 = emulate1()
module1 = D.ROOT / 'build' / 'kernels-hw-scratch' / (IMPORT_SYM + '.co')
rc1, msg1, out1 = D.gpu(module1, IMPORT_SYM, 'chain_import_%d_%d_%d' % (tx, ty, N), ka1b, init1, grid1)
if out1 is not None:
    V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg1)[1], 16)
    ka1b, init1, ref1, steps1 = emulate1()
diff1 = np.nonzero(out1 != ref1)[0] if out1 is not None else None
results['import'] = dict(rc=rc1, gpu=msg1, mismatches=int(len(diff1)) if diff1 is not None else None,
                          status='PASS' if out1 is not None and len(diff1) == 0 else 'FAIL')
print('STAGE1 import:', json.dumps(results['import']), flush=True)
if out1 is None or len(diff1):
    print(json.dumps(results)); raise SystemExit(1)
rgb = ref1[V.SLOT:V.SLOT + N * N * 12].view('<f4').copy()   # real network-input floats, GPU-verified

# ---- stage 2: pre-block, fed the real import output ----
weight_path = os.environ['DLSSNR_WEIGHT_BLOB']
weights = np.frombuffer(open(weight_path, 'rb').read(), np.uint8)
grid2 = ((N + 7) // 8, (N + 7) // 8)
IN_SLOT = V.PTR_FIELDS.index(0x40)

def kernarg2():
    ka2 = V.make_kernarg(H=N, W=N, offy=0, offx=0, flags=0x14, grid=grid2)
    struct.pack_into('<Q', ka2, 0x00, 0)
    for off in (0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68): struct.pack_into('<I', ka2, off, 0)
    struct.pack_into('<f', ka2, 0x50, 1.0)
    for off in range(0x70, 0xa0, 8): struct.pack_into('<Q', ka2, off, 0)
    return ka2

def emulate2():
    ka2 = kernarg2()
    prog = E.load_program(R.DIS, {PRE_SYM}); g, KA = V.build(1, ka2, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    a[2 * V.SLOT:2 * V.SLOT + len(weights)] = weights
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + rgb.nbytes] = rgb.view(np.uint8)
    init = a.copy(); steps = 0
    for wy in range(grid2[1]):
        for wx in range(grid2[0]):
            steps += E.run_workgroup(prog, g, PRE_LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)['steps']
    return bytes(ka2), init, a.copy(), steps

ka2b, init2, ref2, steps2 = emulate2()
module2 = D.ROOT / 'build' / 'kernels-hw-scratch' / (PRE_SYM + '.co')
rc2, msg2, out2 = D.gpu(module2, PRE_SYM, 'chain_preblock_%d_%d_%d' % (tx, ty, N), ka2b, init2, grid2)
if out2 is not None:
    V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg2)[1], 16)
    ka2b, init2, ref2, steps2 = emulate2()
diff2 = np.nonzero(out2 != ref2)[0] if out2 is not None else None
results['preblock'] = dict(rc=rc2, gpu=msg2, mismatches=int(len(diff2)) if diff2 is not None else None,
                            bytes_written=int((ref2 != init2).sum()),
                            status='PASS' if out2 is not None and len(diff2) == 0 and (ref2 != init2).sum() > 0 else 'FAIL')
print('STAGE2 preblock:', json.dumps(results['preblock']), flush=True)
print('SUMMARY', json.dumps(results))
raise SystemExit(0 if results['import']['status'] == results['preblock']['status'] == 'PASS' else 1)
