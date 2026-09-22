"""GPU hardware verification for k_ffwd2 (_Z7k_ffwd211Ffwd2Params), the second FFN dispatch in a C=512 block -
the last unresolved kernel in that block's forward path. Ffwd2Params is 48 bytes per its own .s metadata;
layout decoded from the host launcher (0x18003399c-0x180033a03) and confirmed field-for-field by the kernel's
own prologue and by trace_ffwd2_kernarg.py's exhaustive read trace (reads exactly +0x00/32B, +0x20/16B,
+0x3c/4B = hidden_group_size_x, nothing else; terminated cleanly in 2.1s):
  +0x00 ptr (ctx+0x228), +0x08 ptr (r14, host null-checks it so it is optional), +0x10 ptr (ctx+0x230),
  +0x18 weight ptr (layer=0), +0x20 H, +0x24 W, +0x28 scalar ([[rdi]]), +0x2c padding.
Weight: real block23.layer0.layer bytes (524,288 B = exactly 512*1024, a clean C=512->1024 FFN expand weight).
usage: difftest_ffwd2.py [seed]   (inside sandbox.py)
"""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z7k_ffwd211Ffwd2Params'
KSIZE = 304
NSLOT = 6
LDS = 21504
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
GX, GY = 1, 1
H = W = 8
NGROUP = 4          # 4*16 == H*W, so the whole 64-token tile is live (n=0 is a degenerate no-op: nothing is written)
WEIGHT_PATH = D.ROOT / 'build' / 'weights' / 'block23_layer0.bin'


def kernarg():
    ka = bytearray(KSIZE)
    S = lambda k: V.ARENA + k * V.SLOT
    for off, idx in ((0x00, 0), (0x08, 1), (0x10, 2), (0x18, 3)):
        struct.pack_into('<Q', ka, off, S(idx))
    struct.pack_into('<i', ka, 0x20, H)               # H
    struct.pack_into('<i', ka, 0x24, W)               # W
    struct.pack_into('<i', ka, 0x28, NGROUP)          # count of 16-token groups; effective count is min(n*16, H*W)
    struct.pack_into('<III', ka, 0x30, GX, GY, 1)     # hidden_block_count_x/y/z
    struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)     # hidden_group_size_x/y/z
    struct.pack_into('<HHH', ka, 0x42, 0, 0, 0)       # hidden_remainder_x/y/z
    struct.pack_into('<QQQ', ka, 0x58, 0, 0, 0)       # hidden_global_offset_x/y/z
    struct.pack_into('<H', ka, 0x70, 2)               # hidden_grid_dims
    return ka


def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM}); g, KA = V.build(seed, ka, NSLOT)
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 29)
    a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)
    w = np.frombuffer(WEIGHT_PATH.read_bytes(), np.uint8)
    a[3 * V.SLOT:3 * V.SLOT + len(w)] = w
    init = a.copy(); steps = 0
    for wy in range(GY):
        for wx in range(GX):
            steps += E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=30_000_000)['steps']
    return bytes(ka), init, a.copy(), steps


ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'ffwd2_s%d' % seed
rc, msg, out = D.gpu(module, SYM, tag, ka, init, (GX, GY))
if out is not None:
    _m = re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg); V.ARENA = int(_m[1], 16)
    ka2, init2, ref, steps = emulate()
    changed = np.nonzero(ref != init)[0]
row = dict(kernel=SYM, seed=seed, grid=(GX, GY), emu_steps=steps, emu_bytes_written=int(len(changed)),
           gpu_rc=rc, gpu=msg, written_slots=sorted({int(i // V.SLOT) for i in changed}))
if out is None: row['status'] = 'GPU_FAIL'
else:
    diff = np.nonzero(out != ref)[0]; row['mismatches'] = int(len(diff)); row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
    if len(diff): row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:6]]
    row['out_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
print(json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
