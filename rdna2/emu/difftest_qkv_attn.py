"""GPU hardware verification for k_qkv_attn (AttnParams), using the kernarg contract recovered by reading the
real host launcher (see VARPARAMS_HOST_CONTRACT.md's 'k_qkv_attn (AttnParams): RESOLVED' section):
  +0x00 input ptr, +0x08 output ptr, +0x10 weight ptr (block's layer2 = qkv_weight+attn_scale+attn_bias),
  +0x18 H (i32), +0x1c W (i32), +0x20 origin X (i32), +0x24 origin Y (i32), +0x34 scalar (channel-count-like).
Weight: real block23.layer2.layer bytes (C=512, the first block of the post-encoder window-attention stage that
continues the already-GPU-verified block0-22 chain). Input/output activations are still random (structural
verification only - not yet chained to real pixel data). Kernel descriptor enables both dispatch_ptr and
kernarg_segment_ptr, same convention as the pre-block kernel (see difftest_pre.py).
usage: difftest_qkv_attn.py [seed]   (inside sandbox.py)
"""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z10k_qkv_attn10AttnParams'
KSIZE = 296
NSLOT = 6
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
H, W = 4, 4          # stage-5 spatial size, matches block22's already-GPU-verified pooled output
GX, GY = 1, 1        # (4 - 0 + 7)//8 = 1 window
WEIGHT_PATH = D.ROOT / 'build' / 'weights' / 'block23_layer2.bin'


def kernarg():
    ka = bytearray(KSIZE)
    S = lambda k: V.ARENA + k * V.SLOT
    struct.pack_into('<Q', ka, 0x00, S(0))
    struct.pack_into('<Q', ka, 0x08, S(1))
    struct.pack_into('<Q', ka, 0x10, S(2))
    struct.pack_into('<i', ka, 0x18, H)
    struct.pack_into('<i', ka, 0x1c, W)
    struct.pack_into('<i', ka, 0x20, 0)
    struct.pack_into('<i', ka, 0x24, 0)
    struct.pack_into('<i', ka, 0x34, 512)
    struct.pack_into('<III', ka, KSIZE - 20, GX, GY, 1)
    struct.pack_into('<HHH', ka, KSIZE - 8, 256, 1, 1)
    return ka


def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM}); g, KA = V.build(seed, ka, NSLOT)
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 11)
    a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)
    w = np.frombuffer(WEIGHT_PATH.read_bytes(), np.uint8)
    a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
    init = a.copy(); steps = 0
    DP = 0x7100_0000_0000
    pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
    struct.pack_into('<III', pkt, 12, GX * 256, GY, 1); struct.pack_into('<II', pkt, 24, 64, 58368)
    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
    for wy in range(GY):
        for wx in range(GX):
            steps += E.run_workgroup(prog, g, 58368, 256, {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: wy}, max_steps=30_000_000)['steps']
    return bytes(ka), init, a.copy(), steps


ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'qkv_attn_s%d' % seed
rc, msg, out = D.gpu(module, SYM, tag, ka, init, (GX, GY))
if out is not None:
    _m = re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg); V.ARENA = int(_m[1], 16)
    ka2, init2, ref, steps = emulate()
    changed = np.nonzero(ref != init)[0]
row = dict(kernel=SYM, seed=seed, hw=(H, W), grid=(GX, GY), emu_steps=steps, emu_bytes_written=int(len(changed)),
           gpu_rc=rc, gpu=msg, written_slots=sorted({int(i // V.SLOT) for i in changed}))
if out is None: row['status'] = 'GPU_FAIL'
else:
    diff = np.nonzero(out != ref)[0]; row['mismatches'] = int(len(diff)); row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
    if len(diff): row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:6]]
    row['out_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
print(json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
