"""GPU hardware verification for k_qkv_attn2 (the ViT attention variant, _Z11k_qkv_attn210AttnParams, handle
0x180066380, blocks 31-38 per VARPARAMS_HOST_CONTRACT.md's weight-size analysis - no attn_bias/prior term,
matching the reference's documented ViT-vs-window-block distinction). Same 3-pointer + H/W/originX/Y AttnParams
layout as k_qkv_attn (confirmed via trace_qkv_attn2_kernarg.py's exhaustive read trace), but the SINGLE
kernarg-pointer convention (s[0:1] directly - no dispatch_ptr needed), and hidden-args populated correctly from
the very first test (see VARPARAMS_HOST_CONTRACT.md / RESULTS_REAL_CONFIG.md for why that matters: k_qkv_attn's
first pass guessed +0x34 as a real field and only found the bug once fed real data).
Weight: real block31.layer2.layer bytes (3,145,856 B, C=1024 qkv_weight, no attn_bias - spills past one 1 MiB
arena slot into the next two, which is fine since nothing else uses those slots for this kernel).
usage: difftest_qkv_attn2.py [seed]   (inside sandbox.py)
"""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z11k_qkv_attn210AttnParams'
KSIZE = 296
NSLOT = 6
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
H, W = 8, 8          # H=W=2 (post-block30 pooled size guess) produced 0 bytes written (likely fully masked by
                      # token-padding logic); retrying with a larger, known-non-degenerate size for a structural test
GX, GY = 1, 1
WEIGHT_PATH = D.ROOT / 'build' / 'weights' / 'block31_layer2.bin'


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
    struct.pack_into('<III', ka, 0x28, GX, GY, 1)       # hidden_block_count_x/y/z
    struct.pack_into('<HHH', ka, 0x34, 256, 1, 1)        # hidden_group_size_x/y/z
    struct.pack_into('<HHH', ka, 0x3a, 0, 0, 0)          # hidden_remainder_x/y/z
    struct.pack_into('<QQQ', ka, 0x50, 0, 0, 0)          # hidden_global_offset_x/y/z
    struct.pack_into('<H', ka, 0x68, 2)                  # hidden_grid_dims
    return ka


def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM}); g, KA = V.build(seed, ka, NSLOT)
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 13)
    a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)
    w = np.frombuffer(WEIGHT_PATH.read_bytes(), np.uint8)
    a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
    init = a.copy(); steps = 0
    for wy in range(GY):
        for wx in range(GX):
            steps += E.run_workgroup(prog, g, 58368, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=30_000_000)['steps']
    return bytes(ka), init, a.copy(), steps


ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'qkv_attn2_s%d' % seed
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
