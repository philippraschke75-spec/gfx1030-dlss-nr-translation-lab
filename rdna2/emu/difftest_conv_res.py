"""GPU hardware verification for k_conv_res (_Z10k_conv_res10ConvParams), the projection kernel needed to
complete a C=512 block's forward pass (FFN and attention are already resolved; this is the reference's
documented "Proj(WindowAttention(...))" step). Single kernarg-pointer convention. ConvParams's explicit struct
is 40 bytes like the other kernels, but the exhaustive kernarg-read trace (trace_conv_res_kernarg.py) showed it
only reads +0x00 (32 B = 4 pointers) and +0x34 (hidden_group_size_x, correctly populated, not a real field) -
no H/W scalars at all. Pointer order was guessed as input/output/weight/? matching the k_ffwd/k_swin_var convention - this test
PASSES (proving the translation itself is correct) but `written_slots` came back as [2], meaning the real
output lands where this guess placed the "weight" pointer, so the guessed role mapping is WRONG. This test
only establishes structural GPU==emulator equivalence, not the real pointer semantics - see
VARPARAMS_HOST_CONTRACT.md for what's still open.
Weight: real block23.layer3.layer bytes (263,168 B = 512*512 projection_weight + 512*2 attn_cos_skip, an exact
size match for C=512).
usage: difftest_conv_res.py [seed]   (inside sandbox.py)
"""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z10k_conv_res10ConvParams'
KSIZE = 296
NSLOT = 6
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
GX, GY = 1, 1
WEIGHT_PATH = D.ROOT / 'build' / 'weights' / 'block23_layer3.bin'
LDS = 8192


def kernarg():
    ka = bytearray(KSIZE)
    S = lambda k: V.ARENA + k * V.SLOT
    struct.pack_into('<Q', ka, 0x00, S(0))
    struct.pack_into('<Q', ka, 0x08, S(1))
    struct.pack_into('<Q', ka, 0x10, S(2))
    struct.pack_into('<Q', ka, 0x18, S(3))
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
    rng = np.random.default_rng(seed + 17)
    a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)
    w = np.frombuffer(WEIGHT_PATH.read_bytes(), np.uint8)
    a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
    init = a.copy(); steps = 0
    for wy in range(GY):
        for wx in range(GX):
            steps += E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=30_000_000)['steps']
    return bytes(ka), init, a.copy(), steps


ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'conv_res_s%d' % seed
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
