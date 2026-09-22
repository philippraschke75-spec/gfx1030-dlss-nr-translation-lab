"""Differential test for k_pre_block_1h_32_fp8 (PreParams, partially recovered layout - see VARPARAMS_HOST_CONTRACT.md).
Input: float RGB in [0,1) (what k_import mode>=1 produces); weights: real WEIGHTS_HT block0 record (DLSSNR_WEIGHT_BLOB).
usage: difftest_pre.py [seed] [H W]   (inside sandbox.py)   Unknown +0x38/+0x40 pointers get arena slots 3/4 (random bytes)."""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z21k_pre_block_1h_32_fp89PreParams'
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (16, 16)
grid = ((W + 7) // 8, (H + 7) // 8)

def kernarg():
    ka = bytearray(0x100)
    S = lambda k: V.ARENA + k * V.SLOT
    struct.pack_into('<QQQ', ka, 0x00, S(0), S(1), S(2))
    struct.pack_into('<ii', ka, 0x18, H, W)
    struct.pack_into('<fi', ka, 0x20, 1.0, 0)
    struct.pack_into('<ff', ka, 0x28, 0.0, 0.0)
    struct.pack_into('<i', ka, 0x30, 0)
    struct.pack_into('<QQ', ka, 0x38, S(3), S(4))
    struct.pack_into('<ff', ka, 0x48, 0.0, 0.0)
    struct.pack_into('<III', ka, 0x58, grid[0], grid[1], 1)
    struct.pack_into('<HHH', ka, 0x64, 256, 1, 1)
    return ka

def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM, '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'}); g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 7)
    a[:H * W * 12] = rng.random(H * W * 3, dtype=np.float32).view(np.uint8)   # RGB floats in [0,1)
    init = a.copy(); steps = 0; lds = D.group_size(SYM)
    DP = 0x7100_0000_0000                                  # modeled AQL dispatch packet (kd enables dispatch_ptr in s[0:1])
    pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
    struct.pack_into('<III', pkt, 12, grid[0] * 256, grid[1], 1); struct.pack_into('<II', pkt, 24, 64, lds)
    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            steps += E.run_workgroup(prog, g, lds, 256, {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)['steps']
    return bytes(ka), init, a.copy(), steps

ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'pre_s%d_%dx%d' % (seed, H, W)
rc, msg, out = D.gpu(module, SYM, tag, ka, init, grid)
if out is not None:              # kernels may consume pointer bits numerically: recompute the reference at the device's arena base
    _m = re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg); V.ARENA = int(_m[1], 16)
    ka2, init2, ref, steps = emulate()
    assert ka2 == ka or True and np.array_equal(init2, init), 'device-base fixture differs'
    changed = np.nonzero(ref != init)[0]
row = dict(kernel=SYM, seed=seed, hw=(H, W), grid=grid, emu_steps=steps, emu_bytes_written=int(len(changed)), gpu_rc=rc, gpu=msg,
           written_slots=sorted({int(i // V.SLOT) for i in changed}))
if out is None: row['status'] = 'GPU_FAIL'
else:
    diff = np.nonzero(out != ref)[0]; row['mismatches'] = int(len(diff)); row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
    if len(diff): row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:6]]
    row['out_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
print(json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
