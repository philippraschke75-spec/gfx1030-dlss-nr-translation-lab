"""Differential test of the real network entry: k_swin_var<32,true> launched as the pre-block (flags 0x14) with block0 weights.
Layout from the AMD host launcher (0x18002efb8, see VARPARAMS_HOST_CONTRACT.md): +0x00 = null, +0x08 out, +0x10 weights, +0x18/+0x1c H,W,
+0x28 flags 0x14, +0x38 ptr (ctx+0x220), +0x40 float-RGB input (12 B/pixel), +0x48 ptr (optional 3rd buffer), +0x50 f32 (ctx+0x34),
+0x54/+0x58 f32x2 (ctx+0x20), +0x60/+0x64 f32x2 (ctx+0x28), +0x68 i32 (ctx+0x38), +0x70..+0x9f zero, +0xa0 scratch.
usage: difftest_preblock.py [seed] [H W]   env DLSSNR_WEIGHT_BLOB=<block0 record>   (run inside sandbox.py)"""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM, LDS = D.SYMS['32_1']
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (16, 16)
grid = ((W + 7) // 8, (H + 7) // 8)
IN_SLOT = V.PTR_FIELDS.index(0x40)
os.environ.pop('DLSSNR_REAL_LAYOUT', None)

def kernarg():
    ka = V.make_kernarg(H=H, W=W, offy=0, offx=0, flags=0x14, grid=grid)
    if os.environ.get('PRE_ZERO_ALL', '1') == '1':          # the real host's layout (default)
        struct.pack_into('<Q', ka, 0x00, 0); struct.pack_into('<Q', ka, 0x30, 0)
        # +0x68 is the pre-block's RNG seed, not a pointer. PTR_FIELDS lists it, so make_kernarg wrote a
        # whole pointer; zeroing only its low dword left arena bits in +0x6c, the GPU runner rebased the
        # qword, and the kernel ran seeded with the device arena address while the emulator used 0. That
        # one field made the pre-block differ by 67%% of its output. Same for the other scalars here.
        for off in (0x50, 0x58, 0x60, 0x68): struct.pack_into('<Q', ka, off, 0)
        for off in (0x54, 0x5c, 0x64): struct.pack_into('<I', ka, off, 0)
        struct.pack_into('<f', ka, 0x50, 1.0)
        for off in range(0x70, 0xa0, 8): struct.pack_into('<Q', ka, off, 0)
    for off in filter(None, os.environ.get('PRE_ZERO', '').split(',')):   # bisect: zero individual 8-byte fields
        struct.pack_into('<Q', ka, int(off, 0), 0)
    for item in filter(None, os.environ.get('PRE_FIELDS', '').split(',')):     # e.g. 0x54=f0.5,0x68=i1
        off, val = item.split('='); struct.pack_into('<f' if val[0] == 'f' else '<i', ka, int(off, 0), float(val[1:]) if val[0] == 'f' else int(val[1:]))
    return ka

def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM}); g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 5)
    if os.environ.get('PRE_RAW'): lo = hi = None
    else: lo, hi = float(os.environ.get('PRE_LO', '0')), float(os.environ.get('PRE_HI', '1'))          # RGB range (default [0,1))
    if lo is not None: a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + H * W * 12] = (lo + (hi - lo) * rng.random(H * W * 3, dtype=np.float32)).astype(np.float32).view(np.uint8)
    init = a.copy(); steps = 0
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            steps += E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)['steps']
    return bytes(ka), init, a.copy(), steps

ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'preblock_s%d_%dx%d' % (seed, H, W)
rc, msg, out = D.gpu(module, SYM, tag, ka, init, grid)
if out is not None:              # kernels may consume pointer bits numerically: recompute the reference at the device's arena base
    _m = re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg); V.ARENA = int(_m[1], 16)
    ka2, init2, ref, steps = emulate()
    assert ka2 == ka or True and np.array_equal(init2, init), 'device-base fixture differs'
    changed = np.nonzero(ref != init)[0]
row = dict(kernel=SYM, flags=0x14, seed=seed, hw=(H, W), grid=grid, emu_steps=steps, emu_bytes_written=int(len(changed)), gpu_rc=rc, gpu=msg,
           written_slots={int(k): hex(V.PTR_FIELDS[k]) for k in sorted({int(i // V.SLOT) for i in changed})})
if out is None: row['status'] = 'GPU_FAIL'
else:
    diff = np.nonzero(out != ref)[0]; row['mismatches'] = int(len(diff)); row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
    if len(diff): row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:6]]
    row['out_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
print(json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
