"""Differential test (emulator reference vs translated gfx1030 kernel) for k_import (ImportParams).
ImportParams (from the AMD host launcher 0x18002d3b0 and the kernel prologue; see VARPARAMS_HOST_CONTRACT.md):
 +0x00 src ptr | +0x08 pitch(B) | +0x0c format | +0x10 srcH | +0x14 srcW | +0x18 padH | +0x1c padW | +0x20 dst ptr | +0x28 mode | +0x2c f32 scale
 hidden: +0x30 block counts, +0x3c group size.  Grid = (ceil(padW/256), padH), 256 threads.
usage: difftest_import.py <format> <mode> [seed] [srcH srcW padH padW]   (run inside sandbox.py)"""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z8k_import12ImportParams'
ROOT = D.ROOT
fmt, mode = int(sys.argv[1]), int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H, W, PH, PW = (int(x) for x in sys.argv[4:8]) if len(sys.argv) > 7 else (13, 21, 16, 32)
scale = float(os.environ.get('IMPORT_SCALE', '1.5'))
bpp = {0: 4, 1: 4, 2: 4, 3: 8, 4: 4, 5: 4, 6: 4}.get(fmt, 4)   # bytes/pixel guess; harmless (src slot is 1 MiB)
pitch = W * bpp
grid = ((PW + 255) // 256, PH)

def kernarg():
    ka = bytearray(0x130)
    struct.pack_into('<Q', ka, 0x00, V.ARENA)
    struct.pack_into('<iiiiii', ka, 0x08, pitch, fmt, H, W, PH, PW)
    struct.pack_into('<Q', ka, 0x20, V.ARENA + V.SLOT)
    struct.pack_into('<if', ka, 0x28, mode, scale)
    struct.pack_into('<III', ka, 0x30, grid[0], grid[1], 1)
    struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
    return ka

def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM}); g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 100)
    a[:V.SLOT] = rng.integers(0, 256, V.SLOT, dtype=np.uint8)          # arbitrary source image bytes
    init = a.copy(); steps = 0
    lds = D.group_size(SYM)
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            steps += E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=2_000_000)['steps']
    return bytes(ka), init, a.copy(), steps

ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'import_f%d_m%d_s%d_%dx%d_p%dx%d' % (fmt, mode, seed, H, W, PH, PW)
rc, msg, out = D.gpu(module, SYM, tag, ka, init, grid)
if out is not None:              # kernels may consume pointer bits numerically: recompute the reference at the device's arena base
    _m = re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg); V.ARENA = int(_m[1], 16)
    ka2, init2, ref, steps = emulate()
    assert ka2 == ka or True and np.array_equal(init2, init), 'device-base fixture differs'
    changed = np.nonzero(ref != init)[0]
row = dict(kernel=SYM, fmt=fmt, mode=mode, seed=seed, src=(H, W), pad=(PH, PW), grid=grid, emu_steps=steps, emu_bytes_written=int(len(changed)),
           emu_distinct=int(len(np.unique(ref[changed]))) if len(changed) else 0, gpu_rc=rc, gpu=msg)
if out is None: row['status'] = 'GPU_FAIL'
else:
    diff = np.nonzero(out != ref)[0]; row['mismatches'] = int(len(diff))
    row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
    if len(diff): row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:5]]
    row['out_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
    n = PH * PW * 3                                                     # dst = padded H*W*3 float32 (12 B/pixel)
    r = ref[V.SLOT:V.SLOT + n * 4].view(np.float32).astype(np.float64); g = out[V.SLOT:V.SLOT + n * 4].view(np.float32).astype(np.float64)
    ordr = lambda x: (lambda i: np.where(i < 0, np.int64(-2**31) - i, i))(x.astype(np.float32).view(np.int32).astype(np.int64))
    ulp = np.abs(ordr(r) - ordr(g)); fin = np.isfinite(r) & np.isfinite(g)
    row['float_cmp'] = dict(elements=int(n), differing=int((r != g).sum()), max_ulp=int(ulp[fin].max()) if fin.any() else None,
                            max_rel=float((np.abs(r - g)[fin] / np.maximum(np.abs(r[fin]), 1e-30)).max()) if fin.any() else None,
                            nonfinite_ref=int((~np.isfinite(r)).sum()), nonfinite_gpu=int((~np.isfinite(g)).sum()),
                            ref_range=[float(np.nanmin(r[np.isfinite(r)])), float(np.nanmax(r[np.isfinite(r)]))] if np.isfinite(r).any() else None)
print(json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
