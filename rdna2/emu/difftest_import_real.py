"""Differential test of k_import on REAL captured Cyberpunk colour pixels (format=0, confirmed RGBA16F decode by
verify_import_fmt0.py). Crops a tile from the real capture (row_bytes=13656=1707*8, no padding) so a Python-emulator
preflight stays fast; the GPU dispatch covers the same tile. Reference recomputed at the device's actual base.
usage: difftest_import_real.py <color.bin> <src_w> <src_h> [tile_x tile_y tile_w tile_h] [mode] [seed]"""
import sys, os, json, struct, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z8k_import12ImportParams'
color_path = sys.argv[1]; SRC_W, SRC_H = int(sys.argv[2]), int(sys.argv[3])
tx, ty, TW, TH = (int(x) for x in sys.argv[4:8]) if len(sys.argv) > 7 else (400, 300, 32, 24)
mode = int(sys.argv[8]) if len(sys.argv) > 8 else 0
seed = int(sys.argv[9]) if len(sys.argv) > 9 else 1
full = np.fromfile(color_path, np.uint8).reshape(SRC_H, SRC_W, 8)   # RGBA16F, 8 B/pixel, tight row pitch
tile = np.ascontiguousarray(full[ty:ty + TH, tx:tx + TW])            # real captured pixels
PH, PW = TH, TW    # padded == tile size here (already a multiple of nothing in particular; k_import clamps OOB reads)
pitch = TW * 8
grid = ((PW + 255) // 256, PH)

def kernarg():
    ka = bytearray(0x130)
    struct.pack_into('<Q', ka, 0x00, V.ARENA)
    struct.pack_into('<iiiiii', ka, 0x08, pitch, 0, TH, TW, PH, PW)   # format = 0 (RGBA16F)
    struct.pack_into('<Q', ka, 0x20, V.ARENA + V.SLOT)
    struct.pack_into('<if', ka, 0x28, mode, 1.0)
    struct.pack_into('<III', ka, 0x30, grid[0], grid[1], 1)
    struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
    return ka

def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM}); g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    a[:tile.size] = tile.reshape(-1)
    init = a.copy(); steps = 0; lds = D.group_size(SYM)
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            steps += E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=2_000_000)['steps']
    return bytes(ka), init, a.copy(), steps

ka, init, ref, steps = emulate()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
tag = 'import_real_t%d_%d_%dx%d_m%d_s%d' % (tx, ty, TW, TH, mode, seed)
rc, msg, out = D.gpu(module, SYM, tag, ka, init, grid)
if out is not None:
    import re
    V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
    ka2, init2, ref, steps = emulate()
    changed = np.nonzero(ref != init)[0]
row = dict(kernel=SYM, tile=(tx, ty, TW, TH), mode=mode, seed=seed, grid=grid, emu_steps=steps, emu_bytes_written=int(len(changed)), gpu_rc=rc, gpu=msg)
if out is None: row['status'] = 'GPU_FAIL'
else:
    diff = np.nonzero(out != ref)[0]; row['mismatches'] = int(len(diff)); row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
    if len(diff): row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:6]]
    row['out_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
    r = ref[V.SLOT:V.SLOT + PH * PW * 12].view('<f4'); o = out[V.SLOT:V.SLOT + PH * PW * 12].view('<f4')
    row['float_finite'] = dict(ref=float(np.isfinite(r).mean()), gpu=float(np.isfinite(o).mean()))
    row['float_range'] = dict(ref=[float(r[np.isfinite(r)].min()), float(r[np.isfinite(r)].max())] if np.isfinite(r).any() else None)
    ordr = lambda x: (lambda i: np.where(i < 0, np.int64(-2**31) - i, i))(x.view(np.int32).astype(np.int64))
    fin = np.isfinite(r) & np.isfinite(o)
    ulp = np.abs(ordr(r) - ordr(o))
    row['ulp'] = dict(max=int(ulp[fin].max()) if fin.any() else None, mean=float(ulp[fin].mean()) if fin.any() else None,
                       differing=int((r != o).sum()))
print(json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
