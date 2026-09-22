"""Chain the pre-block and all four stage-1 encoder blocks (k_swin_var<32,*>) on a real captured Cyberpunk tile.
Extends chain_import_preblock_real.py: real pixels -> k_import -> pre-block (block0, true-variant) -> block1..block4
(false-variant, flags/origin-mode per VARPARAMS_HOST_CONTRACT.md's stage-1 schedule). Every stage's GPU output is
independently checked against the emulator at the device's real allocation base before being fed to the next stage.
usage: chain_encoder_stage1_real.py <color.bin> <src_w> <src_h> <tile_x> <tile_y> <N>   (N x N, <=64 for the 1 MiB scratch slot)
weights: rdna2/build/weights/block0.bin .. block4.bin must exist."""
import sys, os, json, struct, hashlib, re
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

IMPORT_SYM = '_Z8k_import12ImportParams'
PRE_SYM, PRE_LDS = D.SYMS['32_1']
ENC_SYM, ENC_LDS = D.SYMS['32_0']
ENC_LDS = ENC_LDS or D.group_size(ENC_SYM)
color_path, SRC_W, SRC_H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tx, ty, N = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
assert N % 8 == 0
WEIGHTS_DIR = D.ROOT / 'build' / 'weights'
full = np.fromfile(color_path, np.uint8).reshape(SRC_H, SRC_W, 8)
tile = np.ascontiguousarray(full[ty:ty + N, tx:tx + N])
IN_SLOT = V.PTR_FIELDS.index(0x40)
results = []


def run_and_verify(sym, lds, kernarg_fn, patch_arena, grid, tag):
    """kernarg_fn() -> kernarg bytes (using current V.ARENA); patch_arena(a) writes real inputs into the fresh arena."""
    def emulate():
        import time as _t; _t0 = _t.time()
        ka = kernarg_fn()
        print('  [%s] load_program...' % tag, flush=True)
        prog = E.load_program(R.DIS, {sym}); g, KA = V.build(1, ka, len(V.PTR_FIELDS))
        print('  [%s] load_program done %.1fs' % (tag, _t.time() - _t0), flush=True)
        a = g.regions[1].arr
        patch_arena(a)
        init = a.copy(); steps = 0
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                steps += E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)['steps']
            print('  [%s] row wy=%d/%d done, %.1fs elapsed' % (tag, wy, grid[1], _t.time() - _t0), flush=True)
        print('  [%s] emulate() total %.1fs' % (tag, _t.time() - _t0), flush=True)
        return bytes(ka), init, a.copy(), steps

    kab, init, ref, steps = emulate()
    module = D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.co')
    print('  [%s] calling GPU dispatch...' % tag, flush=True)
    rc, msg, out = D.gpu(module, sym, tag, kab, init, grid)
    print('  [%s] GPU dispatch returned rc=%s' % (tag, rc), flush=True)
    if out is not None:
        V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
        kab, init, ref, steps = emulate()
    diff = np.nonzero(out != ref)[0] if out is not None else None
    row = dict(stage=tag, rc=rc, gpu=msg, mismatches=int(len(diff)) if diff is not None else None,
               bytes_written=int((ref != init).sum()),
               status='PASS' if out is not None and len(diff) == 0 and (ref != init).sum() > 0 else 'FAIL')
    results.append(row)
    print('STAGE', tag, json.dumps(row), flush=True)
    if row['status'] != 'PASS':
        print(json.dumps(dict(chain=results))); raise SystemExit(1)
    return ref   # the verified reference == what a subsequent stage should consume


# ---- stage 0: k_import (format=0/RGBA16F) ----
pitch = N * 8; grid0 = ((N + 255) // 256, N)


def kernarg0():
    ka = bytearray(0x130)
    struct.pack_into('<Q', ka, 0x00, V.ARENA)
    struct.pack_into('<iiiiii', ka, 0x08, pitch, 0, N, N, N, N)
    struct.pack_into('<Q', ka, 0x20, V.ARENA + V.SLOT)
    struct.pack_into('<if', ka, 0x28, 0, 1.0)
    struct.pack_into('<III', ka, 0x30, grid0[0], grid0[1], 1)
    struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
    return ka


def patch0(a):
    a[:tile.size] = tile.reshape(-1)


ref0 = run_and_verify(IMPORT_SYM, D.group_size(IMPORT_SYM), kernarg0, patch0, grid0, 'import')
rgb = ref0[V.SLOT:V.SLOT + N * N * 12].view('<f4').copy()   # real, GPU-verified network-input floats

# ---- pre-block (block0, k_swin_var<32,true>, flags 0x14) ----
grid_pb = ((N + 7) // 8, (N + 7) // 8)
w0 = np.frombuffer((WEIGHTS_DIR / 'block0.bin').read_bytes(), np.uint8)


def kernarg_pb():
    ka = V.make_kernarg(H=N, W=N, offy=0, offx=0, flags=0x14, grid=grid_pb)
    struct.pack_into('<Q', ka, 0x00, 0)
    for off in (0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68): struct.pack_into('<I', ka, off, 0)
    struct.pack_into('<f', ka, 0x50, 1.0)
    for off in range(0x70, 0xa0, 8): struct.pack_into('<Q', ka, off, 0)
    return ka


def patch_pb(a):
    a[2 * V.SLOT:2 * V.SLOT + len(w0)] = w0
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + rgb.nbytes] = rgb.view(np.uint8)


ref_pb = run_and_verify(PRE_SYM, PRE_LDS, kernarg_pb, patch_pb, grid_pb, 'preblock')
# The pre-block (and every regular block) writes its output to kernarg +0x08 = pointer-field slot index 1
# (VARPARAMS_HOST_CONTRACT.md). Between blocks the real host just re-targets pointers (ping-pong), it does not
# reinterpret the bytes - the exact C-channel/4x4-tile encoding is only partly understood (see the host contract's
# "inferred" note), so we copy the WHOLE verified output slot forward untouched rather than guess a byte count.
act = ref_pb[1 * V.SLOT:2 * V.SLOT].copy()

# ---- stage-1 encoder blocks 1-4 (k_swin_var<32,false>), each fed the previous block's verified output ----
SCHEDULE = [  # (block_weight_file, flags, mode/origin)
    ('block1.bin', 1, (0, 0)),
    ('block2.bin', 0, (-4, -4)),
    ('block3.bin', 0, (-4, 0)),
    ('block4.bin', 4, (0, -4)),
]
cur = act
for i, (wfile, flags, (ox, oy)) in enumerate(SCHEDULE, start=1):
    w = np.frombuffer((WEIGHTS_DIR / wfile).read_bytes(), np.uint8)
    grid_b = ((N + 7) // 8, (N + 7) // 8)
    cur_local = cur   # bind for closures

    def kernarg_b(flags=flags, ox=ox, oy=oy, grid_b=grid_b):
        return V.make_kernarg(H=N, W=N, offy=ox, offx=oy, flags=flags, grid=grid_b)

    def patch_b(a, w=w, cur_local=cur_local):
        a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
        a[2 * V.SLOT:2 * V.SLOT + len(w)] = w


    ref_b = run_and_verify(ENC_SYM, ENC_LDS, kernarg_b, patch_b, grid_b, 'block%d' % i)
    cur = ref_b[1 * V.SLOT:2 * V.SLOT].copy()

print(json.dumps(dict(chain=results)))
raise SystemExit(0 if all(r['status'] == 'PASS' for r in results) else 1)
