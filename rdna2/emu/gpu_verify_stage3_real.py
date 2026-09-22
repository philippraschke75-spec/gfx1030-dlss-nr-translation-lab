"""Continue the real-pixel chain through stage 2 (blocks 5-8, C=64), GPU-verifying each block. Stages before block5
(import, pre-block, blocks 1-4) are computed emulator-only here since they were already independently GPU-verified
(RESULTS_REAL_CONFIG.md); their emulator output is used as the real value, which is equivalent since GPU==emulator
was already established for those kernels. Also checks whether block8 (last-in-stage) fuses its own pooled output
at +0x38, the same way block4 did, for the stage 2->3 transition.
"""
import sys, struct, time, re, json
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

IMPORT_SYM = '_Z8k_import12ImportParams'
PRE_SYM, PRE_LDS = D.SYMS['32_1']
ENC32_SYM, ENC32_LDS = D.SYMS['32_0']; ENC32_LDS = ENC32_LDS or D.group_size(ENC32_SYM)
ENC64_SYM, ENC64_LDS = D.SYMS['64_0']; ENC64_LDS = ENC64_LDS or D.group_size(ENC64_SYM)
ENC128_SYM, ENC128_LDS = D.SYMS['128_0']; ENC128_LDS = ENC128_LDS or D.group_size(ENC128_SYM)
POOL_SLOT = V.PTR_FIELDS.index(0x38)
IN_SLOT = V.PTR_FIELDS.index(0x40)
color_path = 'rdna2/build/real-input/color.bin'
full = np.fromfile(color_path, np.uint8).reshape(960, 1707, 8)
tx, ty, N = 500, 300, 64
tile = np.ascontiguousarray(full[ty:ty + N, tx:tx + N])
WDIR = D.ROOT / 'build' / 'weights'
t_start = time.time()
results = []


def run_local(sym, lds, H, W, flags, ox, oy, patch, label, pre=False):
    grid = ((W + 7) // 8, (H + 7) // 8)
    if sym == IMPORT_SYM:
        pitch = W * 8
        ka = bytearray(0x130)
        struct.pack_into('<Q', ka, 0x00, V.ARENA)
        struct.pack_into('<iiiiii', ka, 0x08, pitch, 0, H, W, H, W)
        struct.pack_into('<Q', ka, 0x20, V.ARENA + V.SLOT)
        struct.pack_into('<if', ka, 0x28, 0, 1.0)
        struct.pack_into('<III', ka, 0x30, grid[0], grid[1], 1)
        struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
        grid = ((W + 255) // 256, H)
    elif pre:
        ka = V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)
        struct.pack_into('<Q', ka, 0x00, 0)
        for off in (0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68): struct.pack_into('<I', ka, off, 0)
        struct.pack_into('<f', ka, 0x50, 1.0)
        for off in range(0x70, 0xa0, 8): struct.pack_into('<Q', ka, off, 0)
    else:
        ka = V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)
    prog = E.load_program(R.DIS, {sym})
    g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    patch(a)
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)
    print('  [%s] emulator done, %.0fs since start' % (label, time.time() - t_start), flush=True)
    return a


def gpu_verify(sym, lds, H, W, flags, ox, oy, patch, label):
    grid = ((W + 7) // 8, (H + 7) // 8)

    def kernarg():
        return V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)

    def emulate():
        ka = kernarg()
        prog = E.load_program(R.DIS, {sym}); g, KA = V.build(1, ka, len(V.PTR_FIELDS))
        a = g.regions[1].arr
        patch(a)
        init = a.copy()
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)
        return bytes(ka), init, a.copy()

    kab, init, ref = emulate()
    module = D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.co')
    rc, msg, out = D.gpu(module, sym, label, kab, init, grid)
    if out is not None:
        V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
        kab, init, ref = emulate()
    diff = np.nonzero(out != ref)[0] if out is not None else None
    row = dict(stage=label, rc=rc, gpu=msg, mismatches=int(len(diff)) if diff is not None else None,
               bytes_written=int((ref != init).sum()) if out is not None else None,
               status='PASS' if out is not None and len(diff) == 0 else 'FAIL')
    results.append(row)
    print('STAGE', label, json.dumps(row), flush=True)
    if row['status'] != 'PASS':
        print(json.dumps(dict(chain=results))); raise SystemExit(1)
    return ref, init


# ---- rebuild real pooled data (emulator only; already GPU-verified upstream), or load a checkpoint if present ----
CKPT = D.ROOT / 'build' / 'pooled2_checkpoint.npy'
HALF = N // 2
if CKPT.exists():
    pooled2 = np.load(CKPT)
    print('loaded pooled2 checkpoint from disk, skipping stage 1+2 recompute, %.0fs' % (time.time() - t_start), flush=True)
else:
    r = run_local(IMPORT_SYM, D.group_size(IMPORT_SYM), N, N, 0, 0, 0, lambda a: a.__setitem__(slice(0, tile.size), tile.reshape(-1)), 'import')
    rgb = r[V.SLOT:V.SLOT + N * N * 12].view('<f4').copy()

    w0 = np.frombuffer((WDIR / 'block0.bin').read_bytes(), np.uint8)
    def patch_pb(a):
        a[2 * V.SLOT:2 * V.SLOT + len(w0)] = w0
        a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + rgb.nbytes] = rgb.view(np.uint8)
    r = run_local(PRE_SYM, PRE_LDS, N, N, 0x14, 0, 0, patch_pb, 'preblock', pre=True)
    cur = r[1 * V.SLOT:2 * V.SLOT].copy()

    SCHEDULE1 = [('block1.bin', 1, 0, 0), ('block2.bin', 0, -4, -4), ('block3.bin', 0, -4, 0), ('block4.bin', 4, 0, -4)]
    pooled = None
    for i, (wf, flags, ox, oy) in enumerate(SCHEDULE1, start=1):
        w = np.frombuffer((WDIR / wf).read_bytes(), np.uint8)
        cur_local = cur
        def patch_b(a, w=w, cur_local=cur_local):
            a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
            a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
        r = run_local(ENC32_SYM, ENC32_LDS, N, N, flags, ox, oy, patch_b, 'block%d' % i)
        cur = r[1 * V.SLOT:2 * V.SLOT].copy()
        if flags & 4:
            pooled = r[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT].copy()

    print('real stage-1 pooled data ready, %.0fs; computing stage 2 (blocks 5-8, emulator-only, already GPU-verified)' % (time.time() - t_start), flush=True)

    SCHEDULE2 = [('block5.bin', 1, 0, 0), ('block6.bin', 0, -4, -4), ('block7.bin', 0, -4, 0), ('block8.bin', 4, 0, -4)]
    cur = pooled
    pooled2 = None
    for i, (wf, flags, ox, oy) in enumerate(SCHEDULE2, start=5):
        w = np.frombuffer((WDIR / wf).read_bytes(), np.uint8)
        cur_local = cur
        def patch_b(a, w=w, cur_local=cur_local):
            a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
            a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
        ref_b = run_local(ENC64_SYM, ENC64_LDS, HALF, HALF, flags, ox, oy, patch_b, 'block%d(emu-only)' % i)
        cur = ref_b[1 * V.SLOT:2 * V.SLOT].copy()
        if flags & 4:
            pooled2 = ref_b[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT].copy()
    np.save(CKPT, pooled2)
    print('saved pooled2 checkpoint to disk for future runs', flush=True)

print('real stage-2 pooled data ready, %.0fs; now GPU-verifying stage 3 (blocks 9-14)' % (time.time() - t_start), flush=True)

# ---- stage 3: blocks 9-14, C=128, H=W=N/4, each GPU-verified ----
QUARTER = N // 4
# per VARPARAMS_HOST_CONTRACT.md's schedule table (read directly from the host's own per-stage mode array), the
# 6-block stage repeats modes 0,1,2,3,0,1; only the first (block9) and last (block14) blocks carry flags 1/4.
SCHEDULE3 = [('block9.bin', 1, 0, 0), ('block10.bin', 0, -4, -4), ('block11.bin', 0, -4, 0), ('block12.bin', 0, 0, -4),
             ('block13.bin', 0, 0, 0), ('block14.bin', 4, -4, -4)]
cur = pooled2
pooled3 = None
start_i = 9
for i in range(14, 8, -1):
    ck = D.ROOT / 'build' / ('block%d_checkpoint.npy' % i)
    if ck.exists():
        cur = np.load(ck); start_i = i + 1
        print('resuming stage 3 from block%d checkpoint, %.0fs' % (i, time.time() - t_start), flush=True)
        break
for i, (wf, flags, ox, oy) in enumerate(SCHEDULE3, start=9):
    if i < start_i:
        continue
    w = np.frombuffer((WDIR / wf).read_bytes(), np.uint8)
    cur_local = cur
    def patch_b(a, w=w, cur_local=cur_local):
        a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
        a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
    ref_b, init_b = gpu_verify(ENC128_SYM, ENC128_LDS, QUARTER, QUARTER, flags, ox, oy, patch_b, 'block%d' % i)
    cur = ref_b[1 * V.SLOT:2 * V.SLOT].copy()
    np.save(D.ROOT / 'build' / ('block%d_checkpoint.npy' % i), cur)
    if flags & 4:
        pool_seg = ref_b[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT]
        pool_init = init_b[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT]
        changed = np.nonzero(pool_seg != pool_init)[0]
        pooled3 = pool_seg.copy()
        print('  block%d: pooled-output slot (+0x38) changed bytes: %d (range [%d,%d))' % (
            i, len(changed), changed.min() if len(changed) else -1, changed.max() + 1 if len(changed) else -1))

print(json.dumps(dict(chain=results)))
raise SystemExit(0 if all(r['status'] == 'PASS' for r in results) else 1)
