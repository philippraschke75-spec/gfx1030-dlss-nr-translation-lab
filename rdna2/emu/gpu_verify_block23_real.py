"""Continue the real-pixel chain past the already-GPU-verified 4-stage encoder (import, pre-block, blocks 1-22,
all emulator-only here since already independently GPU-verified per RESULTS_REAL_CONFIG.md) into block23, the
first block of the C=512 window-attention stage. Uses k_qkv_attn's kernarg contract recovered and GPU-verified
in VARPARAMS_HOST_CONTRACT.md. GPU-verifies block23's k_qkv_attn dispatch against real chained input activations
(block22's pooled output) and real block23 weight bytes, checkpointing the pooled4 input so a rerun can skip the
whole upstream recompute.
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
ENC256_SYM, ENC256_LDS = D.SYMS['256_0']; ENC256_LDS = ENC256_LDS or D.group_size(ENC256_SYM)
QKV_ATTN_SYM = '_Z10k_qkv_attn10AttnParams'
POOL_SLOT = V.PTR_FIELDS.index(0x38)
IN_SLOT = V.PTR_FIELDS.index(0x40)
color_path = 'rdna2/build/real-input/color.bin'
full = np.fromfile(color_path, np.uint8).reshape(960, 1707, 8)
tx, ty, N = 500, 300, 64
tile = np.ascontiguousarray(full[ty:ty + N, tx:tx + N])
WDIR = D.ROOT / 'build' / 'weights'
t_start = time.time()


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


CKPT = D.ROOT / 'build' / 'pooled4_checkpoint.npy'
if CKPT.exists():
    pooled4 = np.load(CKPT)
    print('loaded pooled4 checkpoint from disk, skipping the whole encoder recompute, %.0fs' % (time.time() - t_start), flush=True)
else:
    r = run_local(IMPORT_SYM, D.group_size(IMPORT_SYM), N, N, 0, 0, 0, lambda a: a.__setitem__(slice(0, tile.size), tile.reshape(-1)), 'import')
    rgb = r[V.SLOT:V.SLOT + N * N * 12].view('<f4').copy()

    w0 = np.frombuffer((WDIR / 'block0.bin').read_bytes(), np.uint8)
    def patch_pb(a):
        a[2 * V.SLOT:2 * V.SLOT + len(w0)] = w0
        a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + rgb.nbytes] = rgb.view(np.uint8)
    r = run_local(PRE_SYM, PRE_LDS, N, N, 0x14, 0, 0, patch_pb, 'preblock', pre=True)
    cur = r[1 * V.SLOT:2 * V.SLOT].copy()

    STAGES = [
        (ENC32_SYM, ENC32_LDS, N, N, [('block1.bin', 1, 0, 0), ('block2.bin', 0, -4, -4), ('block3.bin', 0, -4, 0), ('block4.bin', 4, 0, -4)]),
        (ENC64_SYM, ENC64_LDS, N // 2, N // 2, [('block5.bin', 1, 0, 0), ('block6.bin', 0, -4, -4), ('block7.bin', 0, -4, 0), ('block8.bin', 4, 0, -4)]),
        (ENC128_SYM, ENC128_LDS, N // 4, N // 4, [('block9.bin', 1, 0, 0), ('block10.bin', 0, -4, -4), ('block11.bin', 0, -4, 0), ('block12.bin', 0, 0, -4),
                                                    ('block13.bin', 0, 0, 0), ('block14.bin', 4, -4, -4)]),
        (ENC256_SYM, ENC256_LDS, N // 8, N // 8, [('block15.bin', 1, 0, 0), ('block16.bin', 0, -4, -4), ('block17.bin', 0, -4, 0), ('block18.bin', 0, 0, -4),
                                                    ('block19.bin', 0, 0, 0), ('block20.bin', 0, -4, -4), ('block21.bin', 0, -4, 0), ('block22.bin', 4, 0, -4)]),
    ]
    pooled = None
    start_i = 1
    for i in range(22, 0, -1):
        ck = D.ROOT / 'build' / ('block%d_checkpoint.npy' % i)
        if ck.exists() and i not in (4, 8, 14, 22):   # last-in-stage checkpoints hold the regular output, not the pooled one the next stage needs - always recompute those
            cur = np.load(ck); start_i = i + 1
            print('resuming encoder chain from block%d checkpoint, %.0fs' % (i, time.time() - t_start), flush=True)
            break
    idx = 1
    for sym, lds, H, W, schedule in STAGES:
        for wf, flags, ox, oy in schedule:
            if idx < start_i:
                idx += 1
                continue
            w = np.frombuffer((WDIR / wf).read_bytes(), np.uint8)
            cur_local = cur
            def patch_b(a, w=w, cur_local=cur_local):
                a[0 * V.SLOT:0 * V.SLOT + len(cur_local)] = cur_local
                a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
            r = run_local(sym, lds, H, W, flags, ox, oy, patch_b, wf)
            cur = r[1 * V.SLOT:2 * V.SLOT].copy()
            np.save(D.ROOT / 'build' / ('block%d_checkpoint.npy' % idx), cur)
            if flags & 4:
                pooled = r[POOL_SLOT * V.SLOT:(POOL_SLOT + 1) * V.SLOT].copy()
            idx += 1
        cur = pooled
    pooled4 = pooled
    np.save(CKPT, pooled4)
    print('saved pooled4 checkpoint to disk for future runs, %.0fs' % (time.time() - t_start), flush=True)

print('real pooled4 (block22 output, C=512 H=W=4) ready, %.0fs; now GPU-verifying block23 (k_qkv_attn)' % (time.time() - t_start), flush=True)

# ---- block23: k_qkv_attn, real chained input, real weight ----
w23 = np.frombuffer((WDIR / 'block23_layer2.bin').read_bytes(), np.uint8)
KSIZE = 296
H, W = 4, 4
GX, GY = 1, 1


def kernarg23():
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


def emulate23():
    ka = kernarg23()
    prog = E.load_program(R.DIS, {QKV_ATTN_SYM}); g, KA = V.build(1, ka, 6)
    a = g.regions[1].arr
    a[:] = 0
    a[0 * V.SLOT:0 * V.SLOT + len(pooled4)] = pooled4
    a[2 * V.SLOT:2 * V.SLOT + len(w23)] = w23
    init = a.copy()
    DP = 0x7100_0000_0000
    pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
    struct.pack_into('<III', pkt, 12, GX * 256, GY, 1); struct.pack_into('<II', pkt, 24, 64, 58368)
    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
    steps = 0
    for wy in range(GY):
        for wx in range(GX):
            steps += E.run_workgroup(prog, g, 58368, 256, {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: wy}, max_steps=30_000_000)['steps']
    return bytes(ka), init, a.copy(), steps


ka, init, ref, steps = emulate23()
changed = np.nonzero(ref != init)[0]
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (QKV_ATTN_SYM + '.co')
rc, msg, out = D.gpu(module, QKV_ATTN_SYM, 'block23_real', ka, init, (GX, GY))
if out is not None:
    V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
    ka, init, ref, steps = emulate23()
    changed = np.nonzero(ref != init)[0]
diff = np.nonzero(out != ref)[0] if out is not None else None
row = dict(stage='block23_real', rc=rc, gpu=msg, mismatches=int(len(diff)) if diff is not None else None,
           bytes_written=int(len(changed)), status='PASS' if out is not None and len(diff) == 0 and len(changed) > 0 else 'FAIL')
if diff is not None and len(diff):
    row['first_diffs'] = [D.where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:12]]
    outslot = 1 * V.SLOT
    row['diff_range'] = [int(diff.min() - outslot), int(diff.max() - outslot)]
    row['diff_min_max_idx'] = [int(diff.min()), int(diff.max())]
print('STAGE block23_real', json.dumps(row), flush=True)
raise SystemExit(0 if row['status'] == 'PASS' else 1)
