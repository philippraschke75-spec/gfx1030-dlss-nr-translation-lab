"""Run the real captured frame through the network on the gfx1030 at full resolution, GPU only.

This is the frame path, not a difftest. The emulator runs at ~31k instructions/second, so checking a
1707x960 chain against it would take days; every kernel here is instead covered by its own small
fixture, with one loud exception recorded below.

Buffers are allocated by the documented activation formula C * ceil(H/4) * ceil(W/4) * 16 with
(C,H,W) from the ctx+0x190 stage tuple, so nothing here reproduces the host's allocator.

HONEST STATUS, kept in the output so a passing run cannot imply more than it shows:
  * k_import (format 0) is GPU-verified at this exact resolution.
  * block 0, the pre-block k_swin_var<32,true>, FAILS its own difftest by 28,825 bytes on valid
    float input. It is the network's entry point, so anything downstream inherits that.
  * k_export has never been validated - it needs the network's own tiled output at +0x00.

usage: net_frame_full.py <color.bin> <src_w> <src_h> [--stop=<stage>]   (inside sandbox.py)
"""
import sys, os, struct, subprocess, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_var as V, difftest_var as D, kernelspec as K

IMPORT_SYM = '_Z8k_import12ImportParams'
RUN = D.ROOT / 'build' / 'net_run.exe'
# NET_FRAME_OUT gives a run its own scratch dir. Parallel runs sharing one dir overwrite each other's
# tm.txt/tk.bin/ta.bin mid-flight, which produced inconsistent per-prefix results.
OUT = D.ROOT / 'build' / os.environ.get('NET_FRAME_OUT', 'net_frame_full'); OUT.mkdir(parents=True, exist_ok=True)
MOD = lambda s: D.ROOT / 'build' / 'kernels-hw-scratch' / (s + '.co')
WEIGHTS = D.ROOT / 'build' / 'weights'

color_path, SRC_W, SRC_H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stop = 'preblock'
for a in sys.argv[4:]:
    if a.startswith('--stop='):
        stop = a.split('=', 1)[1]

src = np.fromfile(color_path, np.uint8)
assert src.size == SRC_H * SRC_W * 8, (src.size, SRC_H * SRC_W * 8)

# ---------------------------------------------------------------- arena, in real bytes
A = 0
placements = []


def place(name, n, align=1 << 16):
    global A
    o = (A + align - 1) // align * align
    A = o + n
    placements.append((name, o, n))
    return o


ELEM = int(__import__('os').environ.get('ACT_ELEM', '2'))     # activations are f16


# The host pads the frame to a multiple of 128 in both dimensions (0x18002d3b0) and builds the
# stage table from the padded size: stage k has C = 32 * 2^k and (H, W) halved (k+1) times with
# truncating division. Stage 0 is therefore at PADDED/2, not at the capture resolution. Every
# geometry here was previously inferred as SRC >> k, which is one halving short and uses the
# unpadded size - four times the pixels at every stage.
PAD_H, PAD_W = ((SRC_H + 127) // 128) * 128, ((SRC_W + 127) // 128) * 128


def stage_hw(k):
    h, w = PAD_H, PAD_W
    for _ in range(k + 1):
        h //= 2
        w //= 2
    return h, w


def act_bytes(C, h, w):
    """C * ceil(H/4) * ceil(W/4) * 16, which counts ELEMENTS - 16 per 4x4 tile per channel.

    The contract states the formula without a unit. Sized as bytes the pre-block writes past the end
    of the arena and trips the guards, so the 16 is an element count and the buffer is that times the
    element width. ACT_ELEM overrides it for bisecting.
    """
    return C * (-(-h // 4)) * (-(-w // 4)) * 16 * ELEM


off_src = place('src RGBA16F', src.size)
off_rgb = place('import RGB32F', PAD_H * PAD_W * 12)   # host allocates PADDED H*W*12

# stage 1 is at render resolution; each later stage halves spatially and doubles channels
S1 = act_bytes(32, *stage_hw(0))      # stage 0 is PADDED/2, not the capture size
# act_bytes is the ACTIVATION size. +0x38, +0x48 and +0xa0 (scratch) are not activations and there
# is no evidence they follow that formula; the verified difftest just gives every pointer field a
# 1 MiB slot, which is ample at 64 workgroups and says nothing about 25,680. AUX_MULT scales them
# so the question can be answered by measurement instead of assumption.
AUX = S1 * int(os.environ.get('AUX_MULT', '2'))
# The pre-block is non-deterministic at 214x120 workgroups and reproducible at 8x8. If
# act_bytes under-sizes its output, workgroups overlap and race. ACT_MULT tests that.
# act_bytes under-sizes the pre-block's buffers at frame scale: at 214x120 workgroups
# the chain is not reproducible, and doubling every one of them makes all 170
# dispatches reproducible over 4 runs. 2x is measured, not derived - the exact
# requirement is unknown, and 1x demonstrably races.
S1 = S1 * int(os.environ.get('ACT_MULT', '2'))
off_a = place('act A (C=32)', S1)
off_b = place('act B (C=32)', S1)
off_scratch = place('pre-block scratch', AUX)
off_p38 = place('pre-block +0x38', AUX)

off_p48 = place('pre-block +0x48', AUX)          # the "optional 3rd buffer"; difftest_preblock
                                                # leaves make_kernarg's pointer here rather than
                                                # nulling it, so it is not optional in practice
# Encoder: 4 stages, C doubling as the spatial size halves. Each stage needs a ping-pong pair; the
# stage input is the previous stage's pooled output.
ENC_STAGES = [('32_0', [1, 2, 3, 4], 32), ('64_0', [5, 6, 7, 8], 64),
              ('128_0', [9, 10, 11, 12, 13, 14], 128), ('256_0', [15, 16, 17, 18, 19, 20, 21, 22], 256)]
ENC_MODES = [(0, 0), (-4, -4), (-4, 0), (0, -4)]

stage_geom, stage_buf = [], []
for si, (key, blocks, C) in enumerate(ENC_STAGES):
    h, w = stage_hw(si)
    n = act_bytes(C, h, w)
    stage_geom.append((h, w, C, n))
    stage_buf.append((place('enc s%d ping' % (si + 1), n), place('enc s%d pong' % (si + 1), n),
                      place('enc s%d pool' % (si + 1), n)))

# Every PTR_FIELDS entry make_kernarg fills must point at real memory: the defaults are 1 MiB
# difftest slots that do not exist here, and a kernel touching one writes outside the arena. That is
# what tripped the guards on +0x48. Unused fields share one buffer sized for the largest stage.
off_spare = place('spare (unused ptr fields)', S1)

# Everything past the encoder runs at stage-5 geometry: the last encoder stage pools once more.
# 1707>>4 = 106 and 960>>4 = 60, neither a multiple of the 8-wide window. The grid then covers
# 112x64 and the overhanging windows may be empty, which a softmax turns into 0/0 = NaN.
# C512_HW allows testing geometries that tile exactly.
if os.environ.get('C512_HW'):
    H5, W5 = [int(x) for x in os.environ['C512_HW'].split(',')]
else:
    H5, W5 = stage_hw(4)               # the C=512 stage is stage 4
N512, N1024 = act_bytes(512, H5, W5), act_bytes(1024, H5, W5)

c512_1 = [place('c512_1 w%d' % i, N512) for i in range(4)]
vit_buf = [place('vit b%d' % i, N1024) for i in range(6)]
c512_2 = [place('c512_2 w%d' % i, N512) for i in range(4)]
off_b39 = place('block39 out', N512)

# decoder mirrors the encoder: 4 stages, channels halving as the size doubles back up
DEC_STAGES = [('256_0', list(range(48, 56)), 256), ('128_0', list(range(56, 62)), 128),
              ('64_0', list(range(62, 66)), 64), ('32_0', list(range(66, 70)), 32)]
dec_geom, dec_buf = [], []
for di, (key, blocks, C) in enumerate(DEC_STAGES):
    h, w = stage_hw(3 - di)            # decoder walks the encoder's stages in reverse
    n = act_bytes(C, h, w)
    dec_geom.append((h, w, C, n))
    dec_buf.append((place('dec s%d ping' % (di + 1), n), place('dec s%d pong' % (di + 1), n),
                    place('dec s%d pool' % (di + 1), n)))

off_head = place('head out', act_bytes(32, *stage_hw(0)) * 4)
off_dst = place('export dst RGBA16F', SRC_H * SRC_W * 8)

wt = {}


def wplace(name):
    f = WEIGHTS / (name + '.bin')
    n = f.stat().st_size
    wt[name] = (place('w:' + name, n), n)
    return wt[name][0]


for _b in range(23, 31):
    for _l in range(4):
        wplace('block%d_layer%d' % (_b, _l))
for _b in range(31, 39):
    for _l in (0, 1, 2, 4):
        wplace('block%d_layer%d' % (_b, _l))
wplace('block39')
for _b in range(40, 48):
    for _l in range(4):
        wplace('block%d_layer%d' % (_b, _l))
for _b in range(48, 70):
    wplace('block%d' % _b)
wplace('block70_layer0')

w_block0 = (WEIGHTS / 'block0.bin').read_bytes()
off_w0 = place('block0 weights', len(w_block0))

enc_weight_off = {}
for _key, _blocks, _C in ENC_STAGES:
    for _blk in _blocks:
        _n = (WEIGHTS / ('block%d.bin' % _blk)).stat().st_size
        enc_weight_off[_blk] = (place('block%d w' % _blk, _n), _n)

ARENA_SZ = (A + (1 << 20) - 1) // (1 << 20) * (1 << 20)
BASE = V.ARENA

print('=== arena map ===')
for nm, o, n in placements:
    print('  %-22s @0x%08x  %10d B  %7.1f MB' % (nm, o, n, n / 1e6))
print('  total %.1f MB\n' % (ARENA_SZ / 1e6))

arena = np.zeros(ARENA_SZ, np.uint8)
arena[off_src:off_src + src.size] = src
arena[off_w0:off_w0 + len(w_block0)] = np.frombuffer(w_block0, np.uint8)
for _nm, (_o, _n) in wt.items():
    arena[_o:_o + _n] = np.frombuffer((WEIGHTS / (_nm + '.bin')).read_bytes(), np.uint8)
for _blk, (_o, _n) in enc_weight_off.items():
    arena[_o:_o + _n] = np.frombuffer((WEIGHTS / ('block%d.bin' % _blk)).read_bytes(), np.uint8)

steps = []
C512_PROBE = []      # (label, buffer offset, dispatch count) probes, populated when C512_TRACE is set

# ---------------------------------------------------------------- k_import (verified at this size)
ka = bytearray(0x130)
struct.pack_into('<Q', ka, 0x00, BASE + off_src)
# +0x10/+0x14 are the source dimensions, +0x18/+0x1c the PADDED ones - they are separate
# fields and were being given the same values.
struct.pack_into('<iiiiii', ka, 0x08, SRC_W * 8, 0, SRC_H, SRC_W, PAD_H, PAD_W)
struct.pack_into('<Q', ka, 0x20, BASE + off_rgb)
# k_import's scale. The difftest feeds the pre-block RGB in [0,1); the captured frame is HDR and
# reaches 65.12, and with the ctx exposure scalars unknown (we pass zeros) that saturates f16
# downstream. IMPORT_SCALE exists to test whether that is what wrecks the chain.
struct.pack_into('<if', ka, 0x28, 0, float(os.environ.get('IMPORT_SCALE', '1.0')))
g_imp = ((PAD_W + 255) // 256, PAD_H)
struct.pack_into('<III', ka, 0x30, g_imp[0], g_imp[1], 1)
struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
steps.append((IMPORT_SYM, bytes(ka), g_imp, 256))

# ---------------------------------------------------------------- block 0, the pre-block
# Convention from chain_import_preblock_real.py, NOT the encoder one: flags 0x14, +0x00 and +0x30
# null, float-RGB input at +0x40. Built by hand rather than via make_kernarg because that fills the
# pointer fields with 1 MiB difftest slots, which are far too small at this resolution.
PRE_SYM, PRE_LDS = D.SYMS['32_1']
_ph, _pw = stage_hw(0)
g_pre = ((_pw + 7) // 8, (_ph + 7) // 8)
ka = bytearray(424)
struct.pack_into('<Q', ka, 0x00, 0)
struct.pack_into('<Q', ka, 0x08, BASE + off_a)              # output
struct.pack_into('<Q', ka, 0x10, BASE + off_w0)             # block0 weights
struct.pack_into('<Q', ka, 0x30, 0)
struct.pack_into('<Q', ka, 0x38, BASE + off_p38)
struct.pack_into('<Q', ka, 0x40, BASE + off_rgb)            # float RGB from k_import
struct.pack_into('<Q', ka, 0x48, BASE + off_p48)
struct.pack_into('<iiii', ka, 0x18, _ph, _pw, 0, 0)
struct.pack_into('<I', ka, 0x28, 0x14)
struct.pack_into('<f', ka, 0x50, 1.0)
struct.pack_into('<Q', ka, 0xa0, BASE + off_scratch)
struct.pack_into('<III', ka, 0xA8, g_pre[0], g_pre[1], 1)
struct.pack_into('<HHH', ka, 0xB4, 256, 1, 1)
steps.append((PRE_SYM, bytes(ka), g_pre, 256))
DET_PTS = [('k_import RGB', off_rgb, SRC_H * SRC_W * 12, len(steps) - 1), ('block0 out (pre-block)', off_a, S1, len(steps))]

# ---------------------------------------------------------------- encoder blocks 1-22
# Ordinary encoder convention this time: input at +0x00, flags bit0 = first-of-stage,
# bit2 = last-of-stage (which also emits the next stage's input through the pool pointer at +0x38).


def enc_kernarg(h, w, oy, ox, flags, grid, src, dst, wgt, pool):
    ka = bytearray(424)
    for off in V.PTR_FIELDS:                       # every pointer field must be real memory
        struct.pack_into('<Q', ka, off, BASE + off_spare)
    # +0x50/+0x58/+0x60/+0x68 are SCALARS that PTR_FIELDS wrongly lists, and +0x68 is the kernel's
    # RNG seed. Leaving a buffer address there makes net_run rebase it, so the seed becomes the
    # device arena address - which changes between allocations and silently randomises the output.
    for off in (0x50, 0x58, 0x60, 0x68):
        struct.pack_into('<Q', ka, off, 0)
    struct.pack_into('<Q', ka, 0x00, BASE + src)
    struct.pack_into('<Q', ka, 0x08, BASE + dst)
    struct.pack_into('<Q', ka, 0x10, BASE + wgt)
    struct.pack_into('<Q', ka, 0x38, BASE + pool)
    struct.pack_into('<Q', ka, 0xa0, BASE + off_scratch)
    struct.pack_into('<iiii', ka, 0x18, h, w, oy, ox)
    struct.pack_into('<I', ka, 0x28, flags)
    struct.pack_into('<III', ka, 0xA8, grid[0], grid[1], 1)
    struct.pack_into('<HHH', ka, 0xB4, 256, 1, 1)
    return bytes(ka)


if stop != 'preblock':
    stage_in = off_a                                # the pre-block's output feeds stage 1
    for si, (key, blocks, C) in enumerate(ENC_STAGES):
        sym, lds = D.SYMS[key]
        lds = lds or D.group_size(sym)
        h, w, C, _n = stage_geom[si]
        ping, pong, pool = stage_buf[si]
        pp = [ping, pong]
        for i, blk in enumerate(blocks):
            ox, oy = ENC_MODES[i % 4]
            last = (i == len(blocks) - 1)
            flags = (1 if i == 0 else 0) | (4 if last else 0)
            grid = ((w - ox + 7) // 8, (h - oy + 7) // 8)
            src = stage_in if i == 0 else pp[(i + 1) % 2]
            dst = pp[i % 2]
            wgt = enc_weight_off[blk][0]
            steps.append((sym, enc_kernarg(h, w, oy, ox, flags, grid, src, dst, wgt, pool),
                          grid, 256))
            if os.environ.get('C512_TRACE') == '2' and os.environ.get('PROBE_ENC') == '1':
                # Block 0 hands the encoder absmean 1.7 and stage 1 returns 27 pinned to +/-448.
                # Probe every encoder block to see whether that is one step or an accumulation.
                C512_PROBE.append(('enc block %-3d-> dst' % blk, dst, len(steps)))
        stage_in = pool                             # the pooled output is the next stage's input
        DET_PTS.append(('enc stage %d pool' % (si + 1), pool, stage_geom[si][3], len(steps)))

# ---------------------------------------------------------------- C=512, ViT, 39, decoder, head
FFWD_IV, FFWD2 = '_Z14k_ffwd_inpview12FfwdPlParams', '_Z7k_ffwd211Ffwd2Params'
CONVV, QKV = '_Z16k_conv_res_views12ConvPlParams', '_Z10k_qkv_attn10AttnParams'
EXPAND2, CONTRACT2 = '_Z9k_expand212ExpandParams', '_Z11k_contract212ConvParams1d'
QKV2, ATTN2 = '_Z6k_qkv29QkvParams', '_Z12k_attention212AttnParams1d'
DECUP, HEAD = '_Z14k_dec_upsample11DecUpParams', '_Z12k_final_head10HeadParams'
REPACK = '_Z8k_repack12RepackParams'
EXPORT = '_Z8k_export12ExportParams'
M = {k: K.kernel_meta(k) for k in (FFWD_IV, FFWD2, CONVV, QKV, EXPAND2, CONTRACT2,
                                   QKV2, ATTN2, DECUP, HEAD, EXPORT, REPACK)}


_WG2D = {}


def wants_2d(sym):
    """True if the translated kernel actually has workgroup_id_y.

    The translation disables workgroup_id_y for some kernels and re-materialises the original's
    s15 from s2, i.e. the y index arrives in the hardware X id. Those kernels are 1-D: a grid in Y
    gives every workgroup the same index and they all write the same place. It is per-kernel, not
    universal - k_ffwd_inpview, k_conv_res_views, k_dec_upsample, k_repack and k_final_head are 1-D
    while k_ffwd2, k_qkv_attn, k_contract2, k_qkv2, k_attention2 and k_swin_var are genuinely 2-D.
    """
    if sym not in _WG2D:
        t = (D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.s')).read_text(errors='replace')
        kd = t.split('.amdhsa_kernel ' + sym, 1)[1].split('.end_amdhsa_kernel', 1)[0]
        _WG2D[sym] = '.amdhsa_system_sgpr_workgroup_id_y 1' in kd
    return _WG2D[sym]


def grid_for(sym, nbytes, hw=None, per_wg=16384):
    """Extent for one kernel: 1-D kernels are sized by the buffer, 2-D ones by the geometry.

    Leaving the 2-D kernels at (1,1) is the same bug in the other direction - one workgroup covers
    one 8x8 window, not a 60x106 stage.
    """
    if wants_2d(sym):
        if hw is None:
            return (1, 1)
        h, w = hw
        return ((w + 7) // 8, (h + 7) // 8)
    return (max(1, -(-nbytes // per_wg)), 1)


def ka_for(sym, ptrs, ints=(), grid=(1, 1)):
    # Kernarg sized from the kernel metadata, with the HSA hidden args filled from it too.
    n = M[sym][1]
    ka = bytearray(max(n, 424))
    for off, val in ptrs:
        struct.pack_into('<Q', ka, off, BASE + val)
    for off, fmt, vals in ints:
        struct.pack_into(fmt, ka, off, *vals)
    for name, val in (('block_count_x', grid[0]), ('block_count_y', grid[1]), ('block_count_z', 1),
                      ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                      ('grid_dims', 2)):
        if name in M[sym][3]:
            o, sz = M[sym][3][name]
            struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
    return bytes(ka)




def c512_stage(blocks, work, src_in):
    # One C=512 attention block is 5 dispatches; recipe per VARPARAMS_HOST_CONTRACT.md.
    g = (1, 1)
    # weight layer per dispatch: contract table = ffwd_iv 0, ffwd2 0, conv_res 1, qkv_attn 2, conv_res 3
    W5L = [int(x) for x in os.environ.get('C512_WMAP', '0,1,2,3,2').split(',')]
    for bi, blk in enumerate(blocks):
        a = src_in if bi == 0 else work[0]
        L = lambda n: wt['block%d_layer%d' % (blk, n)][0]
        # 16384 B per workgroup was measured for k_final_head and then applied to every 1-D kernel
        # without checking. If the real span here is smaller, the extra workgroups overlap and race,
        # which is what run-to-run variation looks like. C512_WG makes it measurable.
        _wg = int(os.environ.get('C512_WG', '16384'))
        g1 = grid_for(FFWD_IV, N512, per_wg=_wg)      # 1-D: sized by the buffer
        gc = grid_for(CONVV, N512, per_wg=_wg)
        g2 = grid_for(FFWD2, N512, (H5, W5))          # 2-D: sized by the stage geometry
        gq = grid_for(QKV, N512, (H5, W5))
        steps.append((FFWD_IV, ka_for(FFWD_IV, [(0x00, a), (0x08, work[1]), (0x10, L(W5L[0]))],
                                      [(0x18, '<ii', (H5, W5))], g1), g1, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('1 ffwd_inpview -> w1', work[1], len(steps)))
        # work[0] is written only by this block's LAST dispatch, so on the first block +0x00
        # reads a buffer nothing has written. net_block512.py never notices: V.build fills its
        # arena with e4m3-shaped bytes while this runner zero-fills, and a normalisation over
        # an all-zero buffer gives 0/0 = NaN, which then spreads through every later block.
        w0 = work[0] if bi else a
        steps.append((FFWD2, ka_for(FFWD2, [(0x00, w0), (0x08, a), (0x10, work[1]),
                                            (0x18, L(W5L[1]))],
                                    [(0x20, '<iii', (H5, W5, 4))], g2), g2, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('2 ffwd2        -> w1', work[1], len(steps)))
        steps.append((CONVV, ka_for(CONVV, [(0x00, work[1]), (0x08, a), (0x10, w0),
                                            (0x18, work[2]), (0x28, L(W5L[2]))],
                                    [(0x20, '<i', (0,)), (0x30, '<ii', (H5, W5))], gc), gc, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('3 conv_res_1   -> w2', work[2], len(steps)))
        steps.append((QKV, ka_for(QKV, [(0x00, work[2]), (0x08, work[3]), (0x10, L(W5L[3]))],
                                  [(0x18, '<ii', (H5, W5)),
                                   (0x20, '<ii', ENC_MODES[bi % 4] if os.environ.get('C512_SHIFT') == '1'
                                    else (0, 0))], gq), gq, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('4 qkv_attn     -> w3', work[3], len(steps)))
        steps.append((CONVV, ka_for(CONVV, [(0x00, work[3]), (0x10, work[2]), (0x18, work[0]),
                                            (0x28, L(W5L[4]))],
                                    [(0x20, '<i', (0,)), (0x30, '<ii', (H5, W5)),
                                     (0x40, '<ii', (H5, W5))], gc), gc, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('5 conv_res_2   -> w0', work[0], len(steps)))
        if os.environ.get('C512_TRACE') == '2' and                 str(blk) in os.environ.get('PROBE_BLOCKS', '23,24,25,26').split(','):
            # Each probe re-runs a prefix and rewrites the whole 2.26 GB arena, so probing every
            # block costs tens of GB of disk writes. Limit it to the blocks in question.
            C512_PROBE.append(('after block %-3d-> w0' % blk, work[0], len(steps)))


if stop in ('full', 'all'):
    # enc -> mid is `k_repack, k_final_head` in the driver's phase table. The runner had neither, so
    # the C=512 stage was reading the last encoder pool at C=256/120x213 while running at 60x106.
    # k_repack is a pure relayout (2 pointers + four i32, difftest PASS), which is what bridges them.
    # Derive from the stage table rather than a literal: the geometry correction moved this
    # stage from 106x60 to 56x32 and a hard-coded default would have silently gone stale.
    _rp = [int(x) for x in os.environ.get('REPACK_DIMS',
                                          '%d,%d,256,512' % (H5, W5)).split(',')]
    # 16384 B/workgroup is the figure derived for k_final_head, not for k_repack. REPACK_WG
    # lets the real per-workgroup span be found by measurement.
    g_rp = grid_for(REPACK, N512, per_wg=int(os.environ.get('REPACK_WG', '16384')))
    steps.append((REPACK, ka_for(REPACK, [(0x00, stage_buf[3][2]), (0x08, c512_1[0])],
                                 [(0x10, '<iiii', tuple(_rp))], g_rp), g_rp, 256))
    REPACK_END = len(steps)
    if os.environ.get('STOP_AFTER_REPACK') != '1':
        c512_stage(range(23, 31), c512_1, c512_1[0])

    # ViT blocks 31-38: six contiguous buffers, 5 dispatches each (net_vit.py, verified bit-exact)
    g = (1, 1)
    BIN_ = c512_1[0]
    B260, B268, B270, B278, B280, B288 = vit_buf
    for blk in range(31, 39):
        L = lambda n: wt['block%d_layer%d' % (blk, n)][0]
        steps.append((EXPAND2, ka_for(EXPAND2, [(0x00, BIN_), (0x08, B260), (0x10, L(0))],
                                      [], g), g, 256))
        steps.append((CONTRACT2, ka_for(CONTRACT2, [(0x00, B260), (0x08, BIN_), (0x10, B268),
                                                    (0x18, L(1))],
                                        [(0x20, '<ii', (H5, W5)), (0x28, '<i', (4,))], g), g, 256))
        steps.append((QKV2, ka_for(QKV2, [(0x00, B268), (0x08, B270), (0x10, B278), (0x18, B280),
                                          (0x20, L(2))], [], g), g, 256))
        steps.append((ATTN2, ka_for(ATTN2, [(0x00, B270), (0x08, B278), (0x10, B280), (0x18, B288)],
                                    [(0x20, '<ii', (W5, H5))], g), g, 256))
        steps.append((CONTRACT2, ka_for(CONTRACT2, [(0x00, B288), (0x08, B270), (0x10, BIN_),
                                                    (0x18, L(4))],
                                        [(0x20, '<ii', (H5, W5)), (0x28, '<i', (4,))], g), g, 256))

        if os.environ.get('C512_TRACE') == '2' and os.environ.get('PROBE_VIT') == '1':
            C512_PROBE.append(('after ViT %-5d -> BIN' % blk, BIN_, len(steps)))

    # block 39: the real k_dec_upsample, not the k_ffwd_inpview stand-in net_full.py uses
    # k_dec_upsample (disassembly): +0x00 read tile-wise, +0x08 read linearly (wg*8192), +0x10 WRITTEN (wg*8192,
    # the only global store), +0x18 weights; +0x20 is TWO i32 (s2, s3; s4 = s3/4 = tiles per row), not a pointer.
    # 1-D (workgroup_id_y disabled), 8192 B of output per workgroup.
    _dd = [int(x) for x in os.environ.get('DECUP_DIMS', '%d,%d' % (H5, W5)).split(',')]
    _dout, _dskip = (off_b39, c512_2[0]) if os.environ.get('DECUP_SWAP') == '1' else (c512_2[0], off_b39)
    g_du = (int(os.environ.get('DECUP_GRID', str(-(-N512 // 8192)))), 1)
    steps.append((DECUP, ka_for(DECUP, [(0x00, BIN_), (0x08, _dskip), (0x10, _dout),
                                        (0x18, wt['block39'][0])],
                                [(0x20, '<ii', tuple(_dd))], g_du), g_du, 256))

    if os.environ.get('C512_TRACE') == '2' and os.environ.get('PROBE_VIT') == '1':
        C512_PROBE.append(('after block39 -> b39', off_b39, len(steps)))
        C512_PROBE.append(('after block39 -> c512_2[0]', c512_2[0], len(steps)))
    c512_stage(range(40, 48), c512_2, off_b39)

    # decoder blocks 48-69, the encoder mirrored
    stage_in = c512_2[0]
    for di, (key, blocks, C) in enumerate(DEC_STAGES):
        sym, lds = D.SYMS[key]
        lds = lds or D.group_size(sym)
        h, w, C, _n = dec_geom[di]
        ping, pong, pool = dec_buf[di]
        pp = [ping, pong]
        for i, blk in enumerate(blocks):
            ox, oy = ENC_MODES[i % 4]
            last = (i == len(blocks) - 1)
            flags = (1 if i == 0 else 0) | (4 if last else 0)
            grid = ((w - ox + 7) // 8, (h - oy + 7) // 8)
            src = stage_in if i == 0 else pp[(i + 1) % 2]
            steps.append((sym, enc_kernarg(h, w, oy, ox, flags, grid, src, pp[i % 2],
                                           wt['block%d' % blk][0], pool), grid, 256))
            last_dst = pp[i % 2]
        # The pooled output feeds the NEXT stage. After the last decoder stage there is no next
        # stage, so the head must read the last block's own output, not a pool buffer that nothing
        # downsampled into. HEAD_SRC=pool restores the old wiring for comparison.
        stage_in = pool if (di + 1 < len(DEC_STAGES) or
                            os.environ.get('HEAD_SRC', 'last') == 'pool') else last_dst

    # block 70: the real k_final_head
    # k_final_head was dispatched with grid (1,1) - one workgroup cannot cover 1707x960. Its output
    # is what k_export reads at 16 B/pixel, so an unwritten head buffer is exactly what makes the
    # export surface non-finite. HEAD_GRID picks the convention: 'swin' = (ceil(W/8), ceil(H/8)),
    # 'lin' = (ceil(W/256), H) as k_import and k_export use, 'one' = the old (1,1).
    if os.environ.get('HEAD_SRC') == 'enc1':          # isolation test: a buffer known to vary, same size as the head input
        stage_in = stage_buf[0][2]
    _hg = os.environ.get('HEAD_GRID', 'x16k')
    # The head writes ~7 x 16330 B at grid (7,960): the x workgroups land in distinct places and
    # all 960 y rows overwrite each other, so its address does not depend on workgroup_id_y.
    # HeadParams carries no dimensions, so a 1D grid that encodes the whole extent in x is the
    # obvious alternative to a 2D one.
    _npix = SRC_W * SRC_H
    g_head = {'swin': ((SRC_W + 7) // 8, (SRC_H + 7) // 8),
              'lin': ((SRC_W + 255) // 256, SRC_H),
              'one': (1, 1),
              'flat': (((SRC_W + 255) // 256) * SRC_H, 1),
              'pix': ((_npix + 255) // 256, 1),
              # The kernel does s_lshl_b64 s[12:13], {0, wg_id_y}, 13 and adds that to the
              # input pointer: 8192 bytes of input per y-workgroup. So gy must cover the
              # input buffer, not the image height.
              'stride8k': (1, (act_bytes(32, SRC_H, SRC_W) + 8191) // 8192),
              # output pointer += wg_id_y * 16384 (0xa42a4), workgroup_id_x is never read: gx must be 1
              'out16k': (1, (SRC_W * SRC_H * 16 + 16383) // 16384),
              # The TRANSLATED kernel enables workgroup_id_x only (system_sgpr_workgroup_id_y 0) and its prologue does
              # s_mov_b32 s15, s2, so the original's workgroup_id_y arrives in the hardware X id: count in X.
              'x16k': ((SRC_W * SRC_H * 16 + 16383) // 16384, 1)}[_hg]
    steps.append((HEAD, ka_for(HEAD, [(0x00, stage_in), (0x08, off_head),
                                      (0x10, wt['block70_layer0'][0])], [], g_head), g_head, 256))

    # k_export, fed the network's own output at +0x00 - the first time it has had that
    # k_export's kernarg is 320 B (kernel metadata), not 280, and grid_dims lives at +0x80 -
    # it was never set, so the kernel saw 0 instead of 2 and treated a 2D grid as something
    # else. It wrote rows 0..69 of 960 and stopped.
    ka = bytearray(M[EXPORT][1] if EXPORT in M else 320)
    struct.pack_into('<Q', ka, 0x00, BASE + off_head)
    struct.pack_into('<i', ka, 0x08, 0)
    # +0x0c is HEIGHT and +0x10 is WIDTH, not the other way round: the launcher builds the grid
    # as (ceil(width/256), height) from exactly these two fields, so swapping them makes the kernel
    # address the surface with the wrong stride.
    struct.pack_into('<ii', ka, 0x0c, SRC_H, SRC_W)
    # +0x08 = input row stride in ELEMENTS, +0x14 = output row PITCH in BYTES, +0x18 = mode (see EXPORT_FINDINGS.md)
    struct.pack_into('<ii', ka, 0x14, SRC_W * 8, 0)
    struct.pack_into('<Q', ka, 0x20, BASE + off_dst)
    # +0x28 is the export format/mode (ebp in the launcher) and +0x08 an i32 from xmm10; both
    # were guessed as 0. Now that the network feeds real data in, coverage is a usable signal.
    struct.pack_into('<i', ka, 0x28, int(os.environ.get('EXPORT_MODE', '0')))
    struct.pack_into('<i', ka, 0x08, SRC_W)
    struct.pack_into('<Q', ka, 0x30, BASE + off_rgb)
    struct.pack_into('<ff', ka, 0x38, 1.0, 1.0)
    g_exp = ((SRC_W + 255) // 256, SRC_H)
    struct.pack_into('<III', ka, 0x40, g_exp[0], g_exp[1], 1)
    struct.pack_into('<HHH', ka, 0x4c, 256, 1, 1)
    struct.pack_into('<H', ka, 0x80, 2)          # grid_dims
    steps.append((EXPORT, bytes(ka), g_exp, 256))

# ---------------------------------------------------------------- dispatch
# FIX_IN_LOAD=<file>: start at the C=512 stage from a SAVED k_repack output (steps before it are skipped), so the
# stage can be tested with an input that is identical in every run. FIX_IN_SAVE=<file> writes that input.
SKIP = 0
if os.environ.get('FIX_IN_LOAD') and stop in ('full', 'all'):
    SKIP = REPACK_END
    _fx = np.fromfile(os.environ['FIX_IN_LOAD'], np.uint8)
    assert _fx.size == N512, (_fx.size, N512)
    arena[c512_1[0]:c512_1[0] + N512] = _fx


def _run_prefix(nsteps):
    """Dispatch only the first nsteps and return the resulting arena."""
    b = b''; ln = []
    for sym, k, grid, thr in steps[SKIP:nsteps]:
        o = len(b); b += k
        ln.append('%s|%s|%d|%d|%d|%d|%d' % (MOD(sym), sym, o, len(k), grid[0], grid[1], thr))
    (OUT / 'tm.txt').write_text(chr(10).join(ln) + chr(10))
    (OUT / 'tk.bin').write_bytes(b)
    (OUT / 'ta.bin').write_bytes(arena.tobytes())
    rr = subprocess.run([str(RUN), str(OUT / 'tm.txt'), str(OUT / 'tk.bin'),
                         str(OUT / 'ta.bin'), '%x' % BASE], capture_output=True, text=True, timeout=1800)
    if rr.returncode != 0:
        return None
    return np.fromfile(OUT / 'ta.bin', np.uint8)


if os.environ.get('DET_TEST'):
    # Determinism probe: run the SAME prefix DET_TEST times and hash the buffer each dispatch produced. The first
    # label whose hash differs between identical runs is where the non-determinism enters.
    import hashlib
    _n = int(os.environ['DET_TEST'])
    _pts = (DET_PTS if os.environ.get('DET_ENC', '1') == '1' else []) + [('k_repack -> c512_1[0]', c512_1[0], N512, REPACK_END)] + [(l, b, N512, n) for l, b, n in C512_PROBE[:5]]      # block 23's five dispatches
    print()
    print('=== determinism: %d identical runs per point ===' % _n)
    for label, buf, size, nst in _pts:
        row = []
        for _ in range(_n):
            aa = _run_prefix(nst)
            if aa is None:
                row.append('FAIL'); continue
            dd = aa[buf:buf + size]
            if os.environ.get('FIX_IN_SAVE') and label.startswith('k_repack') and not Path(os.environ['FIX_IN_SAVE']).exists():
                dd.tofile(os.environ['FIX_IN_SAVE'])
            row.append('%s d=%d nan=%.2f%%' % (hashlib.sha1(dd.tobytes()).hexdigest()[:8], len(np.unique(dd)), 100.0 * float((dd == 0x7f).mean())))
        print('  %-30s %s   %s' % (label, 'SAME' if len(set(r.split()[0] for r in row)) == 1 else 'DIFFERS', ' | '.join(row)), flush=True)
    raise SystemExit(0)

if os.environ.get('DET_BISECT') == '1':
    # Find the FIRST dispatch after which two identical runs disagree. Everything measured in this
    # chain is worthless until that point is known: yesterday four identical runs gave 94, 245, 244
    # and 2 distinct byte values, so several parameter studies were reading noise.
    import hashlib

    def _hash_prefix(n):
        a = _run_prefix(n)
        return None if a is None else hashlib.sha256(a.tobytes()).hexdigest()[:16]

    # Two agreeing runs is the evidence that produced a false 'deterministic' twice already.
    def _stable(n, tries=int(os.environ.get('DET_TRIES', '2'))):
        h0 = _hash_prefix(n)
        if h0 is None:
            return None
        for _ in range(tries - 1):
            if _hash_prefix(n) != h0:
                return False
        return True

    total = len(steps)
    print()
    print('=== determinism bisect over %d dispatches ===' % total)
    if _stable(total):
        print('  the FULL chain is reproducible over %d runs - nothing to bisect' % int(os.environ.get('DET_TRIES','2')))
        raise SystemExit(0)

    lo, hi = 0, total                      # lo known stable, hi known unstable
    if not _stable(1):
        lo, hi = 0, 1
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            st = _stable(mid)
            print('  prefix %-4d -> %s' % (mid, 'stable' if st else 'DIFFERS'), flush=True)
            if st:
                lo = mid
            else:
                hi = mid
    sym, _k, grid, _t = steps[hi - 1]
    print()
    print('  last reproducible prefix : %d dispatches' % lo)
    print('  first non-reproducible   : %d dispatches' % hi)
    print('  the dispatch that breaks it: #%d  %s  grid %dx%d'
          % (hi - 1, sym, grid[0], grid[1]))
    raise SystemExit(0)

if os.environ.get('C512_TRACE') in ('1', '2'):
    # Measure each of block 23's five dispatches in turn. The stage turns 253 distinct byte values
    # into 2; this says which dispatch does it, instead of inferring from the stage's final state.
    print()
    print('=== C=512 block 23, per dispatch ===')
    base_n = C512_PROBE[0][2] - 1 if C512_PROBE else 0
    a0 = _run_prefix(base_n)
    if a0 is not None:
        d0 = a0[c512_1[0]:c512_1[0] + N512]
        print('  0 before stage  -> w0  nonzero=%6.2f%%  distinct=%3d'
              % (100 * float((d0 != 0).mean()), len(np.unique(d0))))
    for label, buf, nst in C512_PROBE[:int(os.environ.get('C512_PROBE_MAX', '9999'))]:
        aa = _run_prefix(nst)
        if aa is None:
            print('  %s  GPU FAIL' % label); continue
        dd = aa[buf:buf + N512]
        _u, _c = np.unique(dd, return_counts=True); _o = np.argsort(-_c)[:3]
        print('  %s  nonzero=%6.2f%%  distinct=%3d  top: %s'
              % (label, 100 * float((dd != 0).mean()), len(_u),
                 ' '.join('0x%02x=%.2f%%' % (_u[i], 100.0 * _c[i] / dd.size) for i in _o)))
        _nan = np.nonzero(dd == 0x7f)[0]
        if len(_nan) and os.environ.get('C512_NANPOS') == '1':
            # assumed layout [C][ceil(H/4)][ceil(W/4)][16]: report where the NaN codes sit
            _per = ((H5 + 3) // 4) * ((W5 + 3) // 4) * 16
            _ch, _r = _nan // _per, (_nan % _per) // 16
            _ty, _tx = _r // ((W5 + 3) // 4), _r % ((W5 + 3) // 4)
            print('     NaN positions: %d bytes; first idx %s; channels %d..%d (%d distinct); tile rows %d..%d; tile cols %d..%d'
                  % (len(_nan), _nan[:4].tolist(), _ch.min(), _ch.max(), len(np.unique(_ch)), _ty.min(), _ty.max(), _tx.min(), _tx.max()))
    raise SystemExit(0)

blob = b''; lines = []
for sym, k, grid, thr in steps[SKIP:]:
    o = len(blob); blob += k
    lines.append('%s|%s|%d|%d|%d|%d|%d' % (MOD(sym), sym, o, len(k), grid[0], grid[1], thr))
(OUT / 'manifest.txt').write_text('\n'.join(lines) + '\n')
(OUT / 'kernargs.bin').write_bytes(blob)
(OUT / 'arena.bin').write_bytes(arena.tobytes())

print('=== dispatches ===')
for l in lines:
    f = l.split('|')
    print('  %-44s grid %sx%s' % (f[1], f[4], f[5]))

r = subprocess.run([str(RUN), str(OUT / 'manifest.txt'), str(OUT / 'kernargs.bin'),
                    str(OUT / 'arena.bin'), '%x' % BASE], capture_output=True, text=True, timeout=1800)
msg = ((r.stdout or '') + (r.stderr or '')).strip()
print('\n' + msg)
if r.returncode != 0:
    raise SystemExit(1)

res = np.fromfile(OUT / 'arena.bin', np.uint8)


def e4m3(b):
    """Decode the network's activation format. Reading these bytes as f16 reports NaN and +/-65504
    saturation for data that is entirely finite - that misreading was taken as evidence of a broken
    kernel for hours. One byte per value: sign, 4-bit exponent (bias 7), 3-bit mantissa."""
    s_ = np.where(b & 128, -1.0, 1.0).astype(np.float32)
    e = ((b >> 3) & 15).astype(np.float32)
    m = (b & 7).astype(np.float32)
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * np.power(2.0, e - 7))
    return np.where((e == 15) & (m == 7), np.nan, s_ * v)


def report(name, buf):
    f = e4m3(buf).astype(np.float64)
    nz = 100.0 * float((buf != 0).mean())
    print('  %-18s nan=%-6.2f%% min=%-10.4g max=%-10.4g absmean=%-10.4g nonzero=%.1f%%'
          % (name, 100.0 * float(np.isnan(f).mean()), float(np.nanmin(f)), float(np.nanmax(f)),
             float(np.nanmean(np.abs(f))), nz))
    return float(np.isnan(f).mean()), nz


print('\n=== buffers after the run ===')
rgb = res[off_rgb:off_rgb + SRC_H * SRC_W * 12].view(np.float32).reshape(SRC_H, SRC_W, 3)
report('k_import RGB', res[off_rgb:off_rgb + SRC_H * SRC_W * 12])
a_nan, a_nz = report('block0 out', res[off_a:off_a + S1])
report('block0 +0x38', res[off_p38:off_p38 + S1])
if stop != 'preblock':
    for si, (key, blocks, C) in enumerate(ENC_STAGES):
        h, w, C, n = stage_geom[si]
        ping, pong, pool = stage_buf[si]
        report('enc s%d pool C=%d' % (si + 1, C), res[pool:pool + n])

if a_nz < 1.0:
    print('\n  block0 wrote essentially nothing - the pre-block did not run as intended.')


def write_png(path, img8):
    h, w, _ = img8.shape
    raw = b''.join(b'\x00' + img8[y].tobytes() for y in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'
                           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def tonemap(lin):
    lin = np.clip(np.nan_to_num(lin, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


write_png(OUT / 'imported.png', tonemap(rgb))
print('\nwrote', OUT / 'imported.png')

# The pre-block output is C=32 in the network's tiled layout, not an image. Showing the first three
# channels of each 4x4 tile is not a picture of the frame - it is a sanity view: structure here means
# the kernel saw the image, noise means it did not.
tiles_y, tiles_x = -(-SRC_H // 4), -(-SRC_W // 4)
a16 = res[off_a:off_a + S1].view(np.float16)   # f16 confirmed: f32 gives 1e38 nonsense
if a16.size >= tiles_y * tiles_x * 8:
    v = a16[:tiles_y * tiles_x * 8].reshape(tiles_y, tiles_x, 8)[:, :, :3].astype(np.float32)
    if np.isfinite(v).any():
        lo, hi = float(np.nanpercentile(v, 1)), float(np.nanpercentile(v, 99))
        if hi > lo:
            write_png(OUT / 'block0_preview.png',
                      tonemap((np.nan_to_num(v) - lo) / (hi - lo)))
            print('wrote', OUT / 'block0_preview.png', '(channel view, NOT the frame)')

if stop in ('full', 'all'):
    for _nm, _o, _n in ([('c512_1 w%d' % i, o, N512) for i, o in enumerate(c512_1)] + [('vit b%d' % i, o, N1024) for i, o in enumerate(vit_buf)]
                        + [('c512_2 w%d' % i, o, N512) for i, o in enumerate(c512_2)] + [('block39 out', off_b39, N512)]):
        _x = res[_o:_o + _n]
        print('mid %-12s %9d B: nonzero=%6.2f%% distinct=%3d' % (_nm, _n, 100.0 * float((_x != 0).mean()), len(np.unique(_x))))
    for _di, (_k, _b, _C) in enumerate(DEC_STAGES):
        _h, _w, _C, _n = dec_geom[_di]
        for _nm, _o in zip(('ping', 'pong', 'pool'), dec_buf[_di]):
            _x = res[_o:_o + _n]
            print('dec s%d %-4s C=%-3d %10d B: nonzero=%6.2f%% distinct=%3d' % (_di + 1, _nm, _C, _n, 100.0 * float((_x != 0).mean()), len(np.unique(_x))))
    hb = res[off_head:off_head + SRC_W * SRC_H * 16]
    _n8 = 1601
    _hi = res[stage_in:stage_in + _n8 * 8192].reshape(_n8, 8192)
    _nzc = (_hi != 0).any(axis=1)
    print('head input       : stage_in=%#x, %d chunks of 8192 B: %d nonzero chunks (last %s), %.2f%% bytes nonzero, distinct=%d'
          % (stage_in, _n8, int(_nzc.sum()), int(np.nonzero(_nzc)[0][-1]) if _nzc.any() else None, 100.0 * float((_hi != 0).mean()), len(np.unique(_hi))))
    print('head buffer      : %d B, nonzero=%.3f%%, distinct bytes=%d' % (hb.size, 100.0 * float((hb != 0).mean()), len(np.unique(hb))))
    print()
    d16 = res[off_dst:off_dst + SRC_H * SRC_W * 8].view(np.float16).reshape(SRC_H, SRC_W, 4)
    rep = d16[:, :, :3].astype(np.float32)
    print('k_export surface : finite=%s  nonzero=%.1f%%  min=%.4g max=%.4g'
          % (bool(np.isfinite(rep).all()), 100.0 * float((rep != 0).mean()),
             float(np.nanmin(rep)), float(np.nanmax(rep))))
    rows = np.nonzero((d16 != 0).reshape(SRC_H, -1).any(axis=1))[0]
    full = int(((d16 != 0).reshape(SRC_H, SRC_W, 4)[:, :, :3].any(axis=2).all(axis=1)).sum())
    print('k_export rows    : %d of %d touched (last %s), %d fully written across all %d columns'
          % (len(rows), SRC_H, int(rows[-1]) if len(rows) else None, full, SRC_W))
    write_png(OUT / 'rendered.png', tonemap(rep))
    print('wrote', OUT / 'rendered.png')
