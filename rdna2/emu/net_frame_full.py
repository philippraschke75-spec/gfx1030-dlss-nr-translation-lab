"""Run the real captured frame through the network on the gfx1030 at full resolution, GPU only.

This is the frame path, not a difftest. The emulator runs at ~31k instructions/second, so checking a
1707x960 chain against it would take days; every kernel here is instead covered by its own small
fixture, with one loud exception recorded below.

Buffers are allocated by the documented activation formula C * ceil(H/4) * ceil(W/4) * 16 with
(C,H,W) from the ctx+0x190 stage tuple, so nothing here reproduces the host's allocator.

HONEST STATUS, kept in the output so a passing run cannot imply more than it shows:
  * k_import (format 0) is GPU-verified at this exact resolution.
  * block 0 is k_pre_block_1h_32_fp8 (PRE_1H=1), not k_swin_var<32,true> - see FRAME_STATE's
    "k_pre_block_1h_32_fp8 as block 0" fix. k_swin_var<32,true>'s old difftest failure (28,825/
    42,812 bytes) is retired: VARPARAMS_HOST_CONTRACT.md's "SOLVED: block 0 was never broken"
    found it was a fixture bug (+0x68's high dword left uncleared, corrupting the RNG seed) and
    it now PASSES at 0 mismatches. Neither finding says anything about k_pre_block_1h_32_fp8,
    which is the kernel actually dispatched here and has its own, separate verification status.
  * k_export is GPU-verified: net_export.py difftests it against the emulator across every
    format/mode and history-flag combination at 0 mismatches, including the exact fields this
    file packs by default (mode 0, +0x28=0, hist=0). See FRAME_STATE Update 16.
  * post_block (writes ctx+0x100, the sole network-result buffer k_export reads) PASSES its
    difftest except on e4m3 NaN inputs (NaN propagation differs; see FRAME_STATE Update 18).
  * k_pre_block_1h_32_fp8 (the block 0 dispatched here) PASSES difftest_pre.py at 8x8/16x16/32x32
    with the host scalars this file packs (FRAME_STATE Update 18).

usage: net_frame_full.py <color.bin> <src_w> <src_h> [--stop=<stage>]   (inside sandbox.py)
"""
import sys, os, struct, subprocess, zlib, hashlib
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
# R9: the prologue computes the pre-block's grid from ctx+0x18 with the SAME idiom as the
# post-block (movq xmm0,[r10+0x18] / psrad 0x1f / psrld 0x1d / pshufd 0xe1 / psrad 3 at
# 0x18002ee72 and 0x18002efd0), and passes ctx+0x18/0x1c as VarParams +0x18/+0x1c
# (0x18002edbd/0x18002edc1). So the pre-block runs at the PADDED FULL FRAME, 1792x1024,
# not at stage_hw(0). PRE_FULL=0 restores the old half-resolution wiring.
PRE_FULL = os.environ.get('PRE_FULL', '1') == '1'
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
# ctx+0x218, the pre-block's +0x08: full-resolution C=32 features, which the post-block
# reads back at its +0x08 as the outermost U-net skip.
off_a = place('act A (C=32)', max(S1, act_bytes(32, PAD_H, PAD_W)) if PRE_FULL else S1)
off_b = place('act B (C=32)', S1)
# 8 KiB per workgroup (host rule, R6c 0x180030d3e..0x180030d8f). At PRE_FULL the pre-block
# dispatches (PAD_W//8)x(PAD_H//8) = 28,672 workgroups, four times the old count.
off_scratch = place('pre-block scratch', max(AUX, 8192 * (PAD_W // 8) * (PAD_H // 8)))
# At PRE_FULL this holds the pre-block's half-resolution stage-0 tensor, the encoder's input.
off_p38 = place('pre-block +0x38', max(AUX, act_bytes(64, *stage_hw(0))))

off_p48 = place('pre-block +0x48', AUX)          # the "optional 3rd buffer"; difftest_preblock
                                                # leaves make_kernarg's pointer here rather than
                                                # nulling it, so it is not optional in practice
# Encoder: 4 stages, C doubling as the spatial size halves. Each stage needs a ping-pong pair; the
# stage input is the previous stage's pooled output.
ENC_STAGES = [('32_0', [1, 2, 3, 4], 32), ('64_0', [5, 6, 7, 8], 64),
              ('128_0', [9, 10, 11, 12, 13, 14], 128), ('256_0', [15, 16, 17, 18, 19, 20, 21, 22], 256)]
ENC_MODES = [(0, 0), (-4, -4), (-4, 0), (0, -4)]

stage_geom, stage_buf = [], []
enc_stage_out = {}   # si -> the buffer holding that encoder stage's output (the U-net skip)
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
off_2a0 = place('ctx+0x2a0 (decoder input, block 47 +0x20)', N512)
# ctx+0x250, the ViT's own buffer ('vit1d'). Its input ctx+0x228 ('vit512a') must survive the ViT: block 39 reads it
# back at +0x08 as the skip around the ViT.
off_vit = place('vit out (ctx+0x250)', N1024)
# MID_HOST=1 (default): the middle section as the host runs it (FRAME_STATE Update 4/5). The ViT works on the
# (C=1024, H/64, W/64) stage of the host's table at ctx+0x1cc (0x18002de5c), i.e. 16x28 = 448 tokens, not at
# the C=512 geometry. Every S_mid/S_fine score in FRAME_STATE was measured with MID_HOST=1 explicitly set, so
# the default (previously 0) was inconsistent with what this file's own numbers describe.
MID_HOST = os.environ.get('MID_HOST', '1') == '1'
H6, W6 = stage_hw(5)
NTOK = -(-(H6 * W6) // 64) * 64                     # ctx+0x314: H*W rounded up to 64 (0x18002e468-0x18002e47f)
HEAD_TILES = (-(-H6 // 4)) * (-(-W6 // 4))          # ctx+0x30c (0x18002e437-0x18002e461)
C512_TILES = (-(-H5 // 4)) * (-(-W5 // 4))          # ctx+0x308 (0x18002e3ed-0x18002e430)
off_pooled = place('pooled (ctx+0x248)', HEAD_TILES * 8192 * 2)
off_headb = place('head (ctx+0x250)', HEAD_TILES * 16384 * 2)
off_tok = [place('vit tok%d (ctx+0x%x)' % (i, a), NTOK * 1024 * 2) for i, a in enumerate((0x258, 0x290))]


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

# M2 run 5: constant stand-ins for every post-block input, so any x%8 structure left in A's
# output is produced by A from uniform data. POST_CONST selects which inputs are replaced.
off_k_act = place('const e4m3 act', max(act_bytes(32, PAD_H, PAD_W), act_bytes(32, *stage_hw(0))))
off_k_rgb = place('const rgb 0.5', PAD_H * PAD_W * 12)
off_k_f32 = place('const f32 1.0', PAD_H * PAD_W * 4)
off_head = place('head out', act_bytes(32, *stage_hw(0)) * 4)
# ctx+0x100, the network result. The setup fn allocates ctx[0x18]*ctx[0x1c]*16 at
# 0x18002ded7..0x18002df37 - 16 B/pixel over the PADDED frame. k_export reads it (0x18002d7ca).
off_netout = place('net result ctx+0x100', PAD_H * PAD_W * 16)
# ctx+0x118, the post-block's +0x40. 0x18002dd04 aliases it onto ctx+0x108, allocated
# H*W*4 at 0x18002df3e - one f32 per pixel, not the pre-block aux.
off_ctx118 = place('ctx+0x118 (= ctx+0x108)', PAD_H * PAD_W * 4)
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
wplace('block30_layer4')   # k_final_head's weight 'block30.layer4.layer' (MID_HOST, 0x18002fb0a-0x18002fb50)
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
# SENTINEL=1 fills the pre-block's output buffers with 0xA5 so the run can report each
# kernel's true written extent, not the extent the activation formula predicts.
SENT_BUFS = ('act A (C=32)', 'pre-block +0x38', 'act B (C=32)', 'pre-block +0x48')
if os.environ.get('SENTINEL') == '1':
    for _nm, _o, _n in placements:
        if _nm in SENT_BUFS:
            arena[_o:_o + _n] = 0xA5
arena[off_k_act:off_k_rgb] = 0x38                         # e4m3 0x38 = 1.0
arena[off_k_rgb:off_k_f32] = np.full((off_k_f32 - off_k_rgb) // 4, 0.5,
                                     np.float32).view(np.uint8)
arena[off_k_f32:off_k_f32 + PAD_H * PAD_W * 4] = np.full(PAD_H * PAD_W, 1.0,
                                                         np.float32).view(np.uint8)
for _nm, (_o, _n) in wt.items():
    arena[_o:_o + _n] = np.frombuffer((WEIGHTS / (_nm + '.bin')).read_bytes(), np.uint8)
for _blk, (_o, _n) in enc_weight_off.items():
    arena[_o:_o + _n] = np.frombuffer((WEIGHTS / ('block%d.bin' % _blk)).read_bytes(), np.uint8)

steps = []
C512_PROBE = []      # (label, buffer offset, dispatch count) probes, populated when C512_TRACE is set
ENC_BLK_INFO = {}    # blk -> (dispatch index, h, w, C, ox, oy, grid, sym, lds) for ENC_EMU_CHECK
ENC_PROBE_GEOM = {}  # label -> (h, w, C) for the encoder-block probes (DEFECT_MASK per-block trace)

# ---------------------------------------------------------------- k_import (verified at this size)
ka = bytearray(0x130)
struct.pack_into('<Q', ka, 0x00, BASE + off_src)
# +0x10/+0x14 are the source dimensions, +0x18/+0x1c the PADDED ones - they are separate
# fields and were being given the same values.
struct.pack_into('<iiiiii', ka, 0x08, SRC_W * 8, 0, SRC_H, SRC_W, PAD_H, PAD_W)
struct.pack_into('<Q', ka, 0x20, BASE + off_rgb)
# +0x28 mode, +0x2c scale. The host packs +0x28 = ebp = the frame function's 9th argument ([rsp+0x318],
# 0x18002d533 -> 0x18002d5ff), which on the first-frame paths is the global dword 0x18009a488
# (0x180015add / 0x18001a880) - initialised to -1 by the ctx constructor (0x18001f26b) and never
# written directly elsewhere. Any mode != 0 tonemaps in k_import: max(0, scale*x), x/(1+x) clamped,
# then the sRGB OETF (0xaa494-0xaa518), so the network sees [0,1]. Mode 0 passed raw HDR (up to 65 here),
# which blew activations up to the e4m3 limit from encoder block 9 on and printed as dark 16-px blocks.
# The same ebp is k_export's +0x28 (the inverse path), so EXPORT_MODE defaults to -1 too.
# +0x2c: 1.0 (0x18006d2c8), overridden by a positive job value (0x18002d432-0x18002d488).
struct.pack_into('<if', ka, 0x28, int(os.environ.get('IMPORT_MODE', '-1')), float(os.environ.get('IMPORT_SCALE', '1.0')))
g_imp = ((PAD_W + 255) // 256, PAD_H)
struct.pack_into('<III', ka, 0x30, g_imp[0], g_imp[1], 1)
struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
steps.append((IMPORT_SYM, bytes(ka), g_imp, 256))

# ---------------------------------------------------------------- block 0, the pre-block
# Convention from chain_import_preblock_real.py, NOT the encoder one: flags 0x14, +0x00 and +0x30
# null, float-RGB input at +0x40. Built by hand rather than via make_kernarg because that fills the
# pointer fields with 1 MiB difftest slots, which are far too small at this resolution.
# R10: the prologue's two launches are EXCLUSIVE, and the contract has the branches the wrong
# way round. 0x18002ee4a `cmp byte [0x18009b1f0], 1` / 0x18002ee5b `jne 0x18002efb8` sends the
# NOT-equal case to the <32,true> path; the fall-through (==1) sets up handle 0x180066348
# (k_pre_block_1h_32_fp8) at 0x18002efac and then `jmp 0x18002f141` PAST the <32,true> launch.
# C:67 attributes PreParams to `byte != 1`; that is inverted.
# The epilogue is gated on the same byte (0x180030d96 `cmp` / `jne 0x180030f19`), and its
# fall-through is the post-block. So the two branches are coherent pairs:
#     byte == 1 -> k_pre_block_1h_32_fp8  +  k_post_block_1h_32_fp8
#     byte != 1 -> k_swin_var<32,true>    in both places
# The runner was taking block 0 from one branch and the tail from the other. Since the post-block
# is the only writer of ctx+0x100 and k_export reads it, the ==1 pair is the live one.
# PRE_1H=0 restores the <32,true> block 0.
PRE_1H = os.environ.get('PRE_1H', '1') == '1'
PRE1H_SYM = '_Z21k_pre_block_1h_32_fp89PreParams'
PRE_SYM, PRE_LDS = D.SYMS['32_1']
_ph, _pw = (PAD_H, PAD_W) if PRE_FULL else stage_hw(0)   # mirror of A: H,W from ctx+0x18
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

if PRE_1H:
    # PreParams, 0x58 B (C:60-63): +0x00 float-RGB in (ctx+0xf8), +0x08 out (ctx+0x218), +0x10
    # weight, +0x18 H,W, +0x20 f32 (ctx+0x34), +0x24 i32 (ctx+0x38), +0x28 two f32 (ctx+0x20),
    # +0x30 i32 0, +0x38 ptr (ctx+0x220), +0x40 ptr (the ctx+0x180 scratch, 8 KiB per workgroup),
    # +0x48 two 32-bit (ctx+0x28). Same grid rule as the post-block.
    PRE_LDS = D.group_size(PRE1H_SYM)
    PRE_SYM = PRE1H_SYM
    ka = bytearray(K.kernel_meta(PRE1H_SYM)[1])
    struct.pack_into('<Q', ka, 0x00, BASE + off_rgb)
    struct.pack_into('<Q', ka, 0x08, BASE + off_a)
    struct.pack_into('<Q', ka, 0x10, BASE + off_w0)
    struct.pack_into('<ii', ka, 0x18, _ph, _pw)
    # Host values for the scalars (prologue loads at 0x18002ef05-0x18002ef57; ctx = BSS global 0x18009a100):
    #   +0x20 = ctx+0x34 = 0.0625, set once by the ctx constructor (0x18001f211 -> 0x180020468/0x180020470,
    #           an 8-byte store of {0.03125, 0.0625} from 0x18006d2e0 into ctx+0x30/0x34). At 0 the kernel
    #           ignores its RGB input entirely (impulse vs black frame: 0 differing output bytes).
    #   +0x24 = ctx+0x38: 0 (0x18002d6d3)
    #   +0x28 = ctx+0x20: ini [LocalTone, LocalStructure] masked off when ToneChannels == 0 (default) -> 0, 0
    #   +0x48 = ctx+0x28/0x2c: f32 pair from 0x18001a6f2-0x18001a725. With UseAutoMask defaulting to 1
    #           and SkinStructure to -1, both are LocalStructure = 1.0 (was packed as int 0, 0)
    struct.pack_into('<f', ka, 0x20, float(os.environ.get('PRE_F20', '0.0625')))
    struct.pack_into('<i', ka, 0x24, 0)
    struct.pack_into('<ff', ka, 0x28, 0.0, 0.0)
    struct.pack_into('<i', ka, 0x30, 0)
    struct.pack_into('<Q', ka, 0x38, BASE + off_p38)
    # +0x40 is NOT scratch. The host passes ctx+0x118 (0x18002ed8d -> [rbp+0x618] -> 0x18002ef4a), and the
    # kernel reads it as a second 12 B/pixel RGB source at the same index as +0x00, falling back to +0x00
    # when it is null (s_cmp_eq_u64 s[26:27], 0 / v_cndmask at 0x2EFC-0x2FE8). With scratch there, the
    # pre-block's output was byte-identical for a real frame, a black frame and an impulse frame.
    # ctx+0x118 is 0 on the path at 0x18002d6da and ctx+0x108 on the path at 0x18002dd04.
    # PRE_40=null (default) | ctx118 | scratch.
    _p40 = os.environ.get('PRE_40', 'null')
    struct.pack_into('<Q', ka, 0x40, 0 if _p40 == 'null' else BASE + (off_ctx118 if _p40 == 'ctx118' else off_scratch))
    struct.pack_into('<ff', ka, 0x48, *[float(x) for x in os.environ.get('PRE_F48', '1.0,1.0').split(',')])
    # Explicit struct is 0x50 B (kernel metadata), so the hidden block counts sit at +0x50 and
    # the group sizes at +0x5c - not the VarParams 0xa8/0xb4.
    struct.pack_into('<III', ka, 0x50, g_pre[0], g_pre[1], 1)
    struct.pack_into('<HHH', ka, 0x5c, 256, 1, 1)
steps.append((PRE_SYM, bytes(ka), g_pre, 256))
DET_PTS = [('k_import RGB', off_rgb, SRC_H * SRC_W * 12, len(steps) - 1), ('block0 out (pre-block)', off_a, S1, len(steps))]

# ---------------------------------------------------------------- encoder blocks 1-22
# Ordinary encoder convention this time: input at +0x00, flags bit0 = first-of-stage,
# bit2 = last-of-stage (which also emits the next stage's input through the pool pointer at +0x38).


def enc_kernarg(h, w, oy, ox, flags, grid, src, dst, wgt, pool, ptrA=None):
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
    # R5/R5b: launcher A's args 10 and 11 land at +0x30 and +0x38 (the callee stores
    # [rsp+0x4d8] -> [rsp+0x328] and [rsp+0x4e0] -> [rsp+0x330] with the struct based at
    # rsp+0x2f8, which puts flags at +0x28 and zeroes +0x40 onward). The decoder's first
    # block gets the previous stage's output as ptrA at +0x30, and ptrB null.
    if ptrA is None:
        struct.pack_into('<Q', ka, 0x38, BASE + pool)
    else:
        struct.pack_into('<Q', ka, 0x30, BASE + ptrA)
        struct.pack_into('<Q', ka, 0x38, 0)
    struct.pack_into('<Q', ka, 0xa0, BASE + off_scratch)
    # F12: the contract fixes +0x20 as the X origin (C:15, C:20; run_var.py:30-33 reads it from
    # the kernel at PC 0xb0360..0xb036c), but the runner has packed oy there. SWAP_XY=1 packs
    # ox first. The window-origin cycle has length 4, which is the period of the column
    # artefact present in every encoder ping/pong.
    if os.environ.get('SWAP_XY') == '1':
        struct.pack_into('<iiii', ka, 0x18, h, w, ox, oy)
    else:
        struct.pack_into('<iiii', ka, 0x18, h, w, oy, ox)
    struct.pack_into('<I', ka, 0x28, flags)
    struct.pack_into('<III', ka, 0xA8, grid[0], grid[1], 1)
    struct.pack_into('<HHH', ka, 0xB4, 256, 1, 1)
    return bytes(ka)


if stop != 'preblock':
    # PRE_FULL: encoder stage 1 runs at stage_hw(0) = half the padded frame, so its input is
    # the pre-block's HALF-resolution +0x38 output (ctx+0x220), not the full-resolution +0x08.
    # HYPOTHESIS - PRE_SRC=a feeds off_a instead, for comparison in the same build.
    stage_in = (off_p38 if os.environ.get('PRE_SRC', 'p38') == 'p38' else off_a) if PRE_FULL else off_a
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
            # ctx+0x2a8[r9d] is passed to launcher A as its in_ptr, and the encoder indexes
            # that array as [3-stage] while the decoder indexes it as [stage] - so decoder
            # iteration di reads the ENCODER stage (3-di) buffer, matching channel width.
            # The runner was reading the previous decoder stage's output and never touching
            # the encoder buffers at all, i.e. no U-net skip connections.
            # The U-net skip belongs to the DECODER loop, not here. This block used to read
            # `enc_stage_out.get(3 - di, ...)` with `di` leaking from the buffer-allocation loop
            # (FF:147), where it is always 3 - so encoder stages 2-4 took stage 1's output as
            # their first block's input instead of the previous stage's pool.
            src = stage_in if i == 0 else pp[(i + 1) % 2]
            dst = pp[i % 2]
            wgt = enc_weight_off[blk][0]
            steps.append((sym, enc_kernarg(h, w, oy, ox, flags, grid, src, dst, wgt, pool),
                          grid, 256))
            ENC_BLK_INFO[blk] = (len(steps) - 1, h, w, C, ox, oy, grid, sym, lds)
            if os.environ.get('C512_TRACE') == '2' and os.environ.get('PROBE_ENC') == '1':
                # Block 0 hands the encoder absmean 1.7 and stage 1 returns 27 pinned to +/-448.
                # Probe every encoder block to see whether that is one step or an accumulation.
                C512_PROBE.append(('enc block %-3d-> dst' % blk, dst, len(steps)))
                ENC_PROBE_GEOM['enc block %-3d-> dst' % blk] = (h, w, C)
        # The host never passes a skip pointer: encoder stage s ping-pongs between ctx+0x1d8[s]
        # and ctx+0x2a8[3-s] (0x18002f637/0x18002f64f, swapped at 0x18002f693..0x18002f6af),
        # and decoder stage d reads ctx+0x1d8[3-d]. The two are the same pair, aliased.
        # The stage's FIRST block writes ctx+0x1d8[s] (out_ptr = [rbp+0x690], initialised from
        # ctx+0x1d8[s] at 0x18002f6af), so with an even block count the buffer the decoder
        # reads holds the SECOND-TO-LAST block's output, not the last one.
        # SKIP_PARITY=0 restores the last-block choice.
        _sp = 1 if os.environ.get('SKIP_PARITY', '1') == '1' else 0
        enc_stage_out[si] = pp[(len(blocks) - 1 - _sp) % 2]
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
POST = '_Z22k_post_block_1h_32_fp810PostParams'   # epilogue dispatch A, handle 0x180066350
# The host's real C=512 default path (VIT512_OLD unset, FRAME_STATE Update 11): CONVV2/ATTN2_C512
# replace k_conv_res_views/k_qkv_attn for steps 3-5 of every non-pool30 block. k_ffwd_inpview is
# not dispatched at all; k_ffwd2 (FFWD2, unchanged symbol) is reused with different kernarg fields
# and grid. ATTN2_C512 has a different mangled name from the ViT's own k_qkv2 (QKV2 above).
CONVV2 = '_Z11k_conv_res211Conv2Params'
ATTN2_C512 = '_Z11k_qkv_attn210AttnParams'
M = {k: K.kernel_meta(k) for k in (POST, FFWD_IV, FFWD2, CONVV, QKV, EXPAND2, CONTRACT2,
                                   QKV2, ATTN2, DECUP, HEAD, EXPORT, REPACK, CONVV2, ATTN2_C512)}


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
    # grid may be (gx, gy) or (gx, gy, gz); k_qkv_attn2 is the first kernel needing gz>1
    # (FRAME_STATE Update 11: grid = ((W+7-ox)>>3, (H+7-oy)>>3, 16), workgroup_id_z enabled).
    n = M[sym][1]
    ka = bytearray(max(n, 424))
    for off, val in ptrs:
        struct.pack_into('<Q', ka, off, BASE + val)
    for off, fmt, vals in ints:
        struct.pack_into(fmt, ka, off, *vals)
    gz = grid[2] if len(grid) > 2 else 1
    for name, val in (('block_count_x', grid[0]), ('block_count_y', grid[1]), ('block_count_z', gz),
                      ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                      ('grid_dims', 3 if gz > 1 else 2)):
        if name in M[sym][3]:
            o, sz = M[sym][3][name]
            struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
    return bytes(ka)




def c512_stage(blocks, work, src_in):
    # One C=512 attention block is 5 dispatches; recipe per VARPARAMS_HOST_CONTRACT.md.
    g = (1, 1)
    # C512_HOST=1 (default): the host's REAL default path (FRAME_STATE Update 11). Every branch
    # in the per-block launcher (0x180033660) is gated on byte [0x18009b208] = atoi(getenv(
    # 'VIT512_OLD')), 0 when unset. This runner used to dispatch the VIT512_OLD=1 kernels
    # (k_ffwd_inpview/k_conv_res_views/k_qkv_attn) unconditionally - not the host's default.
    # C512_HOST=0 restores that old (wrong-by-default) path for comparison. Defaults to 1: the
    # C=512 stage is now fully difftested (net_block512_2.py, all 4 shift-window origins, 0
    # mismatches - FRAME_STATE Updates 15/16) and scores higher (S_mid +0.71 vs +0.53 for the
    # old kernels).
    if os.environ.get('C512_HOST', '1') != '1':
        _c512_stage_old(blocks, work, src_in)
        return
    T = (-(-H5 // 4)) * (-(-W5 // 4))              # ctx[0x308]
    gf = (-(-T // 4), 8)                            # k_ffwd2:     grid ((T+3)/4, 8, 1)
    gc2 = (-(-T // 4), 4)                           # k_conv_res2: grid ((T+3)/4, 4, 1)
    for bi, blk in enumerate(blocks):
        # host: the launcher's 3rd arg (the stage input) is [rbp+0x630] for block 23 only (0x18002f9e7-
        # 0x18002f9f0) and 0 for EVERY block 40-47 (xor r8d,r8d at 0x18003063a): stage 2 has no first block,
        # it reads ctx+0x228, which the host swaps with k_dec_upsample's ctx+0x298 (0x1800305c4-0x1800305e0).
        first = (bi == 0) and blocks[0] == 23
        # Stage 40-47's own "first" call site was not read (Update 11 flags this as an assumption);
        # using per-stage bi==0 is the only choice consistent with each stage's work[] buffers
        # being separately allocated and uninitialised until something writes them.
        a = src_in if first else work[0]
        L = lambda n: wt['block%d_layer%d' % (blk, n)][0]

        # Step 2: k_ffwd2 (0x18003399c-0x1800339f4). +0x00 = first?NULL:ctx+0x228, +0x08 =
        # first?input:NULL, +0x10 = ctx+0x230, +0x18 = layer0, +0x20 = (H,W), +0x28 = T.
        _p2 = [(0x10, work[1]), (0x18, L(0))]
        _i2 = [(0x20, '<ii', (H5, W5)), (0x28, '<i', (T,))]
        if first:
            _p2.append((0x08, a))
            _i2.append((0x00, '<Q', (0,)))
        else:
            _p2.append((0x00, work[0]))
            _i2.append((0x08, '<Q', (0,)))
        steps.append((FFWD2, ka_for(FFWD2, _p2, _i2, gf), gf, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('2 ffwd2        -> w1', work[1], len(steps)))

        # Step 3: k_conv_res2 (0x18003413e-0x1800341bd). +0x00 = ctx+0x230, +0x08 =
        # first?NULL:ctx+0x228, +0x10 = first?input:NULL, +0x18 = ctx+0x238, +0x20 = 0 (qword),
        # +0x28 = layer1, +0x30 = (H,W), +0x38 = T.
        # +0x28 is the weight POINTER 0x180031bc0(ctx, blk, 1) (0x180034193 -> 0x180034198), not the
        # layer index: the kernel adds offsets to it (s_add_u32 s74, s50, 0x40000 in the .s).
        _p3 = [(0x00, work[1]), (0x18, work[2]), (0x28, L(1))]
        _i3 = [(0x20, '<Q', (0,)), (0x30, '<ii', (H5, W5)), (0x38, '<i', (T,))]
        if first:
            _p3.append((0x10, a))
            _i3.append((0x08, '<Q', (0,)))
        else:
            _p3.append((0x08, work[0]))
            _i3.append((0x10, '<Q', (0,)))
        steps.append((CONVV2, ka_for(CONVV2, _p3, _i3, gc2), gc2, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('3 conv_res2_1  -> w2', work[2], len(steps)))

        # Step 4: k_qkv_attn2 (0x180033c0f-0x180033cda). +0x00/+0x08 = ctx+0x238/+0x240,
        # +0x10 = layer2, +0x18 = (H,W), +0x20 = origin (shift-window table, same ENC_MODES the
        # encoder uses). Grid = ((W+7-ox)>>3, (H+7-oy)>>3, 16) - the first kernel in this project
        # needing a 3-D dispatch (net_run.cpp / ka_for both support gz now).
        ox, oy = ENC_MODES[bi % 4]
        # Grid from the host (0x180033c22-0x180033c65): (H,W) = stage5 dwords [+4],[+8], swapped
        # to (W,H) by pshufd 0xe1, + (7,7,0,0) from 0x18006d570 (paddd at 0x180033c3c: the
        # rounding addend, not a grid literal - Update 11 misread it), - (ox,oy) from 0x180066410,
        # signed /8 -> grid = ((W+7-ox)>>3, (H+7-oy)>>3, 16), gz literal at 0x180033c65.
        # Same launcher (0x180033660) for blocks 23-30 and 40-47, so this holds at every stage size.
        _qgy = int(os.environ.get('QKV2_GY', str((H5 + 7 - oy) >> 3)))
        _qgz = int(os.environ.get('QKV2_GZ', '16'))
        gq2 = ((W5 + 7 - ox) >> 3, _qgy, _qgz)
        steps.append((ATTN2_C512, ka_for(ATTN2_C512, [(0x00, work[2]), (0x08, work[3]), (0x10, L(2))],
                                         [(0x18, '<ii', (H5, W5)), (0x20, '<ii', (ox, oy))], gq2),
                      gq2, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('4 qkv_attn2    -> w3', work[3], len(steps)))

        # Step 5: block 30 keeps the OLD k_conv_res_views + pooled path unmodified (Update 11:
        # "block 30 only: k_conv_res_views + pooled" - this is not part of the VIT512_OLD switch).
        # Every other block's step 5 is k_conv_res2 (0x180033f0d-0x180033f7b): +0x00 = ctx+0x240,
        # +0x08 = ctx+0x238, +0x10 = 0, +0x18 = ctx+0x228, +0x20 = first?input:NULL, +0x28 = layer3,
        # +0x30 = (H,W), +0x38 = T.
        _pool30 = MID_HOST and blk == 30
        if _pool30:
            _wg = int(os.environ.get('C512_WG', '16384'))
            gc = grid_for(CONVV, N512, per_wg=_wg)
            steps.append((CONVV, ka_for(CONVV, [(0x00, work[3]), (0x10, work[2]), (0x18, work[0]),
                                                (0x28, L(3)), (0x38, off_pooled)],
                                        [(0x20, '<i', (0,)), (0x30, '<ii', (H5, W5)),
                                         (0x40, '<ii', (H6, W6))], gc), gc, 256))
        else:
            # +0x28 = weight pointer 0x180031bc0(ctx, blk, 3) (0x180033f51 -> 0x180033f56), as in step 3.
            _p5 = [(0x00, work[3]), (0x08, work[2]), (0x18, work[0]), (0x28, L(3))]
            _i5 = [(0x10, '<Q', (0,)), (0x30, '<ii', (H5, W5)), (0x38, '<i', (T,))]
            # +0x20 is the launcher's 5th arg (r14 reloaded at 0x180033e8d), NOT the stage input: an optional
            # second output (the kernel null-checks it, s_cmp_lg_u64 s[48:49], 0, and stores through it). It
            # is 0 in stage 1 (0x18002f9fd) and ctx+0x2a0 at block 47 (0x180030605-0x180030627), which is the
            # decoder's input (0x180030849).
            if blk == 47:
                _p5.append((0x20, off_2a0))
            else:
                _i5.append((0x20, '<Q', (0,)))
            steps.append((CONVV2, ka_for(CONVV2, _p5, _i5, gc2), gc2, 256))
        if os.environ.get('C512_TRACE') == '1' and bi == 0:
            C512_PROBE.append(('5 conv_res2_2  -> w0', work[0], len(steps)))
        if os.environ.get('C512_TRACE') == '2' and                 str(blk) in os.environ.get('PROBE_BLOCKS', '23,24,25,26').split(','):
            C512_PROBE.append(('after block %-3d-> w0' % blk, work[0], len(steps)))


def _c512_stage_old(blocks, work, src_in):
    # C512_HOST=0: the runner's original VIT512_OLD=1 path, kept for comparison.
    W5L = [int(x) for x in os.environ.get('C512_WMAP', '0,0,1,2,3').split(',')]
    for bi, blk in enumerate(blocks):
        a = src_in if bi == 0 else work[0]
        L = lambda n: wt['block%d_layer%d' % (blk, n)][0]
        _wg = int(os.environ.get('C512_WG', '16384'))
        g1 = grid_for(FFWD_IV, N512, per_wg=_wg)
        gc = grid_for(CONVV, N512, per_wg=_wg)
        g2 = grid_for(FFWD2, N512, (H5, W5))
        gq = grid_for(QKV, N512, (H5, W5))
        steps.append((FFWD_IV, ka_for(FFWD_IV, [(0x00, a), (0x08, work[1]), (0x10, L(W5L[0]))],
                                      [(0x18, '<ii', (H5, W5))], g1), g1, 256))
        w0 = work[0] if bi else a
        steps.append((FFWD2, ka_for(FFWD2, [(0x00, w0), (0x08, a), (0x10, work[1]),
                                            (0x18, L(W5L[1]))],
                                    [(0x20, '<iii', (H5, W5, 4))], g2), g2, 256))
        steps.append((CONVV, ka_for(CONVV, [(0x00, work[1]), (0x08, a), (0x10, w0),
                                            (0x18, work[2]), (0x28, L(W5L[2]))],
                                    [(0x20, '<i', (0,)), (0x30, '<ii', (H5, W5))], gc), gc, 256))
        steps.append((QKV, ka_for(QKV, [(0x00, work[2]), (0x08, work[3]), (0x10, L(W5L[3]))],
                                  [(0x18, '<ii', (H5, W5)),
                                   (0x20, '<ii', ENC_MODES[bi % 4] if os.environ.get('C512_SHIFT') == '1'
                                    else (0, 0))], gq), gq, 256))
        _pool30 = MID_HOST and blk == 30
        steps.append((CONVV, ka_for(CONVV, [(0x00, work[3]), (0x10, work[2]), (0x18, work[0]),
                                            (0x28, L(W5L[4]))] + ([(0x38, off_pooled)] if _pool30 else []),
                                    [(0x20, '<i', (0,)), (0x30, '<ii', (H5, W5)),
                                     (0x40, '<ii', (H6, W6) if _pool30 else (H5, W5))], gc), gc, 256))


if stop in ('full', 'all'):
    # enc -> mid is `k_repack, k_final_head` in the driver's phase table. The runner had neither, so
    # the C=512 stage was reading the last encoder pool at C=256/120x213 while running at 60x106.
    # k_repack is a pure relayout (2 pointers + four i32, difftest PASS), which is what bridges them.
    # Derive from the stage table rather than a literal: the geometry correction moved this
    # stage from 106x60 to 56x32 and a hard-coded default would have silently gone stale.
    # +0x18 is the work count in units of 1024 elements: the kernel's bounds check is
    # 'active while global_id < [+0x18] << 10' (s_lshl_b64 s[8:9], s[4:5], 10 at +9,
    # v_cmpx_gt_u64 at +16). So it is C*ceil(H/4)*ceil(W/4)*16 / 1024. Passing 256 covered
    # 13.78% of the buffer; the derived 896 covers 48.21%, which is the 50% the element
    # count predicts. +0x1c is only compared against zero, i.e. a flag, and makes no
    # difference here. +0x14 is a divisor and also taken ceil(/4); +0x10 multiplies it.
    _elems = 512 * (-(-H5 // 4)) * (-(-W5 // 4)) * 16
    _rp = [int(x) for x in os.environ.get('REPACK_DIMS',
                                          '%d,%d,%d,0' % (H5, W5, _elems // 1024)).split(',')]
    # 16384 B/workgroup is the figure derived for k_final_head, not for k_repack. REPACK_WG
    # lets the real per-workgroup span be found by measurement.
    g_rp = grid_for(REPACK, N512, per_wg=int(os.environ.get('REPACK_WG', '16384')))
    # The host feeds block 23 the encoder's stage-4 pool directly: no k_repack here. The launcher's 3rd
    # argument (r8, 0x18002f9f0 cmove for edi==0x17) is [rbp+0x630] = ctx+0x1f8[k] (0x18002f846/0x18002f84e),
    # and between 0x18002f800 and the launcher call 0x18002fa0b no k_repack handle (0x1800663e0) is
    # referenced. The only k_repack in the "enc -> mid" range is 0x18002fcfe, AFTER the C=512 stage and
    # k_final_head - the pre-ViT repack MID_HOST already runs. The extra repack scrambled every
    # pixel: impulse lift at c512_1 w0 1.26 -> 3.72, S_mid +0.7106 -> +0.8912 (FRAME_STATE Update 20).
    # Default is now the host's. REPACK_ENC=1 restores the old extra repack; NO_REPACK=1 is kept as alias.
    if os.environ.get('REPACK_ENC') != '1' or os.environ.get('NO_REPACK') == '1':
        REPACK_END = len(steps)
        c512_stage(range(23, 31), c512_1, stage_buf[3][2])
    else:
        steps.append((REPACK, ka_for(REPACK, [(0x00, stage_buf[3][2]), (0x08, c512_1[0])],
                                     [(0x10, '<iiii', tuple(_rp))], g_rp), g_rp, 256))
        REPACK_END = len(steps)
        if os.environ.get('STOP_AFTER_REPACK') != '1':
            c512_stage(range(23, 31), c512_1, c512_1[0])

    if MID_HOST:
        # Host middle section, 0x18002fa17-0x1800305b8. Pointers named by their ctx slot.
        B260, B268, B270, B278, B280, B288 = vit_buf
        L30_4 = wt['block30_layer4'][0]
        # k_final_head: ctx+0x248 'pooled' -> ctx+0x250 'head', weight 'block30.layer4.layer', grid (ctx[0x30c],1,1)
        # (0x18002fa90-0x18002fbbf)
        g_h = (HEAD_TILES, 1)
        steps.append((HEAD, ka_for(HEAD, [(0x00, off_pooled), (0x08, off_headb), (0x10, L30_4)], [], g_h), g_h, 256))
        # k_repack forward: ctx+0x250 -> ctx+0x258, +0x10 (H6,W6) from [rbp+0x4a4], +0x18 ctx[0x314], +0x1c 1,
        # grid (256,1,1) (0x18002fc31-0x18002fcfe)
        g_r = (256, 1)
        steps.append((REPACK, ka_for(REPACK, [(0x00, off_headb), (0x08, off_tok[0])],
                                     [(0x10, '<iiii', (H6, W6, NTOK, 1))], g_r), g_r, 256))
        # ViT blocks 31-38 (0x18002fd47-0x1800303a0): in/out ping-pong ctx+0x258 / ctx+0x290, swapped at the loop head.
        gy32, gy8 = (NTOK // 64, 32), (NTOK // 64, 8)
        vin, vout = off_tok
        for blk in range(31, 39):
            L = lambda n: wt['block%d_layer%d' % (blk, n)][0]
            # k_expand2 (0x18002fda7-0x18002fe68): in -> ctx+0x260, layer 0, grid (ntok/64, 32)
            steps.append((EXPAND2, ka_for(EXPAND2, [(0x00, vin), (0x08, B260), (0x10, L(0))], [], gy32), gy32, 256))
            # k_contract2 (0x18002fe85-0x18002ff83): ctx+0x260, in, -> ctx+0x268, layer 1, +0x20 (4096, 4096*1024),
            # +0x28 4, grid (ntok/64, 8)
            steps.append((CONTRACT2, ka_for(CONTRACT2, [(0x00, B260), (0x08, vin), (0x10, B268), (0x18, L(1))],
                                            [(0x20, '<ii', (4096, 4096 * 1024)), (0x28, '<i', (4,))], gy8), gy8, 256))
            # k_qkv2 (0x18002ffa0-0x180030070): ctx+0x268 -> 0x270/0x278/0x280, layer 2, grid (ntok/64, 32)
            steps.append((QKV2, ka_for(QKV2, [(0x00, B268), (0x08, B270), (0x10, B278), (0x18, B280),
                                              (0x20, L(2))], [], gy32), gy32, 256))
            # k_attention2 (0x18003008d-0x18003015d): 0x270/0x278/0x280 -> 0x288, +0x20 = ctx+0x310 pshufd'd
            # (ntok, H6*W6), grid (ntok/64, 32)
            steps.append((ATTN2, ka_for(ATTN2, [(0x00, B270), (0x08, B278), (0x10, B280), (0x18, B288)],
                                        [(0x20, '<ii', (NTOK, H6 * W6))], gy32), gy32, 256))
            # k_contract2 (0x18003017a-0x180030278): ctx+0x288, ctx+0x268 -> out, layer 4, +0x20 (1024, 1024*1024),
            # +0x28 4, grid (ntok/64, 8)
            steps.append((CONTRACT2, ka_for(CONTRACT2, [(0x00, B288), (0x08, B268), (0x10, vout), (0x18, L(4))],
                                            [(0x20, '<ii', (1024, 1024 * 1024)), (0x28, '<i', (4,))], gy8), gy8, 256))
            vin, vout = vout, vin
        # k_repack back: last output -> ctx+0x250, same dims, +0x1c 0, grid (256,1,1) (0x1800303a0-0x18003046a)
        steps.append((REPACK, ka_for(REPACK, [(0x00, vin), (0x08, off_headb)],
                                     [(0x10, '<iiii', (H6, W6, NTOK, 0))], g_r), g_r, 256))
        # k_dec_upsample (0x180030509-0x1800305b8): +0x00 ctx+0x250, +0x08 ctx+0x228 (vit512a = C=512 output),
        # +0x10 ctx+0x298, +0x18 block 39, +0x20 [rbp+0x544] = (H5,W5), grid (ctx[0x308],1,1)
        g_du = (C512_TILES, 1)
        steps.append((DECUP, ka_for(DECUP, [(0x00, off_headb), (0x08, c512_1[0]), (0x10, c512_2[0]),
                                            (0x18, wt['block39'][0])],
                                    [(0x20, '<ii', (H5, W5))], g_du), g_du, 256))
    else:
        # ViT blocks 31-38: six contiguous buffers, 5 dispatches each (net_vit.py, verified bit-exact)
        # The ViT kernels are all 2-D (workgroup_id_y enabled). At (1,1) only one workgroup's tile
        # of each buffer is written and the rest keeps block 30's output, which is why the
        # distinct-byte counts looked healthy - most of it was passthrough.
        g = grid_for(CONTRACT2, N1024, (H5, W5))
        # The host dumps ctx+0x228 as 'vit512a' (0x18002fa17-0x18002fa6a) and passes it to k_dec_upsample +0x08,
        # with the ViT's output ctx+0x250 ('vit1d') at +0x00 (0x180030509-0x180030555). The runner ran the ViT in
        # place on c512_1[0], destroying that skip, and gave +0x08 the never-written off_b39. VIT_SEP=0 restores that.
        # Default off: with the ViT itself mis-launched (see FRAME_STATE Update 4) its output is all zero, so VIT_SEP=1
        # feeds block 39 a zero +0x00. Neither wiring is right until the ViT section is rebuilt from the host.
        VIT_SEP = os.environ.get('VIT_SEP', '0') == '1'
        BIN_ = off_vit if VIT_SEP else c512_1[0]
        B260, B268, B270, B278, B280, B288 = vit_buf
        for blk in range(31, 39):
            L = lambda n: wt['block%d_layer%d' % (blk, n)][0]
            _vin = c512_1[0] if (VIT_SEP and blk == 31) else BIN_   # block 31 reads the C=512 output, writes BIN_
            steps.append((EXPAND2, ka_for(EXPAND2, [(0x00, _vin), (0x08, B260), (0x10, L(0))],
                                          [], g), g, 256))
            steps.append((CONTRACT2, ka_for(CONTRACT2, [(0x00, B260), (0x08, _vin), (0x10, B268),
                                                        (0x18, L(1))],
                                            [(0x20, '<ii', (H5, W5)), (0x28, '<i', (4,))], g), g, 256))
            steps.append((QKV2, ka_for(QKV2, [(0x00, B268), (0x08, B270), (0x10, B278), (0x18, B280),
                                              (0x20, L(2))], [], g), g, 256))
            steps.append((ATTN2, ka_for(ATTN2, [(0x00, B270), (0x08, B278), (0x10, B280), (0x18, B288)],
                                        [(0x20, '<ii', (W5, H5))], g), g, 256))
            # +0x08 is ctx+0x268, step 2's output - net_vit.py:120 passes B268 here and passes its
            # difftest. B270 is k_qkv2's first output, so the block's second contract input was
            # taking the QKV projection instead of the post-FFN activation, in all eight blocks.
            steps.append((CONTRACT2, ka_for(CONTRACT2, [(0x00, B288), (0x08, B268), (0x10, BIN_),
                                                        (0x18, L(4))],
                                            [(0x20, '<ii', (H5, W5)), (0x28, '<i', (4,))], g), g, 256))

            if os.environ.get('C512_TRACE') == '2' and os.environ.get('PROBE_VIT') == '1':
                C512_PROBE.append(('after ViT %-5d -> BIN' % blk, BIN_, len(steps)))

        # block 39: the real k_dec_upsample, not the k_ffwd_inpview stand-in net_full.py uses
        # k_dec_upsample (disassembly): +0x00 read tile-wise, +0x08 read linearly (wg*8192), +0x10 WRITTEN (wg*8192,
        # the only global store), +0x18 weights; +0x20 is TWO i32 (s2, s3; s4 = s3/4 = tiles per row), not a pointer.
        # 1-D (workgroup_id_y disabled), 8192 B of output per workgroup.
        _dd = [int(x) for x in os.environ.get('DECUP_DIMS', '%d,%d' % (H5, W5)).split(',')]
        _skip39 = c512_1[0] if VIT_SEP else off_b39
        _dout, _dskip = (_skip39, c512_2[0]) if os.environ.get('DECUP_SWAP') == '1' else (c512_2[0], _skip39)
        g_du = (int(os.environ.get('DECUP_GRID', str(-(-N512 // 8192)))), 1)
        steps.append((DECUP, ka_for(DECUP, [(0x00, BIN_), (0x08, _dskip), (0x10, _dout),
                                            (0x18, wt['block39'][0])],
                                    [(0x20, '<ii', tuple(_dd))], g_du), g_du, 256))

    if os.environ.get('C512_TRACE') == '2' and os.environ.get('PROBE_VIT') == '1':
        C512_PROBE.append(('after block39 -> b39', off_b39, len(steps)))
        C512_PROBE.append(('after block39 -> c512_2[0]', c512_2[0], len(steps)))
    # k_dec_upsample writes through +0x10, not +0x08: after block 39 c512_2[0] holds 243
    # distinct byte values while off_b39 is entirely zero. The second C=512 stage was being
    # fed the empty buffer.
    c512_stage(range(40, 48), c512_2, c512_2[0])

    # The mid->dec transition added earlier is withdrawn: the phase table has ONE k_repack
    # and ONE k_dec_upsample there, and block 39 already is that dispatch. The decoder's
    # odd first-block records are the upsample + skip concat, so the upsample happens
    # inside block 48's own k_swin_var call, not in a separate kernel. The second
    # k_dec_upsample was also running on block48's k_swin_var weight record.
    # C512_HOST: the decoder reads ctx+0x2a0 (0x180030849), block 47's second output, not ctx+0x228.
    stage_in = off_2a0 if os.environ.get('C512_HOST', '1') == '1' else c512_2[0]

    # decoder blocks 48-69, the encoder mirrored
    for di, (key, blocks, C) in enumerate(DEC_STAGES):
        sym, lds = D.SYMS[key]
        lds = lds or D.group_size(sym)
        h, w, C, _n = dec_geom[di]
        ping, pong, pool = dec_buf[di]
        pp = [ping, pong]
        for i, blk in enumerate(blocks):
            ox, oy = ENC_MODES[i % 4]
            last = (i == len(blocks) - 1)
            # The decoder's flag convention is NOT the encoder's. The odd-sized record in each
            # decoder stage is the FIRST block, not the last (block 66 = 22,784 B matches the
            # f=8 extent, block 69 = 20,672 B the ordinary f=0/1/2 one), so first blocks take
            # bit 3 and last blocks take no pool bit. With flags=4 the last block runs the
            # fused pool and emits a C-doubled, H/W-halved tensor the next stage cannot use.
            _dflags = int(os.environ.get('DEC_FLAGS', '1'))
            # R5, read from the host's decoder loop: the first block takes flag 8 (literal 0x8 at
            # 0x180030933) and the LAST block takes flag 2 (xor r8d,r8d / cmp ebx,edi / sete r8b /
            # add r8d,r8d at 0x180030a0e..0x180030a22). Every block in between takes 0. The runner
            # has never set bit 1 on any dispatch. DEC_LAST2=0 restores the old 8/0/0 sequence.
            _last2 = 2 if (last and os.environ.get('DEC_LAST2', '1') == '1') else 0
            flags = (((8 if i == 0 else 0) | _last2) if _dflags else
                     ((1 if i == 0 else 0) | (4 if last else 0)))
            grid = ((w - ox + 7) // 8, (h - oy + 7) // 8)
            if i == 0 and os.environ.get('DEC_SKIP', '1') == '1':
                # Decoder stage d's first block (flag 8) takes the U-net skip at +0x00: the
                # encoder buffer at this stage's own C and resolution. The previous stage's
                # 2C/half-resolution output goes to +0x38 - HYPOTHESIS, the mirror of the
                # encoder's fused pool write (C:113-119). DEC_SKIP=0 restores the old wiring.
                src, p38 = enc_stage_out.get(3 - di, stage_in), stage_in
            else:
                src, p38 = (stage_in if i == 0 else pp[(i + 1) % 2]), pool
            # DEC_PTRA=1 follows R5: the previous stage goes to +0x30 (ptrA), not +0x38.
            _pa = p38 if (i == 0 and os.environ.get('DEC_PTRA', '1') == '1'
                          and os.environ.get('DEC_SKIP', '1') == '1') else None
            steps.append((sym, enc_kernarg(h, w, oy, ox, flags, grid, src, pp[i % 2],
                                           wt['block%d' % blk][0],
                                           pool if _pa is not None else p38, ptrA=_pa), grid, 256))
            if os.environ.get('C512_TRACE') == '2' and os.environ.get('PROBE_DEC') == '1':
                C512_PROBE.append(('dec block %-3d-> dst' % blk, pp[i % 2], len(steps)))
            last_dst = pp[i % 2]
        # The pooled output feeds the NEXT stage. After the last decoder stage there is no next
        # stage, so the head must read the last block's own output, not a pool buffer that nothing
        # downsampled into. HEAD_SRC=pool restores the old wiring for comparison.
        # Without the pool bit nothing writes `pool`, so every decoder stage hands on its last
        # block's own output - to the next stage and, at the end, to the head.
        stage_in = pool if _dflags == 0 and di + 1 < len(DEC_STAGES) else last_dst

    # HEAD_SRC=enc1 is an isolation test: feed the post-block a buffer known to vary (same size as
    # its real input) instead of the decoder's output, to tell a wiring bug from an upstream one.
    if os.environ.get('HEAD_SRC') == 'enc1':
        stage_in = stage_buf[0][2]
    # (k_final_head / HEAD_GRID dispatch removed here: dead code, never appended to steps. The
    # host's tail is NOT k_final_head. The driver's epilogue runs the post-block (handle
    # 0x180066350) and then k_swin_var<32,true> with flags 0x20. Only the post-block writes
    # ctx+0x100, which k_export reads, so B is deferred until its flags are understood.
    # Layout from the stores at 0x180030e13..0x180030ea4, base rbp+0x280:
    #   +0x00 [rbp+0x668] <- [rbp+0x608] at 0x180030b0a, the decoder loop's carried output
    #   +0x08 ctx+0x218   the pre-block output, i.e. the outermost U-net skip
    #   +0x10 ctx+0x100   THE OUTPUT
    #   +0x18 block 70 layer 0 (both lookups pass r8d=0; blend_scale is never asked for)
    #   +0x20/+0x24 a qword load of ctx+0x18: H, W
    #   +0x30 f32 ctx+0x30, +0x34 i32 1, +0x48 f32 xmm6 - frame constants, values unknown
    #   +0x38 ctx+0xf8    the k_import float RGB
    #   +0x40 ctx+0x118   = ctx+0x108, H*W*4
    # Grid: 256x1x1 threads over (W//8, H//8). The /8 TRUNCATES (psrad/psrld/paddd/psrad at
    # 0x180030dbe), which is exact only because the dims are padded to a multiple of 128.
    # ctx+0x30 = [DlssNrOnAmd] Scale from the ini, default 0.03125 (0x1800082b0-0x1800082d3 -> 0x18009ad14,
    # copied to ctx+0x30 each frame at 0x18001a735). Not unwritten: the store is RIP-relative.
    POST_F30 = float(os.environ.get('POST_F30', '0.03125'))
    # xmm6 at 0x180030ea4 has two paths: `pxor xmm6,xmm6` (0x180030cf0) and `movaps xmm6,xmm7`
    # (0x180030cfd). The zero path is the right one: at 1.0 the frame comes out 17x too dark
    # (mean 6.4 vs the import's 108.7) with even columns carrying 1.88x the odd ones; at 0.0
    # the mean is 46.6 and the column ratio falls to 1.12.
    # Host read overturns that: the zero path is taken only when env DLSSNR_NOBLEND is set
    # (getenv at 0x1800314d6-0x1800314e5 -> byte [0x18009b1f8]). Otherwise xmm6 = ctx+0xd8, which is the
    # f16 tensor 'block70.layer0.blend_scale' read back to the host at 0x180030b7c-0x180030c4f.
    POST_F48 = float(os.environ['POST_F48']) if 'POST_F48' in os.environ else float(np.fromfile(WEIGHTS / 'block70_layer0_blend_scale.bin', np.float16)[0])
    print('post-block +0x30 Scale=%g  +0x48 blend_scale=%g' % (POST_F30, POST_F48))
    g_post = (PAD_W // 8, PAD_H // 8)
    # POST_CONST is a set of field offsets to replace with constant buffers, e.g.
    # POST_CONST=0x00,0x08,0x38,0x40 for M2 run 5.
    _pc = {int(x, 0) for x in os.environ.get('POST_CONST', '').split(',') if x.strip()}
    # POST_A00_FILE loads a raw byte image into the constant buffer and feeds it to A at +0x00.
    # That lets a baseline decoder output be replayed with a few bytes changed, so A's response to
    # a single input byte can be measured without intervening mid-chain.
    _a00f = os.environ.get('POST_A00_FILE')
    if _a00f:
        _blob = np.fromfile(_a00f, np.uint8)
        arena[off_k_act:off_k_act + _blob.size] = _blob
        print('A +0x00 loaded from %s (%d B)' % (_a00f, _blob.size))
    _p00 = off_k_act if (_a00f or 0x00 in _pc) else stage_in
    _p08 = off_k_act if 0x08 in _pc else off_a
    _p38 = off_k_rgb if 0x38 in _pc else off_rgb
    _p40 = off_k_f32 if 0x40 in _pc else off_ctx118
    steps.append((POST, ka_for(POST, [(0x00, _p00), (0x08, _p08), (0x10, off_netout),
                                      (0x18, wt['block70_layer0'][0]),
                                      (0x38, _p38), (0x40, _p40)],
                               [(0x20, '<ii', (PAD_H, PAD_W)), (0x28, '<Q', (0,)),
                                (0x30, '<f', (POST_F30,)), (0x34, '<i', (1,)),
                                (0x48, '<f', (POST_F48,))], g_post), g_post, 256))

    # k_export, fed the network's own output at +0x00 - the first time it has had that
    # k_export's kernarg is 320 B (kernel metadata), not 280, and grid_dims lives at +0x80 -
    # it was never set, so the kernel saw 0 instead of 2 and treated a 2D grid as something
    # else. It wrote rows 0..69 of 960 and stopped.
    ka = bytearray(M[EXPORT][1] if EXPORT in M else 320)
    struct.pack_into('<Q', ka, 0x00, BASE + off_netout)   # ctx+0x100 (0x18002d7ca)
    struct.pack_into('<i', ka, 0x08, 0)
    # +0x0c is HEIGHT and +0x10 is WIDTH, not the other way round: the launcher builds the grid
    # as (ceil(width/256), height) from exactly these two fields, so swapping them makes the kernel
    # address the surface with the wrong stride.
    struct.pack_into('<ii', ka, 0x0c, SRC_H, SRC_W)
    # +0x08 = input row stride in ELEMENTS, +0x14 = output row PITCH in BYTES, +0x18 = mode (see EXPORT_FINDINGS.md)
    struct.pack_into('<ii', ka, 0x14, SRC_W * 8, 0)
    struct.pack_into('<Q', ka, 0x20, BASE + off_dst)
    # +0x28 is ebp in the launcher. It is NOT the output format - that is +0x18 (s7 from
    # s_load_b128 s[4:7], 0xc at 0xab20c, compared at 0xab2d4ff). The kernel reads +0x28 only when
    # +0x38 == 0 (0xab76c), where nonzero selects the v_exp_f32 branch. EXPORT_MODE keeps its name.
    struct.pack_into('<i', ka, 0x28, int(os.environ.get('EXPORT_MODE', '-1')))   # same ebp as k_import +0x28
    struct.pack_into('<i', ka, 0x08, PAD_W)   # input row stride: ctx+0x100 is PAD_W wide
    struct.pack_into('<Q', ka, 0x30, BASE + off_rgb)
    # +0x38/+0x3c come from xmm6/xmm7 in the frame function (0x18002d476-0x18002d488), which reads
    # them from the job struct (its 10th argument). The host read (0x18001a0a4-0x18001a822) settles both:
    #   +0x38 = job +0x40, an INT history-valid flag, not a float: 0 at init, set to 1 only when
    #           byte [0x18009ac5c] == 1 (setns over the history resource creation calls, 0x1800152e4).
    #           The null-job call site (0x180015b24) gives 0 as well.
    #   +0x3c = job +0x3c = f32 table [0x180096cd0 + 4*slot], which is 1.0 for all four slots; 1.0
    #           (0x18006d2c8) again for the null job.
    # EXP_HIST packs +0x38 as the int flag. EXP_S0 still overrides it with a raw float, for comparison.
    struct.pack_into('<i', ka, 0x38, int(os.environ.get('EXP_HIST', '0')))
    if 'EXP_S0' in os.environ:
        struct.pack_into('<f', ka, 0x38, float(os.environ['EXP_S0']))
    struct.pack_into('<f', ka, 0x3c, float(os.environ.get('EXP_S1', '1.0')))
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
    # Saved fixtures are k_repack outputs placed in c512_1[0]. Without the (non-host) encoder repack the
    # stage reads the encoder pool instead, so loading one there would silently do nothing.
    if os.environ.get('REPACK_ENC') != '1' or os.environ.get('NO_REPACK') == '1':
        raise SystemExit('FIX_IN_LOAD needs REPACK_ENC=1: the fixtures are k_repack outputs, and the default '
                         'no longer runs that repack (FRAME_STATE Update 20)')
    SKIP = REPACK_END
    _fx = np.fromfile(os.environ['FIX_IN_LOAD'], np.uint8)
    assert _fx.size == N512, (_fx.size, N512)
    arena[c512_1[0]:c512_1[0] + N512] = _fx


def _run_prefix(nsteps):
    """Dispatch only the first nsteps and return the resulting arena."""
    b = b''; ln = []
    for sym, k, grid, thr in steps[SKIP:nsteps]:
        o = len(b); b += k
        line = '%s|%s|%d|%d|%d|%d|%d' % (MOD(sym), sym, o, len(k), grid[0], grid[1], thr)
        if len(grid) > 2:
            line += '|%d' % grid[2]
        ln.append(line)
    (OUT / 'tm.txt').write_text(chr(10).join(ln) + chr(10))
    (OUT / 'tk.bin').write_bytes(b)
    (OUT / 'ta.bin').write_bytes(arena.tobytes())
    rr = subprocess.run([str(RUN), str(OUT / 'tm.txt'), str(OUT / 'tk.bin'),
                         str(OUT / 'ta.bin'), '%x' % BASE], capture_output=True, text=True, timeout=1800)
    if rr.returncode != 0:
        return None
    return np.fromfile(OUT / 'ta.bin', np.uint8)


def e4m3_dec(b):
    e = ((b >> 3) & 15).astype(np.float32); m = (b & 7).astype(np.float32)
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * np.power(2.0, e - 7))
    return np.where((e == 15) & (m == 7), np.nan, np.where(b & 128, -v, v))


if os.environ.get('ENC_EMU_CHECK'):
    # Check ONE encoder block against the gfx1100 original on the frame's real data, at chosen workgroups.
    # The GPU runs the chain up to just before and just after the block; the emulator then runs the block's
    # exact kernarg on the before-arena, for the workgroups covering DEFECT_MASK (plus ENC_EMU_CTRL controls),
    # and every byte the emulator writes is compared with the GPU's after-arena.
    import gfx11emu as E, run_emu as R, time as _time
    _blk = int(os.environ['ENC_EMU_CHECK'])
    _idx, _h, _w, _C, _ox, _oy, _grid, _sym, _lds = ENC_BLK_INFO[_blk]
    _before, _after = _run_prefix(_idx), _run_prefix(_idx + 1)
    assert _before is not None and _after is not None, 'GPU prefix run failed'
    _f = PAD_H // _h
    _dm = np.load(os.environ['DEFECT_MASK']); _bl = int(os.environ.get('DEFECT_BLOCK', '16'))
    _full = np.zeros((PAD_H, PAD_W), bool)
    _up = np.repeat(np.repeat(_dm, _bl, 0), _bl, 1); _full[:_up.shape[0], :_up.shape[1]] = _up
    _wgs = []
    for wy in range(_grid[1]):
        for wx in range(_grid[0]):
            y0, x0 = (wy * 8 + _oy) * _f, (wx * 8 + _ox) * _f      # the 8x8 window this workgroup owns
            if _full[max(0, y0):max(0, y0 + 8 * _f), max(0, x0):max(0, x0 + 8 * _f)].any():
                _wgs.append((wx, wy, 'defect'))
    _nd = len(_wgs)
    for c in os.environ.get('ENC_EMU_CTRL', '2,2;10,5').split(';'):
        wx, wy = (int(t) for t in c.split(',')); _wgs.append((wx, wy, 'control'))
    _wgs = _wgs[:int(os.environ.get('ENC_EMU_MAX', '12'))]
    print('ENC_EMU_CHECK block %d (%s) %dx%d C=%d origin (%d,%d) grid %s: %d defect workgroups, running %d'
          % (_blk, _sym, _h, _w, _C, _ox, _oy, _grid, _nd, len(_wgs)), flush=True)
    _prog = E.load_program(R.DIS, {_sym})
    _g = E.GMem()
    _KA = 0x7000_0000_0000
    _g.add('kernarg', _KA, np.frombuffer(steps[_idx][1], np.uint8).copy())
    _g.add('arena', BASE, _before.copy())
    sys.path.insert(0, str(D.ROOT))
    import translate_final_head as _tb
    _, _secs, _ = _tb.kd.parse_elf(str(_tb.INPUT))
    _ro = next(x for x in _secs if x['name'] == '.rodata')
    _g.add('rodata', _ro['addr'], np.frombuffer(bytearray((D.ROOT / 'build' / 'kernels-hw-scratch' / 'initialized-rodata.bin').read_bytes()), np.uint8).copy())
    _arr = _g.regions[1].arr
    for wx, wy, kind in _wgs:
        _t0 = _time.time()
        _st = E.run_workgroup(_prog, _g, _lds, 256, {0: _KA & 0xffffffff, 1: _KA >> 32, 14: wx, 15: wy},
                              max_steps=60_000_000)['steps']
        print('  wg (%3d,%3d) %-7s steps %9d  %5.0fs' % (wx, wy, kind, _st, _time.time() - _t0), flush=True)
    _w_ = np.nonzero(_arr != _before)[0]
    _mis = _w_[_after[_w_] != _arr[_w_]]
    print('  emulator wrote %d B across these workgroups; mismatches vs GPU: %d' % (len(_w_), len(_mis)))
    if len(_mis):
        _ev = np.nan_to_num(e4m3_dec(_arr[_mis])); _gv = np.nan_to_num(e4m3_dec(_after[_mis]))
        print('  first: ' + '  '.join('@%#x emu=%02x gpu=%02x' % (int(i), _arr[i], _after[i]) for i in _mis[:6]))
        print('  mismatching values: emu |x| mean %.3g max %.3g   gpu |x| mean %.3g max %.3g'
              % (np.abs(_ev).mean(), np.abs(_ev).max(), np.abs(_gv).mean(), np.abs(_gv).max()))
    raise SystemExit(0)

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
        if os.environ.get('DEFECT_MASK') and label in ENC_PROBE_GEOM:
            # Per-block defect trace: mean |x| and saturated codes inside the DEFECT_MASK blocks vs outside,
            # reading the encoder ping/pong in its footprint-settled layout (4x4-tile-major, FRAME_STATE Update 20).
            _h, _w, _C = ENC_PROBE_GEOM[label]
            _b = aa[buf:buf + _C * _h * _w]
            _bl = int(os.environ.get('DEFECT_BLOCK', '16'))
            _dm = np.load(os.environ['DEFECT_MASK'])
            _full = np.zeros((PAD_H, PAD_W), bool)
            _up = np.repeat(np.repeat(_dm, _bl, 0), _bl, 1); _full[:_up.shape[0], :_up.shape[1]] = _up
            _f = PAD_H // _h
            _m = _full[:_h * _f, :_w * _f].reshape(_h, _f, _w, _f).any(axis=(1, 3))
            _val = np.zeros((_h, _w), bool); _val[:-(-SRC_H // _f), :-(-SRC_W // _f)] = True
            _B = _b.reshape(_h // 4, _w // 4, _C, 4, 4).transpose(2, 0, 3, 1, 4).reshape(_C, _h, _w)
            _e = ((_B >> 3) & 15).astype(np.float32); _mm = (_B & 7).astype(np.float32)
            _V = np.where(_e == 0, _mm / 8 * 2.0 ** -6, (1 + _mm / 8) * np.power(2.0, _e - 7))
            _ax = _V.mean(0); _sat = ((_B & 0x7f) == 0x7e).mean(0)
            _in, _out = _m & _val, ~_m & _val
            print('  %s  |x| in/out %7.3f / %7.3f (x%5.2f)   sat in/out %.2e / %.2e   max|x| in %.0f'
                  % (label, float(_ax[_in].mean()), float(_ax[_out].mean()), float(_ax[_in].mean() / _ax[_out].mean()),
                     float(_sat[_in].mean()), float(_sat[_out].mean()), float(_V[:, _in].max())), flush=True)
            continue
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
    line = '%s|%s|%d|%d|%d|%d|%d' % (MOD(sym), sym, o, len(k), grid[0], grid[1], thr)
    if len(grid) > 2:
        line += '|%d' % grid[2]
    lines.append(line)
(OUT / 'manifest.txt').write_text('\n'.join(lines) + '\n')
(OUT / 'kernargs.bin').write_bytes(blob)
# REUSE_ARENA=<file>: skip the GPU run and analyse a saved post-run arena instead (same env, so the same
# buffer offsets). For iterating on the probes below without re-running the frame. Nothing is dispatched.
_reuse = os.environ.get('REUSE_ARENA')
if not _reuse:
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
else:
    print('REUSE_ARENA=%s: no dispatch, analysing a saved arena' % _reuse)

res = np.fromfile(_reuse or (OUT / 'arena.bin'), np.uint8)


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
# k_import writes at the PADDED pitch (it is given PAD_H/PAD_W and a padded grid, and the buffer is
# PAD_H*PAD_W*12). Reading it packed at SRC_W shears every row by PAD_W-SRC_W = 85 px, and imported.png -
# the reference smid.py scores against - was that sheared image: adjacent-row corr 0.21 vs 0.75.
rgb = res[off_rgb:off_rgb + PAD_H * PAD_W * 12].view(np.float32).reshape(PAD_H, PAD_W, 3)[:SRC_H, :SRC_W]
report('k_import RGB', res[off_rgb:off_rgb + SRC_H * SRC_W * 12])
a_nan, a_nz = report('block0 out', res[off_a:off_a + S1])
report('block0 +0x38', res[off_p38:off_p38 + S1])
if os.environ.get('SENTINEL') == '1':
    print()
    print('=== sentinel extents (0xA5 = untouched) ===')
    for _nm, _o, _n in placements:
        if _nm not in SENT_BUFS:
            continue
        _x = res[_o:_o + _n]
        _w = np.nonzero(_x != 0xA5)[0]
        print('  %-18s alloc %10d  written %10d  highest offset %10s  sha %s'
              % (_nm, _n, _w.size, (int(_w[-1]) if _w.size else -1),
                 hashlib.sha256(_x.tobytes()).hexdigest()[:16]))
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
                        + [('c512_2 w%d' % i, o, N512) for i, o in enumerate(c512_2)]
                        # off_vit is only written with MID_HOST=0 and VIT_SEP=1 (line ~755). With MID_HOST=1
                        # (default) it is never touched, so reporting it there showed an empty buffer for
                        # a path that did not run. The MID_HOST=1 ViT lives in off_tok (ctx+0x258/0x290)
                        # and its result is repacked into off_headb (ctx+0x250), block 39's +0x00.
                        + ([('vit tok%d' % i, o, NTOK * 1024 * 2) for i, o in enumerate(off_tok)]
                           + [('head ctx+0x250', off_headb, HEAD_TILES * 16384 * 2)] if MID_HOST else [])
                        + ([('vit out (off_vit)', off_vit, N1024)]
                           if not MID_HOST and os.environ.get('VIT_SEP') == '1' else [])
                        # off_b39 is only written with MID_HOST=0. With MID_HOST=1 block 39 writes c512_2[0]
                        # (+0x10), so reporting off_b39 there showed an unused, all-zero buffer as 'block39 out'.
                        + ([] if MID_HOST else [('block39 out', off_b39, N512)])):
        _x = res[_o:_o + _n]
        print('mid %-12s %9d B: nonzero=%6.2f%% distinct=%3d' % (_nm, _n, 100.0 * float((_x != 0).mean()), len(np.unique(_x))))
    for _di, (_k, _b, _C) in enumerate(DEC_STAGES):
        _h, _w, _C, _n = dec_geom[_di]
        for _nm, _o in zip(('ping', 'pong', 'pool'), dec_buf[_di]):
            _x = res[_o:_o + _n]
            print('dec s%d %-4s C=%-3d %10d B: nonzero=%6.2f%% distinct=%3d' % (_di + 1, _nm, _C, _n, 100.0 * float((_x != 0).mean()), len(np.unique(_x))))
    # off_head/HEAD (k_final_head, grid HEAD_GRID) is dead: the host's real tail is POST_BLOCK +
    # k_swin_var (see the "block 70" comment above), which writes off_netout, not off_head. Nothing
    # in the live pipeline writes off_head, so printing it here only produced a misleading always-
    # empty "head buffer" line - removed. off_netout's content is covered by the k_export line below.
    _n8 = 1601
    _hi = res[stage_in:stage_in + _n8 * 8192].reshape(_n8, 8192)
    _nzc = (_hi != 0).any(axis=1)
    print('head input       : stage_in=%#x, %d chunks of 8192 B: %d nonzero chunks (last %s), %.2f%% bytes nonzero, distinct=%d'
          % (stage_in, _n8, int(_nzc.sum()), int(np.nonzero(_nzc)[0][-1]) if _nzc.any() else None, 100.0 * float((_hi != 0).mean()), len(np.unique(_hi))))
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

    if os.environ.get('STRUCT_PROBE') == '1':
        # Where does the network lose the input's structure? Per stage buffer: decode NCHW16c e4m3, band-pass
        # each channel and the input luminance (resampled to that buffer's grid), and report the best |corr| over
        # channels plus the fraction of channels above 0.2. A stage that still sees the image has a channel
        # well above noise; the first stage where both collapse is where it is lost.
        def _blur(a, r):
            k = np.ones(2 * r + 1) / (2 * r + 1)
            a = np.apply_along_axis(lambda m: np.convolve(m, k, 'same'), 0, a)
            return np.apply_along_axis(lambda m: np.convolve(m, k, 'same'), 1, a)

        def _band(a):
            return _blur(a, 1) - _blur(a, 3)

        _lum = np.nan_to_num(rgb) @ np.array([0.2126, 0.7152, 0.0722], np.float32)

        def _lum_at(h, w):
            # Pad to the padded frame, then block-average to h x w.
            L = np.zeros((PAD_H, PAD_W), np.float32); L[:SRC_H, :SRC_W] = _lum
            fy, fx = PAD_H // h, PAD_W // w
            return L[:h * fy, :w * fx].reshape(h, fy, w, fx).mean(axis=(1, 3)), fy

        # Candidate byte layouts, each mapped to (C, h, w). NCHW16c is the one proven by the 16-impulse probe
        # (FRAME_STATE, corrections). The tile layouts are candidates for buffers where NCHW16c reads ~0 while
        # the pool computed from them reads ~1 (Update 3), and for the C=512 stage (one 4x4 tile x 512 ch
        # = 8192 B per workgroup). STRUCT_LAYOUTS=all tries every one; the right layout is the one that
        # sees the image, since a wrong layout scrambles pixels and destroys spatial correlation.
        _LAYOUTS = {
            'nchw16c':    lambda b, C, h, w: b.reshape(C // 16, h, w, 16).transpose(0, 3, 1, 2),
            'cb_t4_c16':  lambda b, C, h, w: b.reshape(C // 16, h // 4, w // 4, 4, 4, 16).transpose(0, 5, 1, 3, 2, 4),
            'tile_c_pix': lambda b, C, h, w: b.reshape(h // 4, w // 4, C, 4, 4).transpose(2, 0, 3, 1, 4),
            'tile_pix_c': lambda b, C, h, w: b.reshape(h // 4, w // 4, 4, 4, C).transpose(4, 0, 2, 1, 3),
            'tile_cb_pix_c16': lambda b, C, h, w: b.reshape(h // 4, w // 4, C // 16, 4, 4, 16).transpose(2, 5, 0, 3, 1, 4),
            # the same tile orders with tiles enumerated column-major (x outer), in case the grid walks x fastest
            'tileT_c_pix': lambda b, C, h, w: b.reshape(w // 4, h // 4, C, 4, 4).transpose(2, 1, 3, 0, 4),
            'tileT_pix_c': lambda b, C, h, w: b.reshape(w // 4, h // 4, 4, 4, C).transpose(4, 1, 2, 0, 3),
        }
        # IMPULSE_BASE=<arena> (with REUSE_ARENA=<impulse arena>, IMPULSE_CELLS='y,x;y,x;...' in 32-px cells of
        # the padded frame): instead of correlating with luminance, map |impulse - base| per pixel under each
        # layout and report how much of that energy lands near the impulse centres ('in') against the area
        # those neighbourhoods cover ('area'). lift = in/area: ~1 means scattered (wrong layout, or no spatial
        # signal left), >>1 means the footprint sits where the impulses were.
        _ib = os.environ.get('IMPULSE_BASE')
        _base = np.fromfile(_ib, np.uint8) if _ib else None
        _cells = [tuple(int(t) for t in c.split(',')) for c in os.environ.get('IMPULSE_CELLS', '').split(';') if c]
        _lay_sel = os.environ.get('STRUCT_LAYOUTS', 'nchw16c')
        _lay_sel = list(_LAYOUTS) if _lay_sel == 'all' else _lay_sel.split(',')
        _maxch = int(os.environ.get('STRUCT_MAXCH', '64'))   # channels sampled per buffer (speed)

        def _impulse(name, off, h, w, C):
            n = C * h * w
            d = np.abs(np.nan_to_num(e4m3(res[off:off + n]).astype(np.float32))
                       - np.nan_to_num(e4m3(_base[off:off + n]).astype(np.float32)))
            f = PAD_H // h
            yy, xx = np.mgrid[0:h, 0:w]
            rad = max(1.5, 0.08 * w)                       # neighbourhood radius in this buffer's pixels
            mask = np.zeros((h, w), bool)
            for cy, cx in _cells:
                mask |= (yy - (cy * 32 + 16) / f) ** 2 + (xx - (cx * 32 + 16) / f) ** 2 <= rad * rad
            area = float(mask.mean())
            for lay in _lay_sel:
                try:
                    E_ = _LAYOUTS[lay](d, C, h, w).reshape(C, h, w).sum(axis=0)
                except ValueError:
                    print('  %-22s %-15s n/a for %dx%d' % (name, lay, h, w)); continue
                tot = float(E_.sum())
                if tot <= 0:
                    print('  %-22s %-15s no difference at all' % (name, lay)); continue
                inn = float(E_[mask].sum()) / tot
                print('  %-22s %-15s %4dx%-4d changed=%5.1f%%  in=%.3f area=%.3f lift=%5.2f'
                      % (name, lay, h, w, 100.0 * float((d != 0).mean()), inn, area, inn / max(area, 1e-9)))

        # DEFECT_MASK=<.npy>: a boolean map over the source frame in DEFECT_BLOCK-px blocks (default 16) marking
        # a visible defect (e.g. the dark blocks). Per buffer and layout, compare per-pixel statistics inside the
        # mask with outside: NaN codes (0x7f/0xff), saturated codes (|x| = 448, 0x7e/0xfe), zero bytes, mean |x|.
        # The first buffer whose statistics differ inside the mask is where the defect enters.
        _dm = os.environ.get('DEFECT_MASK')
        _dmask = np.load(_dm) if _dm else None
        _dblk = int(os.environ.get('DEFECT_BLOCK', '16'))

        def _defect(name, off, h, w, C):
            n = C * h * w
            b = res[off:off + n]
            full = np.zeros((PAD_H, PAD_W), bool)
            up = np.repeat(np.repeat(_dmask, _dblk, 0), _dblk, 1)
            full[:up.shape[0], :up.shape[1]] = up
            f = PAD_H // h
            m = full[:h * f, :w * f].reshape(h, f, w, f).any(axis=(1, 3))
            valid = np.zeros((h, w), bool); valid[:-(-SRC_H // f), :-(-SRC_W // f)] = True
            inside, outside = m & valid, ~m & valid
            if not inside.any():
                print('  %-22s mask empty at %dx%d' % (name, h, w)); return
            v = np.nan_to_num(e4m3(b).astype(np.float32))
            for lay in _lay_sel:
                try:
                    B_ = _LAYOUTS[lay](b, C, h, w).reshape(C, h, w)
                    V_ = _LAYOUTS[lay](v, C, h, w).reshape(C, h, w)
                except ValueError:
                    print('  %-22s %-15s n/a for %dx%d' % (name, lay, h, w)); continue
                feats = (('nan', ((B_ & 0x7f) == 0x7f).mean(0)), ('sat', ((B_ & 0x7f) == 0x7e).mean(0)),
                         ('zero', (B_ == 0).mean(0)), ('|x|', np.abs(V_).mean(0)))
                print('  %-22s %-15s %4dx%-4d in/out: %s' % (name, lay, h, w, '  '.join(
                    '%s %.3g/%.3g' % (k, float(a[inside].mean()), float(a[outside].mean())) for k, a in feats)))

        def _probe(name, off, h, w, C):
            if _dmask is not None:
                return _defect(name, off, h, w, C)
            if _base is not None:
                return _impulse(name, off, h, w, C)
            n = C * h * w
            raw = np.nan_to_num(e4m3(res[off:off + n]).astype(np.float32))
            L, f = _lum_at(h, w)
            vy, vx = min(h, -(-SRC_H // f)), min(w, -(-SRC_W // f))   # valid (unpadded) region
            bl = _band(L)[:vy, :vx].ravel(); bl = (bl - bl.mean()) / max(bl.std(), 1e-12)
            chs = np.unique(np.linspace(0, C - 1, min(C, _maxch)).astype(int))
            for lay in _lay_sel:
                try:
                    v = _LAYOUTS[lay](raw, C, h, w).reshape(C, h, w)
                except ValueError:
                    print('  %-22s %-15s n/a for %dx%d' % (name, lay, h, w)); continue
                rs, p8 = [], []
                for c in chs:
                    bc = _band(v[c])[:vy, :vx].ravel()
                    s = bc.std()
                    rs.append(0.0 if s < 1e-12 else float(np.dot(bl, (bc - bc.mean()) / s) / bl.size))
                    # ph8: share of the column profile's variance that is periodic in x mod 8 (the output's
                    # stripe period, in this buffer's own pixels). ~8/width for no periodicity, 1 for pure.
                    m = v[c][:vy, :vx].mean(axis=0); m = m - m.mean(); tv = float((m * m).mean())
                    if tv > 1e-20 and vx >= 16:
                        ph = np.array([m[k::8].mean() for k in range(8)])
                        p8.append(float((ph * ph).mean()) / tv)
                rs = np.abs(np.array(rs))
                print('  %-22s %-15s %4dx%-4d C=%-4d best|r|=%.3f (ch %3d)  ch>0.2: %5.1f%%  ph8(med)=%.3f'
                      % (name, lay, h, w, C, rs.max(), int(chs[rs.argmax()]), 100.0 * float((rs > 0.2).mean()),
                         float(np.median(p8)) if p8 else float('nan')))

        print('\n=== structure probe (band-passed corr with input luminance) ===')
        for si, (key, blocks, C) in enumerate(ENC_STAGES):
            h, w, C, n = stage_geom[si]
            for _nm, _o in zip(('ping', 'pong', 'pool'), stage_buf[si]):
                try:
                    _probe('enc s%d %s' % (si + 1, _nm), _o, h if _nm != 'pool' else h // 2,
                           w if _nm != 'pool' else w // 2, C)
                except Exception as e:
                    print('  enc s%d %s: %s' % (si + 1, _nm, e))
        for _nm, _bufs in (('c512_1', c512_1), ('c512_2', c512_2)):
            for i, _o in enumerate(_bufs):
                try:
                    _probe('%s w%d' % (_nm, i), _o, H5, W5, 512)
                except Exception as e:
                    print('  %s w%d: %s' % (_nm, i, e))
        for di, (_k, _b, C) in enumerate(DEC_STAGES):
            h, w, C, n = dec_geom[di]
            for _nm, _o in zip(('ping', 'pong'), dec_buf[di][:2]):
                try:
                    _probe('dec s%d %s' % (di + 1, _nm), _o, h, w, C)
                except Exception as e:
                    print('  dec s%d %s: %s' % (di + 1, _nm, e))
