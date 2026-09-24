"""Run one complete C=512 attention block (block 23) as a five-dispatch chain on the gfx1030.

The recipe and every buffer here come from decoding the launcher at 0x180033660, not from guessing:

    1. k_ffwd_inpview   in            -> ctx+0x230   layer 0   (first block of the stage only)
    2. k_ffwd2          ctx+0x228     -> ctx+0x230   layer 0
    3. k_conv_res_views ctx+0x230     -> ctx+0x238   layer 1   (closes the FFN residual)
    4. k_qkv_attn       ctx+0x238     -> ctx+0x240   layer 2
    5. k_conv_res_views ctx+0x240     -> ctx+0x228   layer 3   (closes the attention residual)

ctx+0x228 is the block's persistent activation buffer, so step 5 writing back into it is what makes
block n's output block n+1's input. The four weight layers are the real block23 records.

Checked against the emulator running the identical five-step chain, which is the same standard the
encoder stage-1 chain met.

usage: net_block512.py    (inside sandbox.py)
"""
import sys, re, struct, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D, kernelspec as K

# Which block's weights to run. The recipe is shared across a whole stage, so this is how
# we check it generalises past the single block it was decoded on.
import os as _os
BLOCK = int(_os.environ.get('BLOCK', '23'))

# The C=512 stage is stage 4 of the padded table: at 1707x960 that is 56x32, i.e. 7x4
# workgroups - small enough for the emulator. Verifying the recipe at its REAL geometry is
# the point: passing at 8x8 says nothing about the size it actually runs at.
#
# That intent was never finished: block_count_x/y below were hardcoded to (1,1) for every
# kernel regardless of H,W - both in the kernarg hidden args and in the GPU dispatch
# manifest's own separate |1|1| - so every run at any BH/BW still only ever launched one
# workgroup. Same blind spot as net_vit.py (FRAME_STATE Update 7), same fix: these are 2-D
# k_swin_var-family kernels, grid = (ceil(W/8), ceil(H/8)).
H = int(_os.environ.get('BH', '8'))
W = int(_os.environ.get('BW', '8'))
GX, GY = -(-W // 8), -(-H // 8)
# k_ffwd_inpview and k_conv_res_views are 1-D (workgroup_id_y disabled): the translated prologue
# does s_mov_b32 s15, s2, i.e. the original reads its tile index from s15 and the hardware X id
# supplies it. Giving them (GX, GY) and the emulator {14: wx, 15: wy} made every emulated
# workgroup compute tile 0 while the GPU computed tiles 0..GX-1 - the grid=(2,1) FAIL of
# FRAME_STATE Update 8. The emulator now puts the index in s15. The grid is the host's: the launcher
# 0x180033660 gives both kernels ([rdi], 1, 1) with [rdi] -> ctx[0x308] = ceil(H/4)*ceil(W/4)
# (0x180033787, 0x18003395b, 0x18003400b; ctx[0x308] set at 0x18002e3ed-0x18002e430).
N1D = -(-H // 4) * -(-W // 4)
# STEPS=0,1,2 runs a prefix of the chain (net_vit.py's incremental check); default all five.
STEPS = [int(x) for x in _os.environ.get('STEPS', '0,1,2,3,4').split(',')]
NGROUP = 4                      # min(n*16, H*W); 4*16 == 8*8 so the whole tile is live
# arena slots: the four ctx buffers, then the four weight records
B228, B230, B238, B240, BIN = 0, 1, 2, 3, 4
WL0, WL1, WL2, WL3 = 5, 6, 7, 8
NSLOT = 10
S = lambda i: V.ARENA + i * V.SLOT
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_block512'; OUT.mkdir(parents=True, exist_ok=True)
MOD = lambda s: D.ROOT / 'build' / 'kernels-hw-scratch' / (s + '.co')

FFWD_IV = '_Z14k_ffwd_inpview12FfwdPlParams'
FFWD2 = '_Z7k_ffwd211Ffwd2Params'
CONVV = '_Z16k_conv_res_views12ConvPlParams'
QKV = '_Z10k_qkv_attn10AttnParams'
META = {s: K.kernel_meta(s) for s in (FFWD_IV, FFWD2, CONVV, QKV)}


def steps():
    out = []

    # 1. k_ffwd_inpview: stage input -> ctx+0x230, weight layer 0
    n = META[FFWD_IV][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(BIN))
    struct.pack_into('<Q', ka, 0x08, S(B230))
    struct.pack_into('<Q', ka, 0x10, S(WL0))
    struct.pack_into('<ii', ka, 0x18, H, W)
    out.append((FFWD_IV, ka, META[FFWD_IV]))

    # 2. k_ffwd2: ctx+0x228 -> ctx+0x230, weight layer 0
    n = META[FFWD2][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B228))
    struct.pack_into('<Q', ka, 0x08, S(BIN))
    struct.pack_into('<Q', ka, 0x10, S(B230))
    struct.pack_into('<Q', ka, 0x18, S(WL0))
    struct.pack_into('<iii', ka, 0x20, H, W, NGROUP)
    out.append((FFWD2, ka, META[FFWD2]))

    # 3. k_conv_res_views: ctx+0x230 -> ctx+0x238, weight layer 1
    n = META[CONVV][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B230))
    struct.pack_into('<Q', ka, 0x08, S(BIN))
    struct.pack_into('<Q', ka, 0x10, S(B228))
    struct.pack_into('<Q', ka, 0x18, S(B238))
    struct.pack_into('<i', ka, 0x20, 0)
    struct.pack_into('<Q', ka, 0x28, S(WL1))
    struct.pack_into('<ii', ka, 0x30, H, W)
    out.append((CONVV, ka, META[CONVV]))

    # 4. k_qkv_attn: ctx+0x238 -> ctx+0x240, weight layer 2
    n = META[QKV][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B238))
    struct.pack_into('<Q', ka, 0x08, S(B240))
    struct.pack_into('<Q', ka, 0x10, S(WL2))
    struct.pack_into('<ii', ka, 0x18, H, W)
    struct.pack_into('<ii', ka, 0x20, 0, 0)          # window origin, mode 0 for block 23
    out.append((QKV, ka, META[QKV]))

    # 5. k_conv_res_views: ctx+0x240 -> ctx+0x228, weight layer 3
    n = META[CONVV][1]; ka = bytearray(n)
    struct.pack_into('<Q', ka, 0x00, S(B240))
    struct.pack_into('<Q', ka, 0x10, S(B238))
    struct.pack_into('<Q', ka, 0x18, S(B228))
    struct.pack_into('<i', ka, 0x20, 0)
    struct.pack_into('<Q', ka, 0x28, S(WL3))
    struct.pack_into('<ii', ka, 0x30, H, W)
    struct.pack_into('<ii', ka, 0x40, H, W)
    out.append((CONVV, ka, META[CONVV]))

    final = []
    for sym, ka, (explicit, ksize, lds, hid) in out:
        gx, gy = grid(sym)
        for name, val in (('block_count_x', gx), ('block_count_y', gy), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in hid:
                o, sz = hid[name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        final.append((sym, bytes(ka), lds))
    return [final[i] for i in STEPS]


def is_2d(sym):
    t = (D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.s')).read_text(errors='replace')
    kd = t.split('.amdhsa_kernel ' + sym, 1)[1].split('.end_amdhsa_kernel', 1)[0]
    return '.amdhsa_system_sgpr_workgroup_id_y 1' in kd


def grid(sym):
    return (GX, GY) if is_2d(sym) else (N1D, 1)


def arena(seed):
    a = np.random.default_rng(seed + 55).integers(0, 256, V.SLOT * NSLOT, dtype=np.uint8)
    for slot, fn in ((WL0, ('block%d_layer0.bin' % BLOCK)), (WL1, ('block%d_layer1.bin' % BLOCK)),
                     (WL2, ('block%d_layer2.bin' % BLOCK)), (WL3, ('block%d_layer3.bin' % BLOCK))):
        w = np.frombuffer((D.ROOT / 'build' / 'weights' / fn).read_bytes(), np.uint8)
        a[slot * V.SLOT:slot * V.SLOT + len(w)] = w
    return a


def wants_dispatch_ptr(sym):
    """k_qkv_attn enables dispatch_ptr, so its kernarg arrives in s[2:3] rather than s[0:1]."""
    t = (D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.s')).read_text(errors='replace')
    kd = t.split('.amdhsa_kernel ' + sym, 1)[1].split('.end_amdhsa_kernel', 1)[0]
    return '.amdhsa_user_sgpr_dispatch_ptr 1' in kd


def emulate(seed):
    cur = arena(seed); base = cur.copy()
    for sym, ka, lds in steps():
        prog = E.load_program(R.DIS, {sym})
        g, KA = V.build(seed, ka, NSLOT)
        g.regions[1].arr[:] = cur
        dp = wants_dispatch_ptr(sym)
        gx, gy = grid(sym); two_d = is_2d(sym)
        for wy in range(gy):
            for wx in range(gx):
                ids = {14: wx, 15: wy} if two_d else {14: 0, 15: wx}
                if dp:
                    DP = 0x7100_0000_0000
                    pkt = bytearray(64)
                    struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
                    struct.pack_into('<III', pkt, 12, 256, 1, 1)
                    struct.pack_into('<II', pkt, 24, 64, lds)
                    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
                    sgpr = {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, **ids}
                else:
                    sgpr = {0: KA & 0xffffffff, 1: KA >> 32, **ids}
                E.run_workgroup(prog, g, lds, 256, sgpr, max_steps=30_000_000)
        cur = g.regions[1].arr.copy()
    return base, cur


def net_run(seed):
    blob = b''; lines = []
    for sym, ka, lds in steps():
        o = len(blob); blob += ka
        lines.append('%s|%s|%d|%d|%d|%d|256' % ((MOD(sym), sym, o, len(ka)) + grid(sym)))
    (OUT / 'm.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'k.bin').write_bytes(blob)
    (OUT / 'a.bin').write_bytes(arena(seed).tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'm.txt'), str(OUT / 'k.bin'), str(OUT / 'a.bin'), '%x' % V.ARENA],
                       capture_output=True, text=True, timeout=1800)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'a.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out


print('block %d: C=512 attention block, steps %s, H=%d W=%d' % (BLOCK, STEPS, H, W))
for i, (sym, ka, lds) in zip(STEPS, steps()):
    print('  %d. %-42s lds=%d grid=%s' % (i + 1, sym.split('E')[0][:42], lds, grid(sym)))
rc, msg, out = net_run(1)
print('\n', msg.splitlines()[-1] if msg else 'no output')
if out is None:
    print('GPU FAIL rc=%d\n%s' % (rc, msg)); raise SystemExit(1)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
base, ref = emulate(1)
d = np.nonzero(out != ref)[0]
touched = sorted({int(i // V.SLOT) for i in np.nonzero(ref != base)[0]})
names = {B228: 'ctx+0x228(out)', B230: 'ctx+0x230', B238: 'ctx+0x238', B240: 'ctx+0x240'}
print('emulator chain wrote %d bytes into slots %s  (%s)'
      % (int((ref != base).sum()), touched, ', '.join(names.get(s, 'slot%d' % s) for s in touched)))
print('net_run vs emulator over steps %s: %d mismatches -> %s'
      % (STEPS, len(d), 'PASS' if len(d) == 0 and len(touched) else 'FAIL'))
raise SystemExit(0 if (len(d) == 0 and len(touched)) else 1)
