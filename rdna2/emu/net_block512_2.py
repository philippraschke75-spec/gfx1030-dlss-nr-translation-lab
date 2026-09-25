"""Difftest the host's DEFAULT C=512 block (VIT512_OLD unset, FRAME_STATE Update 11) on the gfx1030.

    step 0  k_ffwd2      in/ctx+0x228 -> ctx+0x230   layer 0   grid ((T+3)/4, 8)
    step 1  k_conv_res2  ctx+0x230    -> ctx+0x238   layer 1   grid ((T+3)/4, 4)
    step 2  k_qkv_attn2  ctx+0x238    -> ctx+0x240   layer 2   grid ((W+7-ox)>>3, (H+7-oy)>>3, 16)
    step 3  k_conv_res2  ctx+0x240    -> ctx+0x228   layer 3   grid ((T+3)/4, 4)

Every field is packed exactly as net_frame_full.py's C512_HOST=1 c512_stage packs it, so a PASS here
covers the kernarg layout the frame uses, not a re-derivation of it. GPU output is compared with the
original gfx1100 code run in the emulator on the same arena and grid.

Workgroup ids, from each translated prologue: k_ffwd2 / k_conv_res2 map hw (x, y) -> s14, s15;
k_qkv_attn2 maps hw (x, y, z) -> s13, s14, s15 (s_mov_b32 s15, s6 ...), z being the head index.

env: BH, BW   geometry (default 32, 56 = stage_hw(4) at 1707x960)
     STEPS    comma list of step indices to run as a prefix chain (default 0,1,2,3)
     FIRST    1 = first block of the stage (input via +0x08/+0x10/+0x20), 0 = later block (default 1)
     MODE     shift-window mode 0..3 -> origin ENC_MODES[MODE] (default 0, block 23)
     BLOCK    weights to use (default 23)

Exit status: 1 if this run failed, else coverage_guard's verdict over build/coverage/net_block512_2.json:
0 only once all four origins have a PASS under the current code (full chain, 32x56, FIRST=1), 2 while any
origin is missing or stale. Each run prints its own result and then the counted verdict.

usage: net_block512_2.py
"""
import sys, re, struct, subprocess, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D, kernelspec as K
import coverage_guard as CG

BLOCK = int(os.environ.get('BLOCK', '23'))
H = int(os.environ.get('BH', '32'))
W = int(os.environ.get('BW', '56'))
FIRST = os.environ.get('FIRST', '1') == '1'
ENC_MODES = [(0, 0), (-4, -4), (-4, 0), (0, -4)]       # table 0x180066410
OX, OY = ENC_MODES[int(os.environ.get('MODE', '0'))]
STEPS = [int(x) for x in os.environ.get('STEPS', '0,1,2,3').split(',')]
T = (-(-H // 4)) * (-(-W // 4))                         # ctx[0x308]

B228, B230, B238, B240, BIN = 0, 1, 2, 3, 4
WL0, WL1, WL2, WL3 = 5, 6, 7, 8
NSLOT = 10
S = lambda i: V.ARENA + i * V.SLOT
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_block512_2'; OUT.mkdir(parents=True, exist_ok=True)
MOD = lambda s: D.ROOT / 'build' / 'kernels-hw-scratch' / (s + '.co')

FFWD2 = '_Z7k_ffwd211Ffwd2Params'
CONV2 = '_Z11k_conv_res211Conv2Params'
QKVA2 = '_Z11k_qkv_attn210AttnParams'
META = {s: K.kernel_meta(s) for s in (FFWD2, CONV2, QKVA2)}
IDS = {FFWD2: (14, 15), CONV2: (14, 15), QKVA2: (13, 14, 15)}


def pack(sym, ptrs, ints, grid):
    explicit, ksize, lds, hid = META[sym]
    ka = bytearray(ksize)
    for off, slot in ptrs:
        struct.pack_into('<Q', ka, off, S(slot))
    for off, fmt, vals in ints:
        struct.pack_into(fmt, ka, off, *vals)
    gx, gy, gz = (tuple(grid) + (1,))[:3]
    for name, val in (('block_count_x', gx), ('block_count_y', gy), ('block_count_z', gz),
                      ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                      ('grid_dims', 3 if gz > 1 else 2)):
        if name in hid:
            o, sz = hid[name]
            struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
    return (sym, bytes(ka), lds, (gx, gy, gz))


def steps():
    gT = -(-T // 4)
    out = []
    # k_ffwd2 (0x18003399c-0x1800339f4)
    p = [(0x10, B230), (0x18, WL0)]
    i = [(0x20, '<ii', (H, W)), (0x28, '<i', (T,))]
    if FIRST:
        p.append((0x08, BIN)); i.append((0x00, '<Q', (0,)))
    else:
        p.append((0x00, B228)); i.append((0x08, '<Q', (0,)))
    out.append(pack(FFWD2, p, i, (gT, 8)))
    # k_conv_res2, layer 1 (0x18003413e-0x1800341bd)
    p = [(0x00, B230), (0x18, B238), (0x28, WL1)]
    i = [(0x20, '<Q', (0,)), (0x30, '<ii', (H, W)), (0x38, '<i', (T,))]
    if FIRST:
        p.append((0x10, BIN)); i.append((0x08, '<Q', (0,)))
    else:
        p.append((0x08, B228)); i.append((0x10, '<Q', (0,)))
    out.append(pack(CONV2, p, i, (gT, 4)))
    # k_qkv_attn2 (0x180033c0f-0x180033cda)
    out.append(pack(QKVA2, [(0x00, B238), (0x08, B240), (0x10, WL2)],
                    [(0x18, '<ii', (H, W)), (0x20, '<ii', (OX, OY))],
                    ((W + 7 - OX) >> 3, (H + 7 - OY) >> 3, 16)))
    # k_conv_res2, layer 3 (0x180033f0d-0x180033f7b)
    p = [(0x00, B240), (0x08, B238), (0x18, B228), (0x28, WL3)]
    i = [(0x10, '<Q', (0,)), (0x30, '<ii', (H, W)), (0x38, '<i', (T,))]
    # +0x20 = the launcher's 5th arg (0x180033e8d): an optional second output, 0 in stage 1
    # (0x18002f9fd) and ctx+0x2a0 only at block 47. OUT2=1 exercises it (into the BIN slot).
    if os.environ.get('OUT2') == '1':
        p.append((0x20, BIN))
    else:
        i.append((0x20, '<Q', (0,)))
    out.append(pack(CONV2, p, i, (gT, 4)))
    return [out[k] for k in STEPS]


def arena(seed):
    a = np.random.default_rng(seed + 55).integers(0, 256, V.SLOT * NSLOT, dtype=np.uint8)
    for slot, n in ((WL0, 0), (WL1, 1), (WL2, 2), (WL3, 3)):
        w = np.frombuffer((D.ROOT / 'build' / 'weights' / ('block%d_layer%d.bin' % (BLOCK, n))).read_bytes(), np.uint8)
        assert len(w) <= V.SLOT
        a[slot * V.SLOT:slot * V.SLOT + len(w)] = w
    return a


def emulate(seed, upto):
    cur = arena(seed); base = cur.copy()
    for sym, ka, lds, (gx, gy, gz) in steps()[:upto]:
        prog = E.load_program(R.DIS, {sym})
        g, KA = V.build(seed, ka, NSLOT)
        g.regions[1].arr[:] = cur
        ids = IDS[sym]
        for wz in range(gz):
            for wy in range(gy):
                for wx in range(gx):
                    sgpr = {0: KA & 0xffffffff, 1: KA >> 32}
                    sgpr.update(zip(ids, (wx, wy, wz)))
                    E.run_workgroup(prog, g, lds, 256, sgpr, max_steps=60_000_000)
                # ~2.2 s per workgroup at 32x56: without this the run looks hung for minutes
                print('    emu %s wz=%d wy=%d/%d' % (sym[:16], wz, wy + 1, gy), flush=True)
        cur = g.regions[1].arr.copy()
    return base, cur


def net_run(seed, upto):
    blob = b''; lines = []
    for sym, ka, lds, (gx, gy, gz) in steps()[:upto]:
        o = len(blob); blob += ka
        lines.append('%s|%s|%d|%d|%d|%d|256|%d' % (MOD(sym), sym, o, len(ka), gx, gy, gz))
    (OUT / 'm.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'k.bin').write_bytes(blob)
    (OUT / 'a.bin').write_bytes(arena(seed).tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'm.txt'), str(OUT / 'k.bin'), str(OUT / 'a.bin'), '%x' % V.ARENA],
                       capture_output=True, text=True, timeout=1800)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'a.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out


NAMES = {B228: 'ctx+0x228', B230: 'ctx+0x230', B238: 'ctx+0x238', B240: 'ctx+0x240'}
print('block %d host-default C=512 chain: H=%d W=%d T=%d first=%d origin=(%d,%d) steps=%s'
      % (BLOCK, H, W, T, FIRST, OX, OY, STEPS))
for k, (sym, ka, lds, grid) in zip(STEPS, steps()):
    print('  %d. %-32s lds=%-6d grid=%s' % (k, sym, lds, grid))
ok = True
for upto in range(1, len(STEPS) + 1):
    rc, msg, out = net_run(1, upto)
    if out is None:
        print('GPU FAIL rc=%d\n%s' % (rc, msg)); raise SystemExit(1)
    V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
    base, ref = emulate(1, upto)
    d = np.nonzero(out != ref)[0]
    wrote = np.nonzero(ref != base)[0]
    slots = sorted({int(i // V.SLOT) for i in wrote})
    bad = sorted({int(i // V.SLOT) for i in d})
    good = len(d) == 0 and len(wrote) > 0
    ok &= good
    print('steps %-10s wrote %8d B in %-28s mismatches %8d %s -> %s'
          % (STEPS[:upto], len(wrote), ','.join(NAMES.get(s, 'slot%d' % s) for s in slots), len(d),
             ('in ' + ','.join(NAMES.get(s, 'slot%d' % s) for s in bad)) if bad else '',
             'PASS' if good else 'FAIL'))
    if not good:
        break

# Counted coverage: "the C=512 stage is bit-exact" needs the full chain at every shift-window origin,
# at the frame's geometry. A run with fewer steps, another geometry or OUT2 records its own key and
# never fills a required slot. The verdict below refuses PASS until all four origins have a current PASS.
_combo = dict(MODE=int(os.environ.get('MODE', '0')), FIRST=int(FIRST), STEPS=','.join(map(str, STEPS)),
              BH=H, BW=W, BLOCK=BLOCK, OUT2=os.environ.get('OUT2', '0'))
cov = CG.Coverage('net_block512_2',
                  required=[dict(MODE=m, FIRST=1, STEPS='0,1,2,3', BH=32, BW=56, BLOCK=23, OUT2='0') for m in range(4)],
                  fingerprint_files=[__file__, E.__file__, R.DIS] + [MOD(s) for s in (FFWD2, CONV2, QKVA2)])
cov.record(_combo, 'PASS' if ok else 'FAIL')
print('this run: %s -> %s' % (_combo, 'PASS' if ok else 'FAIL'))
_v = cov.verdict()
raise SystemExit(1 if not ok else _v)
