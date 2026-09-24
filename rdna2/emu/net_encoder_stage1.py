"""Run encoder stage 1 (blocks 1-4, C=32) as ONE net_run dispatch sequence, with real block weights,
and check it against the emulator running the same chain.

This is the first time multiple real network blocks run as a sequence on the gfx1030 rather than as
isolated per-kernel difftests. The chain is what matters: block n's output buffer is block n+1's
input buffer, so an error in the hand-off shows up as a mismatch even when every kernel in
isolation is already verified.

Schedule facts used here, all from VARPARAMS_HOST_CONTRACT.md:
  * stage 1 is blocks 1-4 at C=32, dispatched by launcher A (k_swin_var<32,false>)
  * per-block shifted-window modes cycle 0,1,2,3 -> origins (0,0), (-4,-4), (-4,0), (0,-4)
  * flags = (block==first_in_stage) | 4*(block==last_in_stage)
  * grid = ((W - Xoff + 7)/8, (H - Yoff + 7)/8)
  * activation buffers ping-pong between blocks

usage: net_encoder_stage1.py    (inside sandbox.py)
"""
import sys, re, struct, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

# argv[2] picks the stage's channel count, so decoder stages (C=256/128/64/32) can be run
# through the same chain harness as the encoder's.
KEY = sys.argv[2] if len(sys.argv) > 2 else '32_0'
SYM, _lds = D.SYMS[KEY]
LDS = _lds or D.group_size(SYM)
# H/W were fixed at 16, i.e. two 8-px windows per axis. A column artefact with a period of 4
# cannot be distinguished from noise at that width, so every PASS this harness has recorded is
# blind to it. DIFF_H/DIFF_W make the geometry settable.
import os as _os
H = int(_os.environ.get('DIFF_H', '16'))
W = int(_os.environ.get('DIFF_W', '16'))
MODES = [(0, 0), (-4, -4), (-4, 0), (0, -4)]          # table 0x180066410
# Any same-width run of blocks dispatched through launcher A. The decoder uses the identical
# launcher and the identical k_swin_var kernels as the encoder - only the weights and the stage
# tuple differ - so a decoder stage chains exactly the same way.
BLOCKS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 2, 3, 4]
NSLOT = len(V.PTR_FIELDS) + len(BLOCKS)               # default fields, then one slot per block weight
WSLOT0 = len(V.PTR_FIELDS)
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_encoder'; OUT.mkdir(parents=True, exist_ok=True)
S = lambda i: V.ARENA + i * V.SLOT


def plan():
    """(kernarg, grid) per block, ping-ponging activations between slots 0 and 1."""
    out = []
    for i, blk in enumerate(BLOCKS):
        ox, oy = MODES[i % 4]
        flags = (1 if i == 0 else 0) | (4 if i == len(BLOCKS) - 1 else 0)
        grid = ((W - ox + 7) // 8, (H - oy + 7) // 8)
        ka = V.make_kernarg(H=H, W=W, offy=oy, offx=ox, flags=flags, grid=grid)
        struct.pack_into('<Q', ka, 0x00, S(i % 2))          # input  <- previous block's output
        struct.pack_into('<Q', ka, 0x08, S((i + 1) % 2))    # output -> the other buffer
        struct.pack_into('<Q', ka, 0x10, S(WSLOT0 + i))     # this block's real weight record
        out.append((bytes(ka), grid))
    return out


def arena(seed):
    g, _ = V.build(seed, plan()[0][0], NSLOT)
    a = g.regions[1].arr.copy()
    for i, blk in enumerate(BLOCKS):
        w = np.frombuffer((D.ROOT / 'build' / 'weights' / ('block%d.bin' % blk)).read_bytes(), np.uint8)
        a[(WSLOT0 + i) * V.SLOT:(WSLOT0 + i) * V.SLOT + len(w)] = w
    return a


def emulate(seed):
    """The same four dispatches over one emulator arena, exactly as net_run will do on the GPU."""
    prog = E.load_program(R.DIS, {SYM})
    steps = plan()
    base = arena(seed)
    cur = base.copy()
    for ka, grid in steps:
        g, KA = V.build(seed, ka, NSLOT)
        g.regions[1].arr[:] = cur
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                                max_steps=8_000_000)
        cur = g.regions[1].arr.copy()
    return base, cur


def net_run(seed):
    steps = plan()
    blob = b''; lines = []
    mod = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
    for ka, grid in steps:
        off = len(blob); blob += ka
        lines.append('%s|%s|%d|%d|%d|%d|%d' % (mod, SYM, off, len(ka), grid[0], grid[1], 256))
    (OUT / 'manifest.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'kernargs.bin').write_bytes(blob)
    (OUT / 'arena.bin').write_bytes(arena(seed).tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'manifest.txt'), str(OUT / 'kernargs.bin'),
                        str(OUT / 'arena.bin'), '%x' % V.ARENA],
                       capture_output=True, text=True, timeout=1800)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'arena.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out


print('stage chain: blocks %s, C=32, H=W=%d, %d dispatches' % (BLOCKS, H, len(BLOCKS)))
for i, (ka, grid) in enumerate(plan()):
    ox, oy = MODES[i % 4]
    print('  block%-2d mode=%d origin=(%d,%d) flags=%d grid=%s in=slot%d out=slot%d'
          % (BLOCKS[i], i % 4, ox, oy, struct.unpack_from('<I', ka, 0x28)[0], grid, i % 2, (i + 1) % 2))

rc, msg, out = net_run(1)
print('\n', msg.splitlines()[-1] if msg else 'no output')
if out is None:
    print('GPU FAIL rc=%d\n%s' % (rc, msg)); raise SystemExit(1)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)   # rebase, then re-reference
base, ref = emulate(1)
d = np.nonzero(out != ref)[0]
changed = int((ref != base).sum())
final_slot = len(BLOCKS) % 2
print('emulator chain wrote %d bytes; final activations in slot %d' % (changed, final_slot))
print('net_run vs emulator over the 4-block chain: %d mismatches -> %s'
      % (len(d), 'PASS' if len(d) == 0 and changed else 'FAIL'))
# The artefact this harness was blind to: a period-4 column structure. Report it for the
# EMULATOR's output, i.e. the original gfx1100 kernel's own result. If the reference shows it too,
# the period is what the kernel does with these launch parameters, not a translation defect.
def _e4m3(b):
    b = b.astype(np.uint8); sg = np.where(b >> 7, -1.0, 1.0)
    e = ((b >> 3) & 0xF).astype(np.int32); m = (b & 7).astype(np.float64)
    return np.where(b == 0x7f, np.nan,
                    sg * np.where(e == 0, (m / 8.0) * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e - 7)))
for _nm, _buf in (('emulator (reference)', ref), ('gpu (translated)', out)):
    _a = _buf[final_slot * V.SLOT:final_slot * V.SLOT + 32 * H * W]
    _v = _e4m3(_a.reshape(2, H, W, 16))
    _cm = np.nanmean(np.abs(_v), axis=(0, 1, 3)); _dd = _cm - _cm.mean()
    _ac = np.correlate(_dd, _dd, 'full')[len(_dd) - 1:]; _ac /= max(_ac[0], 1e-12)
    _ph = np.array([np.nanmean(_cm[k::4]) for k in range(4)])
    print('  %-22s lag2 %+.2f lag4 %+.2f  phase means %s  phase-var/var %.3f'
          % (_nm, _ac[2] if len(_ac) > 2 else 0, _ac[4] if len(_ac) > 4 else 0,
             ' '.join('%.2f' % x for x in _ph), _ph.var() / max(_cm.var(), 1e-12)))
raise SystemExit(0 if (len(d) == 0 and changed) else 1)
