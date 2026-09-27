"""Impulse-response footprint of k_post_block (emu only, no GPU, no kernel changes).

Baseline = the post_block_const difftest fixture (e4m3 0x38 inputs, f32 constants,
real block70 weights) run to completion over the whole grid. An impulse run writes
`nbytes` x `val` at slot*SLOT + off; the set of arena bytes that differ is the
kernel's response footprint for that input byte.

usage: impulse_post.py [SLOT OFF] ...     default: slot0 +0x0, +0x1000 and slot1 +0x0
       env: NBYTES (default 1), VAL (default 0x00)
"""
import sys, struct, os
from pathlib import Path
import numpy as np

_myargs = sys.argv[1:]
sys.argv = sys.argv[:1]                     # keep helper modules from parsing our args
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_spec as DS

SYM = '_Z22k_post_block_1h_32_fp810PostParams'
SWIN = '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'
seed = 1
spec = DS.SPECS['post_block_const']()
grid, threads = spec.grid, spec.threads
NBYTES = int(os.environ.get('NBYTES', '1'))
VAL = int(os.environ.get('VAL', '0'), 0)

nums = [int(x, 0) for x in _myargs]
pos = [(nums[k], nums[k + 1]) for k in range(0, len(nums) - 1, 2)]
if not pos:
    pos = [(0, 0), (0, 0x1000), (1, 0)]


def run_full(impulse=None):
    """Full-grid run; returns the final arena bytes."""
    ka = spec.kernarg()
    p = E.load_program(R.DIS, {SYM, SWIN})
    entry = E.load_program(R.DIS, {SYM})[0][0]
    k = min(idx for idx, x in enumerate(p) if x[0] == entry)
    prog = p[k:] + p[:k]
    g, KA = V.build(seed, bytes(ka), spec.nslot)
    a = g.regions[1].arr
    a[:] = spec.arena(seed)[:len(a)]
    if impulse:
        slot, off = impulse
        a[slot * V.SLOT + off:slot * V.SLOT + off + NBYTES] = np.full(NBYTES, VAL, np.uint8)
    lds = spec.lds
    DP = 0x7100_0000_0000
    pkt = bytearray(64)
    struct.pack_into('<HHHHHH', pkt, 0, 0, 3, threads, 1, 1, 0)
    struct.pack_into('<III', pkt, 12, grid[0] * threads, grid[1], 1)
    struct.pack_into('<II', pkt, 24, 64, lds)
    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
    steps = 0
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            steps += E.run_workgroup(prog, g, lds, threads,
                                     {0: DP & 0xffffffff, 1: DP >> 32,
                                      2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: wy},
                                     max_steps=30_000_000)['steps']
    return a.copy(), steps


def compress(off):
    off = sorted(set(int(x) for x in off))
    out, s, p = [], off[0], off[0]
    for x in off[1:]:
        if x == p + 1:
            p = x
        else:
            out.append((s, p)); s = p = x
    out.append((s, p))
    return out


base, bsteps = run_full()
nfin = int(np.isfinite(base.view(np.float32)).sum())
print('baseline: %d bytes, finite-f32 overall %d/%d, steps=%d' %
      (len(base), nfin, len(base) // 4, bsteps))

for (slot, off) in pos:
    out, _ = run_full((slot, off))
    d = np.nonzero(out != base)[0]
    if not len(d):
        print('impulse slot%d[+0x%x]: NO CHANGE' % (slot, off)); continue
    slots = {}
    for idx in d:
        slots.setdefault(int(idx // V.SLOT), []).append(int(idx % V.SLOT))
    print('impulse slot%d[+0x%x]=%#04x: %d changed bytes, slots %s' %
          (slot, off, VAL, len(d), sorted(slots)))
    for sl in sorted(slots):
        rr = compress(slots[sl])
        shown = ' '.join('%d-%d' % r if r[0] != r[1] else str(r[0]) for r in rr[:24])
        print('  slot %d: %d ranges, first bytes %d..%d: %s%s' %
              (sl, len(rr), rr[0][0], rr[-1][1], shown, ' ...' if len(rr) > 24 else ''))
    # value previews at the first few changed elements (4-byte aligned view)
    els = []
    for idx in d:
        e = int(idx) - int(idx) % 4
        if e not in els:
            els.append(e)
        if len(els) >= 8:
            break
    for e in els:
        b = out[e:e + 4].tobytes(); b0 = base[e:e + 4].tobytes()
        f32 = np.frombuffer(b, np.float32)[0] if len(b) == 4 else float('nan')
        f320 = np.frombuffer(b0, np.float32)[0] if len(b0) == 4 else float('nan')
        print('    off %d (+%d slot): %s -> %s  f32 %g -> %g' %
              (e, e % V.SLOT, b0.hex(), b.hex(), f320, f32))