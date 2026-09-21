"""Host-only: which kernarg pointer slots does a k_swin_var kernel read/write, at what widths and how densely?
usage: trace_reads.py <sym-key 32_1|...> <flags> [H W] -- emulator only, synthetic arena; recovers access pattern, not semantics."""
import sys, collections, struct
import numpy as np
import gfx11emu as E, run_var as V, difftest_var as D
key = sys.argv[1]; flags = int(sys.argv[2], 0)
H = int(sys.argv[3]) if len(sys.argv) > 3 else 16; W = int(sys.argv[4]) if len(sys.argv) > 4 else 16
sym, lds = D.SYMS[key]; lds = lds or D.group_size(sym)
ka = V.make_kernarg(H=H, W=W, flags=flags, grid=(2, 2), offy=0, offx=0)
prog = E.load_program(D.R.DIS, {sym}); g, KA = V.build(1, ka, len(V.PTR_FIELDS))
nslot = len(V.PTR_FIELDS)
rd = np.zeros((nslot, V.SLOT), bool); wr = np.zeros((nslot, V.SLOT), bool); widths = collections.defaultdict(collections.Counter)
orig_r, orig_w = g.read, g.write
acc = collections.defaultdict(set)
def hook(store, which):
    def f(addr, x):
        a = np.asarray(addr, np.uint64); n = x if which == 'r' else x.shape[1]
        off = a.astype(np.int64) - V.ARENA
        ok = (off >= 0) & (off + n <= nslot * V.SLOT)
        for o in off[ok]:
            s, b = divmod(int(o), V.SLOT)
            (rd if which == 'r' else wr)[s, b:min(b + n, V.SLOT)] = True; widths[(which, s)][n] += 1
            if which == "r" and s == 2: acc[n].add(b)
        return store(addr, x)
    return f
g.read = hook(orig_r, 'r'); g.write = hook(orig_w, 'w')
for wy in range(2):
    for wx in range(2):
        E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)
print('kernel', sym, 'flags', flags, 'H,W', H, W)
for s in range(nslot):
    r, w = rd[s], wr[s]
    if r.any() or w.any():
        def ext(m): i = np.nonzero(m)[0]; return '[%#x..%#x) cnt=%d' % (i[0], i[-1] + 1, len(i)) if len(i) else '-'
        print('slot %2d kernarg+%#04x  READ %-32s WRITE %-32s rw=%s %s' % (s, V.PTR_FIELDS[s], ext(r), ext(w),
              dict(widths[('r', s)]), dict(widths[('w', s)])))

# ---- per-width extents inside the weight slot (2): which byte ranges are read with which access width
import os
if os.environ.get('SEGMENTS'):
    pass
print('slot2 read segments by access width (merged runs of touched start offsets, gap<=64):')
for n, offs in sorted(acc.items()):
    o = sorted(offs); runs = []; st = pr = o[0]
    for x in o[1:]:
        if x - pr > 64: runs.append((st, pr + n)); st = x
        pr = x
    runs.append((st, pr + n)); print(' width', n, 'starts', len(o), 'runs', [(hex(a), hex(b)) for a, b in runs][:12])
