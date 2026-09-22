"""Host-only: does k_swin_var<32,false> write to kernarg +0x30/+0x38 (the 'unexplained' pointer fields) specifically
when flags=4 (last-in-stage)? If OpenDLSS-NR's "2x2 box pool fused into the last block" description is right, one of
these should get a WRITE only for flags&4, sized like a half-resolution, double-width buffer."""
import sys, struct, collections
import numpy as np
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM, LDS = D.SYMS['32_0']; LDS = LDS or D.group_size(SYM)
N = 32   # small, fast
grid = ((N + 7) // 8, (N + 7) // 8)

def probe(flags):
    ka = V.make_kernarg(H=N, W=N, offy=0, offx=0, flags=flags, grid=grid)
    prog = E.load_program(R.DIS, {SYM})
    g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    orig_w = g.write
    touched = collections.defaultdict(set)
    def hook(addr, data):
        off = np.asarray(addr, np.uint64).astype(np.int64) - V.ARENA
        ok = (off >= 0) & (off < len(V.PTR_FIELDS) * V.SLOT)
        for o in off[ok]:
            s = int(o) // V.SLOT
            touched[s].add(int(o) % V.SLOT)
        return orig_w(addr, data)
    g.write = hook
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=500_000)
    print('flags=%d  slots written: %s' % (flags, {s: (len(v), min(v), max(v)) for s, v in sorted(touched.items())}))

for f in (0, 1, 4):
    probe(f)
