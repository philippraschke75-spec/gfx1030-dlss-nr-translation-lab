"""Host-only: for each k_import source format code, trace the per-pixel access width/count on the source buffer.
Distinguishes 8 B/pixel (matches RGBA16F: two bytes x4 channels) formats from 4 B/pixel and other formats.
usage: trace_import_format.py [fmt0] [fmt1] ... (default 0..7)"""
import sys, struct, collections
import numpy as np
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z8k_import12ImportParams'
fmts = [int(x) for x in sys.argv[1:]] or list(range(8))
H, W, PH, PW = 13, 21, 16, 32
for fmt in fmts:
    ka = bytearray(0x130)
    pitch = W * 4
    grid = ((PW + 255) // 256, PH)
    struct.pack_into('<Q', ka, 0x00, V.ARENA)
    struct.pack_into('<iiiiii', ka, 0x08, pitch, fmt, H, W, PH, PW)
    struct.pack_into('<Q', ka, 0x20, V.ARENA + V.SLOT)
    struct.pack_into('<if', ka, 0x28, 0, 1.0)
    struct.pack_into('<III', ka, 0x30, grid[0], grid[1], 1)
    struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
    prog = E.load_program(R.DIS, {SYM})
    g, KA = V.build(1, ka, len(V.PTR_FIELDS))
    widths = collections.Counter()
    orig_r = g.read
    def hook(addr, n, widths=widths):
        a = np.asarray(addr, np.uint64)
        off = a.astype(np.int64) - V.ARENA
        ok = (off >= 0) & (off < V.SLOT)
        widths[n] += int(ok.sum())
        return orig_r(addr, n)
    g.read = hook
    lds = D.group_size(SYM)
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=2_000_000)
    print('fmt', fmt, 'source-buffer read widths (bytes: count):', dict(widths))
