"""Host-only: trace exactly which kernarg byte offsets k_ffwd reads (scalar s_load's off the kernarg pointer),
not just which arena pointer-slots get touched. Ground truth for the struct layout beyond the 3 pointers already
confirmed (+0x00/+0x08/+0x10) and the channel-count-like scalar at +0x2c.
"""
import sys, struct, collections
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z6k_ffwd10FfwdParams'
KSIZE = 288
NSLOT = 6
ka = bytearray(KSIZE)
for off, idx in ((0x00, 0), (0x08, 1), (0x10, 2)):
    struct.pack_into('<Q', ka, off, V.ARENA + idx * V.SLOT)
# fill every OTHER 4-byte slot with a distinct small positive sentinel so any use shows up plausibly (not zero -> div fault)
for off in range(0x18, KSIZE - 20, 4):
    struct.pack_into('<i', ka, off, 8)
struct.pack_into('<i', ka, 0x2c, 32)
struct.pack_into('<III', ka, KSIZE - 20, 4, 1, 1)
struct.pack_into('<HHH', ka, KSIZE - 8, 256, 1, 1)

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, bytes(ka), NSLOT)
a = g.regions[1].arr
rng = np.random.default_rng(1)
a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)

kernarg_reads = collections.Counter()
orig_r = g.read
def hookr(addr, n):
    off = np.asarray(addr, np.uint64).astype(np.int64) - int(KA)
    ok = (off >= 0) & (off < KSIZE)
    for o in off[ok]: kernarg_reads[(int(o), n)] += 1
    return orig_r(addr, n)
g.read = hookr

try:
    for wx in range(2):
        E.run_workgroup(prog, g, 24576, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: 0}, max_steps=3_000_000)
    print('terminated cleanly')
except Exception as e:
    print('FAULT:', e)

for (off, n), cnt in sorted(kernarg_reads.items()):
    print('kernarg +0x%02x  width=%d  count=%d' % (off, n, cnt))
