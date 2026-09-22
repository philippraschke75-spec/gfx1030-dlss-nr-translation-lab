"""Host-only: trace which kernarg byte offsets k_qkv_attn reads. This kernel enables BOTH dispatch_ptr (s[0:1])
and kernarg_segment_ptr (s[2:3]) per its kernel descriptor - same convention as the pre-block kernel - so a
modeled AQL dispatch packet must be present too, not just the kernarg.
"""
import sys, struct, collections
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z10k_qkv_attn10AttnParams'
KSIZE = 296
NSLOT = 6
ka = bytearray(KSIZE)
for off, idx in ((0x00, 0), (0x08, 1), (0x10, 2), (0x18, 3), (0x20, 4)):
    struct.pack_into('<Q', ka, off, V.ARENA + idx * V.SLOT)
for off in range(0x00, KSIZE - 20, 4):
    if struct.unpack_from('<I', ka, off)[0] == 0:
        struct.pack_into('<i', ka, off, 8)
struct.pack_into('<III', ka, KSIZE - 20, 2, 1, 1)
struct.pack_into('<HHH', ka, KSIZE - 8, 256, 1, 1)

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, bytes(ka), NSLOT)
a = g.regions[1].arr
rng = np.random.default_rng(1)
a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)

DP = 0x7100_0000_0000
pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
struct.pack_into('<III', pkt, 12, 2 * 256, 1, 1); struct.pack_into('<II', pkt, 24, 64, 58368)
g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())

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
        E.run_workgroup(prog, g, 58368, 256, {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: 0}, max_steps=3_000_000)
    print('terminated cleanly')
except Exception as e:
    print('FAULT:', e)

for (off, n), cnt in sorted(kernarg_reads.items()):
    print('kernarg +0x%02x  width=%d  count=%d' % (off, n, cnt))
