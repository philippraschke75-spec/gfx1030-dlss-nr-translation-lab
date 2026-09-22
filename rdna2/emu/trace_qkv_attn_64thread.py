"""Retest k_qkv_attn with nthreads=64 (one 8x8=64-token attention window per workgroup, per OpenDLSS-NR's
network.md: "The window is 8x8 tokens... Tokens are padded to a multiple of 64") instead of the 256 threads
copied from the encoder kernels, which caused an outer exec-mask loop to never terminate.
"""
import sys, struct, collections, time
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
struct.pack_into('<III', ka, KSIZE - 20, 1, 1, 1)
struct.pack_into('<HHH', ka, KSIZE - 8, 64, 1, 1)

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, bytes(ka), NSLOT)
a = g.regions[1].arr
a[:] = 0

DP = 0x7100_0000_0000
pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 64, 1, 1, 0)
struct.pack_into('<III', pkt, 12, 64, 1, 1); struct.pack_into('<II', pkt, 24, 64, 58368)
g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())

kernarg_reads = collections.Counter()
orig_r = g.read
def hookr(addr, n):
    off = np.asarray(addr, np.uint64).astype(np.int64) - int(KA)
    ok = (off >= 0) & (off < KSIZE)
    for o in off[ok]: kernarg_reads[(int(o), n)] += 1
    return orig_r(addr, n)
g.read = hookr

t0 = time.time()
try:
    E.run_workgroup(prog, g, 58368, 64, {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: 0, 15: 0}, max_steps=8_000_000)
    print('terminated cleanly, %.1fs' % (time.time() - t0))
except Exception as e:
    print('FAULT:', e, '%.1fs' % (time.time() - t0))

for (off, n), cnt in sorted(kernarg_reads.items()):
    print('kernarg +0x%02x  width=%d  count=%d' % (off, n, cnt))
