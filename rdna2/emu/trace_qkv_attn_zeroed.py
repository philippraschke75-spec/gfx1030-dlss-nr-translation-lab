"""Same as trace_qkv_attn_kernarg.py but with a ZEROED arena instead of random bytes, to test the hypothesis that
random-byte f16 data (frequently NaN) is causing data-dependent per-wave divergence in k_qkv_attn's barrier loop
(waves must all hit s_barrier the same number of times; observed one wave racing ahead of the other 7).
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
struct.pack_into('<III', ka, KSIZE - 20, 2, 1, 1)
struct.pack_into('<HHH', ka, KSIZE - 8, 256, 1, 1)

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, bytes(ka), NSLOT)
a = g.regions[1].arr
a[:] = 0   # <-- zeroed instead of random

DP = 0x7100_0000_0000
pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
struct.pack_into('<III', pkt, 12, 2 * 256, 1, 1); struct.pack_into('<II', pkt, 24, 64, 58368)
g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())

t0 = time.time()
try:
    for wx in range(2):
        E.run_workgroup(prog, g, 58368, 256, {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: 0}, max_steps=80_000_000)
    print('terminated cleanly, %.1fs' % (time.time() - t0))
except Exception as e:
    print('FAULT:', e, '%.1fs' % (time.time() - t0))
