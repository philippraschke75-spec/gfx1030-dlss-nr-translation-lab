"""Exhaustive kernarg-read trace for k_expand (_Z8k_expand12ExpandParams). ExpandParams is only 24 bytes per its
own .s metadata (.offset 0, .size 24) - the smallest AttnParams-family-adjacent kernel seen yet. Prologue reads
confirm 3 pointers only (input, output, weight - s_load_b128 at +0x00 for the pair, s_load_b64 at +0x10 for the
third) plus +0x24 = hidden_group_size_x (used as a genuine divisor, same pattern as k_conv_res/k_qkv_attn2).
"""
import sys, struct, collections, time
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z8k_expand12ExpandParams'
KSIZE = 280
NSLOT = 6
LDS = 16384
ka = bytearray(KSIZE)
for off, idx in ((0x00, 0), (0x08, 1), (0x10, 2)):
    struct.pack_into('<Q', ka, off, V.ARENA + idx * V.SLOT)
struct.pack_into('<III', ka, 0x18, 1, 1, 1)     # hidden_block_count_x/y/z
struct.pack_into('<HHH', ka, 0x24, 256, 1, 1)    # hidden_group_size_x/y/z
struct.pack_into('<HHH', ka, 0x2a, 0, 0, 0)      # hidden_remainder_x/y/z
struct.pack_into('<QQQ', ka, 0x40, 0, 0, 0)      # hidden_global_offset_x/y/z
struct.pack_into('<H', ka, 0x58, 2)              # hidden_grid_dims

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

t0 = time.time()
try:
    E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: 0, 15: 0}, max_steps=30_000_000)
    print('terminated cleanly, %.1fs' % (time.time() - t0))
except Exception as e:
    print('FAULT: %s, %.1fs' % (e, time.time() - t0))

for (off, n), cnt in sorted(kernarg_reads.items()):
    print('kernarg +0x%02x  width=%d  count=%d' % (off, n, cnt))
