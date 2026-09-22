"""Exhaustive kernarg-read trace for k_qkv_attn2 (_Z11k_qkv_attn210AttnParams), the ViT-variant attention kernel
(handle 0x180066380, blocks 31-38 per VARPARAMS_HOST_CONTRACT.md's weight-size analysis). Single kernarg-pointer
convention (s[0:1] directly, confirmed from its own prologue: 's_load_b128 s[4:7],s[0:1],0x18'), unlike
k_qkv_attn's dual dispatch_ptr/kernarg_ptr convention. AttnParams is 40 bytes per its own .s metadata (identical
to k_qkv_attn), so hidden-args are populated correctly here from the start rather than guessed.
"""
import sys, struct, collections, time
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z11k_qkv_attn210AttnParams'
KSIZE = 296
NSLOT = 6
ka = bytearray(KSIZE)
for off, idx in ((0x00, 0), (0x08, 1), (0x10, 2), (0x18, 3), (0x20, 4)):
    struct.pack_into('<Q', ka, off, V.ARENA + idx * V.SLOT)
struct.pack_into('<III', ka, 0x28, 1, 1, 1)     # hidden_block_count_x/y/z
struct.pack_into('<HHH', ka, 0x34, 256, 1, 1)    # hidden_group_size_x/y/z
struct.pack_into('<HHH', ka, 0x3a, 0, 0, 0)      # hidden_remainder_x/y/z
struct.pack_into('<QQQ', ka, 0x50, 0, 0, 0)      # hidden_global_offset_x/y/z
struct.pack_into('<H', ka, 0x68, 2)              # hidden_grid_dims

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
    E.run_workgroup(prog, g, 58368, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: 0, 15: 0}, max_steps=30_000_000)
    print('terminated cleanly, %.1fs' % (time.time() - t0))
except Exception as e:
    print('FAULT: %s, %.1fs' % (e, time.time() - t0))

for (off, n), cnt in sorted(kernarg_reads.items()):
    print('kernarg +0x%02x  width=%d  count=%d' % (off, n, cnt))
