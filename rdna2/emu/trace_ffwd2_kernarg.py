"""Exhaustive kernarg-read trace for k_ffwd2 (_Z7k_ffwd211Ffwd2Params), the second FFN dispatch in a C=512
block. Ffwd2Params is 48 bytes per its own .s metadata (.offset 0, .size 48), so hidden-args start at +0x30
and hidden_group_size_x lands at +0x3c. The kernel's own prologue reads exactly that:
  s_load_b256 s[16:23], s[0:1], null   -> 4 pointers at +0x00/+0x08/+0x10/+0x18
  s_load_b128 s[8:11],  s[0:1], 0x20   -> 4 scalars  at +0x20/+0x24/+0x28/+0x2c
  s_load_b32  s2,       s[0:1], 0x3c   -> hidden_group_size_x
Host launcher (0x18003399c-0x180033a03) builds the same struct field-for-field:
  +0x00 ptr (ctx+0x228, same buffer k_ffwd reads as input), +0x08 r14 (host null-checks it, so an optional ptr),
  +0x10 ptr (ctx+0x230, the same buffer k_ffwd writes as output), +0x18 weight ptr (call 0x180031bc0(ctx,block,
  layer=0) - block23.layer0.layer is 524,288 B = exactly 512*1024, a clean C=512->1024 FFN expand weight),
  +0x20 H (ctx_sub1+4), +0x24 W (ctx_sub1+8), +0x28 scalar ([[rdi]]), +0x2c padding.
"""
import sys, struct, collections, time
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z7k_ffwd211Ffwd2Params'
KSIZE = 304
NSLOT = 6
LDS = 21504

ka = bytearray(KSIZE)
for off, idx in ((0x00, 0), (0x08, 1), (0x10, 2), (0x18, 3)):
    struct.pack_into('<Q', ka, off, V.ARENA + idx * V.SLOT)
struct.pack_into('<i', ka, 0x20, 8)               # H
struct.pack_into('<i', ka, 0x24, 8)               # W
struct.pack_into('<i', ka, 0x28, 0)               # scalar from [[rdi]]
struct.pack_into('<III', ka, 0x30, 1, 1, 1)       # hidden_block_count_x/y/z
struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)     # hidden_group_size_x/y/z
struct.pack_into('<HHH', ka, 0x42, 0, 0, 0)       # hidden_remainder_x/y/z
struct.pack_into('<QQQ', ka, 0x58, 0, 0, 0)       # hidden_global_offset_x/y/z
struct.pack_into('<H', ka, 0x70, 2)               # hidden_grid_dims

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, bytes(ka), NSLOT)
a = g.regions[1].arr
rng = np.random.default_rng(1)
a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)
w = np.frombuffer((D.ROOT / 'build' / 'weights' / 'block23_layer0.bin').read_bytes(), np.uint8)
a[3 * V.SLOT:3 * V.SLOT + len(w)] = w

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
