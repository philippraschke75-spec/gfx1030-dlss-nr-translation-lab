"""Retest k_qkv_attn with the CORRECTED kernarg layout, recovered by reading the actual host x86 launcher
(embedded PE in G:\\dlss\\dlssnr_on_amd_setup.exe at file offset 0x47c00, disassembled with llvm-objdump) at
0x180033c22-0x180033cda:

  +0x00 (8B): input ptr
  +0x08 (8B): output ptr
  +0x10 (8B): weight ptr (this block's layer2 = qkv_weight+attn_scale+attn_bias, combined blob)
  +0x18 (4B): H (i32)            <- previously wrongly guessed as part of a second pointer pair
  +0x1c (4B): W (i32)            <- previously wrongly guessed as part of a second pointer pair
  +0x20 (4B): window origin X (i32, from the same mode table 0x180066410[mode] k_swin_var uses)
  +0x24 (4B): window origin Y (i32)
  +0x34 (4B): scalar, still unconfirmed - left at 0

The earlier trace fed ARENA+3*SLOT / ARENA+4*SLOT (huge 64-bit heap addresses, ~10^11) into +0x18/+0x20 as if
they were a second pointer pair - which the kernel actually reads as H/W/originX/originY 32-bit scalars. That
explains the non-terminating loop: window/grid arithmetic on an astronomically large "H" or "W" never converges
within any practical step budget.
"""
import sys, struct, collections, time
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z10k_qkv_attn10AttnParams'
KSIZE = 296
NSLOT = 6
ka = bytearray(KSIZE)
struct.pack_into('<Q', ka, 0x00, V.ARENA + 0 * V.SLOT)   # input
struct.pack_into('<Q', ka, 0x08, V.ARENA + 1 * V.SLOT)   # output
struct.pack_into('<Q', ka, 0x10, V.ARENA + 2 * V.SLOT)   # weight (layer2)
struct.pack_into('<i', ka, 0x18, 8)                       # H (one 8x8 window)
struct.pack_into('<i', ka, 0x1c, 8)                       # W
struct.pack_into('<i', ka, 0x20, 0)                       # origin X (phase 0)
struct.pack_into('<i', ka, 0x24, 0)                       # origin Y
struct.pack_into('<i', ka, 0x34, 32)
struct.pack_into('<III', ka, KSIZE - 20, 1, 1, 1)
struct.pack_into('<HHH', ka, KSIZE - 8, 256, 1, 1)

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, bytes(ka), NSLOT)
a = g.regions[1].arr
a[:] = 0

DP = 0x7100_0000_0000
pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
struct.pack_into('<III', pkt, 12, 256, 1, 1); struct.pack_into('<II', pkt, 24, 64, 58368)
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
    E.run_workgroup(prog, g, 58368, 256, {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: 0, 15: 0}, max_steps=20_000_000)
    print('terminated cleanly, %.1fs' % (time.time() - t0))
except Exception as e:
    print('FAULT: %s, %.1fs' % (e, time.time() - t0))

for (off, n), cnt in sorted(kernarg_reads.items()):
    print('kernarg +0x%02x  width=%d  count=%d' % (off, n, cnt))
