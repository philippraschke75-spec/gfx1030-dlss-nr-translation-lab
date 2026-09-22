"""Exhaustive kernarg-read trace + structural GPU check for k_conv_res_views (_Z16k_conv_res_views12ConvPlParams),
the REAL kernel the host launcher dispatches for a C=512 block's projection step (handle 0x1800663d0) - not
plain k_conv_res (handle 0x180066360, confirmed to be an init-time-only warmup probe with no real per-block
call site anywhere in the host .text section). Field layout decoded directly from the host launcher disassembly
at 180033f76-1800340e9 (embedded PE in dlssnr_on_amd_setup.exe):
  +0x00 ptr (ctx+0x240), +0x08 null (always 0), +0x10 ptr (ctx+0x238), +0x18 ptr (ctx+0x228),
  +0x20 r14 (role TBD), +0x28 weight ptr (layer3, real launcher: call 0x180031bc0(ctx,block,layer=3)),
  +0x30 H (ctx_sub1+4), +0x34 W (ctx_sub1+8), +0x38 r15 (role TBD),
  +0x40 H2 (ctx_sub2+4), +0x44 W2 (ctx_sub2+8).
ConvPlParams is exactly 72 bytes per its own .s metadata (.offset 0, .size 72) - matches the decoded field count
exactly. Hidden-args populated correctly from the start, shifted by the 72-vs-40-byte explicit-size difference.
"""
import sys, struct, collections, time
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z16k_conv_res_views12ConvPlParams'
KSIZE = 328
NSLOT = 6
LDS = 24576

ka = bytearray(KSIZE)
struct.pack_into('<Q', ka, 0x00, V.ARENA + 0 * V.SLOT)
struct.pack_into('<Q', ka, 0x08, 0)
struct.pack_into('<Q', ka, 0x10, V.ARENA + 1 * V.SLOT)
struct.pack_into('<Q', ka, 0x18, V.ARENA + 2 * V.SLOT)
struct.pack_into('<i', ka, 0x20, 0)
struct.pack_into('<Q', ka, 0x28, V.ARENA + 3 * V.SLOT)
struct.pack_into('<i', ka, 0x30, 4)
struct.pack_into('<i', ka, 0x34, 4)
struct.pack_into('<i', ka, 0x38, 0)
struct.pack_into('<i', ka, 0x40, 4)
struct.pack_into('<i', ka, 0x44, 4)
struct.pack_into('<III', ka, 0x48, 1, 1, 1)       # hidden_block_count_x/y/z (shifted +32 vs the 40-byte kernels)
struct.pack_into('<HHH', ka, 0x54, 256, 1, 1)      # hidden_group_size_x/y/z
struct.pack_into('<HHH', ka, 0x5a, 0, 0, 0)        # hidden_remainder_x/y/z
struct.pack_into('<QQQ', ka, 0x70, 0, 0, 0)        # hidden_global_offset_x/y/z
struct.pack_into('<H', ka, 0x88, 2)                # hidden_grid_dims

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
