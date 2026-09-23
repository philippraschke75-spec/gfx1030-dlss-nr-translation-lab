"""Characterise HOW the pre-block's emulator and GPU outputs differ, rather than how much.

difftest_preblock reports a count (28,825) that has stayed byte-identical across four separate
emulator ALU fixes. A count that never moves under arithmetic changes is not arithmetic. This asks
the shape question instead: same values in the wrong places (addressing/layout), a subset of lanes or
waves, or genuinely different numbers.

usage: preblock_diffshape.py [seed] [H W]   env DLSSNR_WEIGHT_BLOB=<block0 record>
"""
import sys, os, re, struct
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM, LDS = D.SYMS['32_1']
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (16, 16)
grid = ((W + 7) // 8, (H + 7) // 8)
IN_SLOT = V.PTR_FIELDS.index(0x40)


def kernarg():
    ka = V.make_kernarg(H=H, W=W, offy=0, offx=0, flags=0x14, grid=grid)
    struct.pack_into('<Q', ka, 0x00, 0)
    struct.pack_into('<Q', ka, 0x30, 0)
    # +0x50/+0x58/+0x60/+0x68 are 4-byte scalars that PTR_FIELDS lists as pointers, so
    # make_kernarg wrote whole pointers there. Clearing only the low dword leaves arena bits
    # in the high dword; the GPU runner rebases the qword and the kernel is seeded with the
    # device arena address while the emulator sees 0. +0x68 IS the RNG seed - that alone
    # made the pre-block mismatch by 67% of its output.
    for off in (0x50, 0x58, 0x60, 0x68):
        struct.pack_into('<Q', ka, off, 0)
    for off in (0x54, 0x5c, 0x64):
        struct.pack_into('<I', ka, off, 0)
    struct.pack_into('<f', ka, 0x50, 1.0)
    for off in range(0x70, 0xa0, 8):
        struct.pack_into('<Q', ka, off, 0)
    return ka


def emulate():
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM})
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 5)
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + H * W * 12] = \
        rng.random(H * W * 3, dtype=np.float32).astype(np.float32).view(np.uint8)
    init = a.copy()
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                            max_steps=8_000_000)
    return bytes(ka), init, a.copy()


ka, init, ref = emulate()
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
rc, msg, out = D.gpu(module, SYM, 'diffshape_s%d_%dx%d' % (seed, H, W), ka, init, grid)
if out is None:
    raise SystemExit('GPU failed: %s' % msg)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
ka, init, ref = emulate()          # rebuild at the device's real arena base

OUT_SLOT = V.PTR_FIELDS.index(0x8)
lo, hi = OUT_SLOT * V.SLOT, OUT_SLOT * V.SLOT + H * W * 32 * 2
e, g_, i0 = ref[lo:hi], out[lo:hi], init[lo:hi]

wrote = (e != i0) | (g_ != i0)
diff = e != g_
print('output slot (+0x8), %d bytes examined' % e.size)
print('  emulator wrote   : %d' % int((e != i0).sum()))
print('  gpu wrote        : %d' % int((g_ != i0).sum()))
print('  both untouched   : %d' % int((~wrote).sum()))
print('  differing        : %d' % int(diff.sum()))
print('  differ but ONLY ONE side wrote : %d'
      % int((diff & (((e != i0) & (g_ == i0)) | ((g_ != i0) & (e == i0)))).sum()))

print('\nsame multiset of byte values?')
he = np.bincount(e[wrote], minlength=256)
hg = np.bincount(g_[wrote], minlength=256)
print('  histogram L1 distance: %d of %d written bytes (%.1f%%)'
      % (int(np.abs(he - hg).sum()), int(wrote.sum()), 100.0 * np.abs(he - hg).sum() / max(1, wrote.sum())))

print('\nis it a shift? (emu[k:] vs gpu[:-k])')
best = None
for k in range(1, 65):
    m = int((e[k:] == g_[:-k]).sum())
    if best is None or m > best[1]:
        best = (k, m)
print('  best shift %d bytes -> %d matching of %d (%.1f%%)'
      % (best[0], best[1], e.size - best[0], 100.0 * best[1] / (e.size - best[0])))
print('  aligned (shift 0)  -> %d matching (%.1f%%)'
      % (int((e == g_).sum()), 100.0 * (e == g_).sum() / e.size))

print('\nhow far apart, as 16-bit words?')
ew = e[:e.size // 2 * 2].view(np.uint16).astype(np.int32)
gw = g_[:g_.size // 2 * 2].view(np.uint16).astype(np.int32)
d = np.abs(ew - gw)
nz = d != 0
if nz.any():
    print('  differing words: %d' % int(nz.sum()))
    for t in (1, 2, 4, 16, 256):
        print('    |delta| <= %-4d : %5.1f%%' % (t, 100.0 * float((d[nz] <= t).mean())))
    print('  high halves equal in %.1f%% of differing dwords'
          % (100.0 * float(((e.view(np.uint32) ^ g_.view(np.uint32)) & 0xffff0000 == 0)[
              (e.view(np.uint32) != g_.view(np.uint32))].mean())))

print('\nper-64-byte-block (one wave-ish granule) share of differing bytes:')
blk = diff[:diff.size // 64 * 64].reshape(-1, 64).mean(axis=1)
print('  blocks fully equal      : %d' % int((blk == 0).sum()))
print('  blocks fully differing  : %d' % int((blk == 1).sum()))
print('  blocks partly differing : %d' % int(((blk > 0) & (blk < 1)).sum()))

print('\npattern of the 128 clean vs 128 dirty 64-byte blocks:')
clean = (blk == 0).astype(int)
print('  first 64 blocks:', ''.join('.' if c else 'X' for c in clean[:64]))
print('  next  64 blocks:', ''.join('.' if c else 'X' for c in clean[64:128]))
print('  alternating?  even-index clean %d/%d, odd-index clean %d/%d'
      % (int(clean[0::2].sum()), len(clean[0::2]), int(clean[1::2].sum()), len(clean[1::2])))
print('  contiguous?   first half clean %d/%d, second half clean %d/%d'
      % (int(clean[:128].sum()), 128, int(clean[128:].sum()), 128))

# within a dirty block, which 2-byte lanes differ?
dirty = np.nonzero(blk < 1)[0]
lanes = diff[:diff.size // 64 * 64].reshape(-1, 64)[dirty]
per_lane = lanes.reshape(len(dirty), 32, 2).any(axis=2).mean(axis=0)
print('\n  within dirty blocks, share of blocks where each 2-byte lane differs:')
print('   ', ' '.join('%.0f' % (100 * x) for x in per_lane))
