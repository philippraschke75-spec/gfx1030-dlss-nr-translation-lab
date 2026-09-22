"""Host-only: probe k_ffwd's real kernarg layout empirically. Confirmed from disassembly (0x180033660 host launcher):
+0x00 input ptr, +0x08 output ptr, +0x10 weight ptr (via the same per-block weight-lookup helper k_swin_var uses),
+0x18 a scalar (from a nested *ctx dereference). kernarg_size=288, 1D grid (workgroup_id_x only), 256 threads assumed.
This script fills a plausible kernarg (real-looking pointers, small positive scalars) and traces which kernarg
offsets the kernel itself treats as pointers (loaded then dereferenced) vs plain scalars, and what it reads/writes.
"""
import sys, struct, collections
import numpy as np
sys.path.insert(0, '.')
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

SYM = '_Z6k_ffwd10FfwdParams'
KSIZE = 288
NSLOT = 6   # a few generous 1 MiB slots for any pointer-looking field
ka = bytearray(KSIZE)
for off, idx in ((0x00, 0), (0x08, 1), (0x10, 2)):
    struct.pack_into('<Q', ka, off, V.ARENA + idx * V.SLOT)
struct.pack_into('<i', ka, 0x18, 64)      # guess: a channel count or row count
struct.pack_into('<i', ka, 0x1c, 0)
struct.pack_into('<i', ka, 0x2c, 32)      # confirmed from k_ffwd's own prologue: s_load_b32 s2,[0x2c], masked &0xffff, used as a fixed-point division denominator (channel count?)
# hidden args (HIP appends them after the explicit struct end; kernarg_size already includes them for this ELF)
struct.pack_into('<III', ka, KSIZE - 20, 4, 1, 1)   # grid (workgroup_id_x enabled only) - guess 4 workgroups
struct.pack_into('<HHH', ka, KSIZE - 8, 256, 1, 1)

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(1, bytes(ka), NSLOT)
a = g.regions[1].arr
rng = np.random.default_rng(1)
a[:] = rng.integers(0, 256, len(a), dtype=np.uint8)   # random content everywhere so any read is "valid-ish"

reads = collections.defaultdict(set); writes = collections.defaultdict(set)
orig_r, orig_w = g.read, g.write
def hookr(addr, n):
    off = np.asarray(addr, np.uint64).astype(np.int64) - V.ARENA
    ok = (off >= 0) & (off < NSLOT * V.SLOT)
    for o in off[ok]: reads[int(o) // V.SLOT].add(int(o) % V.SLOT)
    return orig_r(addr, n)
def hookw(addr, data):
    off = np.asarray(addr, np.uint64).astype(np.int64) - V.ARENA
    ok = (off >= 0) & (off < NSLOT * V.SLOT)
    for o in off[ok]: writes[int(o) // V.SLOT].add(int(o) % V.SLOT)
    return orig_w(addr, data)
g.read = hookr; g.write = hookw

try:
    for wx in range(4):
        E.run_workgroup(prog, g, 24576, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: 0}, max_steps=2_000_000)
    print('terminated cleanly')
except Exception as e:
    print('FAULT:', e)

for s in sorted(set(reads) | set(writes)):
    r = reads.get(s, set()); w = writes.get(s, set())
    print('slot', s, 'kernarg+0x%02x' % (s * 8 if s < 3 else -1), 'read', (min(r), max(r) + 1, len(r)) if r else None,
          'write', (min(w), max(w) + 1, len(w)) if w else None)
