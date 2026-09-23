"""Find the store instruction responsible for the first byte where the pre-block diverges.

statebisect compares REGISTERS at first visits to each PC. The pre-block's divergence is invisible
that way: registers match through the whole first-visit trace while memory still differs by 67% of
written bytes, which means it happens on repeat visits inside k_swin_var's loops.

This asks the question from the memory side instead. Every global store the emulator performs is
logged with the PC that issued it. The GPU result is then diffed against the emulator's, and for the
first differing byte we report which store wrote it, at which PC, on which occurrence of that PC.
No GPU instrumentation is needed, so it is unaffected by the statebisect rework.

usage: preblock_firststore.py [seed] [H W]   env DLSSNR_WEIGHT_BLOB=<block0 record>
"""
import sys, os, re, struct, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

VARIANT = os.environ.get('PRE_VARIANT', '32_1')   # '32_0' is the passing encoder variant
FLAGS = int(os.environ.get('PRE_FLAGS', '0x14'), 0)
SYM, LDS = D.SYMS[VARIANT]
LDS = LDS or D.group_size(SYM)   # '32_0' declares None and falls back to the group size
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (8, 8)
grid = ((W + 7) // 8, (H + 7) // 8)
IN_SLOT = V.PTR_FIELDS.index(0x40)
OUT_SLOT = V.PTR_FIELDS.index(0x8)

CUR = {'pc': -1, 'n': 0}
STORES = []            # (byte_addr_lo, byte_addr_hi, pc, occurrence)
PC_SEEN = collections.Counter()
EXEC_OPS = collections.Counter()

_orig_compile = E.Exec.compile


def compile_hook(self, pc):
    fn = _orig_compile(self, pc)
    addr = self.prog[pc][0]

    op = self.prog[pc][1]

    def wrapped(w, _fn=fn, _a=addr, _o=op):
        CUR['pc'] = _a
        EXEC_OPS[_o] += 1
        return _fn(w)
    return wrapped


_orig_write = E.GMem.write


def write_hook(self, addr, data):
    a = np.asarray(addr, np.uint64)
    if a.size:
        n = data.shape[1]
        pc = CUR['pc']
        PC_SEEN[pc] += 1
        STORES.append((int(a.min()), int(a.max()) + n, pc, PC_SEEN[pc]))
    return _orig_write(self, addr, data)


def kernarg():
    ka = V.make_kernarg(H=H, W=W, offy=0, offx=0, flags=FLAGS, grid=grid)
    if FLAGS == 0x14:                    # pre-block layout only; the encoder reads +0x00
        struct.pack_into('<Q', ka, 0x00, 0)
        struct.pack_into('<Q', ka, 0x30, 0)
        for off in (0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68):
            struct.pack_into('<I', ka, off, 0)
        struct.pack_into('<f', ka, 0x50, 1.0)
        for off in range(0x70, 0xa0, 8):
            struct.pack_into('<Q', ka, off, 0)
    return ka


def emulate(log=False):
    del STORES[:]
    PC_SEEN.clear()
    EXEC_OPS.clear()
    ka = kernarg()
    prog = E.load_program(R.DIS, {SYM})
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    rng = np.random.default_rng(seed + 5)
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + H * W * 12] = \
        rng.random(H * W * 3, dtype=np.float32).astype(np.float32).view(np.uint8)
    init = a.copy()
    if log:
        E.Exec.compile, E.GMem.write = compile_hook, write_hook
    try:
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                E.run_workgroup(prog, g, LDS, 256,
                                {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                                max_steps=8_000_000)
    finally:
        E.Exec.compile, E.GMem.write = _orig_compile, _orig_write
    return bytes(ka), init, a.copy(), g.regions[1].base


ka, init, ref, arena_base = emulate()
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
rc, msg, out = D.gpu(module, SYM, 'firststore_s%d_%dx%d' % (seed, H, W), ka, init, grid)
if out is None:
    raise SystemExit('GPU failed: %s' % msg)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
ka, init, ref, arena_base = emulate(log=True)          # rebuild at the device base, with logging

_PROG = list(E.load_program(R.DIS, {SYM}))
diff = np.nonzero(ref != out)[0]
print('total differing bytes: %d of %d written'
      % (diff.size, int((ref != init).sum())))
if not diff.size:
    raise SystemExit('no divergence')

first = int(diff[0])
print('first differing byte : arena offset %#x  (slot %d, +%#x)'
      % (first, first // V.SLOT, first % V.SLOT))
print('   emu=%02x  gpu=%02x  init=%02x' % (ref[first], out[first], init[first]))

addr = arena_base + first
hits = [s for s in STORES if s[0] <= addr < s[1]]
print('\nemulator stores covering that address: %d' % len(hits))
for lo, hi, pc, occ in hits[:10]:
    ins = None
    for _e in _PROG:
        if _e[0] == pc:
            ins = '%s %s' % (_e[1], _e[2])
            break
    print('   pc=%#x  occurrence #%d  range %#x..%#x   %s' % (pc, occ, lo, hi, ins or '?'))

print('\nstore PCs by volume (top 8):')
vol = collections.Counter()
for lo, hi, pc, occ in STORES:
    vol[pc] += 1
for pc, c in vol.most_common(8):
    print('   pc=%#x  %d stores' % (pc, c))

# The divergent store is global_store_b32: one dword per lane. Comparing the stored dwords shows
# which byte lanes of the assembled value disagree, which points at the producing instruction.
print('\nstored dwords at that address (first 16 lanes):')
ew = ref[first:first + 64].view(np.uint32)
gw = out[first:first + 64].view(np.uint32)
for i in range(16):
    x = int(ew[i]) ^ int(gw[i])
    print('   lane%-3d emu=%08x gpu=%08x  xor=%08x  bytes differing: %s'
          % (i, ew[i], gw[i], x,
             ''.join('b%d ' % k for k in range(4) if (x >> (8 * k)) & 0xff) or 'none'))
mask = 0
for i in range(len(ew)):
    mask |= int(ew[i]) ^ int(gw[i])
print('   union of differing bits across these lanes: %08x' % mask)

# Measured in the RIGHT unit. The output is e4m3 (one byte per value); an earlier pass compared
# these as uint16 words and concluded "not rounding", which the unit made meaningless.
print('\ne4m3 byte-code distance over all differing bytes:')
de = np.abs(ref[diff].astype(np.int32) - out[diff].astype(np.int32))
for t in (1, 2, 3, 4, 8, 16):
    print('   |code delta| <= %-3d : %5.1f%%' % (t, 100.0 * float((de <= t).mean())))
print('   sign bit differs     : %5.1f%%'
      % (100.0 * float((((ref[diff] ^ out[diff]) & 0x80) != 0).mean())))
print('   exponent differs     : %5.1f%%'
      % (100.0 * float((((ref[diff] ^ out[diff]) & 0x78) != 0).mean())))
print('   mantissa only        : %5.1f%%'
      % (100.0 * float((((ref[diff] ^ out[diff]) & 0xf8) == 0).mean())))

# First divergence in EXECUTION order, not address order. The address-ordered "first" store says
# little about causality; the earliest store whose bytes end up wrong is the place to look.
print('\nearliest store (in execution order) whose bytes differ in the final arena:')
bad = np.zeros(len(ref), bool)
bad[diff] = True
shown = 0
for idx, (lo, hi, pc, occ) in enumerate(STORES):
    o_lo, o_hi = lo - arena_base, hi - arena_base
    if o_lo < 0 or o_hi > len(bad):
        continue
    if bad[o_lo:o_hi].any():
        ins = next(('%s %s' % (e[1], e[2]) for e in _PROG if e[0] == pc), '?')
        print('   store #%d of %d  pc=%#x  occurrence #%d  %d/%d bytes wrong   %s'
              % (idx, len(STORES), pc, occ, int(bad[o_lo:o_hi].sum()), o_hi - o_lo, ins))
        shown += 1
        if shown >= 6:
            break
if not shown:
    print('   none - every differing byte was written by a store outside the logged range')

# Which transcendentals actually EXECUTE here? The emulator computes them exactly; AMD hardware
# approximates them (~1 ULP per the RDNA ISA), so any that run are a guaranteed source of divergence
# that is NOT a translation defect.
print('\ntranscendental / approximate ops executed by the pre-block:')
TR = ('v_rcp_', 'v_rsq_', 'v_sqrt_', 'v_exp_', 'v_log_', 'v_sin_', 'v_cos_')
tot = sum(EXEC_OPS.values())
hit = [(o, c) for o, c in EXEC_OPS.most_common() if o.startswith(TR)]
for o, c in hit:
    print('   %-16s %8d executions' % (o, c))
print('   (%d executions across %d distinct opcodes total)' % (tot, len(EXEC_OPS)))
if not hit:
    print('   none executed')
