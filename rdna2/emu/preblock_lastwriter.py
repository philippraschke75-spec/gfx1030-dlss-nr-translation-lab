"""Attribute the pre-block's wrong output bytes to the store that LAST wrote them.

preblock_firststore.py reports the earliest store (in execution order) whose bytes differ in the FINAL
arena. That blames a store for bytes it may have written correctly and that a later stage then rewrote.
Here every emulator store is logged (pc, wave, per-wave occurrence, addresses, data) and, per arena byte,
the last store to write it is tracked. A byte that differs from the GPU is then charged to that last
writer, so the stores listed are the ones whose OWN data is wrong in memory.

usage: preblock_lastwriter.py [seed] [H W]   env DLSSNR_WEIGHT_BLOB=<block0 record>; PRE_VARIANT, PRE_FLAGS
"""
import sys, os, re, struct, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

VARIANT = os.environ.get('PRE_VARIANT', '32_1')
FLAGS = int(os.environ.get('PRE_FLAGS', '0x14'), 0)
SYM, LDS = D.SYMS[VARIANT]
LDS = LDS or D.group_size(SYM)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
H, W = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (8, 8)
grid = ((W + 7) // 8, (H + 7) // 8)
IN_SLOT = V.PTR_FIELDS.index(0x40)
CUR = {'pc': -1, 'wid': -1}
OCC = collections.Counter()               # (wave, pc) -> arrivals so far
STORES = []                               # (pc, wave, occurrence, nlanes, nbytes)
LAST = None                               # int32 per arena byte: index into STORES of the last writer
ARENA_REGION = 1

_orig_compile, _orig_write = E.Exec.compile, E.GMem.write


def compile_hook(self, pc):
    fn = _orig_compile(self, pc); addr = self.prog[pc][0]
    def wrapped(w, _fn=fn, _a=addr):
        CUR['pc'] = _a; CUR['wid'] = w.wid; OCC[(w.wid, _a)] += 1
        return _fn(w)
    return wrapped


def write_hook(self, addr, data):
    a = np.asarray(addr, np.uint64)
    if a.size and LAST is not None:
        r = self.regions[ARENA_REGION]
        n = data.shape[1]
        m = (a >= r.base) & (a + np.uint64(n) <= r.base + len(r.arr))
        if m.any():
            off = (a[m] - np.uint64(r.base)).astype(np.int64)
            STORES.append((CUR['pc'], CUR['wid'], OCC[(CUR['wid'], CUR['pc'])], int(m.sum()), n))
            LAST[off[:, None] + np.arange(n)] = len(STORES) - 1
    return _orig_write(self, addr, data)


def kernarg():
    ka = V.make_kernarg(H=H, W=W, offy=0, offx=0, flags=FLAGS, grid=grid)
    if FLAGS == 0x14:
        struct.pack_into('<Q', ka, 0x00, 0); struct.pack_into('<Q', ka, 0x30, 0)
        # +0x50/+0x58/+0x60/+0x68 are 4-byte scalars but PTR_FIELDS makes make_kernarg write 8-byte arena pointers
        # there. Zero the whole qword: leaving +0x6c non-zero makes the GPU runner rebase the +0x68 qword, so the
        # kernel is SEEDED (s22 -> s_mul_i32 s17, s22, 0x9e3779b9) with a device address while the emulator is
        # seeded with 0 - which is the entire 7268-byte failure of difftest_preblock.py (see BLOCK0_FINDINGS.md).
        for off in (0x50, 0x58, 0x60, 0x68):
            struct.pack_into('<Q', ka, off, 0)
        struct.pack_into('<f', ka, 0x50, 1.0)
        for off in range(0x70, 0xa0, 8):
            struct.pack_into('<Q', ka, off, 0)
    return ka


def emulate(log=False):
    global LAST
    del STORES[:]; OCC.clear()
    ka = kernarg(); prog = E.load_program(R.DIS, {SYM})
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[ARENA_REGION].arr
    rng = np.random.default_rng(seed + 5)
    a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + H * W * 12] = \
        rng.random(H * W * 3, dtype=np.float32).astype(np.float32).view(np.uint8)
    init = a.copy()
    LAST = np.full(len(a), -1, np.int32) if log else None
    if log:
        E.Exec.compile, E.GMem.write = compile_hook, write_hook
    try:
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)
    finally:
        E.Exec.compile, E.GMem.write = _orig_compile, _orig_write
    return bytes(ka), init, a.copy(), g.regions[ARENA_REGION].base


ka, init, ref, base = emulate()
module = D.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.co')
rc, msg, out = D.gpu(module, SYM, 'lastwriter_s%d_%dx%d' % (seed, H, W), ka, init, grid)
if out is None:
    raise SystemExit('GPU failed: %s' % msg)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
ka, init, ref, base = emulate(log=True)
prog = list(E.load_program(R.DIS, {SYM})); INS = {e[0]: '%s %s' % (e[1], e[2]) for e in prog}

m = re.search(r'(\d+) pointers rebased', msg)
print('GPU rebased %s kernarg pointers (the pre-block launch has 6 real ones)' % (m[1] if m else '?'))
if m and int(m[1]) != 6 and FLAGS == 0x14:
    print('WARNING: unexpected pointer count - a stale qword in the kernarg is being rebased and will seed/alter the kernel')
wrong = ref != out
written = ref != init
print('arena bytes: %d changed by the emulator, %d differ from the GPU' % (int(written.sum()), int(wrong.sum())))
print('unwritten-by-emulator bytes that differ from the GPU: %d' % int((wrong & (LAST < 0)).sum()))

by_store = collections.Counter(LAST[wrong & (LAST >= 0)].tolist())
tot_by_store = collections.Counter(LAST[LAST >= 0].tolist())
by_pc = collections.defaultdict(lambda: [0, 0, 0])       # pc -> [bytes it is last writer of, of which wrong, stores]
for si, (pc, wid, occ, nl, nb) in enumerate(STORES):
    by_pc[pc][2] += 1
for si, c in tot_by_store.items():
    by_pc[STORES[si][0]][0] += c
for si, c in by_store.items():
    by_pc[STORES[si][0]][1] += c
print('\nfinal bytes by LAST WRITER pc (bytes owned / of which wrong):')
for pc, (own, bad, ns) in sorted(by_pc.items(), key=lambda kv: -kv[1][1]):
    print('   pc=%#x  stores=%3d  owned=%6d  wrong=%6d (%5.1f%%)  %s' % (pc, ns, own, bad, 100.0 * bad / max(own, 1), INS.get(pc, '?')))

print('\nearliest stores (execution order) that are the LAST writer of a wrong byte:')
for si in sorted(by_store)[:8]:
    pc, wid, occ, nl, nb = STORES[si]
    print('   store #%d  pc=%#x wave%d occurrence %d  %d lanes x %dB  wrong=%d/%d  %s'
          % (si, pc, wid, occ, nl, nb, by_store[si], tot_by_store[si], INS.get(pc, '?')))
print('\nstore #0 (0xb1598) bytes: how many are still owned by store #0 at the end, and how many wrong')
print('   owned=%d wrong=%d ; overwritten by later stores=%d' % (tot_by_store.get(0, 0), by_store.get(0, 0), 32 - tot_by_store.get(0, 0)))
