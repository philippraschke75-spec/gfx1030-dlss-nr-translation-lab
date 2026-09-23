"""Count approximate-hardware ops executed by a k_swin_var variant (emulator only, no GPU).

AMD's V_RCP/V_RSQ/V_SQRT/V_EXP/V_LOG are ~1 ULP approximations per the RDNA ISA; gfx11emu computes
them exactly with numpy. Any kernel that executes them therefore CANNOT match hardware bit-for-bit,
and a difftest failure there is not evidence of a translation defect.

usage: count_transcendentals.py <32_0|32_1> <flags> [seed] [H W]
"""
import sys, os, struct, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

variant = sys.argv[1] if len(sys.argv) > 1 else '32_1'
flags = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x14
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
H, W = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (8, 8)

SYM, LDS = D.SYMS[variant]
LDS = LDS or D.group_size(SYM)
grid = ((W + 7) // 8, (H + 7) // 8)
IN_SLOT = V.PTR_FIELDS.index(0x40)

OPS = collections.Counter()
_orig = E.Exec.compile


def hook(self, pc):
    fn, op = _orig(self, pc), self.prog[pc][1]

    def wrapped(w, _f=fn, _o=op):
        OPS[_o] += 1
        return _f(w)
    return wrapped


ka = V.make_kernarg(H=H, W=W, offy=0, offx=0, flags=flags, grid=grid)
if flags == 0x14:
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

prog = E.load_program(R.DIS, {SYM})
g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
a = g.regions[1].arr
rng = np.random.default_rng(seed + 5)
a[IN_SLOT * V.SLOT:IN_SLOT * V.SLOT + H * W * 12] = \
    rng.random(H * W * 3, dtype=np.float32).astype(np.float32).view(np.uint8)

E.Exec.compile = hook
try:
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            E.run_workgroup(prog, g, LDS, 256,
                            {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}, max_steps=8_000_000)
finally:
    E.Exec.compile = _orig

TR = ('v_rcp_', 'v_rsq_', 'v_sqrt_', 'v_exp_', 'v_log_', 'v_sin_', 'v_cos_')
hits = [(o, c) for o, c in OPS.most_common() if o.startswith(TR)]
print('%s  flags=%#x  %d instructions executed' % (variant, flags, sum(OPS.values())))
for o, c in hits:
    print('   %-24s %7d' % (o, c))
print('   TOTAL approximate ops: %d' % sum(c for _, c in hits))

import json
out = Path(os.environ.get('OPS_JSON', 'ops_%s.json' % variant))
out.write_text(json.dumps(dict(OPS)))
print('   wrote %s' % out)
