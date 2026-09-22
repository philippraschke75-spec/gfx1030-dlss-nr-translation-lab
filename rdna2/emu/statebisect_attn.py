"""Binary-search the first instruction where k_attention's GPU and emulator state diverge.

k_attention uses the single-pointer ABI (its descriptor has dispatch_ptr 0), so statebisect.py's
original k_swin_var convention - kernarg in s[0:1] - applies directly; only the kernarg contents and
the arena need replacing with this kernel's own.

Context: the mismatch is already known to be an emulator defect rather than a translation one (it is
invariant across random, zeroed and well-conditioned inputs, which rules out numerical sensitivity),
and fixing the dropped stride64 LDS scaling took it from 250 to 214. This finds what is left.

usage: statebisect_attn.py    (inside sandbox.py)
"""
import sys, os, json, struct
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D
import statebisect as SB
import kernelspec as K

SYM = '_Z11k_attention12AttnParams1d'
H = W = 8
NSLOT = 6
WEIGHT = 'block31_layer2.bin'


def source_vgprs(sym):
    import re
    dis = (SB.ROOT.parent / 'analysis' / 'gfx1100-disassembly.txt').read_text(errors='replace').splitlines()
    b = next(i for i, l in enumerate(dis) if l.startswith('0') and '<%s>:' % sym in l)
    e = next((i for i in range(b + 1, len(dis)) if re.match(r'^[0-9a-f]{16} <', dis[i])), len(dis))
    mx = -1
    for l in dis[b + 1:e]:
        t = l.split('//')[0]
        for m in re.finditer(r'v(\d+)', t): mx = max(mx, int(m[1]))
        for m in re.finditer(r'v\[(\d+):(\d+)\]', t): mx = max(mx, int(m[2]))
    return mx + 1


class AttnCtx(SB.Ctx):
    def __init__(self, seed=1):
        self.wg = (0, 0); self.grid = (1, 1)
        self.key, self.flags, self.seed = 'attn', 0, seed
        self.sym = SYM
        explicit, self.ksize, self.lds, self.hidden = K.kernel_meta(SYM)
        # coverage.json only ever holds the last-built kernel, so take the source VGPR count
        # straight from the gfx1100 disassembly: the highest v-register the kernel references.
        self.nreg = source_vgprs(SYM)
        self.src = (SB.ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.s')).read_text()
        self.prog = E.load_program(R.DIS, {SYM})
        self.wgsgpr = 2          # user_sgpr_count is 2 here, so workgroup_id_x is s2
        self.soft = 0
        self.hi_seen = {4}
        self.rebuild_kernarg()

    def rebuild_kernarg(self):
        ka = bytearray(self.ksize)
        S = lambda i: V.ARENA + i * V.SLOT
        for off, slot in ((0x00, 0), (0x08, 1), (0x10, 2), (0x18, 3)):
            struct.pack_into('<Q', ka, off, S(slot))
        struct.pack_into('<ii', ka, 0x20, H, W)
        for name, val in (('block_count_x', 1), ('block_count_y', 1), ('block_count_z', 1),
                          ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1),
                          ('grid_dims', 2)):
            if name in self.hidden:
                o, sz = self.hidden[name]
                struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
        self.ka = ka

    def arena(self):
        g, KA = V.build(self.seed, self.ka, NSLOT)
        a = g.regions[1].arr
        w = np.frombuffer((SB.ROOT / 'build' / 'weights' / WEIGHT).read_bytes(), np.uint8)
        a[2 * V.SLOT:2 * V.SLOT + len(w)] = w
        a[SB.DUMP_OFF:SB.DUMP_OFF + 8 * (self.nreg + 1) * 128] = 0      # dump area starts empty
        return g, KA


if __name__ == '__main__':
    ctx = AttnCtx()
    tr = ctx.trace()
    print('first-visit trace: %d instructions; NREG=%d' % (len(tr), ctx.nreg), flush=True)
    def only_scc(m):
        """Entry SCC legitimately differs: the emulator poisons state so it reads 2, hardware 0.
        Any other field differing at entry would be a real harness problem."""
        parts = [x.strip() for x in (m or '').split(';') if x.strip()]
        return bool(parts) and all(' scc ' in x for x in parts)

    bad0, msg = ctx.compare(tr[0], 'a0'); print('probe at entry:', bad0, msg, flush=True)
    bad0, msg = ctx.compare(tr[0], 'a0b'); print('probe at entry (arena base adopted):', bad0, msg, flush=True)
    if bad0 is not False and not only_scc(msg):
        raise SystemExit('entry state inconsistent -> harness problem')
    if bad0: print('   (entry differs only in scc - the known poisoning artifact; continuing)', flush=True)
    lo, hi = 0, len(tr) - 1
    bad_hi, msg_hi = ctx.compare(tr[hi], 'ahi'); print('probe at last:', bad_hi, (msg_hi or '')[:200], flush=True)
    if bad_hi is None: raise SystemExit('GPU/checkpoint failure: ' + msg_hi)
    if bad_hi is False: raise SystemExit('no divergence visible at the last visited instruction')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        b, m = ctx.compare(tr[mid], 'a%d' % mid)
        print('trace[%d]=%x -> %s %s' % (mid, tr[mid], 'DIVERGED' if b else 'match', (m or '')[:150]), flush=True)
        if b is None: raise SystemExit('GPU failure during bisect: ' + m)
        if b: hi = mid
        else: lo = mid
    instr = {a: (op, args) for a, op, args, _ in ctx.prog}
    print('\nLAST MATCH  trace[%d]=%x  %s' % (lo, tr[lo], instr[tr[lo]]))
    print('FIRST DIVERGENCE trace[%d]=%x  %s' % (hi, tr[hi], instr[tr[hi]]))
    print('detail:', ctx.compare(tr[hi], 'afinal')[1])
