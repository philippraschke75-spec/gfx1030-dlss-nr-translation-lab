"""State-dump bisection: find the first instruction at which the translated kernel's state diverges from
the gfx1100 emulator.

For a checkpoint address X the translated .s is instrumented so that
  * at entry every VGPR below NREG (except v0) is poisoned with 0xDEADBEEF and the workitem id is kept,
  * at label .Lpc_X each wave saves EXEC/SCC/VCC, forces EXEC=-1, dumps v0..v(NREG-1) (32 lanes) and
    s0..s63, writes a marker and ends.
The emulator does the same (poisoned VGPRs and SGPRs, snapshot at first arrival, wave ends) and adopts the
GPU's real arena base so pointer-valued registers are comparable. Only values the emulator has written
(not poison) are compared. A mismatch at X means the two disagree about the state *before* X executes.

usage (inside the sandbox): statebisect.py <32_1|...> <flags> [seed]
"""
import sys, os, re, struct, subprocess, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

ROOT = D.ROOT
BIN = Path(r'C:\Program Files\AMD\ROCm\6.4\bin')
W = ROOT / 'build' / 'emu_bisect'; W.mkdir(parents=True, exist_ok=True)
POISON = 0xDEADBEEF
DUMP_OFF = 0x80000                        # dump area = parameter slot 0 + 512 KiB (kernarg hidden bytes are overwritten by the runtime)
SG_OFF = 0x40000                          # scalar dump offset inside the dump area
SG_LIST = list(range(64)) + [100, 101, 102]   # s0..s63, then saved EXEC, SCC, VCC
MARK = 0x600DF00D


def variant_source(src_text, nreg, addr, name, wg=(0, 0), wgsgpr=4):
    # wgsgpr: the SGPR holding workgroup_id_x, which is just user_sgpr_count - the system SGPRs
    # follow the user ones. k_swin_var has 4 user SGPRs so its id lands in s4, but k_attention has
    # 2 and uses s2. Hard-coding s4 made the guard below compare garbage, so every wave branched to
    # dumpskip and the GPU dump came back empty.
    lines = src_text.split('\n'); out = []
    entry = ['s_load_dwordx2 s[104:105], s[0:1], 0x0', 's_waitcnt lgkmcnt(0)', 's_add_u32 s104, s104, 0x%x' % DUMP_OFF,
             's_addc_u32 s105, s105, 0', 'v_mov_b32 v250, v0', 's_mov_b32 s98, s%d' % wgsgpr, 's_mov_b32 s99, s%d' % (wgsgpr + 1)] + ['v_mov_b32 v%d, 0x%x' % (r, POISON) for r in range(1, nreg)]
    guard = ['s_cselect_b32 s101, 1, 0', 's_cmp_lg_u32 s98, %d' % wg[0], 's_cbranch_scc1 .Ldumpskip_%x' % addr, 's_cmp_lg_u32 s99, %d' % wg[1], 's_cbranch_scc1 .Ldumpskip_%x' % addr]
    dump = ['s_mov_b32 s100, exec_lo', 's_mov_b32 s102, vcc_lo', 's_mov_b32 exec_lo, -1',
            'v_mbcnt_lo_u32_b32 v251, -1, 0', 'v_and_b32 v252, 0x3ff, v250', 'v_lshrrev_b32 v252, 5, v252',
            'v_mul_u32_u24 v253, %d, v252' % (len(SG_LIST) * 4), 'v_add_nc_u32 v253, 0x%x, v253' % SG_OFF,
            'v_mul_u32_u24 v252, %d, v252' % ((nreg + 1) * 128), 'v_lshl_add_u32 v251, v251, 2, v252']
    for r in range(nreg):
        dump += ['global_store_dword v251, v%d, s[104:105]' % r, 'v_add_nc_u32 v251, 128, v251']
    dump += ['s_mov_b32 exec_lo, 1']
    for i in SG_LIST:
        dump += ['v_mov_b32 v252, s%d' % i, 'global_store_dword v253, v252, s[104:105]', 'v_add_nc_u32 v253, 4, v253']
    dump += ['v_mov_b32 v252, 0x%x' % MARK, 'global_store_dword v251, v252, s[104:105]', 's_waitcnt_vscnt null, 0', 's_endpgm', '.Ldumpskip_%x:' % addr, 's_endpgm']
    for l in lines:
        out.append(l)
        if l == name + ':':
            out += entry
        if l == '.Lpc_%x:' % addr:
            out += guard + dump
    txt = '\n'.join(out)
    txt = re.sub(r'\.amdhsa_next_free_vgpr \d+', '.amdhsa_next_free_vgpr 256', txt)
    txt = re.sub(r'\.amdhsa_next_free_sgpr \d+', '.amdhsa_next_free_sgpr 106', txt)
    return txt


class Ctx:
    def __init__(self, key, flags, seed, wg=(0, 0)):
        self.wg = wg; grid = (wg[0] + 1, wg[1] + 1)
        self.key, self.flags, self.seed, self.grid = key, flags, seed, grid
        self.sym, lds = D.SYMS[key]; self.lds = lds or D.group_size(self.sym)
        cov = json.load(open(ROOT / 'build' / 'kernels-hw-scratch' / 'coverage.json'))
        self.nreg = next(k['source_vgprs'] for k in cov['kernels'] if k['symbol'] == self.sym)
        self.src = (ROOT / 'build' / 'kernels-hw-scratch' / (self.sym + '.s')).read_text()
        self.prog = E.load_program(R.DIS, {self.sym})
        self.soft = 0
        self.hi_seen = {4}
        self.rebuild_kernarg()

    def rebuild_kernarg(self):
        self.ka = V.make_kernarg(flags=self.flags, grid=self.grid)

    def adopt(self, dev):
        if dev != V.ARENA:
            V.ARENA = dev; self.rebuild_kernarg()

    def arena(self):
        g, KA = V.build(self.seed, self.ka, len(V.PTR_FIELDS))
        g.regions[1].arr[DUMP_OFF:DUMP_OFF + 8 * (self.nreg + 1) * 128] = 0        # dump area starts empty
        return g, KA

    def trace(self):
        g, KA = self.arena(); tr = []
        sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: self.wg[0], 15: self.wg[1]}
        E.run_workgroup(self.prog, g, self.lds, 256, sg, trace=tr, poison=True)
        return tr

    def emu_snap(self, addr):
        g, KA = self.arena(); snap = {}
        sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: self.wg[0], 15: self.wg[1]}
        E.run_workgroup(self.prog, g, self.lds, 256, sg, stop_at=addr, snap=snap, poison=True)
        return snap

    def validate_allocation(self, addr, arena, dev, rebased):
        self.adopt(dev)
        if bytes(self.ka) != rebased:
            raise RuntimeError('checkpoint rebased kernarg disagrees with emulator')
        actual, _ = self.arena()
        if not np.array_equal(actual.regions[1].arr, arena):
            raise RuntimeError('checkpoint input arena changed during rebase')
        if not self.emu_snap(addr):
            raise RuntimeError('checkpoint unreachable at allocated device address')

    def gpu_dump(self, addr, tag):
        txt = variant_source(self.src, self.nreg, addr, self.sym, self.wg, getattr(self, 'wgsgpr', 4))
        s = W / (tag + '.s'); s.write_text(txt)
        o, co = W / (tag + '.o'), W / (tag + '.co')
        subprocess.run([str(BIN / 'llvm-mc.exe'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1030', '-filetype=obj', str(s), '-o', str(o)], check=True, capture_output=True)
        subprocess.run([str(BIN / 'ld.lld.exe'), '-shared', str(o), '-o', str(co)], check=True, capture_output=True)
        g, KA = self.arena()
        arena = g.regions[1].arr.copy()
        def validate_allocation(dev, rebased):
            self.validate_allocation(addr, arena, dev, rebased)
        rc, msg, out = D.gpu(co, self.sym, tag, bytes(self.ka), arena, self.grid,
                             preflight=validate_allocation)
        if out is None:
            return rc, msg, None, None
        n = 8 * (self.nreg + 1) * 128
        vec = out[DUMP_OFF:DUMP_OFF + n].view(np.uint32).reshape(8, self.nreg + 1, 32)
        sca = out[DUMP_OFF + SG_OFF:DUMP_OFF + SG_OFF + 8 * len(SG_LIST) * 4].view(np.uint32).reshape(8, len(SG_LIST))
        m = re.search(r'arena_dev=0x([0-9a-f]+)', msg)
        if m: self.adopt(int(m[1], 16))
        return rc, msg, vec, sca

    def compare(self, addr, tag):
        exploratory = os.environ.get('BISECT_HEURISTIC_FILTERS', '0') == '1'
        ulp = int(os.environ.get('ULP', '0'))
        if ulp < 0 or (ulp and not exploratory):
            raise ValueError('ULP requires a nonnegative value and BISECT_HEURISTIC_FILTERS=1')
        # Reject faulting/nonterminating checkpoint fixtures before GPU work.
        snap = self.emu_snap(addr)
        if not snap:
            return None, 'checkpoint was not reached in the emulator'
        rc, msg, vec, sca = self.gpu_dump(addr, tag)
        if vec is None:
            return None, 'GPU %s %s' % (rc, msg)
        # gpu_dump may adopt a new device arena base; refresh pointer values.
        snap = self.emu_snap(addr)
        reached_gpu = {w for w in range(8) if vec[w, self.nreg, 0] == MARK}
        if reached_gpu != set(snap):
            return True, 'waves reaching X differ: emu=%s gpu=%s' % (sorted(snap), sorted(reached_gpu))
        bad = []; self.soft = 0
        for w in sorted(snap):
            V_, S_, scc = snap[w]
            e = V_[:self.nreg]; g = vec[w, :self.nreg]
            differ = (e != g) & (e != POISON)
            ptr = differ & (e >= np.uint32(0x8000)) & (e < np.uint32(0x140000))      # original image addresses (code/rodata pointers)
            near = np.zeros_like(differ)
            if ulp:                                                                  # float-noise tolerance (transcendentals differ per arch)
                near = differ & (np.abs(e.view(np.int32).astype(np.int64) - g.view(np.int32).astype(np.int64)) <= ulp)
            # Heuristic filtering is useful for exploration, never a proof of
            # agreement: integer values can look like floats or code pointers.
            soft = ((differ & (((e ^ g) & np.uint32(0xffff)) == 0)) | ptr | near) if exploratory else np.zeros_like(differ)
            idx = np.argwhere(differ & ~soft)
            self.soft += int(soft.sum())
            for r, l in idx[:3]:
                bad.append('wave%d v%d lane%d emu=%08x gpu=%08x' % (w, r, l, e[r, l], g[r, l]))
            if len(idx): bad.append('wave%d: %d hard-differing vector cells' % (w, len(idx)))
            gs = dict(zip(SG_LIST, sca[w]))
            for name, ev, gv in (('exec', S_[E.EXEC], gs[100]), ('scc', scc, gs[101]), ('vcc', S_[E.VCC], gs[102])):
                if ev != POISON and ev != gv:
                    bad.append('wave%d %s emu=%x gpu=%x' % (w, name, ev, gv))
            skip = set()
            for i in range(2, 64):
                if S_[i] == POISON or S_[i] == gs[i]:
                    continue
                if exploratory and 0x8000 <= S_[i] < 0x140000:      # diagnostic-only pointer heuristic
                    skip.update((i, i + 1)); self.hi_seen.add(int(gs[i + 1])); continue
                if exploratory and S_[i] == 0 and i % 2 == 1 and 0 < int(gs[i]) <= 0xff:
                    continue
                bad.append('wave%d s%d emu=%08x gpu=%08x' % (w, i, S_[i], gs[i]))
        return (True, '; '.join(bad[:6])) if bad else (False, 'match on %d waves (%d heuristically ignored cells; not numerical qualification)' % (len(snap), self.soft))


if __name__ == '__main__':
    if os.environ.get('ULP', '0') != '0' and os.environ.get('BISECT_HEURISTIC_FILTERS', '0') != '1':
        raise SystemExit('ULP requires BISECT_HEURISTIC_FILTERS=1 (exploratory diagnostics only)')
    key, flags = sys.argv[1], int(sys.argv[2], 0)
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    wg = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 0)
    ctx = Ctx(key, flags, seed, wg)
    tr = ctx.trace(); print('first-visit trace: %d instructions; NREG=%d' % (len(tr), ctx.nreg), flush=True)
    bad0, msg = ctx.compare(tr[0], 'p0'); print('probe at entry:', bad0, msg, flush=True)
    bad0, msg = ctx.compare(tr[0], 'p0b'); print('probe at entry (arena base adopted):', bad0, msg, flush=True)
    if bad0 is not False: raise SystemExit('entry state inconsistent -> harness problem')
    lo, hi = 0, len(tr) - 1
    bad_hi, msg_hi = ctx.compare(tr[hi], 'phi'); print('probe at last:', bad_hi, msg_hi[:240], flush=True)
    if bad_hi is None: raise SystemExit('GPU/checkpoint failure: ' + msg_hi)
    if bad_hi is False: raise SystemExit('no divergence visible at the last visited instruction')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        b, m = ctx.compare(tr[mid], 'p%d' % mid)
        print('trace[%d]=%x -> %s  %s' % (mid, tr[mid], 'DIVERGED' if b else 'match', (m or '')[:200]), flush=True)
        if b is None: raise SystemExit('GPU failure during bisect: ' + m)
        if b: hi = mid
        else: lo = mid
    print('LAST MATCH trace[%d]=%x ; FIRST DIVERGENCE trace[%d]=%x' % (lo, tr[lo], hi, tr[hi]))
    print('divergence detail:', ctx.compare(tr[hi], 'pfinal')[1])
    instr = {a: (op, args) for a, op, args, _ in ctx.prog}
    print('instruction BEFORE the divergence (executed last, result differs):', hex(tr[lo]), instr[tr[lo]])
    print('instruction at divergence point:', hex(tr[hi]), instr[tr[hi]])
