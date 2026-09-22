"""Adapt statebisect.py's dump/compare machinery (originally built for k_swin_var's single-pointer ABI) to
k_qkv_attn's dual dispatch_ptr/kernarg_ptr ABI, using the exact real block22-pooled-output input and real
block23 weight data that produced a 7,840/8,192-byte GPU-vs-emulator mismatch in gpu_verify_block23_real.py.
Binary-searches the first-visit instruction trace to find the exact instruction where GPU and emulator state
diverge.
usage (inside the sandbox): statebisect_qkv_attn.py
"""
import sys, os, re, struct, subprocess, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D
import statebisect as SB

ROOT = D.ROOT
BIN = Path(r'C:\Program Files\AMD\ROCm\6.4\bin')
W = ROOT / 'build' / 'emu_bisect'; W.mkdir(parents=True, exist_ok=True)
POISON = SB.POISON
DUMP_OFF = SB.DUMP_OFF
SG_OFF = SB.SG_OFF
SG_LIST = SB.SG_LIST
MARK = SB.MARK

def variant_source_dual_ptr(src_text, nreg, addr, name, wg=(0, 0)):
    """Same as statebisect.variant_source, but reads the kernarg pointer from s[2:3] (this kernel's
    kernel descriptor enables both dispatch_ptr in s[0:1] AND kernarg_segment_ptr in s[2:3] - the
    original function assumes the single-pointer k_swin_var convention where s[0:1] IS the kernarg
    pointer, which would read garbage here since s[0:1] is the dispatch packet address instead)."""
    lines = src_text.split('\n'); out = []
    entry = ['s_load_dwordx2 s[104:105], s[2:3], 0x0', 's_waitcnt lgkmcnt(0)', 's_add_u32 s104, s104, 0x%x' % SB.DUMP_OFF,
             's_addc_u32 s105, s105, 0', 'v_mov_b32 v250, v0', 's_mov_b32 s98, s4', 's_mov_b32 s99, s5'] + ['v_mov_b32 v%d, 0x%x' % (r, SB.POISON) for r in range(1, nreg)]
    guard = ['s_cselect_b32 s101, 1, 0', 's_cmp_lg_u32 s98, %d' % wg[0], 's_cbranch_scc1 .Ldumpskip_%x' % addr, 's_cmp_lg_u32 s99, %d' % wg[1], 's_cbranch_scc1 .Ldumpskip_%x' % addr]
    dump = ['s_mov_b32 s100, exec_lo', 's_mov_b32 s102, vcc_lo', 's_mov_b32 exec_lo, -1',
            'v_mbcnt_lo_u32_b32 v251, -1, 0', 'v_and_b32 v252, 0x3ff, v250', 'v_lshrrev_b32 v252, 5, v252',
            'v_mul_u32_u24 v253, %d, v252' % (len(SB.SG_LIST) * 4), 'v_add_nc_u32 v253, 0x%x, v253' % SB.SG_OFF,
            'v_mul_u32_u24 v252, %d, v252' % ((nreg + 1) * 128), 'v_lshl_add_u32 v251, v251, 2, v252']
    for r in range(nreg):
        dump += ['global_store_dword v251, v%d, s[104:105]' % r, 'v_add_nc_u32 v251, 128, v251']
    dump += ['s_mov_b32 exec_lo, 1']
    for i in SB.SG_LIST:
        dump += ['v_mov_b32 v252, s%d' % i, 'global_store_dword v253, v252, s[104:105]', 'v_add_nc_u32 v253, 4, v253']
    dump += ['v_mov_b32 v252, 0x%x' % SB.MARK, 'global_store_dword v251, v252, s[104:105]', 's_waitcnt_vscnt null, 0', 's_endpgm', '.Ldumpskip_%x:' % addr, 's_endpgm']
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


SYM = '_Z10k_qkv_attn10AttnParams'
KSIZE = 296
H, W_ = 4, 4
GX, GY = 1, 1
DP = 0x7100_0000_0000


class QkvCtx:
    def __init__(self):
        self.wg = (0, 0); self.grid = (GX, GY)
        cov = json.load(open(ROOT / 'build' / 'kernels-hw-scratch' / 'coverage.json'))
        self.nreg = next(k['source_vgprs'] for k in cov['kernels'] if k['symbol'] == SYM)
        self.src = (ROOT / 'build' / 'kernels-hw-scratch' / (SYM + '.s')).read_text()
        self.prog = E.load_program(R.DIS, {SYM})
        self.soft = 0
        pooled4 = np.load(ROOT / 'build' / 'pooled4_checkpoint.npy')
        w23 = np.frombuffer((ROOT / 'build' / 'weights' / 'block23_layer2.bin').read_bytes(), np.uint8)
        self.pooled4, self.w23 = pooled4, w23
        self.rebuild_kernarg()

    def rebuild_kernarg(self):
        ka = bytearray(KSIZE)
        S = lambda k: V.ARENA + k * V.SLOT
        struct.pack_into('<Q', ka, 0x00, S(0))
        struct.pack_into('<Q', ka, 0x08, S(1))
        struct.pack_into('<Q', ka, 0x10, S(2))
        struct.pack_into('<i', ka, 0x18, H)
        struct.pack_into('<i', ka, 0x1c, W_)
        struct.pack_into('<i', ka, 0x20, 0)
        struct.pack_into('<i', ka, 0x24, 0)
        struct.pack_into('<i', ka, 0x34, 512)
        struct.pack_into('<III', ka, KSIZE - 20, GX, GY, 1)
        struct.pack_into('<HHH', ka, KSIZE - 8, 256, 1, 1)
        self.ka = ka

    def adopt(self, dev):
        if dev != V.ARENA:
            V.ARENA = dev; self.rebuild_kernarg()

    def arena(self):
        g, KA = V.build(1, self.ka, 6)
        a = g.regions[1].arr
        a[:] = 0
        a[0 * V.SLOT:0 * V.SLOT + len(self.pooled4)] = self.pooled4
        a[2 * V.SLOT:2 * V.SLOT + len(self.w23)] = self.w23
        a[DUMP_OFF:DUMP_OFF + 8 * (self.nreg + 1) * 128] = 0
        g.add('dispatch', DP, self._packet())
        return g, KA

    def _packet(self):
        pkt = bytearray(64); struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
        struct.pack_into('<III', pkt, 12, GX * 256, GY, 1); struct.pack_into('<II', pkt, 24, 64, 58368)
        return np.frombuffer(bytes(pkt), np.uint8).copy()

    def sg(self, KA):
        return {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: 0, 15: 0}

    def trace(self):
        g, KA = self.arena(); tr = []
        E.run_workgroup(self.prog, g, 58368, 256, self.sg(KA), trace=tr, poison=True, max_steps=30_000_000)
        return tr

    def emu_snap(self, addr):
        g, KA = self.arena(); snap = {}
        E.run_workgroup(self.prog, g, 58368, 256, self.sg(KA), stop_at=addr, snap=snap, poison=True, max_steps=30_000_000)
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
        txt = variant_source_dual_ptr(self.src, self.nreg, addr, SYM, self.wg)
        s = W / (tag + '.s'); s.write_text(txt)
        o, co = W / (tag + '.o'), W / (tag + '.co')
        subprocess.run([str(BIN / 'llvm-mc.exe'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1030', '-filetype=obj', str(s), '-o', str(o)], check=True, capture_output=True)
        subprocess.run([str(BIN / 'ld.lld.exe'), '-shared', str(o), '-o', str(co)], check=True, capture_output=True)
        g, KA = self.arena()
        arena = g.regions[1].arr.copy()
        def validate_allocation(dev, rebased):
            self.validate_allocation(addr, arena, dev, rebased)
        rc, msg, out = D.gpu(co, SYM, tag, bytes(self.ka), arena, self.grid, preflight=validate_allocation)
        if out is None:
            return rc, msg, None, None
        n = 8 * (self.nreg + 1) * 128
        vec = out[DUMP_OFF:DUMP_OFF + n].view(np.uint32).reshape(8, self.nreg + 1, 32)
        sca = out[DUMP_OFF + SG_OFF:DUMP_OFF + SG_OFF + 8 * len(SG_LIST) * 4].view(np.uint32).reshape(8, len(SG_LIST))
        m = re.search(r'arena_dev=0x([0-9a-f]+)', msg)
        if m: self.adopt(int(m[1], 16))
        return rc, msg, vec, sca

    def compare(self, addr, tag):
        snap = self.emu_snap(addr)
        if not snap:
            return None, 'checkpoint was not reached in the emulator'
        rc, msg, vec, sca = self.gpu_dump(addr, tag)
        if vec is None:
            return None, 'GPU %s %s' % (rc, msg)
        snap = self.emu_snap(addr)
        reached_gpu = {w for w in range(8) if vec[w, self.nreg, 0] == MARK}
        if reached_gpu != set(snap):
            return True, 'waves reaching X differ: emu=%s gpu=%s' % (sorted(snap), sorted(reached_gpu))
        bad = []; self.soft = 0
        for w in sorted(snap):
            V_, S_, scc = snap[w]
            e = V_[:self.nreg]; g = vec[w, :self.nreg]
            differ = (e != g) & (e != POISON)
            idx = np.argwhere(differ)
            for r, l in idx[:3]:
                bad.append('wave%d v%d lane%d emu=%08x gpu=%08x' % (w, r, l, e[r, l], g[r, l]))
            if len(idx): bad.append('wave%d: %d hard-differing vector cells' % (w, len(idx)))
            gs = dict(zip(SG_LIST, sca[w]))
            for name, ev, gv, pois in (('exec', S_[E.EXEC], gs[100], POISON), ('scc', scc, gs[101], 2), ('vcc', S_[E.VCC], gs[102], POISON)):
                if ev != pois and ev != gv:
                    bad.append('wave%d %s emu=%x gpu=%x' % (w, name, ev, gv))
            for i in range(4, 64):   # skip s0/s1 (dispatch_ptr) and s2/s3 (kernarg_segment_ptr) - both are
                                      # legitimately different device-allocation addresses on each side, not computed state
                if S_[i] == POISON or S_[i] == gs[i]:
                    continue
                bad.append('wave%d s%d emu=%08x gpu=%08x' % (w, i, S_[i], gs[i]))
        return (True, '; '.join(bad[:6])) if bad else (False, 'match on %d waves' % len(snap))


if __name__ == '__main__':
    ctx = QkvCtx()
    tr = ctx.trace(); print('first-visit trace: %d instructions; NREG=%d' % (len(tr), ctx.nreg), flush=True)
    bad0, msg = ctx.compare(tr[0], 'p0'); print('probe at entry:', bad0, msg, flush=True)
    bad0, msg = ctx.compare(tr[0], 'p0b'); print('probe at entry (arena base adopted):', bad0, msg, flush=True)
    if bad0 is not False: raise SystemExit('entry state inconsistent -> harness problem: ' + str(msg))
    lo, hi = 0, len(tr) - 1
    bad_hi, msg_hi = ctx.compare(tr[hi], 'phi'); print('probe at last:', bad_hi, (msg_hi or '')[:240], flush=True)
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
