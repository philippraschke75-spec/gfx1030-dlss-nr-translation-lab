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
import sys, os, re, struct, subprocess, json, collections
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
PREBLOCK_DUMP_SRC = 0x08                  # the pre-block launch nulls kernarg +0x00, the dump's usual base. The HIP
                                          # runtime overwrites everything at +0xa8 and up, so the base is derived from the
                                          # output pointer (arena slot 1) instead: slot 0 == out_ptr - SLOT.


def variant_source(src_text, nreg, addr, name, wg=(0, 0), wgsgpr=4, dump_src_off=0, dump_delta=None, occ=1):
    # wgsgpr: the SGPR holding workgroup_id_x, which is just user_sgpr_count - the system SGPRs
    # follow the user ones. k_swin_var has 4 user SGPRs so its id lands in s4, but k_attention has
    # 2 and uses s2. Hard-coding s4 made the guard below compare garbage, so every wave branched to
    # dumpskip and the GPU dump came back empty.
    # dump base = kernarg pointer at dump_src_off + dump_delta (default: slot 0 + DUMP_OFF, as always)
    if dump_delta is None: dump_delta = DUMP_OFF
    adj = (['s_add_u32 s104, s104, 0x%x' % dump_delta, 's_addc_u32 s105, s105, 0'] if dump_delta >= 0
           else ['s_sub_u32 s104, s104, 0x%x' % -dump_delta, 's_subb_u32 s105, s105, 0'])
    # The harness reserves s97..s105 (occurrence counter, workgroup ids, saved EXEC/SCC/VCC, dump base).
    used = [int(m[1]) for m in re.finditer(r'\bs(\d+)\b', src_text)] + \
           [int(m[2]) for m in re.finditer(r'\bs\[(\d+):(\d+)\]', src_text)]
    assert max(used, default=-1) < 96, 'kernel uses SGPR s%d, which the harness reserves' % max(used)
    lines = src_text.split('\n'); out = []
    entry = ['s_load_dwordx2 s[104:105], s[0:1], 0x%x' % dump_src_off, 's_waitcnt lgkmcnt(0)'] + adj + (['s_mov_b32 s97, 0'] if occ > 1 else []) + ['v_mov_b32 v250, v0', 's_mov_b32 s98, s%d' % wgsgpr, 's_mov_b32 s99, s%d' % (wgsgpr + 1)] + ['v_mov_b32 v%d, 0x%x' % (r, POISON) for r in range(1, nreg)]
    guard = ['s_cselect_b32 s101, 1, 0'] + (['s_add_u32 s97, s97, 1', 's_cmp_lg_u32 s97, %d' % occ,
                                                          's_cbranch_scc1 .Lresume_%x' % addr] if occ > 1 else [])
    guard += ['s_cmp_lg_u32 s98, %d' % wg[0], 's_cbranch_scc1 .Ldumpskip_%x' % addr, 's_cmp_lg_u32 s99, %d' % wg[1], 's_cbranch_scc1 .Ldumpskip_%x' % addr]
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
    if occ > 1:
        dump += ['.Lresume_%x:' % addr, 's_cmp_lg_u32 s101, 0']           # SCC := saved SCC, then the original instruction
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


def source_vgprs(sym):
    """Highest VGPR the kernel actually uses, read from the disassembly.

    coverage.json only ever holds the LAST kernel that was built, so looking the symbol up there
    raises StopIteration for anything else - which is every kernel but one. statebisect_attn.py
    carries this same fix; it belongs in the shared module.
    """
    dis = (ROOT.parent / 'analysis' / 'gfx1100-disassembly.txt').read_text(errors='replace').splitlines()
    b = next(i for i, l in enumerate(dis) if l.startswith('0') and '<%s>:' % sym in l)
    e = next((i for i in range(b + 1, len(dis)) if re.match(r'^[0-9a-f]{16} <', dis[i])), len(dis))
    mx = -1
    for l in dis[b + 1:e]:
        t = l.split('//')[0]
        for m in re.finditer(r'v(\d+)', t):
            mx = max(mx, int(m[1]))
        for m in re.finditer(r'v\[(\d+):(\d+)\]', t):
            mx = max(mx, int(m[2]))
    return mx + 1


class Ctx:
    def __init__(self, key, flags, seed, wg=(0, 0)):
        self.wg = wg; grid = (wg[0] + 1, wg[1] + 1)
        self.H = int(os.environ.get('BISECT_H', '8'))     # 8x8 / one workgroup is the minimal
        self.W = int(os.environ.get('BISECT_W', '8'))     # failing case: mismatches scale linearly
                                                          # with workgroup count, ~67% either way
        self.key, self.flags, self.seed, self.grid = key, flags, seed, grid
        # flags 0x14 is the block0 pre-block. run_var.build() puts the real weights record in slot 2 only when
        # DLSSNR_WEIGHT_BLOB is set; otherwise slot 2 is random e4m3-shaped bytes and the launch is NOT the
        # one that fails (its intermediate values, and so its divergences, are different).
        blob = os.environ.get('DLSSNR_WEIGHT_BLOB')
        if flags == 0x14 and not blob and os.environ.get('BISECT_ALLOW_RANDOM_WEIGHTS') != '1':
            raise SystemExit('flags 0x14 needs DLSSNR_WEIGHT_BLOB=<.../weights/block0.bin> (the failing launch uses the real '
                             'block0 weights). Set BISECT_ALLOW_RANDOM_WEIGHTS=1 to bisect a random-weights launch on purpose.')
        if flags == 0x14:
            import hashlib
            print('weights: %s' % (('%s sha256=%s' % (blob, hashlib.sha256(open(blob, 'rb').read()).hexdigest()[:16])) if blob
                                  else 'RANDOM (BISECT_ALLOW_RANDOM_WEIGHTS=1) - not the failing launch'), flush=True)
        self.sym, lds = D.SYMS[key]; self.lds = lds or D.group_size(self.sym)
        self.nreg = source_vgprs(self.sym)
        self.src = (ROOT / 'build' / 'kernels-hw-scratch' / (self.sym + '.s')).read_text()
        self.prog = E.load_program(R.DIS, {self.sym})
        self.soft = 0
        self.hi_seen = {4}
        self.rebuild_kernarg()

    def rebuild_kernarg(self):
        # flags 0x14 is the PRE-BLOCK launch, which is not make_kernarg's default shape: +0x00 and
        # +0x30 are null, the float-RGB input lives at +0x40, the scalars at +0x50..0x68 are zero
        # except +0x50 = 1.0, and +0x70..0x9f are zero. Bisecting with the encoder defaults (and
        # offx/offy = -4) runs a different launch from the one that fails, so any divergence found
        # belongs to that other configuration. Layout per chain_import_preblock_real.py.
        if self.flags == 0x14:
            self.ka = V.make_kernarg(H=self.H, W=self.W, offy=0, offx=0,
                                     flags=self.flags, grid=self.grid)
            struct.pack_into('<Q', self.ka, 0x00, 0)
            struct.pack_into('<Q', self.ka, 0x30, 0)
            # Zero the FULL 8-byte spans first. +0x50/+0x58/+0x60/+0x68 are 4-byte scalars, but
            # PTR_FIELDS treats them as pointers and make_kernarg wrote whole pointers there.
            # Clearing only the low dword (as difftest_preblock does) leaves stale high bits, which
            # the GPU-side rebaser then shifts - the kernargs disagree and the run aborts.
            if os.environ.get('BISECT_LEGACY_KERNARG') == '1':
                # Reproduces difftest_preblock.py's fixture ON PURPOSE: low dwords only, so +0x6c keeps arena
                # bits, the runner rebases the +0x68 qword and the kernel is seeded with a device address
                # (s22 -> s_mul_i32 s17, s22, 0x9e3779b9 at 0xb0550) while the emulator is seeded with 0.
                # The allocation preflight then aborts with the differing kernarg fields.
                for off in (0x50, 0x54, 0x58, 0x5c, 0x60, 0x64, 0x68):
                    struct.pack_into('<I', self.ka, off, 0)
            else:
                for off in (0x50, 0x58, 0x60, 0x68):
                    struct.pack_into('<Q', self.ka, off, 0)
                for off in (0x54, 0x5c, 0x64):
                    struct.pack_into('<I', self.ka, off, 0)
            struct.pack_into('<f', self.ka, 0x50, 1.0)
            for off in range(0x70, 0xa0, 8):
                struct.pack_into('<Q', self.ka, off, 0)
        else:
            self.ka = V.make_kernarg(flags=self.flags, grid=self.grid)

    def adopt(self, dev):
        if dev != V.ARENA:
            V.ARENA = dev; self.rebuild_kernarg()

    def arena(self):
        g, KA = V.build(self.seed, self.ka, len(V.PTR_FIELDS))
        if self.flags == 0x14:                 # the pre-block reads float RGB, not e4m3-shaped bytes
            IN = V.PTR_FIELDS.index(0x40)
            rng = np.random.default_rng(self.seed + 5)
            n = self.H * self.W * 12
            g.regions[1].arr[IN * V.SLOT:IN * V.SLOT + n] =                 rng.random(self.H * self.W * 3, dtype=np.float32).astype(np.float32).view(np.uint8)
        g.regions[1].arr[DUMP_OFF:DUMP_OFF + 8 * (self.nreg + 1) * 128] = 0        # dump area starts empty
        return g, KA

    def _sgprs(self):
        g, KA = self.arena()
        return g, {0: KA & 0xffffffff, 1: KA >> 32, 14: self.wg[0], 15: self.wg[1]}

    def trace(self):
        """First-visit trace of wave 0 (each address once). Blind to loop iterations: see dynamic_trace."""
        g, sg = self._sgprs(); tr = []
        E.run_workgroup(self.prog, g, self.lds, 256, sg, trace=tr, poison=True)
        return tr

    def dynamic_trace(self, upto=None):
        """Wave 0's executed instruction stream as [(addr, occurrence)], occurrence counted per address from 1.

        upto=<pc> ends the stream just before wave 0's first arrival at that pc (the arrival itself is not
        in the list). This is the sequence a bisection has to walk: the first-visit trace skips every loop
        iteration after the first, so its neighbours are not always consecutive instructions."""
        seq = []; count = collections.Counter(); orig = E.Exec.compile
        def hook(exself, pc):
            fn = orig(exself, pc); a = exself.prog[pc][0]
            def wrapped(w, _fn=fn, _a=a):
                if w.wid == 0:
                    count[_a] += 1; seq.append((_a, count[_a]))
                return _fn(w)
            return wrapped
        g, sg = self._sgprs(); E.Exec.compile = hook
        try:
            E.run_workgroup(self.prog, g, self.lds, 256, sg, stop_at=upto, snap={}, poison=True)
        finally:
            E.Exec.compile = orig
        return seq

    def emu_snap(self, addr, occ=1):
        """Each wave's state just BEFORE its occ-th arrival at addr executes (the wave then ends)."""
        g, sg = self._sgprs(); snap = {}
        self._emu_mem = g.regions[1].arr                       # aliases the run's arena: read after the run below
        if occ == 1:
            E.run_workgroup(self.prog, g, self.lds, 256, sg, stop_at=addr, snap=snap, poison=True)
            return snap
        count = collections.Counter(); orig = E.Exec.compile
        def hook(exself, pc):
            fn = orig(exself, pc)
            if exself.prog[pc][0] != addr:
                return fn
            def wrapped(w, _fn=fn):
                count[w.wid] += 1
                if count[w.wid] == occ:
                    snap[w.wid] = (w.V.copy(), list(w.S), w.scc); w.state = 'done'; return
                return _fn(w)
            return wrapped
        E.Exec.compile = hook
        try:
            E.run_workgroup(self.prog, g, self.lds, 256, sg, poison=True)
        finally:
            E.Exec.compile = orig
        return snap

    def _snap(self, addr, occ):
        return self.emu_snap(addr) if occ == 1 else self.emu_snap(addr, occ)

    def _dump(self, addr, tag, occ):
        return self.gpu_dump(addr, tag) if occ == 1 else self.gpu_dump(addr, tag, occ)

    def validate_allocation(self, addr, arena, dev, rebased, occ=1):
        self.adopt(dev)
        if bytes(self.ka) != rebased:
            import struct as _s
            d = [(o, _s.unpack_from('<Q', bytes(self.ka), o)[0], _s.unpack_from('<Q', rebased, o)[0])
                 for o in range(0, min(len(rebased), len(self.ka)) - 7, 8)
                 if bytes(self.ka)[o:o + 8] != rebased[o:o + 8]]
            raise RuntimeError('checkpoint rebased kernarg disagrees with emulator; differing '
                               'fields (offset, mine, gpu): '
                               + ', '.join('%#x: %#x vs %#x' % t for t in d[:12]))
        actual, _ = self.arena()
        if not np.array_equal(actual.regions[1].arr, arena):
            raise RuntimeError('checkpoint input arena changed during rebase')
        if not self._snap(addr, occ):
            raise RuntimeError('checkpoint unreachable at allocated device address')

    def gpu_dump(self, addr, tag, occ=1):
        if self.flags == 0x14:
            dsrc, ddelta = PREBLOCK_DUMP_SRC, DUMP_OFF - V.SLOT * V.PTR_FIELDS.index(PREBLOCK_DUMP_SRC)
        else:
            dsrc, ddelta = 0, DUMP_OFF
        txt = variant_source(self.src, self.nreg, addr, self.sym, self.wg, getattr(self, 'wgsgpr', 4), dsrc, ddelta, occ)
        s = W / (tag + '.s'); s.write_text(txt)
        o, co = W / (tag + '.o'), W / (tag + '.co')
        subprocess.run([str(BIN / 'llvm-mc.exe'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1030', '-filetype=obj', str(s), '-o', str(o)], check=True, capture_output=True)
        subprocess.run([str(BIN / 'ld.lld.exe'), '-shared', str(o), '-o', str(co)], check=True, capture_output=True)
        g, KA = self.arena()
        arena = g.regions[1].arr.copy()
        def validate_allocation(dev, rebased):
            self.validate_allocation(addr, arena, dev, rebased, occ)
        rc, msg, out = D.gpu(co, self.sym, tag, bytes(self.ka), arena, self.grid,
                             preflight=validate_allocation)
        if out is None:
            return rc, msg, None, None
        self._gpu_mem = out
        n = 8 * (self.nreg + 1) * 128
        vec = out[DUMP_OFF:DUMP_OFF + n].view(np.uint32).reshape(8, self.nreg + 1, 32)
        sca = out[DUMP_OFF + SG_OFF:DUMP_OFF + SG_OFF + 8 * len(SG_LIST) * 4].view(np.uint32).reshape(8, len(SG_LIST))
        m = re.search(r'arena_dev=0x([0-9a-f]+)', msg)
        if m: self.adopt(int(m[1], 16))
        return rc, msg, vec, sca

    def states(self, addr, tag, occ=1):
        """(emulator snapshot, GPU vector dump, GPU scalar dump) for the occ-th arrival at addr, or an error string."""
        if not self._snap(addr, occ):              # reject faulting/nonterminating fixtures before GPU work
            return 'checkpoint (%#x, occurrence %d) was not reached in the emulator' % (addr, occ)
        rc, msg, vec, sca = self._dump(addr, tag, occ)
        if vec is None:
            return 'GPU %s %s' % (rc, msg)
        return self._snap(addr, occ), vec, sca        # gpu_dump may adopt a new arena base: refresh pointers

    def mem_diff(self):
        """Arena bytes that differ between the emulator and the GPU, excluding the harness dump area.

        Both sides stop every wave at its occ-th arrival at the checkpoint, so this is the memory the stores
        executed BEFORE the checkpoint produced (waves that never reach it run to the end on both sides)."""
        e, g = self._emu_mem, self._gpu_mem
        d = np.nonzero(e != g)[0]
        return d[(d < DUMP_OFF) | (d >= DUMP_OFF + SG_OFF + 0x1000)]

    def compare(self, addr, tag, occ=1):
        exploratory = os.environ.get('BISECT_HEURISTIC_FILTERS', '0') == '1'
        ulp = int(os.environ.get('ULP', '0'))
        if ulp < 0 or (ulp and not exploratory):
            raise ValueError('ULP requires a nonnegative value and BISECT_HEURISTIC_FILTERS=1')
        st = self.states(addr, tag, occ)
        if isinstance(st, str):
            return None, st
        snap, vec, sca = st
        mode = os.environ.get('BISECT_PREDICATE', 'regs')       # regs: register/scalar state; mem: arena bytes
        if mode == 'mem':
            d = self.mem_diff()
            if not len(d):
                return False, 'memory matches (arena, dump area excluded)'
            f = int(d[0])
            return True, 'memory: %d differing bytes; first slot %d +%#x emu=%02x gpu=%02x' % (
                len(d), f // V.SLOT, f % V.SLOT, self._emu_mem[f], self._gpu_mem[f])
        reached_gpu = {w for w in range(8) if vec[w, self.nreg, 0] == MARK}
        if reached_gpu != set(snap):
            return True, 'waves reaching X differ: emu=%s gpu=%s' % (sorted(snap), sorted(reached_gpu))
        bad = []; self.soft = 0
        # s_getpc_b64 returns the ORIGINAL image address in the emulator and the translated code address on
        # the GPU, so those two can never agree. This is the only SGPR difference explained in strict
        # mode, and it is exact rather than by register number: a pair is explained only when the
        # emulator's low dword is precisely a getpc return address (instruction address + 4) and its high
        # dword is 0. Blanket-skipping the destination registers would also hide real data in s2..s5,
        # s18/s19 and s38/s39. Every explained pair is counted and printed. NOTE: values COMPUTED from a
        # getpc result (s_add_u32 s38, s38, imm after s_getpc_b64 s[38:39]) are PC-relative pointers that
        # differ for the same reason and are not covered here; they surface as a divergence at the first
        # such add, which is a harness artifact, not a translation defect.
        getpc_ret = {a + 4 for a, op, args, _ in getattr(self, 'prog', ()) if op == 's_getpc_b64'}
        self.pc_explained = 0
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
            # SCC carries its own poison sentinel: run_workgroup sets scc = 2 for "unwritten", since
            # 0 and 1 are both real values. Comparing that against the GPU's real 0 reports every
            # wave as differing at entry and aborts as a harness fault, which is what it did.
            for name, ev, gv, poison in (('exec', S_[E.EXEC], gs[100], POISON),
                                         ('scc', scc, gs[101], 2),
                                         ('vcc', S_[E.VCC], gs[102], POISON)):
                if ev != poison and ev != gv:
                    bad.append('wave%d %s emu=%x gpu=%x' % (w, name, ev, gv))
            skip = set()
            for i in range(2, 64):
                if i in skip or S_[i] == POISON or S_[i] == gs[i]:
                    continue
                if i + 1 < 64 and S_[i] in getpc_ret and S_[i + 1] == 0:
                    self.pc_explained += 1; skip.update((i, i + 1)); continue
                if exploratory and 0x8000 <= S_[i] < 0x140000:      # diagnostic-only pointer heuristic
                    skip.update((i, i + 1)); self.hi_seen.add(int(gs[i + 1])); continue
                if exploratory and S_[i] == 0 and i % 2 == 1 and 0 < int(gs[i]) <= 0xff:
                    continue
                bad.append('wave%d s%d emu=%08x gpu=%08x' % (w, i, S_[i], gs[i]))
        return (True, '; '.join(bad[:6])) if bad else (False, 'match on %d waves (%d heuristically ignored cells; %d getpc-return SGPR pairs explained; not numerical qualification)' % (len(snap), self.soft, self.pc_explained))

    def report_transition(self, dyn, lo, hi):
        """Operand values around the instruction executed between dyn[lo] (states match) and dyn[hi] (they differ)."""
        instr = {a: (op, args) for a, op, args, _ in self.prog}
        (alo, olo), (ahi, ohi) = dyn[lo], dyn[hi]
        op, args = instr[alo]
        print('\nTRANSITION: %#x occurrence %d  %s %s' % (alo, olo, op, args))
        a = self.states(alo, 'rep_lo', olo); b = self.states(ahi, 'rep_hi', ohi)
        if isinstance(a, str) or isinstance(b, str):
            print('  could not collect states:', a if isinstance(a, str) else b); return
        (sa, va, _), (sb, vb, _) = a, b
        regs = sorted({int(m) for m in re.findall(r'\bv(\d+)\b', args)} |
                      {r for m in re.finditer(r'\bv\[(\d+):(\d+)\]', args) for r in range(int(m[1]), int(m[2]) + 1)})
        for w in sorted(sb):
            Eh, Gh = sb[w][0][:self.nreg], vb[w, :self.nreg]
            d = (Eh != Gh) & (Eh != POISON)
            if not d.any():
                continue
            print('  wave %d: %d differing cells after the instruction; registers: %s'
                  % (w, int(d.sum()), ' '.join('v%d(%d lanes)' % (r, int(d[r].sum())) for r in np.nonzero(d.any(axis=1))[0])))
            for r in np.nonzero(d.any(axis=1))[0]:
                for l in np.nonzero(d[r])[0][:6]:
                    print('    AFTER  v%-3d lane%-2d emu=%08x gpu=%08x  xor=%08x' % (r, l, Eh[r, l], Gh[r, l], Eh[r, l] ^ Gh[r, l]))
            lanes = np.nonzero(d.any(axis=0))[0][:3]
            El, Gl = sa[w][0][:self.nreg], va[w, :self.nreg]
            for l in lanes:
                for r in regs:
                    print('    BEFORE v%-3d lane%-2d emu=%08x gpu=%08x  %s' % (r, l, El[r, l], Gl[r, l], 'same' if El[r, l] == Gl[r, l] else 'DIFFERENT'))
            break


if __name__ == '__main__':
    if os.environ.get('ULP', '0') != '0' and os.environ.get('BISECT_HEURISTIC_FILTERS', '0') != '1':
        raise SystemExit('ULP requires BISECT_HEURISTIC_FILTERS=1 (exploratory diagnostics only)')
    key, flags = sys.argv[1], int(sys.argv[2], 0)
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    wg = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 0)
    ctx = Ctx(key, flags, seed, wg)
    instr = {a: (op, args) for a, op, args, _ in ctx.prog}

    # BISECT_UPTO=<pc>: search only [entry, wave 0's first arrival at pc]. A whole-run bisection assumes a
    # divergence persists, but registers get overwritten ("healed") while the wrong bytes stay in memory, so
    # the end of the run can match even though an earlier point does not. Pick the end point at a place
    # already known to differ (e.g. the first store that writes a wrong byte).
    upto = int(os.environ['BISECT_UPTO'], 16) if os.environ.get('BISECT_UPTO') else None
    dyn = ctx.dynamic_trace(upto)
    if upto is not None:
        dyn.append((upto, 1))
    print('dynamic trace of wave 0: %d instructions, %d distinct addresses; NREG=%d'
          % (len(dyn), len({a for a, _ in dyn}), ctx.nreg), flush=True)

    def probe(k, tag):
        a, o = dyn[k]
        b, m = ctx.compare(a, tag, o)
        print('dyn[%d]=%x occ=%d -> %s  %s' % (k, a, o, 'DIVERGED' if b else ('match' if b is False else 'ERROR'), (m or '')[:200]), flush=True)
        return b, m

    for tag in ('p0', 'p0b'):                                   # entry twice: harness sanity, then with the adopted arena base
        b, m = probe(0, tag)
        if b is not False: raise SystemExit('entry state inconsistent -> harness problem')
    lo, hi = 0, len(dyn) - 1
    b, m = probe(hi, 'phi')
    if b is None: raise SystemExit('GPU/checkpoint failure: ' + m)
    if b is False:
        raise SystemExit('the end point matches, so bisection is not valid: either there is no divergence in '
                         'register state or it healed. Set BISECT_UPTO to a pc known to differ.')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        b, m = probe(mid, 'p%d' % mid)
        if b is None: raise SystemExit('GPU failure during bisect: ' + m)
        if b: hi = mid
        else: lo = mid
    print('\nLAST MATCH dyn[%d]=%x occ=%d ; FIRST DIVERGENCE dyn[%d]=%x occ=%d' % (lo, *dyn[lo], hi, *dyn[hi]))
    print('instruction executed between them:', hex(dyn[lo][0]), instr[dyn[lo][0]])
    ctx.report_transition(dyn, lo, hi)
