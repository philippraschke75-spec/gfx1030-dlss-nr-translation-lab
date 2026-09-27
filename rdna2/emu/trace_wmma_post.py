"""post_block WMMA instrumentation: stop the k_post_block emulation at a chosen
v_wmma_f32_16x16x16_f16 and decode its A-operand mapping empirically (mirror of
trace_wmma_pre.py; arena/kernarg come straight from the difftest_spec registry).

The baseline fixture is the post_block_const fill (e4m3 0x38 inputs, f32 constants,
real block70 weights) - the one configuration with a proven 0-mismatch PASS. An
impulse writes a single byte value at slot*SLOT + off instead of leaving the
fixture byte in place.

usage:
  trace_wmma_post.py tapcount                 exec count per WMMA site (kernel symbol only)
  trace_wmma_post.py layout                   per-workgroup read/write extent per arena slot
  trace_wmma_post.py dump [SLOT OFF [VAL]]    baseline A/B/D full dump (+ impulse + A-delta)
  trace_wmma_post.py sweep [SLOT OFF0 OFF1 STEP [VAL]]
                                              dense byte-offset sweep vs site 1's A operand
  trace_wmma_post.py taps SLOT OFF [VAL]      one impulse: A-delta lanes at EVERY site
  trace_wmma_post.py loops SITE SLOT OFF [VAL]
                                              per-iteration A/B/D deltas at one site
  trace_wmma_post.py mat SITE [NTH [WAVE [SLOT OFF [VAL]]]]
                                              readable A/B matrices + D delta (f16/f32)

Pure emulator instrumentation; no kernel/build files are touched.
"""
import sys, re, struct, os
from pathlib import Path
import numpy as np

_myargs = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, kernelspec as K, difftest_spec as DS

SYM = '_Z22k_post_block_1h_32_fp810PostParams'
SWIN = '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'
WMMA_OP = os.environ.get('WMMA_OP', 'v_wmma_f32_16x16x16_f16')
seed = 1
spec = DS.SPECS['post_block_const']()      # kernarg/scalars/grid/fills/weights from the registry
H = spec.scalars[0x20][1]                  # 16
W = spec.scalars[0x24][1]                  # 32
grid, threads = spec.grid, spec.threads    # (4, 2), 256


def f16s(u16):
    """uint16 array -> float values via bit reinterpretation."""
    return np.asarray(u16, np.uint16).view(np.float16).astype(np.float64)


def prepare(impulse=None):
    """impulse = (slot, off, val): overwrite that arena byte in the otherwise
    registry-faithful arena. Returns prog (entry first), g, lds, sgprs (wg 0,0)."""
    ka = spec.kernarg()
    p = E.load_program(R.DIS, {SYM, SWIN})
    entry = E.load_program(R.DIS, {SYM})[0][0]
    k = min(idx for idx, x in enumerate(p) if x[0] == entry)
    prog = p[k:] + p[:k]
    g, KA = V.build(seed, bytes(ka), spec.nslot)
    a = g.regions[1].arr
    a[:] = spec.arena(seed)[:len(a)]
    if impulse:
        slot, off, val = impulse
        a[slot * V.SLOT + off] = val & 0xff
    lds = spec.lds
    DP = 0x7100_0000_0000
    pkt = bytearray(64)
    struct.pack_into('<HHHHHH', pkt, 0, 0, 3, threads, 1, 1, 0)
    struct.pack_into('<III', pkt, 12, grid[0] * threads, grid[1], 1)
    struct.pack_into('<II', pkt, 24, 64, lds)
    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
    sgprs = {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: 0, 15: 0}
    return prog, g, lds, sgprs


def kernel_wmmas(prog):
    """(idx, addr, args) of every WMMA site in the program, in prog order.

    k_post_block itself contains NONE: the WMMA work lives in swin_layer, the
    shared helper both pre/post_block call via s_getpc (same 14 sites the
    pre_block analysis numbered - trace_wmma_pre.py counted them the same way)."""
    return [(idx, a, args) for idx, (a, op, args, _) in enumerate(prog)
            if op == WMMA_OP]


def run_stop(impulse, wmma_n=0, after=False, stop_nth=None, wx=0, wy=0):
    """wmma_n: 0-based index of the kernel's own WMMA to stop at; after=True stops at the
    instruction directly following it (post-execution -> D holds the result);
    stop_nth: fire on the Nth visit of stop_at (per wave) instead of the first."""
    prog, g, lds, sgprs = prepare(impulse)
    sgprs[14], sgprs[15] = wx, wy
    wmmas = kernel_wmmas(prog)
    if not wmmas:
        print('no %s in %s program' % (WMMA_OP, SYM)); sys.exit(1)
    idx0, addr0, args0 = wmmas[wmma_n]
    if after:
        addr0 = prog[idx0 + 1][0]
    snap = {}
    res = E.run_workgroup(prog, g, lds, threads, sgprs, max_steps=16_000_000,
                          stop_at=addr0, snap=snap, stop_nth=stop_nth)
    ranges = re.findall(r'v\[(\d+):(\d+)\]', args0)
    d0, d1 = int(ranges[0][0]), int(ranges[0][1])
    a0, a1 = int(ranges[1][0]), int(ranges[1][1])
    b0, b1 = int(ranges[2][0]), int(ranges[2][1])
    meta = dict(idx=idx0, addr=addr0, args=args0, steps=res['steps'],
                nwmmas=len(wmmas), nwaves=len(snap))
    dump = {wid: s[0].copy() for wid, s in snap.items()}
    return dump, (d0, d1, a0, a1, b0, b1), res['lds'], meta


def regdelta(base, imp, lo, hi):
    """-> dict wid -> sorted list of (reg, lane, half) changed in v[lo..hi]; half 0=lo 1=hi 2=both."""
    out = {}
    for wid in sorted(base):
        if wid not in imp:
            continue
        hits = []
        for r in range(lo, hi + 1):
            for lane in range(32):
                x, y = int(base[wid][r][lane]), int(imp[wid][r][lane])
                if x != y:
                    blo = (x & 0xffff) != (y & 0xffff)
                    bhi = (x >> 16) != (y >> 16)
                    hits.append((r, lane, 2 if (blo and bhi) else (1 if bhi else 0)))
        if hits:
            out[wid] = hits
    return out


def adelta(base, imp, a0, a1):
    return regdelta(base, imp, a0, a1)


def parse_imp(args, start=0, default_val=0x00):
    """positional SLOT OFF [VAL] starting at args[start]; -> (slot, off, val) or None."""
    if len(args) <= start + 1:
        return None
    slot, off = int(args[start]), int(args[start + 1])
    val = int(args[start + 2], 0) if len(args) > start + 2 else default_val
    return (slot, off, val)


def imp_str(imp):
    return 'slot%d[+0x%x]=%#04x' % imp if imp else '(baseline)'


def full_dump(name, dump, regs, lds_arr, meta):
    d0, d1, a0, a1, b0, b1 = regs
    print('=== %s: wmma #%d at prog-idx %d addr 0x%x' % (name, 1, meta['idx'], meta['addr']))
    print('    args: %s | sites=%d steps=%d waves=%d' %
          (meta['args'], meta['nwmmas'], meta['steps'], meta['nwaves']))
    for wid in sorted(dump):
        Vm = dump[wid]
        print('  --- wave %d ---' % wid)
        for label, lo, hi in (('D', d0, d1), ('A', a0, a1)):
            for r in range(lo, hi + 1):
                print('   %s v%-3d u32 %s' % (label, r, ' '.join('%08x' % int(v) for v in Vm[r])))
                print('   %s v%-3d f32 %s' % (label, r, ' '.join('%.6g' % float(v) for v in Vm[r].view(np.float32))))
        for r in range(b0, b1 + 1):
            u = Vm[r].view(np.uint16)                      # 64 halfs: lo0 hi0 lo1 hi1 ...
            print('   B v%-3d u32 %s' % (r, ' '.join('%08x' % int(v) for v in Vm[r])))
            print('   B v%-3d f16 %s' % (r, ' '.join('%.6g' % v for v in f16s(u))))
        for r in range(7, 15):
            print('   W v%-3d f32 %s' % (r, ' '.join('%.6g' % float(v) for v in Vm[r].view(np.float32))))
    nz = np.nonzero(lds_arr)[0]
    if len(nz):
        print('  --- LDS nonzero: %d B, range 0x%x..0x%x ---' % (len(nz), nz[0], nz[-1]))


def main():
    mode = _myargs[0] if _myargs else None

    if mode == 'tapcount':
        # full workgroup run, no stop: dynamic execution count per kernel WMMA site
        prog, g, lds, sgprs = prepare()
        wmmas = kernel_wmmas(prog)
        counts = {}
        res = E.run_workgroup(prog, g, lds, threads, sgprs, max_steps=16_000_000, counts=counts)
        print('run finished: steps=%d' % res.get('steps', -1))
        print('%s has %d %s sites (program total incl. callees: %d)' %
              (SYM, len(wmmas), WMMA_OP,
               sum(1 for _, op, _, _ in prog if op == WMMA_OP)))
        for i, (idx, addr, _) in enumerate(wmmas, 1):
            print('  wmma site #%2d idx %5d addr 0x%x exec=%d' % (i, idx, addr, counts.get(addr, 0)))
        return

    if mode == 'layout':
        # per-workgroup: which byte ranges of which arena slots does the kernel read/write?
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                prog, g, lds, sgprs = prepare()
                sgprs[14], sgprs[15] = wx, wy
                nslot = spec.nslot
                rd = np.zeros((nslot, V.SLOT), bool)
                wr = np.zeros((nslot, V.SLOT), bool)
                orig_r, orig_w = g.read, g.write
                def hook(store, which, rd=rd, wr=wr):
                    def f(addr, x):
                        arr = np.asarray(addr, np.uint64)
                        n = x if which == 'r' else x.shape[1]
                        off = arr.astype(np.int64) - V.ARENA
                        ok = (off >= 0) & (off + n <= nslot * V.SLOT)
                        for o in off[ok]:
                            s, b = divmod(int(o), V.SLOT)
                            (rd if which == 'r' else wr)[s, b:min(b + n, V.SLOT)] = True
                        return store(addr, x)
                    return f
                g.read = hook(orig_r, 'r')
                g.write = hook(orig_w, 'w')
                r = E.run_workgroup(prog, g, lds, threads, sgprs, max_steps=16_000_000)
                g.read, g.write = orig_r, orig_w
                def ext(m):
                    i = np.nonzero(m)[0]
                    return '[+0x%x..+0x%x) n=%d' % (i[0], i[-1] + 1, len(i)) if len(i) else '-'
                bits = []
                for s in range(nslot):
                    if rd[s].any() or wr[s].any():
                        bits.append('s%d R%s W%s' % (s, ext(rd[s]), ext(wr[s])))
                print('wg(%d,%d) steps=%d: %s' % (wx, wy, r['steps'], '; '.join(bits) or 'nothing touched'))
        return

    if mode == 'sweep':
        # sweep [SLOT OFF0 OFF1 STEP [VAL]]: A-delta at site 1 for each byte offset
        slot = int(_myargs[1]) if len(_myargs) > 1 else 0
        o0 = int(_myargs[2], 0) if len(_myargs) > 2 else 0
        o1 = int(_myargs[3], 0) if len(_myargs) > 3 else 8192
        step = int(_myargs[4], 0) if len(_myargs) > 4 else 64
        val = int(_myargs[5], 0) if len(_myargs) > 5 else 0x00
        base, regs, lds0, meta = run_stop(None)
        a0, a1 = regs[2], regs[3]
        print('site 1 args: %s | A=v%d..v%d D=v%d..v%d steps=%d waves=%d | sweep slot%d +0x%x..+0x%x step %d -> %#04x'
              % (meta['args'], a0, a1, regs[0], regs[1], meta['steps'], meta['nwaves'],
                 slot, o0, o1, step, val))
        for off in range(o0, o1, step):
            imp, _, lds_i, _ = run_stop((slot, off, val))
            dl = adelta(base, imp, a0, a1)
            lds_nz = int(np.count_nonzero(lds_i != lds0))
            if not dl:
                print('off +0x%05x: no A delta; lds delta %d' % (off, lds_nz))
                continue
            parts = []
            for wid in sorted(dl):
                hits = dl[wid]
                lanes = sorted(set(l for _, l, _ in hits))
                regs_hit = sorted(set(r for r, _, _ in hits))
                parts.append('w%d v%s lanes%s' % (wid, regs_hit, lanes))
            print('off +0x%05x: %s | lds delta %d' % (off, '; '.join(parts), lds_nz))
        return

    if mode == 'taps':
        # taps SLOT OFF [VAL]: where does this byte land in A at every site?
        imp = parse_imp(_myargs, 1)
        if imp is None:
            print('usage: taps SLOT OFF [VAL]'); return
        base, regs, lds0, meta = run_stop(None)          # regs from call 1 (same ranges all calls)
        print('A=v%d..v%d; impulse %s; per-site A-delta lanes:' %
              (regs[2], regs[3], imp_str(imp)))
        nsites = meta['nwmmas']
        for n in range(1, nsites + 1):
            try:
                b, r2, _, m2 = run_stop(None, wmma_n=n - 1)
                i2, _, _, _ = run_stop(imp, wmma_n=n - 1)
            except Exception as e:
                print('  site %2d: ERROR %s' % (n, e)); continue
            dl = regdelta(b, i2, r2[2], r2[3])
            if not dl:
                print('  site %2d: idx %5d addr 0x%x: no A delta' % (n, m2['idx'], m2['addr']))
                continue
            parts = ['w%d lanes%s' % (wid, sorted(set(l for _, l, _ in h)))
                     for wid, h in sorted(dl.items())]
            print('  site %2d: idx %5d addr 0x%x: %s' %
                  (n, m2['idx'], m2['addr'], '; '.join(parts)))
        return

    if mode == 'loops':
        # loops SITE SLOT OFF [VAL]: per-iteration A/B/D deltas before/after that site
        site = int(_myargs[1]) - 1
        imp = parse_imp(_myargs, 2)
        if imp is None:
            print('usage: loops SITE SLOT OFF [VAL]'); return
        for nth in range(1, 17):
            b, r2, _, m2 = run_stop(None, wmma_n=site, stop_nth=nth)
            if not b:
                print('site %d: no wave reached iteration %d' % (site + 1, nth)); break
            i2, _, _, _ = run_stop(imp, wmma_n=site, stop_nth=nth)
            segs = []
            for label, lo, hi in (('A', r2[2], r2[3]), ('B', r2[4], r2[5])):
                dl = regdelta(b, i2, lo, hi)
                if dl:
                    segs.append(label + ' ' + '; '.join(
                        'w%d lanes%s' % (wid, sorted(set(l for _, l, _ in h)))
                        for wid, h in sorted(dl.items())))
            bd, rd, _, md = run_stop(None, wmma_n=site, after=True, stop_nth=nth)
            idl, _, _, _ = run_stop(imp, wmma_n=site, after=True, stop_nth=nth)
            if bd:
                dd = regdelta(bd, idl, rd[0], rd[1])
                if dd:
                    segs.append('D ' + '; '.join(
                        'w%d regs%s lanes%s' % (wid,
                                                sorted(set(r for r, _, _ in h)),
                                                sorted(set(l for _, l, _ in h)))
                        for wid, h in sorted(dd.items())))
            print('iter %2d: idx %5d addr 0x%x steps=%d: %s' %
                  (nth, m2['idx'], m2['addr'], m2['steps'],
                   ' | '.join(segs) if segs else 'NO DELTA anywhere'))
        return

    if mode == 'mat':
        # mat SITE [NTH [WAVE [SLOT OFF [VAL]]]]: A/B matrices and D delta as values
        site = int(_myargs[1]) - 1 if len(_myargs) > 1 else 0
        nth = int(_myargs[2]) if len(_myargs) > 2 else 1
        want_w = int(_myargs[3]) if len(_myargs) > 3 else 0
        imp = parse_imp(_myargs, 4)
        b, r2, l0, m2 = run_stop(None, wmma_n=site, stop_nth=nth if nth > 1 else None)
        i2, _, l1, _ = run_stop(imp, wmma_n=site, stop_nth=nth if nth > 1 else None)
        if want_w not in b:
            print('wave %d not in snap (%s)' % (want_w, sorted(b))); return
        d0, d1, a0, a1, b0, b1 = r2
        print('== impulse %s ==' % imp_str(imp))

        def f16mat(Vm, lo, hi):
            # rows = (reg - lo) * 2 + half, cols = lane 0..15
            M = np.zeros((2 * (hi - lo + 1), 16), np.float64)
            for ri, r in enumerate(range(lo, hi + 1)):
                u = Vm[r]                                   # (32,) uint32
                M[2 * ri] = f16s(np.uint16(u & 0xffff))[:16]
                M[2 * ri + 1] = f16s(np.uint16(u >> 16))[:16]
            return M

        for label, lo, hi in (('A', a0, a1), ('B', b0, b1)):
            Mb = f16mat(b[want_w], lo, hi)
            Mi = f16mat(i2[want_w], lo, hi)
            print('== %s (wave %d) baseline; cols=lane0..15, rows=(reg-half) ==' % (label, want_w))
            print(np.array2string(Mb, precision=4, suppress_small=True, max_line_width=200))
            Dm = Mi - Mb
            if np.any(Dm):
                print('== %s impulse delta ==' % label)
                print(np.array2string(Dm, precision=4, suppress_small=True, max_line_width=200))
            else:
                print('== %s: no delta ==' % label)
        # D after the call for this wave
        bd, rd, _, _ = run_stop(None, wmma_n=site, after=True, stop_nth=nth if nth > 1 else None)
        idl, _, _, _ = run_stop(imp, wmma_n=site, after=True, stop_nth=nth if nth > 1 else None)
        if want_w in bd:
            Db = bd[want_w]; Di = idl[want_w]
            for r in range(rd[0], rd[1] + 1):
                vb = Db[r][want_w].view(np.float32); vi = Di[r][want_w].view(np.float32)
                if not np.array_equal(vb, vi):
                    print('== D v%d baseline: %s' % (r, np.array2string(vb, precision=5, suppress_small=True, max_line_width=200)))
                    print('== D v%d delta   : %s' % (r, np.array2string(vi - vb, precision=5, suppress_small=True, max_line_width=200)))
        return

    imp = parse_imp(_myargs, 0)
    base, regs, lds0, meta = run_stop(None)
    full_dump('baseline', base, regs, lds0, meta)
    if imp:
        it, _, lds_i, _ = run_stop(imp)
        full_dump('impulse %s' % imp_str(imp), it, regs, lds_i, meta)
        a0, a1 = regs[2], regs[3]
        print('=== A delta ===')
        for wid, hits in adelta(base, it, a0, a1).items():
            print('  wave%d: %s' % (wid, hits))
        print('LDS delta bytes: %d' % int(np.count_nonzero(lds_i != lds0)))
    else:
        nz = np.nonzero(lds0)[0]
        if len(nz):
            rows = sorted(set(int(x) // 16 * 16 for x in nz))[:48]
            for off in rows:
                print('  lds+0x%04x %s' % (off, lds0[off:off + 16].tobytes().hex()))


main()