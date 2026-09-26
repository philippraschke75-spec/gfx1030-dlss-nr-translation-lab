"""Stop the k_pre_block emulation at its FIRST v_wmma_f32_16x16x16_f16 and decode its
im2col mapping empirically: run a sweep of single-pixel impulses and record, per impulse,
which (wave, register, lane, half) of the A operand changes vs the zero baseline.

usage:
  trace_wmma_pre.py                 baseline dump of A/B/D (all lanes) + LDS
  trace_wmma_pre.py Y X             one impulse: full dump + delta vs baseline
  trace_wmma_pre.py sweep           impulse line sweep: (4,0)..(4,7) and (0,4)..(7,4)
                                    -> compact per-impulse A-delta signatures

Pure emulator instrumentation; no kernel/build files are touched.
"""
import sys, re, struct, os
from pathlib import Path
import numpy as np

_myargs = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, kernelspec as K, difftest_var as D

SYM = '_Z21k_pre_block_1h_32_fp89PreParams'
SWIN = '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'
seed = 1
H = int(os.environ.get('H', '16'))
W = int(os.environ.get('W', '16'))
grid = ((W + 7) // 8, (H + 7) // 8)


def f16s(u16):
    """uint16 array -> float values via bit reinterpretation."""
    return np.asarray(u16, np.uint16).view(np.float16).astype(np.float64)


def kernarg():
    explicit, ksize, lds, hid = K.kernel_meta(SYM)
    ka = bytearray(ksize)
    S = lambda k: V.ARENA + k * V.SLOT
    struct.pack_into('<QQQ', ka, 0x00, S(0), S(1), S(2))
    struct.pack_into('<ii', ka, 0x18, H, W)
    struct.pack_into('<fi', ka, 0x20, float(os.environ.get('PRE_F20', '0.0625')), 0)
    struct.pack_into('<ff', ka, 0x28, 0.0, 0.0)
    struct.pack_into('<i', ka, 0x30, 0)
    p40 = os.environ.get('PRE_40', 'null')
    struct.pack_into('<QQ', ka, 0x38, S(3), 0 if p40 == 'null' else S(4))
    struct.pack_into('<ff', ka, 0x48, *[float(x) for x in os.environ.get('PRE_F48', '1.0,1.0').split(',')])
    for name, val in (('block_count_x', grid[0]), ('block_count_y', grid[1]), ('block_count_z', 1),
                      ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1), ('grid_dims', 2)):
        o, sz = hid[name]
        struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
    return ka


def prepare(impulse, wx=0, wy=0):
    ka = kernarg()
    p = E.load_program(R.DIS, {SYM, SWIN})
    entry = E.load_program(R.DIS, {SYM})[0][0]
    k = min(idx for idx, x in enumerate(p) if x[0] == entry)
    prog = p[k:] + p[:k]
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    a = g.regions[1].arr
    a[:H * W * 12] = 0
    if impulse:
        off = (impulse[0] * W + impulse[1]) * 12
        amp = float(os.environ.get('IMP_AMP', '1'))   # linearity probe: 0.5 must halve the delta
        a[off:off + 12] = np.full(3, amp, np.float32).view(np.uint8)
    lds = D.group_size(SYM)
    DP = 0x7100_0000_0000
    pkt = bytearray(64)
    struct.pack_into('<HHHHHH', pkt, 0, 0, 3, 256, 1, 1, 0)
    struct.pack_into('<III', pkt, 12, grid[0] * 256, grid[1], 1)
    struct.pack_into('<II', pkt, 24, 64, lds)
    g.add('dispatch', DP, np.frombuffer(bytes(pkt), np.uint8).copy())
    sgprs = {0: DP & 0xffffffff, 1: DP >> 32, 2: KA & 0xffffffff, 3: KA >> 32, 14: wx, 15: wy}
    return prog, g, lds, sgprs


def run_stop(impulse, wx=0, wy=0, wmma_n=0, after=False, stop_nth=None):
    """wmma_n: 0-based index of the v_wmma to stop at; after=True stops at the
    instruction directly following it (post-execution -> D holds the result);
    stop_nth: fire on the Nth visit of stop_at (per wave) instead of the first."""
    prog, g, lds, sgprs = prepare(impulse, wx, wy)
    wmmas = [(idx, a, args) for idx, (a, op, args, _) in enumerate(prog)
             if op == 'v_wmma_f32_16x16x16_f16']
    if not wmmas:
        print('no v_wmma in program'); sys.exit(1)
    idx0, addr0, args0 = wmmas[wmma_n]
    if after:
        addr0 = prog[idx0 + 1][0]
    snap = {}
    res = E.run_workgroup(prog, g, lds, 256, sgprs, max_steps=16_000_000,
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

    if mode == 'mat':
        # mat [SITE [ITER] [WAVE]] : print A (16 m x 16 k) and B matrices as values,
        # baseline and impulse delta, for readable decoding of features/weights.
        site = int(_myargs[1]) - 1 if len(_myargs) > 1 else 0
        nth = int(_myargs[2]) if len(_myargs) > 2 else 1
        want_w = int(_myargs[3]) if len(_myargs) > 3 else 6
        imp = (int(_myargs[4]), int(_myargs[5])) if len(_myargs) > 5 else (4, 4)
        b, r2, l0, m2 = run_stop(None, wmma_n=site, stop_nth=nth if nth > 1 else None)
        i2, _, l1, _ = run_stop(imp, wmma_n=site, stop_nth=nth if nth > 1 else None)
        if want_w not in b:
            print('wave %d not in snap (%s)' % (want_w, sorted(b))); return
        d0, d1, a0, a1, b0, b1 = r2

        def f16mat(Vm, lo, hi, w):
            # rows = (reg - lo) * 2 + half, cols = lane 0..15 (w = lane for cross-check)
            M = np.zeros((2 * (hi - lo + 1), 16), np.float64)
            for ri, r in enumerate(range(lo, hi + 1)):
                u = Vm[r]                                   # (32,) uint32
                M[2 * ri] = f16s(np.uint16(u & 0xffff))[:16]
                M[2 * ri + 1] = f16s(np.uint16(u >> 16))[:16]
            return M

        for label, lo, hi in (('A', a0, a1), ('B', b0, b1)):
            Mb = f16mat(b[want_w], lo, hi, want_w)
            Mi = f16mat(i2[want_w], lo, hi, want_w)
            print('== %s (wave %d) baseline; cols=lane0..15, rows=(reg-half) ==' % (label, want_w))
            print(np.array2string(Mb, precision=4, suppress_small=True, max_line_width=200))
            Dm = Mi - Mb
            if np.any(Dm):
                print('== %s impulse %s delta ==' % (label, imp))
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

    if mode == 'loops':
        # loops SITE Y X : per-iteration A/B-delta before and D-delta after the
        # given wmma site for impulse (Y,X)
        site = int(_myargs[1]) - 1
        imp = (int(_myargs[2]), int(_myargs[3]))
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

    if mode == 'tapcount':
        # full workgroup run, no stop: dynamic execution count per v_wmma site
        prog, g, lds, sgprs = prepare(None)
        wmmas = [(idx, a) for idx, (a, op, args, _) in enumerate(prog)
                 if op == 'v_wmma_f32_16x16x16_f16']
        counts = {}
        res = E.run_workgroup(prog, g, lds, 256, sgprs, max_steps=16_000_000, counts=counts)
        print('run finished: steps=%d' % res.get('steps', -1))
        for i, (idx, a) in enumerate(wmmas, 1):
            print('  wmma site #%2d idx %5d addr 0x%x exec=%d' % (i, idx, a, counts.get(a, 0)))
        return

    if mode == 'taps':
        # for each wmma call N: where does impulse (y,x) land in A (lane shift = tap offset)?
        imp = (int(_myargs[1]), int(_myargs[2]))
        base, regs, lds0, meta = run_stop(None)          # regs from call 1 (same ranges all calls)
        a0, a1 = regs[2], regs[3]
        print('A=v%d..v%d; impulse %s; per-call A-delta lanes (baseline lane of this impulse = call1):'
              % (a0, a1, imp))
        for n in range(1, 15):
            try:
                b, r2, _, m2 = run_stop(None, wmma_n=n - 1)
                i2, _, _, _ = run_stop(imp, wmma_n=n - 1)
            except Exception as e:
                print('  call %2d: ERROR %s' % (n, e)); continue
            dl = regdelta(b, i2, r2[2], r2[3])
            if not dl:
                print('  call %2d: idx %5d addr 0x%x: no A delta' % (n, m2['idx'], m2['addr']))
                continue
            parts = ['w%d lanes%s' % (wid, sorted(set(l for _, l, _ in h)))
                     for wid, h in sorted(dl.items())]
            print('  call %2d: idx %5d addr 0x%x: %s' %
                  (n, m2['idx'], m2['addr'], '; '.join(parts)))
        return

    if mode == 'probe':
        # probe N A|D : A-operand before the (N+1)-th wmma, or D after it.
        n = int(_myargs[1]) - 1
        which = _myargs[2] if len(_myargs) > 2 else 'A'
        after = which == 'D'
        base, regs, lds0, meta = run_stop(None, wmma_n=n, after=after)
        lo, hi = (regs[0], regs[1]) if after else (regs[2], regs[3])
        print('wmma #%d %s: prog-idx %d addr 0x%x args %s | scan v%d..v%d | steps=%d waves=%d' %
              (n + 1, 'D-after' if after else 'A-before', meta['idx'], meta['addr'],
               meta['args'], lo, hi, meta['steps'], meta['nwaves']))
        positions = [(4, x) for x in range(9)] + [(y, 4) for y in range(9)]
        for pos in positions:
            imp, _, lds_i, _ = run_stop(pos, wmma_n=n, after=after)
            dl = regdelta(base, imp, lo, hi)
            if not dl:
                print('imp (%2d,%2d): no delta' % pos)
                continue
            parts = []
            for wid in sorted(dl):
                hits = dl[wid]
                lanes = sorted(set(l for _, l, _ in hits))
                parts.append('w%d regs%d-%d lanes%s' %
                             (wid, hits[0][0], hits[-1][0], lanes))
            print('imp (%2d,%2d): %s' % (pos[0], pos[1], '; '.join(parts)))
        return

    if mode == 'sweep':
        base, regs, lds0, meta = run_stop(None)
        a0, a1 = regs[2], regs[3]
        print('baseline: first wmma prog-idx %d addr 0x%x args: %s' %
              (meta['idx'], meta['addr'], meta['args']))
        print('A=v%d..v%d  D=v%d..v%d  B=v%d..v%d  steps=%d waves=%d' %
              (a0, a1, regs[0], regs[1], regs[4], regs[5], meta['steps'], meta['nwaves']))
        positions = [(y, x) for y in range(9) for x in range(9) if (y, x) != (8, 8)]
        for pos in positions:
            imp, _, lds_i, _ = run_stop(pos)
            dl = adelta(base, imp, a0, a1)
            lds_nz = int(np.count_nonzero(lds_i != lds0))
            if not dl:
                print('imp (%2d,%2d): no A delta; lds delta %d' % (pos[0], pos[1], lds_nz))
                continue
            parts = []
            for wid in sorted(dl):
                hits = dl[wid]
                lanes = sorted(set(l for _, l, _ in hits))
                parts.append('w%d lanes%s' % (wid, lanes))
            print('imp (%2d,%2d): %s | lds delta %d' % (pos[0], pos[1], '; '.join(parts), lds_nz))
            if pos == (4, 4):          # value-level detail for one representative impulse
                for wid in sorted(dl):
                    for (r, lane, half) in dl[wid][:6]:
                        x0, x1 = int(base[wid][r][lane]), int(imp[wid][r][lane])
                        f = lambda u: '(lo %s hi %s)' % (f16s(np.array([u & 0xffff], np.uint16))[0],
                                                         f16s(np.array([u >> 16], np.uint16))[0])
                        print('     w%d v%d lane%d: %08x->%08x %s -> %s' %
                              (wid, r, lane, x0, x1, f(x0), f(x1)))
                dlo = np.nonzero(lds_i != lds0)[0]
                blocks = sorted(set(int(o) // 64 * 64 for o in dlo))
                for b in blocks[:6]:
                    m = slice(b, b + 64)
                    ch = np.nonzero(lds_i[m] != lds0[m])[0]
                    ex = [(int(b + c), int(lds0[b + c]), int(lds_i[b + c])) for c in ch[:6]]
                    print('     lds+%d: %s' % (b, ex))
        return

    impulse = (int(_myargs[0]), int(_myargs[1])) if mode and mode.lstrip('-').isdigit() else None
    base, regs, lds0, meta = run_stop(None)
    full_dump('baseline', base, regs, lds0, meta)
    if impulse:
        imp, _, lds_i, _ = run_stop(impulse)
        full_dump('impulse %s' % (impulse,), imp, regs, lds_i, meta)
        a0, a1 = regs[2], regs[3]
        print('=== A delta ===')
        for wid, hits in adelta(base, imp, a0, a1).items():
            print('  wave%d: %s' % (wid, hits))
        print('LDS delta bytes: %d' % int(np.count_nonzero(lds_i != lds0)))
    else:
        nz = np.nonzero(lds0)[0]
        if len(nz):
            rows = sorted(set(int(x) // 16 * 16 for x in nz))[:48]
            for off in rows:
                print('  lds+0x%04x %s' % (off, lds0[off:off + 16].tobytes().hex()))


main()
