"""Systematic amplitude-sweep + closed-form fit for the pre_block Site 3 'A' operand
nonlinearity found manually in handover/pre-block-collapse-gates.md (2026-09-27).

Manual point-by-point sweeps found: positive amp exactly linear (slope 1) out to 8,
negative amp linear-but-shifted (-0.5) up to -4, then saturating at -8; and at small
amplitudes (<=0.5) a SECOND register/channel becomes sensitive that is silent at
larger amplitudes - the single-channel assumption breaks down near the origin.
This script automates the sweep (finer grid, tracks ALL changed (reg,lane) not just
one) and fits candidate closed forms (identity, GELU, softplus, tanh) per channel via
least squares, instead of manual grep extraction.

Pure emulator instrumentation; no kernel/build files touched.
"""
import sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_real_argv, _real_stdout = sys.argv, sys.stdout
sys.argv = [_real_argv[0]]          # trace_wmma_pre.py runs main() unconditionally at import time
import io
sys.stdout = io.StringIO()          # swallow its default (mode=None) full dump
import trace_wmma_pre as T
sys.argv, sys.stdout = _real_argv, _real_stdout

AMPS = [-8, -6, -5, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8]


def sweep(site=2, wave=6, pixel=(4, 4)):
    """-> dict (reg, lane) -> list of (amp, delta) for every A-cell that ever changes."""
    base, ranges, _, _ = T.run_stop(None, wmma_n=site)
    if wave not in base:
        raise SystemExit(f'wave {wave} not present')
    d0, d1, a0, a1, b0, b1 = ranges
    base_u = base[wave]
    hits = {}
    for amp in AMPS:
        os.environ['IMP_AMP'] = str(amp)
        imp, _, _, _ = T.run_stop(pixel, wmma_n=site)
        iu = imp[wave]
        for r in range(a0, a1 + 1):
            for lane in range(32):
                bx, ix = int(base_u[r][lane]), int(iu[r][lane])
                if bx == ix:
                    continue
                blo = T.f16s([bx & 0xffff])[0]
                bhi = T.f16s([(bx >> 16) & 0xffff])[0]
                ilo = T.f16s([ix & 0xffff])[0]
                ihi = T.f16s([(ix >> 16) & 0xffff])[0]
                if blo != ilo:
                    hits.setdefault((r, lane, 'lo'), []).append((amp, float(ilo - blo)))
                if bhi != ihi:
                    hits.setdefault((r, lane, 'hi'), []).append((amp, float(ihi - bhi)))
    del os.environ['IMP_AMP']
    return hits


def fit_candidates(amps, deltas):
    """Least-squares fit of delta(amp) against a few closed forms through likely
    baseline+slope combos. Returns (name, params, residual) sorted best first."""
    amps = np.asarray(amps, float)
    deltas = np.asarray(deltas, float)
    results = []

    # identity: delta = k*amp
    k = np.sum(amps * deltas) / np.sum(amps * amps)
    resid = np.sum((k * amps - deltas) ** 2)
    results.append(('linear(k=%.4f)' % k, resid))

    # shifted linear with a floor: delta = amp for amp>=0, amp+c for amp<0 (const shift)
    pos = amps >= 0
    neg = ~pos
    if neg.any():
        c = np.mean(deltas[neg] - amps[neg])
        pred = np.where(pos, amps, amps + c)
        resid = np.sum((pred - deltas) ** 2)
        results.append(('kinked(c=%.4f)' % c, resid))

    # softplus-derivative-like saturating form: delta = a*tanh(amp/b)*b roughly;
    # fit b by grid search, a by linear regression given b
    best = None
    for b in np.linspace(0.5, 20, 80):
        x = np.tanh(amps / b) * b
        denom = np.sum(x * x)
        if denom < 1e-9:
            continue
        a = np.sum(x * deltas) / denom
        resid = np.sum((a * x - deltas) ** 2)
        if best is None or resid < best[1]:
            best = (('tanh_sat(a=%.4f,b=%.4f)' % (a, b)), resid)
    if best:
        results.append(best)

    results.sort(key=lambda t: t[1])
    return results


def main():
    site = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 2
    hits = sweep(site=site)
    print(f'{len(hits)} (reg,lane,half) cells changed across {len(AMPS)} amplitudes')
    for key, pts in sorted(hits.items()):
        if len(pts) < 4:
            continue
        amps = [p[0] for p in pts]
        deltas = [p[1] for p in pts]
        fits = fit_candidates(amps, deltas)
        print(f'--- reg={key[0]} lane={key[1]} half={key[2]} ({len(pts)} pts) ---')
        for name, resid in fits[:3]:
            print(f'    {name}: resid={resid:.5f}')


if __name__ == '__main__':
    main()
