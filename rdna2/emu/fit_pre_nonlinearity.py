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

# Independent single-channel axes: probe R, G, B separately (not the R=G=B
# diagonal) so a genuinely 3-input linear function can be told apart from a
# real nonlinearity instead of being mis-fit as "no clean 1D model exists".
AXES = {'R': ('IMP_R', 'IMP_G', 'IMP_B'), 'G': ('IMP_G', 'IMP_R', 'IMP_B'), 'B': ('IMP_B', 'IMP_R', 'IMP_G')}


def sweep_axis(site, wave, pixel, axis_env, other_envs):
    """-> dict (reg,lane,half) -> list of (amp, delta) for one isolated colour channel."""
    for e in other_envs:
        os.environ[e] = '0'
    base, ranges, _, _ = T.run_stop(None, wmma_n=site)
    if wave not in base:
        raise SystemExit(f'wave {wave} not present')
    d0, d1, a0, a1, b0, b1 = ranges
    base_u = base[wave]
    hits = {}
    for amp in AMPS:
        os.environ[axis_env] = str(amp)
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
    for e in list(other_envs) + [axis_env]:
        os.environ.pop(e, None)
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
    wave = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pixel = (4, 4)
    per_axis = {}
    for name, (axis_env, o1, o2) in AXES.items():
        print(f'=== axis {name} ({axis_env}, others held at 0) ===', file=sys.stderr)
        per_axis[name] = sweep_axis(site, wave, pixel, axis_env, [o1, o2])

    all_keys = set().union(*(h.keys() for h in per_axis.values()))
    print(f'{len(all_keys)} (reg,lane,half) cells changed across any axis, {len(AMPS)} amps each')
    for key in sorted(all_keys):
        row = []
        for name in ('R', 'G', 'B'):
            pts = per_axis[name].get(key, [])
            if len(pts) < 4:
                row.append(f'{name}: (no data)')
                continue
            amps = [p[0] for p in pts]
            deltas = [p[1] for p in pts]
            k = float(np.sum(np.asarray(amps) * np.asarray(deltas)) / np.sum(np.asarray(amps) ** 2))
            resid = float(np.sum((k * np.asarray(amps) - np.asarray(deltas)) ** 2))
            # relative residual: bad linear fit if resid is comparable to the deltas' own spread
            spread = float(np.sum(np.asarray(deltas) ** 2)) + 1e-9
            quality = 'GOOD' if resid / spread < 0.05 else ('OK' if resid / spread < 0.2 else 'POOR')
            row.append(f'{name}: k={k:+.4f} resid/spread={resid/spread:.3f} [{quality}] n={len(pts)}')
        print(f'--- reg={key[0]} lane={key[1]} half={key[2]} ---')
        for r in row:
            print(f'    {r}')


if __name__ == '__main__':
    main()
