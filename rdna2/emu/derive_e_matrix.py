"""Derive the full 16x3 E matrix + 16 baselines for pre_block sites 1/2's A
operand (the im2col input to the WMMA), using the affine model validated in
handover/pre-block-collapse-gates.md Update 8: A[feature] = q_e4m3(baseline +
kR*R + kG*G + kB*B), independently per feature (register/half), at a fixed
pixel/lane.

Usage: derive_e_matrix.py SITE WAVE   (SITE: 1-based, 1 or 2)
Writes handover/pre_block_site{SITE}_E_matrix.json

Pure emulator instrumentation; no kernel/build files touched.
"""
import sys, os, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_real_argv, _real_stdout = sys.argv, sys.stdout
sys.argv = [_real_argv[0]]
import io
sys.stdout = io.StringIO()
import trace_wmma_pre as T
sys.argv, sys.stdout = _real_argv, _real_stdout

ROOT = Path(__file__).resolve().parent.parent.parent


def cell_value(u32, half):
    return float(T.f16s([(u32 >> 16) & 0xffff if half == 'hi' else u32 & 0xffff])[0])


def main():
    site = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
    wave = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pixel = (4, 4)

    base, ranges, _, _ = T.run_stop(None, wmma_n=site)
    if wave not in base:
        raise SystemExit(f'wave {wave} not present')
    d0, d1, a0, a1, b0, b1 = ranges
    bu = base[wave]

    def probe(rgb):
        os.environ['IMP_R'], os.environ['IMP_G'], os.environ['IMP_B'] = (str(x) for x in rgb)
        imp, _, _, _ = T.run_stop(pixel, wmma_n=site)
        return imp[wave]

    AMP = float(os.environ.get('DERIVE_AMP', '2.0'))
    resp = {ax: probe(tuple(AMP if i == ax_i else 0 for i in range(3)))
            for ax_i, ax in enumerate(('R', 'G', 'B'))}
    for k in ('IMP_R', 'IMP_G', 'IMP_B'):
        os.environ.pop(k, None)

    entries = []
    for reg in range(a0, a1 + 1):
        for half in ('lo', 'hi'):
            b_val = cell_value(int(bu[reg][0]), half)
            slopes = {}
            for ax in ('R', 'G', 'B'):
                v = cell_value(int(resp[ax][reg][0]), half)
                slopes[ax] = (v - b_val) / AMP
            entries.append({'reg': reg, 'half': half, 'baseline': b_val,
                             'kR': slopes['R'], 'kG': slopes['G'], 'kB': slopes['B']})

    # out-of-sample validation on a few random RGB triples, all channels at once
    import random
    random.seed(7)
    checks = []
    for _ in range(4):
        rgb = tuple(round(random.uniform(-3, 3), 3) for _ in range(3))
        os.environ['IMP_R'], os.environ['IMP_G'], os.environ['IMP_B'] = (str(x) for x in rgb)
        imp, _, _, _ = T.run_stop(pixel, wmma_n=site)
        for k in ('IMP_R', 'IMP_G', 'IMP_B'):
            os.environ.pop(k, None)
        iu = imp[wave]
        errs = []
        for e in entries:
            predicted = e['baseline'] + e['kR'] * rgb[0] + e['kG'] * rgb[1] + e['kB'] * rgb[2]
            actual = cell_value(int(iu[e['reg']][0]), e['half'])
            errs.append(abs(predicted - actual))
        checks.append({'rgb': rgb, 'max_err': max(errs), 'mean_err': float(np.mean(errs))})

    out = {'site': site + 1, 'wave': wave, 'pixel': list(pixel), 'amp_used': AMP,
           'a_reg_range': [a0, a1], 'entries': entries, 'validation': checks}
    outpath = ROOT / 'handover' / f'pre_block_site{site+1}_E_matrix.json'
    outpath.write_text(json.dumps(out, indent=2))
    print(f'wrote {outpath} ({len(entries)} channels)')
    print('validation max errors:', [c['max_err'] for c in checks])


if __name__ == '__main__':
    main()
