"""Structure of the 14 WMMA sites: C chains, A/B reuse across sites/waves."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ana32 as A
T = np.load('E:/tmp32/trace32_s3.npz')
sites = sorted(set(T['addr'].tolist()))
n = len(T['addr'])
idx = {(sites.index(T['addr'][k]) + 1, int(T['wid'][k]), int(T['visit'][k])): k for k in range(n)}
order0 = [(sites.index(T['addr'][k]) + 1, int(T['visit'][k])) for k in range(n) if T['wid'][k] == 0]
Ds = {}
for pos, (s, v) in enumerate(order0):
    k = idx[(s, 0, v)]
    C, D, Aa, Bb = T['C'][k], T['D'][k], T['A'][k], T['B'][k]
    src = 'zero' if not C.any() else next((f'D{p}' for p, q in Ds.items() if np.array_equal(q, C)), 'other')
    # A/B reuse among earlier visits (wave 0)
    ar = [p for p in range(pos) if np.array_equal(T['A'][idx[(order0[p][0], 0, order0[p][1])]], Aa)]
    br = [p for p in range(pos) if np.array_equal(T['B'][idx[(order0[p][0], 0, order0[p][1])]], Bb)]
    # same A/B in other waves at same site/visit?
    aw = [w for w in range(1, 8) if np.array_equal(T['A'][idx[(s, w, v)]], Aa)]
    bw = [w for w in range(1, 8) if np.array_equal(T['B'][idx[(s, w, v)]], Bb)]
    Ds[(s, v)] = D
    print(f'pos {pos:2d} site {s:2d} visit {v}  C={src:8s} A_reuse={ar} B_reuse={br}  A_same_waves={aw} B_same_waves={bw}  |A|max={np.abs(Aa).max():.3g} |B|max={np.abs(Bb).max():.3g}')
