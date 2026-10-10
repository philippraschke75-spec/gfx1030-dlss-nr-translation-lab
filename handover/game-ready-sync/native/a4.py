"""Brute-force the f32 summation tree of y = x*rs + 4 contract partials against the qkv A operand (token tile 0)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ana32 as A, itertools
T = np.load('E:/tmp32/trace32_s3.npz'); w = A.load(3, 5, 5)
sites = sorted(set(T['addr'].tolist()))
idx = {(sites.index(T['addr'][k]) + 1, int(T['wid'][k]), int(T['visit'][k])): k for k in range(len(T['addr']))}
def F(s, wid, v, what): return T[what][idx[(s, wid, v)]].astype(np.float32)
C = 32; rec = w['rec']; perm = np.array([A.pi32(c) for c in range(32)]); inv = np.argsort(perm)
rm = np.array([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15])
xm = A.LUT[w['x']]; f16 = lambda a: a.astype(np.float16).astype(np.float32)
rs = np.frombuffer(bytes(A.AR[rec + 8 * C * C:rec + 8 * C * C + 128]), np.float16).astype(np.float32); rso = 8 if not rs[:8].any() else 0
def qa(tt):   # Yq columns seen by the qkv matmul for token tile tt: [16 tok][6 tiles * 32 k]
    return np.concatenate([np.concatenate([F(7, i % 8, i // 8 + 1, 'A'), F(8, i % 8, i // 8 + 1, 'A')], 1) for i in range(6 * tt, 6 * tt + 6)], 1)
COLS = [qa(tt) for tt in range(4)]
def trees(leaves):   # all binary trees over an ordered leaf list
    if len(leaves) == 1: yield leaves[0]; return
    for i in range(1, len(leaves)):
        for l in trees(leaves[:i]):
            for r in trees(leaves[i:]): yield (l, r)
def ev(t, env):
    if isinstance(t, str): return env[t]
    return (ev(t[0], env) + ev(t[1], env)).astype(np.float32)


def forms(xx, r, P, order):
    out = {}
    xr64 = xx.astype(np.float64) * r
    for pt in ('id', 'f16', 'q8'):
        Pl = [P[f'P{v}'] for v in order]
        Pl = [q if pt == 'id' else (f16(q) if pt == 'f16' else A.q8(f16(q))) for q in Pl]
        for sp in ('f32', 'f16'):
            s_ = Pl[0].copy()
            for q in Pl[1:]: s_ = (s_ + q).astype(np.float32) if sp == 'f32' else f16(s_ + q)
            out[f'{pt}/{sp}: f16(fma32)'] = f16((xr64 + s_).astype(np.float32))
            out[f'{pt}/{sp}: fma16'] = f16(xr64 + s_)
            out[f'{pt}/{sp}: f16(x*rs)+s'] = f16(f16(xx * r) + s_)
            out[f'{pt}/{sp}: f32 mul add'] = ((xx * r).astype(np.float32) + s_).astype(np.float32)
            out[f'{pt}/{sp}: f16 of f32 mul add'] = f16(((xx * r).astype(np.float32) + s_).astype(np.float32))
            out[f'{pt}/{sp}: f32 fma'] = (xr64 + s_).astype(np.float32)
    return out
res = {}
for nt in range(2):
    r = rs[rso + nt * 16:rso + nt * 16 + 16][rm]
    for order in itertools.permutations(range(1, 5)):
        for tt in range(4):
            P = {f'P{v}': F(6, 2 * tt + nt, v, 'D') for v in range(1, 5)}
            xx = xm[tt * 16:tt * 16 + 16][:, inv][:, nt * 16:nt * 16 + 16][:, rm]
            cols = COLS[tt]
            for name, y in forms(xx, r, P, order).items():
                yq = A.q8(y)
                n = sum(any(np.array_equal(yq[:, c], cols[:, j]) for j in range(cols.shape[1])) for c in range(16))
                res[(name, order)] = res.get((name, order), 0) + n
best = sorted(res.items(), key=lambda kv: -kv[1])[:12]
for k, v in best: print(v, '/128', k)
