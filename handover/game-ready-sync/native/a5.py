"""Verify: y accumulates in f16: y0 = f16(x*rs); y = f16(D_v + f32(y)) for the 4 contract partials; Yq = q8(y)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ana32 as A
T = np.load('E:/tmp32/trace32_s3.npz'); w = A.load(3, 5, 5)
sites = sorted(set(T['addr'].tolist()))
idx = {(sites.index(T['addr'][k]) + 1, int(T['wid'][k]), int(T['visit'][k])): k for k in range(len(T['addr']))}
def F(s, wid, v, what): return T[what][idx[(s, wid, v)]].astype(np.float32)
C = 32; rec = w['rec']; perm = np.array([A.pi32(c) for c in range(32)]); inv = np.argsort(perm)
rm = np.array([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15])
xm = A.LUT[w['x']]; f16 = lambda a: a.astype(np.float16).astype(np.float32)
rs = np.frombuffer(bytes(A.AR[rec + 8 * C * C:rec + 8 * C * C + 128]), np.float16).astype(np.float32); rso = 8 if not rs[:8].any() else 0
def qa(tt):
    return np.concatenate([np.concatenate([F(7, i % 8, i // 8 + 1, 'A'), F(8, i % 8, i // 8 + 1, 'A')], 1) for i in range(6 * tt, 6 * tt + 6)], 1)
tot = 0
for tt in range(4):
    cols = qa(tt)
    for nt in range(2):
        r = rs[rso + nt * 16:rso + nt * 16 + 16][rm]
        xx = xm[tt * 16:tt * 16 + 16][:, inv][:, nt * 16:nt * 16 + 16][:, rm]
        y = f16(xx * r)
        for v in range(1, 5): y = f16((F(6, 2 * tt + nt, v, 'D') + y).astype(np.float32))
        yq = A.q8(y)
        n = sum(any(np.array_equal(yq[:, c], cols[:, j]) for j in range(cols.shape[1])) for c in range(16)); tot += n
        print('tt', tt, 'nt', nt, n, '/16')
print('total', tot, '/128')
