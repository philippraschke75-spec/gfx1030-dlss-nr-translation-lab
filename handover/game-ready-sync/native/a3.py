"""Stage checks of the trace against the record/x: expand, hidden, contract partials, y / Yq."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, ana32 as A
T = np.load('E:/tmp32/trace32_s3.npz'); w = A.load(3, 5, 5)
sites = sorted(set(T['addr'].tolist()))
idx = {(sites.index(T['addr'][k]) + 1, int(T['wid'][k]), int(T['visit'][k])): k for k in range(len(T['addr']))}
def F(s, wid, v, what): return T[what][idx[(s, wid, v)]].astype(np.float32)
C = 32; rec = w['rec']
perm = np.array([A.pi32(c) for c in range(32)])          # mem -> ref
xm = A.LUT[w['x']]                                       # [tok][mem ch]
We = A.LUT[A.mat(rec, 128, 32, 8)]                       # [128 n][32 k_ref]
Wc = A.LUT[A.mat(rec + 4 * C * C, 32, 128, 2)]           # [32 n][128 k]
Wq = A.LUT[A.mat(rec + 8 * C * C + 2 * C + 32, 96, 32, 6)]
def eq(a, b): return f'{np.mean(a == b) * 100:6.2f}%'
print('--- expand: B(1,w,1) vs We[w*16..][pi32(k)] (k 0..15), D(s,w,2) vs seqdot(x_mem, We_perm)')
for wid in range(8):
    Bp = We[wid * 16:wid * 16 + 16][:, perm]
    print(' wave', wid, 'B v1', eq(F(1, wid, 1, 'B'), Bp[:, :16]), 'B v2', eq(F(1, wid, 2, 'B'), Bp[:, 16:]),
          'D', ' '.join(eq(F(s, wid, 2, 'D'), A.seqdot(xm[(s - 1) * 16:(s) * 16], Bp)) for s in range(1, 5)))
def colmap(Bfrag, Wm):   # for each k column of the fragment, which column(s) of Wm [16 n][K] equal it
    return [[k2 for k2 in range(Wm.shape[1]) if np.array_equal(Bfrag[:, k], Wm[:, k2])] for k in range(Bfrag.shape[1])]
rm = np.array([0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15])
print('expand B colmap wave0 v1 (rows permuted):', colmap(F(1, 0, 1, 'B'), We[rm]))
print('expand B colmap wave0 v2 (rows permuted):', colmap(F(1, 0, 2, 'B'), We[rm]))
print('expand B colmap wave3 v1 (rows permuted):', colmap(F(1, 3, 1, 'B'), We[48 + rm]))
for wid in (0, 1):
  for v in (1, 2):
    for s_ in (5, 6):
      print(f'contract B colmap wave{wid} v{v} s{s_} rows perm:', colmap(F(s_, wid, v, 'B'), Wc[(wid & 1) * 16 + rm]))
print('expand B colmap wave0 v2:', colmap(F(1, 0, 2, 'B'), We[0:16]))
print('expand B rowmap wave0 v1:', [[n2 for n2 in range(128) if np.array_equal(np.sort(F(1, 0, 1, 'B')[n]), np.sort(We[n2][perm][:16]))] for n in range(16)])
print('contract B colmap wave0 v1 s5:', colmap(F(5, 0, 1, 'B'), Wc[0:16]))
print('contract B colmap wave1 v1 s5:', colmap(F(5, 1, 1, 'B'), Wc[16:32]))
# hidden = q8(act(D)) ; act as in the native kernel
def act(x):
    t = np.clip(x, -4, 4).astype(np.float32)
    inner = np.float32(-0.055908203125) * np.abs(t) + np.float32(0.447265625)   # fmaf -> f64 then round
    inner = (np.abs(t).astype(np.float64) * -0.055908203125 + 0.447265625).astype(np.float32)
    p = (t.astype(np.float64) * inner + 0.89453125).astype(np.float32)
    return (x * p).astype(np.float32)
Hid = np.zeros((64, 128), np.float32)
for wid in range(8):
    for s in range(1, 5): Hid[(s - 1) * 16:s * 16, wid * 16:wid * 16 + 16] = A.q8(act(F(s, wid, 2, 'D')))
print('--- hidden: A(5/6, w, v) vs q8(act(expand D)) hidden block')
for wid in range(2):
    tt = wid >> 1
    for v in range(1, 5):
        for s in (5, 6):
            a = F(s, wid, v, 'A'); hb = (v - 1) * 32 + (s - 5) * 16
            print(f' wave {wid} v{v} site {s}: tok tile {tt} hid {hb}..: ', eq(a, Hid[tt * 16:tt * 16 + 16, hb:hb + 16]),
                  ' |alt tile', max(eq(a, Hid[t2 * 16:t2 * 16 + 16, h2:h2 + 16]) for t2 in range(4) for h2 in range(0, 128, 16)))
print('--- contract B(5/6,w,v) vs Wc[(w&1)*16..][k block]')
for wid in range(2):
    for v in range(1, 5):
        for s in (5, 6):
            kb = (v - 1) * 32 + (s - 5) * 16
            print(f' wave {wid} v{v} s{s}', eq(F(s, wid, v, 'B'), Wc[(wid & 1) * 16:(wid & 1) * 16 + 16, kb:kb + 16]))
print('--- contract partial D(6,w,v) == seqdot(A5|A6, B5|B6)')
for wid in range(2):
    for v in range(1, 5):
        Aa = np.concatenate([F(5, wid, v, 'A'), F(6, wid, v, 'A')], 1); Bb = np.concatenate([F(5, wid, v, 'B'), F(6, wid, v, 'B')], 1)
        print(f' wave {wid} v{v}', eq(F(6, wid, v, 'D'), A.seqdot(Aa, Bb)))
# y = rs * x + sum(partials): which partial order and which x channel map give A(7) = q8(f16(y))?
rs_off = rec + 8 * C * C
rs = np.frombuffer(bytes(A.AR[rs_off:rs_off + 128]), np.float16).astype(np.float32)
rso = 8 if not rs[:8].any() else 0
print('rs', rs[rso:rso + 32])
f16 = lambda a: a.astype(np.float16).astype(np.float32)
import itertools
inv = np.argsort(perm)     # ref -> mem
cols7 = np.concatenate([F(7, wq, 1, 'A') for wq in range(6)], 1)   # [16 tok][96] token tile 0
def ncols(yq): return sum(any(np.array_equal(yq[:, c], cols7[:, j]) for j in range(96)) for c in range(yq.shape[1]))
res = []
for order in itertools.permutations(range(4)):
    y = np.zeros((16, 32), np.float32)
    for nt in range(2):
        P = [F(6, nt, v, 'D') for v in range(1, 5)]   # waves 0/1: token tile 0, n-tile nt
        ysum = P[order[0]].copy()
        for o in order[1:]: ysum = (ysum + P[o]).astype(np.float32)
        y[:, nt * 16 + rm] = ysum
    for form in ('fma', 'mul_add', 'add_fma'):
        xx = xm[0:16][:, inv]; r = rs[rso:rso + 32]
        if form == 'fma': yy = (xx.astype(np.float64) * r + y).astype(np.float32)
        elif form == 'mul_add': yy = ((xx * r).astype(np.float32) + y).astype(np.float32)
        else: yy = (y.astype(np.float64) + xx.astype(np.float64) * r).astype(np.float32)
        for qf in ('q8f16', 'q8', 'f16'):
            yq = A.q8(f16(yy)) if qf == 'q8f16' else (A.q8(yy) if qf == 'q8' else f16(yy))
            res.append((ncols(yq), order, form, qf))
res.sort(reverse=True); print('y candidates:', res[:6])
