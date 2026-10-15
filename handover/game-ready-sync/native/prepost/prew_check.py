"""CPU proof for PREW (LDS-staged MLP weights in k_pre_native / k_post_native). No GPU.
1) Index map, exhaustive: simulate the double-buffered staging (wfetch per lane, LDS writes/reads in program order,
   barriers between) with Wt[i] = i, and check that every operand the PREW=1 loop reads is the very element the
   PREW=0 loop reads, for every chunk, row and slot (so each dot2 chain gets identical operands in identical order).
2) Arithmetic: emulate the MLP (dot2 = f32 fma pair in the kernel's chain order, cubic SiLU, q8) on the real post
   weights (post_weights.h, local only) and random e4m3 inputs through both operand paths; outputs must be bit-equal.
Run: python -I prew_check.py [post_weights.h]"""
import os, re, sys
import numpy as np

W_EXP, W_CON = 0, 128 * 32

def staged_reads():
    """per chunk: dict (kind, a, b) -> Wt h2-index read by the PREW=1 code, through the simulated LDS"""
    Wt = np.arange(12288 // 2)                       # h2 index -> identity
    def wfetch(hc, t):
        i = (W_EXP + hc * 32) // 2 + 4 * t if t < 32 else (W_CON + (t - 32) * 128 + hc) // 2
        assert i % 4 == 0                             # 16 B aligned dwordx4
        return Wt[i:i + 4].copy()
    Wl = np.full((2, 256), -1)
    for t in range(64): Wl[0, 4 * t:4 * t + 4] = wfetch(0, t)
    out = []                                          # barrier
    for hc in range(0, 128, 8):
        b = (hc >> 3) & 1
        wn = [wfetch(hc + 8, t) for t in range(64)] if hc + 8 < 128 else None
        L = Wl[b].copy()                              # what the reads see (no writes to Wl[b] this iteration)
        r = {}
        for j in range(8):
            for k in range(16): r["e", j, k] = L[j * 16 + k]
        for c in range(32):
            for j in range(4): r["c", c, j] = L[128 + c * 4 + j]
        out.append(r)
        if wn is not None:
            for t in range(64): Wl[b ^ 1, 4 * t:4 * t + 4] = wn[t]   # other buffer: last read in the previous iteration
        # barrier
    return out

def check_index_map():
    st = staged_reads()
    n = 0
    for n_hc, hc in enumerate(range(0, 128, 8)):
        r = st[n_hc]
        for j in range(8):
            for k in range(16):
                assert r["e", j, k] == (W_EXP + (hc + j) * 32) // 2 + k, (hc, j, k); n += 1
        for c in range(32):
            for j in range(4):
                assert r["c", c, j] == (W_CON + c * 128 + hc) // 2 + j, (hc, c, j); n += 1
    # every MLP weight is read exactly once per thread, in both paths
    seen = sorted(v for r in st for v in r.values())
    assert seen == list(range(W_CON // 2 + 32 * 128 // 2)), "coverage"
    return n

def f16(x): return np.float32(np.float16(x))
def q8(x):   # e4m3 RNE, saturating at 448 (the kernel's q8)
    x = np.float32(x); a = min(abs(x), np.float32(448))
    if a < 0.015625: r = np.float32(np.round(a * 512) / 512)
    else:
        e = np.floor(np.log2(a)); s = 2.0 ** (e - 3); r = np.float32(np.round(a / s) * s)
    return np.float32(np.copysign(r, x))
def dot2(a, b, c):   # v_dot2_f32_f16: c + a.x*b.x + a.y*b.y; products exact in f32, modelled as two f32 fmas
    return np.float32(np.float32(c + np.float32(a[0]) * np.float32(b[0])) + np.float32(a[1]) * np.float32(b[1]))

def mlp(xq, get_e, get_c):
    y = [np.float32(0)] * 32
    for hc in range(0, 128, 8):
        a = []
        for j in range(8):
            s = np.float32(0)
            for k in range(16): s = dot2(xq[k], get_e(hc, j, k), s)
            tc = np.float32(min(max(s, -4), 4)); inner = np.float32(abs(tc) * np.float32(-0.055908203125) + np.float32(0.447265625))
            a.append(q8(np.float32(s * np.float32(tc * inner + np.float32(0.89453125)))))
        ap = [(f16(a[2 * j]), f16(a[2 * j + 1])) for j in range(4)]
        for c in range(32):
            for j in range(4): y[c] = dot2(ap[j], get_c(hc, c, j), y[c])
    return np.array(y, np.float32)

def check_arith(path):
    src = open(path).read()
    w = np.array([int(v) for v in re.search(r"gWt\[\d+\] = \{([^}]*)\}", src).group(1).split(",")], np.uint16).view(np.float16)
    W2 = w.reshape(-1, 2)                             # h2 view
    st = staged_reads()
    orig_e = lambda hc, j, k: W2[(W_EXP + (hc + j) * 32) // 2 + k]
    orig_c = lambda hc, c, j: W2[(W_CON + c * 128 + hc) // 2 + j]
    stg_e = lambda hc, j, k: W2[st[hc // 8]["e", j, k]]
    stg_c = lambda hc, c, j: W2[st[hc // 8]["c", c, j]]
    rng = np.random.default_rng(1)
    for _ in range(20):
        xq = [(q8(f16(v0)), q8(f16(v1))) for v0, v1 in rng.normal(0, 1.5, (16, 2)).astype(np.float32)]
        a, b = mlp(xq, orig_e, orig_c), mlp(xq, stg_e, stg_c)
        assert a.tobytes() == b.tobytes()
    return 20

if __name__ == "__main__":
    print("index map: %d operand slots per thread identical (16 chunks x (128 expand + 128 contract))" % check_index_map())
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_weights.h")
    if os.path.exists(p): print("arithmetic: %d random tokens, MLP output bit-identical (real post weights)" % check_arith(p))
    else: print("arithmetic: skipped (no post_weights.h)")
