"""CPU proof that k_ffwd2_n stage 2 (32-wide chunks, 39a6c99 -> perf-c512b) feeds every output the same dot2 sequence
as the 64-wide original. Symbolic emulation of one workgroup: LDS cells hold tags, each dot2 appends (A, B) to the
output's list. Equal lists => equal f32 results bit for bit (stage 1 and epilogues are untouched)."""
ipi32 = lambda n: n   # same storage permutation in both versions; the identity keeps the tags comparable


def cvt4(src, off):   # dword -> uint2 (lo = bytes 0,1, hi = bytes 2,3); tags only
    return ((src, off, 0), (src, off, 1))


def Rs_fill():   # identical code in both: Rs[tx*64 + ty*4 + i] = (pkh(s0,s2), pkh(s1,s3)), r = 4tx+e
    Rs = {}
    for t in range(256):
        ty, tx = t >> 4, t & 15
        for i in range(4):
            tok = ty * 4 + i
            Rs[tx * 64 + tok] = ((("R", tok, 4 * tx), ("R", tok, 4 * tx + 2)), (("R", tok, 4 * tx + 1), ("R", tok, 4 * tx + 3)))
    return Rs


def mma(As, Ws, KD, NB, acc, keyf):   # mmaT<KD,NB> / mma16 = mmaT<16,4>; keyf(tok, e) -> output key
    for t in range(256):
        ty, tx = t >> 4, t & 15
        for kd in range(KD):
            for i in range(4):
                for e in range(NB):
                    a, b = As[kd * 64 + ty * 4 + i], Ws[kd * 16 * NB + tx * NB + e]
                    acc.setdefault(keyf(ty * 4 + i, tx, e), []).extend([(a[0], b[0]), (a[1], b[1])])


def h_tag(hseq):   # H value = function of its dot2 sequence (q8(split_act(.))): tag by the sequence itself
    return ("H", tuple(hseq))


def orig():
    Rs, H, OUT = Rs_fill(), {}, {}
    for jc in range(4):
        Ws = {}
        for t in range(256):
            r5, u = t >> 7, t & 127
            off = 262144 + (r5 << 13) + u * 16 + (jc << 11)
            for e in range(4):
                j = (u >> 2 & 1) | (e >> 1) << 1 | (u >> 5 & 1) << 2 | (u >> 3 & 1) << 3 | (u >> 4 & 1) << 4 | (u >> 6 & 1) << 5
                kd = (u & 1) | (u >> 1 & 1) << 1 | (e & 1) << 2 | r5 << 3
                Ws[kd * 64 + j] = cvt4("Q0", off + 4 * e)
        hc = {}
        mma(Rs, Ws, 16, 4, hc, lambda tok, tx, e: (tok, 64 * jc + 4 * tx + e))
        H.update(hc)
        Hs = {}
        for t in range(256):
            ty, tx = t >> 4, t & 15
            for i in range(4):
                tok = ty * 4 + i
                s = [h_tag(hc[(tok, 64 * jc + 4 * tx + e)]) for e in range(4)]
                Hs[tx * 64 + tok] = ((s[0], s[1]), (s[2], s[3]))
        Ws = {}
        for u in range(256):
            off = 393216 + u * 16 + (jc << 12)
            for e in range(4):
                n = (u >> 2 & 7) | (e >> 1) << 3 | (u >> 5 & 3) << 4
                kd = (e & 1) | (u & 3) << 1 | (u >> 7) << 3
                Ws[kd * 64 + ((n & 32) | ipi32(n & 31))] = cvt4("Q2", off + 4 * e)
        mma(Hs, Ws, 16, 4, OUT, lambda tok, tx, e: (tok, 4 * tx + e))
    return H, OUT


def new():
    Rs, H, OUT = Rs_fill(), {}, {}
    for c in range(8):
        Q0s, Q2s = {}, {}
        for t in range(256):
            lt = t & 127
            if t < 128:
                r5, u = lt >> 6, lt & 63
                off = 262144 + (r5 << 13) + u * 16 + (c << 10)
                for e in range(4):
                    j = (u >> 2 & 1) | (e >> 1) << 1 | (u >> 5 & 1) << 2 | (u >> 3 & 1) << 3 | (u >> 4 & 1) << 4
                    kd = (u & 3) | (e & 1) << 2 | r5 << 3
                    Q0s[kd * 32 + j] = cvt4("Q0", off + 4 * e)
            else:
                u = lt
                off = 393216 + lt * 16 + (c << 11)
                for e in range(4):
                    n = (u >> 2 & 7) | (e >> 1) << 3 | (u >> 5 & 3) << 4
                    kd = (e & 1) | (u & 3) << 1
                    Q2s[kd * 64 + ((n & 32) | ipi32(n & 31))] = cvt4("Q2", off + 4 * e)
        hc = {}
        mma(Rs, Q0s, 16, 2, hc, lambda tok, tx, e: (tok, 32 * c + 2 * tx + e))
        H.update(hc)
        Hs32 = {}
        for t in range(256):
            ty, tx = t >> 4, t & 15
            for i in range(4):
                tok = ty * 4 + i
                Hs32[(tx >> 1) * 128 + tok * 2 + (tx & 1)] = tuple(h_tag(hc[(tok, 32 * c + 2 * tx + e)]) for e in range(2))
        Hs = {k: (Hs32[2 * k], Hs32[2 * k + 1]) for k in range(512)}
        mma(Hs, Q2s, 8, 4, OUT, lambda tok, tx, e: (tok, 4 * tx + e))
    return H, OUT


if __name__ == "__main__":
    (h0, o0), (h1, o1) = orig(), new()
    assert len(h0) == 64 * 256 and len(o0) == 64 * 64
    assert h0 == h1, "hidden dot2 sequences differ"
    assert o0 == o1, "output dot2 sequences differ"
    print("ffwd2 stage 2: identical dot2 sequences for all", len(h0), "hidden and", len(o0), "outputs")
