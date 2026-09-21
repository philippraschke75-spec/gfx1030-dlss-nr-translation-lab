"""One-shot source patch: adds the instruction forms k_swin_var needs to gfx11emu.py (idempotent guard)."""
p = 'gfx11emu.py'
s = open(p, encoding='utf-8').read()
if 'def _ovf_add' in s:
    raise SystemExit('already patched')


def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a[:80], s.count(a))
    s = s.replace(a, b)


# ---------------------------------------------------------------- scalar helpers + table
rep("SALU2 = {", '''def _ovf_add(x, y):
    r = _sh((x + y) & M32)
    return int((_sh(x) >= 0) == (_sh(y) >= 0) and (_sh(x) >= 0) != (r >= 0))


def _ovf_sub(x, y):
    r = _sh((x - y) & M32)
    return int((_sh(x) >= 0) != (_sh(y) >= 0) and (_sh(x) >= 0) != (r >= 0))


def _bfe(x, y, signed):
    off, wd = y & 31, (y >> 16) & 0x7f
    r = (x >> off) & ((1 << wd) - 1) if wd else 0
    if signed and wd and (r >> (wd - 1)) & 1:
        r = (r - (1 << wd)) & M32
    return r & M32, int((r & M32) != 0)


def f32_class(x):
    """v_cmp_class_f32 class bits: 0 sNaN 1 qNaN 2 -inf 3 -norm 4 -denorm 5 -0 6 +0 7 +denorm 8 +norm 9 +inf."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    sign = (bits >> 31).astype(bool)
    e = (bits >> 23) & 0xff
    m = bits & 0x7fffff
    cls = np.zeros(len(bits), np.uint32)
    nan = (e == 0xff) & (m != 0)
    snan = nan & ((m & 0x400000) == 0)
    cls = np.where(snan, 1 << 0, cls)
    cls = np.where(nan & ~snan, 1 << 1, cls)
    inf = (e == 0xff) & (m == 0)
    cls = np.where(inf & sign, 1 << 2, cls)
    cls = np.where(inf & ~sign, 1 << 9, cls)
    norm = (e != 0) & (e != 0xff)
    cls = np.where(norm & sign, 1 << 3, cls)
    cls = np.where(norm & ~sign, 1 << 8, cls)
    den = (e == 0) & (m != 0)
    cls = np.where(den & sign, 1 << 4, cls)
    cls = np.where(den & ~sign, 1 << 7, cls)
    zero = (e == 0) & (m == 0)
    cls = np.where(zero & sign, 1 << 5, cls)
    cls = np.where(zero & ~sign, 1 << 6, cls)
    return cls


SALU2 = {''')
rep("    's_add_i32': lambda x, y, c: ((x + y) & M32, None),", """    's_add_i32': lambda x, y, c: ((x + y) & M32, _ovf_add(x, y)),
    's_sub_i32': lambda x, y, c: ((x - y) & M32, _ovf_sub(x, y)),
    's_sub_u32': lambda x, y, c: ((x - y) & M32, int(x < y)),
    's_nor_b32': lambda x, y, c: ((~(x | y)) & M32, int(((~(x | y)) & M32) != 0)),
    's_mul_hi_u32': lambda x, y, c: (((x * y) >> 32) & M32, None),
    's_mul_hi_i32': lambda x, y, c: (((_sh(x) * _sh(y)) >> 32) & M32, None),
    's_lshl_b64': lambda x, y, c: ((x << (y & 63)) & ((1 << 64) - 1), int(((x << (y & 63)) & ((1 << 64) - 1)) != 0)),
    's_bfe_u32': lambda x, y, c: _bfe(x, y, False),
    's_bfe_i32': lambda x, y, c: _bfe(x, y, True),""")

# ---------------------------------------------------------------- scalar decode additions
rep("        if base in ('s_mov_b32', 's_mov_b64'):", """        if base == 's_movk_i32':
            return lambda w: w.ws(P[0], _sh16(w.rs(P[1])) & M32)
        if base == 's_addk_i32':
            def addk(w):
                a, b = w.rs(P[0]), _sh16(w.rs(P[1])) & M32
                w.ws(P[0], (a + b) & M32)
                w.scc = _ovf_add(a, b)
            return addk
        if base in ('s_sext_i32_i16', 's_sext_i32_i8'):
            bitsz = 16 if base.endswith('i16') else 8

            def sext(w):
                v = w.rs(P[1]) & ((1 << bitsz) - 1)
                w.ws(P[0], (v - (1 << bitsz) if v >> (bitsz - 1) else v) & M32)
            return sext
        if base in ('s_bitcmp1_b32', 's_bitcmp0_b32'):
            one = base.endswith('1_b32')

            def bitcmp(w):
                b = (w.rs(P[0]) >> (w.rs(P[1]) & 31)) & 1
                w.scc = int(b == (1 if one else 0))
            return bitcmp
        if base == 's_or_saveexec_b32':
            def orsav(w):
                old = w.S[EXEC]
                w.ws(P[0], old)
                w.S[EXEC] = (w.rs(P[1]) | old) & M32
                w.scc = int(w.S[EXEC] != 0)
            return orsav
        if base in ('s_mov_b32', 's_mov_b64'):""")
rep("(m := re.fullmatch(r's_cmpk?_(eq|lg|gt|ge|lt|le)_(i|u)32', base))", "(m := re.fullmatch(r's_cmpk?_(eq|lg|gt|ge|lt|le)_(i|u)(32|64)', base))")
rep("""            c, ty = m[1], m[2]

            def scmp(w):
                x, y = w.rs(P[0]), w.rs(P[1])
                if ty == 'i':
                    x, y = _sh(x), _sh(y)""", """            c, ty, is32 = m[1], m[2], m[3] == '32'

            def scmp(w):
                x, y = w.rs(P[0]), w.rs(P[1])
                if ty == 'i' and is32:
                    x, y = _sh(x), _sh(y)""")
rep("def _sh(x):", "def _sh16(x):\n    x &= 0xffff\n    return x - 0x10000 if x & 0x8000 else x\n\n\ndef _sh(x):")

# ---------------------------------------------------------------- memory: d16 loads, bpermute
rep("""        ld, t = m[1] == 'load', m[2]
        n = WIDTH[t]
        sign = t.startswith('i')
        if ld:
            vd, va = P[0], P[1]
        elif kind == 'scratch':""", """        ld, t = m[1] == 'load', m[2]
        d16 = None
        if t.startswith('d16_hi_'):
            d16, t = 'hi', t[7:]
        elif t.startswith('d16_'):
            d16, t = 'lo', t[4:]
        n = WIDTH[t]
        sign = t.startswith('i')
        if ld:
            vd, va = P[0], P[1]
        elif kind == 'scratch':""")
rep("""            if ld:
                v = ld_val(raw, n, sign)
                if v.ndim == 1:
                    v = v[:, None]
                for j in range(v.shape[1]):
                    w.V[vd['i'] + j, idx] = v[:, j]
        return run""", """            if ld:
                v = ld_val(raw, n, sign)
                if d16:
                    cur = w.V[vd['i'], idx]
                    w.V[vd['i'], idx] = ((cur & np.uint32(0xffff)) | (v << np.uint32(16))) if d16 == 'hi' else ((cur & np.uint32(0xffff0000)) | (v & np.uint32(0xffff)))
                    return
                if v.ndim == 1:
                    v = v[:, None]
                for j in range(v.shape[1]):
                    w.V[vd['i'] + j, idx] = v[:, j]
        return run""")
rep("""    def _ds(s, rest, P, mods):
        off = mods.get('offset', 0)""", """    def _ds(s, rest, P, mods):
        off = mods.get('offset', 0)
        if rest == 'bpermute_b32':
            def bperm(w):
                src = ((w.RV[P[1]['i']].astype(np.int64) + off) >> 2) & 31
                data = w.RV[P[2]['i']].copy()
                w.wv(P[0], data[src])
            return bperm""")

# ---------------------------------------------------------------- vector additions
rep("        raise NotImplementedError('valu ' + b)", '''        # ---- packed f16 math (op_sel selects the source half for the low result, op_sel_hi for the high result)
        def pk(f, nsrc):
            osel, ohi = mods.get('op_sel', [0] * 3), mods.get('op_sel_hi', [1] * 3)

            def run(w):
                halves = []
                for k in range(nsrc):
                    raw = w.r32(dict(P[1 + k], neg=False, abs=False))
                    lo, hi = (raw & np.uint32(0xffff)).astype(np.uint16).view(np.float16), (raw >> np.uint32(16)).astype(np.uint16).view(np.float16)
                    L = (hi if osel[k] else lo).astype(np.float32)
                    H = (hi if ohi[k] else lo).astype(np.float32)
                    if P[1 + k]['abs']:
                        L, H = np.abs(L), np.abs(H)
                    if P[1 + k]['neg']:
                        L, H = -L, -H
                    halves.append((L, H))
                rl = f(*[x[0] for x in halves]).astype(np.float16)
                rh_ = f(*[x[1] for x in halves]).astype(np.float16)
                w.wv(P[0], hbits(rl) | (hbits(rh_) << np.uint32(16)))
            return run
        if b == 'v_pk_fma_f16':
            return pk(lambda a, c, d: (a.astype(np.float64) * c.astype(np.float64) + d.astype(np.float64)), 3)
        if b == 'v_pk_add_f16':
            return pk(lambda a, c: a + c, 2)
        if b == 'v_pk_max_f16':
            return pk(np.fmax, 2)
        if b == 'v_pk_min_f16':
            return pk(np.fmin, 2)
        if b == 'v_pk_mul_f16':
            return pk(lambda a, c: a * c, 2)
        if b == 'v_add_f16':
            return lambda w: w.wv(P[0], hbits(w.rh(P[1]) + w.rh(P[2])))
        if b == 'v_pack_b32_f16':
            return lambda w: w.wv(P[0], (w.r32(P[1], True) & np.uint32(0xffff)) | ((w.r32(P[2], True) & np.uint32(0xffff)) << np.uint32(16)))
        if b == 'v_alignbit_b32':
            return lambda w: w.wv(P[0], u32(((w.r32(P[1]).astype(np.uint64) << np.uint64(32)) | w.r32(P[2]).astype(np.uint64)) >> (w.r32(P[3]) & np.uint32(31)).astype(np.uint64)))
        if b == 'v_add_lshl_u32':
            return lambda w: w.wv(P[0], (w.r32(P[1]) + w.r32(P[2])) << (w.r32(P[3]) & np.uint32(31)))
        if b == 'v_xad_u32':
            return lambda w: w.wv(P[0], (w.r32(P[1]) ^ w.r32(P[2])) + w.r32(P[3]))
        if b == 'v_xor3_b32':
            return lambda w: w.wv(P[0], w.r32(P[1]) ^ w.r32(P[2]) ^ w.r32(P[3]))
        if b == 'v_not_b32':
            return lambda w: w.wv(P[0], ~w.r32(P[1]))
        if b == 'v_subrev_nc_u32':
            return lambda w: w.wv(P[0], w.r32(P[2]) - w.r32(P[1]))
        if b == 'v_fmaak_f32':
            return lambda w: w.wv(P[0], fbits(fma32(w.rf(P[1]), w.rf(P[2]), w.rf(P[3]))))
        if b == 'v_fmamk_f32':
            return lambda w: w.wv(P[0], fbits(fma32(w.rf(P[1]), w.rf(P[2]), w.rf(P[3]))))   # operands: D, S0, K, S1 -> S0*K+S1
        if b == 'v_clz_i32_u32':
            def clz(w):
                x = w.r32(P[1])
                out = np.full(32, 0xffffffff, np.uint32)          # ISA: -1 when there is no set bit
                for i in range(32):
                    hit = ((x >> np.uint32(31 - i)) & np.uint32(1)).astype(bool) & (out == 0xffffffff)
                    out = np.where(hit, np.uint32(i), out)
                w.wv(P[0], out)
            return clz
        if b == 'v_sqrt_f32':
            return lambda w: w.wv(P[0], fbits(np.sqrt(w.rf(P[1]))))
        if b == 'v_exp_f32':
            return lambda w: w.wv(P[0], fbits(np.exp2(w.rf(P[1]).astype(np.float64)).astype(np.float32)))
        if b == 'v_mbcnt_lo_u32_b32':
            def mbcnt(w):
                mask = w.r32(P[1])
                lane = np.arange(32, dtype=np.uint32)
                below = (np.uint64(1) << lane.astype(np.uint64)) - np.uint64(1)
                cnt = np.array([bin(int(m) & int(bl)).count('1') for m, bl in zip(mask, below)], np.uint32)
                w.wv(P[0], cnt + w.r32(P[2]))
            return mbcnt
        if b == 'v_mad_i64_i32':
            return lambda w: w.wv64(P[0], (w.r32(P[2]).view(np.int32).astype(np.int64) * w.r32(P[3]).view(np.int32).astype(np.int64) + w.r64(P[4]).view(np.int64)).view(np.uint64))
        if b == 'v_fma_mixhi_f16':
            osel, ohi = mods.get('op_sel', [0, 0, 0]), mods.get('op_sel_hi', [0, 0, 0])

            def mixhi(w):
                vals = []
                for k in range(3):
                    raw = w.r32(dict(P[1 + k], neg=False, abs=False))
                    if ohi[k]:
                        hv = (raw >> np.uint32(16)) if osel[k] else raw
                        x = (hv & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float32)
                    else:
                        x = raw.view(np.float32)
                    if P[1 + k]['abs']:
                        x = np.abs(x)
                    if P[1 + k]['neg']:
                        x = -x
                    vals.append(x)
                r = fma32(*vals)
                w.wv(P[0], (w.V[P[0]['i']] & np.uint32(0xffff)) | (hbits(r.astype(np.float16)) << np.uint32(16)))
            return mixhi
        raise NotImplementedError('valu ' + b)''')
rep("        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|f16|i32|u32|i16|u16)', b)", '''        if (m := re.fullmatch(r'v_cmpx?_class_f32', b)):
            def cls(w):
                bits = w.bits(((f32_class(w.r32(P[-2]).view(np.float32)) & w.r32(P[-1])) != 0))
                if b.startswith('v_cmpx'):
                    w.S[EXEC] = bits
                else:
                    w.wmask(P[0], bits)
            return cls
        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|f16|i32|u32|i16|u16)', b)''')
open(p, 'w', encoding='utf-8').write(s)
print('patched')
