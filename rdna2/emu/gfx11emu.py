"""Independent CPU interpreter for the gfx1100 (RDNA3) instruction stream of the SWIN kernels.

Purpose: reference implementation for differential testing of the gfx1030 translation, and a
safe way to discover every address a dispatch touches. Wave32 only. Not a general emulator:
unknown opcodes raise NotImplementedError instead of being guessed.
"""
import re
from pathlib import Path
import numpy as np
np.seterr(all='ignore')
M32 = 0xffffffff
VCC, M0, EXEC = 106, 124, 126
SH_HI, PR_HI = 0x00010000, 0x00020000          # flat aperture high dwords chosen by the harness
AR = np.arange(32, dtype=np.uint64)
ONE = np.uint64(1)
# Host diagnostic switch; disabled for the reference model. This is a
# hypothesis probe, not an established gfx1100 ISA rule.
MIX_F16_INPUT_FLUSH = False


class MemFault(Exception):
    pass


class Region:
    def __init__(s, name, base, data):
        s.name, s.base = name, base
        s.arr = data if isinstance(data, np.ndarray) else np.frombuffer(bytearray(data), dtype=np.uint8)
        s.lo, s.hi, s.rd, s.wr = 1 << 62, 0, False, False


class GMem:
    def __init__(s):
        s.regions = []

    def add(s, name, base, data):
        r = Region(name, base, data)
        s.regions.append(r)
        return r

    def read(s, addr, n):
        addr = np.asarray(addr, np.uint64)
        res = np.zeros((len(addr), n), np.uint8)
        got = np.zeros(len(addr), bool)
        for r in s.regions:
            m = (addr >= r.base) & (addr + np.uint64(n) <= r.base + len(r.arr)) & ~got
            if m.any():
                off = (addr[m] - np.uint64(r.base)).astype(np.int64)
                res[m] = r.arr[off[:, None] + np.arange(n)]
                r.rd = True
                r.lo = min(r.lo, int(off.min()))
                r.hi = max(r.hi, int(off.max()) + n)
                got |= m
        if not got.all():
            raise MemFault('global read fault at %s (n=%d)' % ([hex(int(a)) for a in addr[~got][:4]], n))
        return res

    def write(s, addr, data):
        addr = np.asarray(addr, np.uint64)
        n = data.shape[1]
        got = np.zeros(len(addr), bool)
        for r in s.regions:
            m = (addr >= r.base) & (addr + np.uint64(n) <= r.base + len(r.arr)) & ~got
            if m.any():
                off = (addr[m] - np.uint64(r.base)).astype(np.int64)
                r.arr[off[:, None] + np.arange(n)] = data[m]
                r.wr = True
                r.lo = min(r.lo, int(off.min()))
                r.hi = max(r.hi, int(off.max()) + n)
                got |= m
        if not got.all():
            raise MemFault('global write fault at %s (n=%d)' % ([hex(int(a)) for a in addr[~got][:4]], n))


# ---------------------------------------------------------------- program loading
LINE = re.compile(r'^\t(\S+)(?: (.*?))?\s*// ([0-9A-Fa-f]+): ((?:[0-9A-Fa-f]{8} ?)+)')


def load_program(path, symbols):
    prog, cur = [], None
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        m = re.fullmatch(r'([0-9a-fA-F]+) <([^>]+)>:', line)
        if m:
            cur = m[2] in symbols
            continue
        if cur:
            m = LINE.match(line)
            if m:
                op, args, addr, raw = m.groups()
                prog.append((int(addr, 16), op, (args or '').strip(), 4 * len(raw.split())))
    return prog


# ---------------------------------------------------------------- operands
SPECIAL = {'vcc_lo': (VCC, 1), 'vcc_hi': (VCC + 1, 1), 'vcc': (VCC, 2), 'exec_lo': (EXEC, 1), 'exec_hi': (EXEC + 1, 1),
           'exec': (EXEC, 2), 'm0': (M0, 1)}


def parse_op(t):
    t = t.strip()
    neg = ab = False
    if t.startswith('-') and len(t) > 1 and not (t[1].isdigit() or t[1] == '.'):
        neg = True
        t = t[1:]
    if t.startswith('|'):
        ab = True
        t = t.strip('|')
    o = dict(neg=neg, abs=ab)
    if (m := re.fullmatch(r'v(\d+)', t)):
        o.update(k='v', i=int(m[1]), n=1)
    elif (m := re.fullmatch(r'v\[(\d+):(\d+)\]', t)):
        o.update(k='v', i=int(m[1]), n=int(m[2]) - int(m[1]) + 1)
    elif (m := re.fullmatch(r's(\d+)', t)):
        o.update(k='s', i=int(m[1]), n=1)
    elif (m := re.fullmatch(r's\[(\d+):(\d+)\]', t)):
        o.update(k='s', i=int(m[1]), n=int(m[2]) - int(m[1]) + 1)
    elif t in SPECIAL:
        o.update(k='s', i=SPECIAL[t][0], n=SPECIAL[t][1])
    elif t == 'null':
        o.update(k='null', n=1)
    elif t == 'off':
        o.update(k='off')
    elif t == 'src_shared_base':
        o.update(k='lit', v=SH_HI << 32, f=False, n=2)
    elif t == 'src_private_base':
        o.update(k='lit', v=PR_HI << 32, f=False, n=2)
    elif re.fullmatch(r'-?\d+\.\d*|-?\.\d+', t):
        o.update(k='lit', v=float(t), f=True, n=1)
    else:
        try:
            o.update(k='lit', v=int(t, 0), f=False, n=1)
        except ValueError:
            raise NotImplementedError('operand ' + t)
    return o


MODS = re.compile(r'\s*(op_sel_hi:\[[^\]]*\]|op_sel:\[[^\]]*\]|offset0:\d+|offset1:\d+|offset:-?\d+|glc|slc|dlc|clamp|mul:\d|div:\d)')


def split_args(args):
    mods = {}
    for m in MODS.findall(args):
        if m.startswith('op_sel_hi'):
            mods['op_sel_hi'] = eval(m.split(':', 1)[1])
        elif m.startswith('op_sel'):
            mods['op_sel'] = eval(m.split(':', 1)[1])
        elif ':' in m:
            k, v = m.split(':')
            mods[k] = int(v)
        else:
            mods[m] = True
    rest = MODS.sub('', args).strip()
    return ([x.strip() for x in rest.split(',')] if rest else []), mods


# ---------------------------------------------------------------- wave
class Wave:
    def __init__(s, gmem, lds, wid, tid0):
        s.g, s.lds, s.wid = gmem, lds, wid
        s.S = [0] * 128
        s.V = np.zeros((256, 32), np.uint32)
        s.RV = s.V
        s.scc = 0
        s.pc = 0
        s.next = 0
        s.state = 'run'
        s.priv = np.zeros((32, 512), np.uint8)
        s.privmax = 0
        s.S[EXEC] = M32
        s.n = 0
        s.V[0] = np.arange(tid0, tid0 + 32, dtype=np.uint32)

    @property
    def em(s):
        return ((np.uint64(s.S[EXEC]) >> AR) & ONE).astype(bool)

    def lanes(s, mask):
        return ((np.uint64(mask) >> AR) & ONE).astype(bool)

    def r32(s, o, f16=False):
        k = o['k']
        if k == 'v':
            x = s.RV[o['i']].copy()
        elif k == 's':
            x = np.full(32, s.S[o['i']], np.uint32)
        elif k == 'null':
            x = np.zeros(32, np.uint32)
        elif k == 'lit':
            v = o['v']
            if o['f']:
                x = np.full(32, np.float16(v).view(np.uint16) if f16 else np.float32(v).view(np.uint32), np.uint32)
            else:
                x = np.full(32, v & M32, np.uint32)
        else:
            raise NotImplementedError(k)
        if o['abs']:
            x &= np.uint32(0x7fff if f16 else 0x7fffffff)
        if o['neg']:
            x ^= np.uint32(0x8000 if f16 else 0x80000000)
        return x

    def rf(s, o):
        return s.r32(o).view(np.float32)

    def rh(s, o):
        return (s.r32(o, True) & np.uint32(0xffff)).astype(np.uint16).view(np.float16)

    def rs(s, o):
        k = o['k']
        if k == 's':
            return s.S[o['i']] if o['n'] == 1 else s.S[o['i']] | (s.S[o['i'] + 1] << 32)
        if k == 'lit':
            if o['f']:
                return int(np.float32(o['v']).view(np.uint32))
            return o['v'] & ((1 << 64) - 1 if o['n'] == 2 else M32)
        if k == 'null':
            return 0
        raise NotImplementedError('rs ' + k)

    def r64(s, o):
        if o['k'] == 'v':
            return s.RV[o['i']].astype(np.uint64) | (s.RV[o['i'] + 1].astype(np.uint64) << np.uint64(32))
        if o['k'] == 's':
            return np.full(32, s.rs(o), np.uint64)
        if o['k'] == 'lit':
            return np.full(32, o['v'] & ((1 << 64) - 1), np.uint64)
        raise NotImplementedError('r64')

    def mask(s, o):
        return s.S[o['i']] if o['k'] == 's' else 0

    def wv(s, o, val):
        if o['k'] == 'null':
            return
        em = s.em
        val = np.asarray(val).astype(np.uint32)
        s.V[o['i']][em] = val[em]

    def wv64(s, o, val):
        em = s.em
        val = np.asarray(val).astype(np.uint64)
        s.V[o['i']][em] = (val & np.uint64(M32)).astype(np.uint32)[em]
        s.V[o['i'] + 1][em] = (val >> np.uint64(32)).astype(np.uint32)[em]

    def ws(s, o, val):
        if o['k'] == 'null':
            return
        if o['n'] == 2:
            s.S[o['i']] = val & M32
            s.S[o['i'] + 1] = (val >> 32) & M32
        else:
            s.S[o['i']] = val & M32

    def wmask(s, o, bits):
        if o['k'] != 'null':
            s.S[o['i']] = bits & M32

    def bits(s, res):
        return int(((np.asarray(res, bool) & s.em).astype(np.uint64) << AR).sum())


# ---------------------------------------------------------------- helpers
def u32(x):
    return np.asarray(x).astype(np.uint32)


def h(x):
    return (np.asarray(x, np.uint32) & np.uint32(0xffff)).astype(np.uint16)


def sh16(x):
    return h(x).view(np.int16)


def fbits(x):
    return np.ascontiguousarray(x, np.float32).view(np.uint32)


def hbits(x):
    return np.ascontiguousarray(x, np.float16).view(np.uint16).astype(np.uint32)


def cvt_i32(x):
    x = x.astype(np.float64)
    r = np.where(np.isnan(x), 0, np.clip(np.trunc(x), -2147483648, 2147483647))
    return r.astype(np.int32).view(np.uint32)


def cvt_u32(x):
    x = x.astype(np.float64)
    r = np.where(np.isnan(x), 0, np.clip(np.trunc(x), 0, 4294967295))
    return r.astype(np.uint32)


def fma32(a, b, c):
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def exp_of(x):
    return ((np.ascontiguousarray(x, np.float32).view(np.uint32) >> 23) & 0xff).astype(np.int32)


def div_scale(s0, s1, s2):
    """v_div_scale_f32: S0 = value being scaled (denominator or numerator), S1 = denominator, S2 = numerator."""
    d = s0.copy()
    vcc = np.zeros(32, bool)
    e1, e2 = exp_of(s1), exp_of(s2)
    tiny = np.float32(2.0 ** -126)
    rcp = np.float32(1) / s1
    q = s2 / s1
    bad = (s2 == 0) | (s1 == 0)
    d[bad] = np.float32(np.nan)
    c1 = ~bad & ((e2 - e1) >= 96)
    vcc |= c1
    sel = c1 & (s0 == s1)
    d[sel] = np.ldexp(s0, 64)[sel]
    c2 = ~bad & ~c1 & (e1 == 0)
    d[c2] = np.ldexp(s0, 64)[c2]
    r_den = (np.abs(rcp) < tiny) & (rcp != 0)
    q_den = (np.abs(q) < tiny) & (q != 0)
    c3 = ~bad & ~c1 & ~c2 & r_den & q_den
    vcc |= c3
    sel = c3 & (s0 == s1)
    d[sel] = np.ldexp(s0, 64)[sel]
    c4 = ~bad & ~c1 & ~c2 & ~c3 & r_den
    d[c4] = np.ldexp(s0, -64)[c4]
    c5 = ~bad & ~c1 & ~c2 & ~c3 & ~c4 & q_den
    vcc |= c5
    sel = c5 & (s0 == s2)
    d[sel] = np.ldexp(s0, 64)[sel]
    c6 = ~bad & ~c1 & ~c2 & ~c3 & ~c4 & ~c5 & (e2 <= 23)
    d[c6] = np.ldexp(s0, 64)[c6]
    return d.astype(np.float32), vcc


def div_fixup(q, den, num):
    sign = ((np.ascontiguousarray(den).view(np.uint32) ^ np.ascontiguousarray(num).view(np.uint32)) >> 31).astype(bool)
    out = np.where(sign, -np.abs(q), np.abs(q)).astype(np.float32)
    inf, nan = np.float32(np.inf), np.float32(np.nan)
    out = np.where((den == 0) | np.isinf(num), np.where(sign, -inf, inf), out)
    out = np.where(np.isinf(den) | (num == 0), np.where(sign, np.float32(-0.0), np.float32(0.0)), out)
    out = np.where(((den == 0) & (num == 0)) | (np.isinf(den) & np.isinf(num)), nan, out)
    out = np.where(np.isnan(num), num, np.where(np.isnan(den), den, out))
    return out.astype(np.float32)


CMP = dict(eq=lambda a, b: a == b, ne=lambda a, b: a != b, lt=lambda a, b: a < b, le=lambda a, b: a <= b,
           gt=lambda a, b: a > b, ge=lambda a, b: a >= b, lg=lambda a, b: (a < b) | (a > b),
           o=lambda a, b: ~(np.isnan(a) | np.isnan(b)), u=lambda a, b: np.isnan(a) | np.isnan(b),
           nlt=lambda a, b: ~(a < b), ngt=lambda a, b: ~(a > b), nge=lambda a, b: ~(a >= b), nle=lambda a, b: ~(a <= b),
           neq=lambda a, b: ~(a == b), nlg=lambda a, b: ~((a < b) | (a > b)))


def typed(w, o, ty):
    if ty == 'f32':
        return w.rf(o)
    if ty == 'f16':
        return w.rh(o)
    if ty == 'u32':
        return w.r32(o)
    if ty == 'i32':
        return w.r32(o).view(np.int32)
    if ty == 'u16':
        return h(w.r32(o))
    if ty == 'i16':
        return h(w.r32(o)).view(np.int16)
    raise NotImplementedError(ty)


def ld_val(raw, n, sign):
    if n == 1:
        v = raw[:, 0].astype(np.uint32)
        return v.astype(np.int8).astype(np.int32).view(np.uint32) if sign else v
    if n == 2:
        v = raw[:, 0].astype(np.uint32) | (raw[:, 1].astype(np.uint32) << 8)
        return v.astype(np.uint16).view(np.int16).astype(np.int32).view(np.uint32) if sign else v
    raw = np.ascontiguousarray(raw)
    return raw.reshape(len(raw), n // 4, 4).view(np.uint32).reshape(len(raw), n // 4)


def stbytes(w, vd, n, idx, hi=False):
    idx = np.asarray(idx)
    out = np.zeros((len(idx), n), np.uint8)
    for j in range((n + 3) // 4):
        word = w.RV[vd['i'] + j][idx]
        if hi and j == 0:
            word = word >> np.uint32(16)
        b4 = np.ascontiguousarray(word.astype('<u4')).view(np.uint8).reshape(len(idx), 4)
        k = min(4, n - 4 * j)
        out[:, 4 * j:4 * j + k] = b4[:, :k]
    return out


WIDTH = dict(b8=1, u8=1, i8=1, b16=2, u16=2, i16=2, b32=4, b64=8, b96=12, b128=16)


def _sh16(x):
    x &= 0xffff
    return x - 0x10000 if x & 0x8000 else x


def _sh(x):
    return x - (1 << 32) if x & 0x80000000 else x


def _ovf_add(x, y):
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


SALU2 = {
    's_or_b32': lambda x, y, c: ((x | y) & M32, int(((x | y) & M32) != 0)),
    's_and_b32': lambda x, y, c: ((x & y) & M32, int(((x & y) & M32) != 0)),
    's_xor_b32': lambda x, y, c: ((x ^ y) & M32, int(((x ^ y) & M32) != 0)),
    's_and_not1_b32': lambda x, y, c: ((x & ~y) & M32, int(((x & ~y) & M32) != 0)),
    's_or_not1_b32': lambda x, y, c: ((x | ~y) & M32, int(((x | ~y) & M32) != 0)),
    's_add_i32': lambda x, y, c: ((x + y) & M32, _ovf_add(x, y)),
    's_sub_i32': lambda x, y, c: ((x - y) & M32, _ovf_sub(x, y)),
    's_sub_u32': lambda x, y, c: ((x - y) & M32, int(x < y)),
    's_nor_b32': lambda x, y, c: ((~(x | y)) & M32, int(((~(x | y)) & M32) != 0)),
    's_mul_hi_u32': lambda x, y, c: (((x * y) >> 32) & M32, None),
    's_mul_hi_i32': lambda x, y, c: (((_sh(x) * _sh(y)) >> 32) & M32, None),
    's_lshl_b64': lambda x, y, c: ((x << (y & 63)) & ((1 << 64) - 1), int(((x << (y & 63)) & ((1 << 64) - 1)) != 0)),
    's_bfe_u32': lambda x, y, c: _bfe(x, y, False),
    's_bfe_i32': lambda x, y, c: _bfe(x, y, True),
    's_add_u32': lambda x, y, c: ((x + y) & M32, int(x + y > M32)),
    's_addc_u32': lambda x, y, c: ((x + y + c) & M32, int(x + y + c > M32)),
    's_mul_i32': lambda x, y, c: ((_sh(x) * _sh(y)) & M32, None),
    # ISA: D = (S0 < S1) ? S0 : S1 (min) / (S0 >= S1) ? S0 : S1 (max); SCC = (D == S0)
    's_min_i32': lambda x, y, c: ((x if _sh(x) < _sh(y) else y), int((x if _sh(x) < _sh(y) else y) == x)),
    's_max_i32': lambda x, y, c: ((x if _sh(x) >= _sh(y) else y), int((x if _sh(x) >= _sh(y) else y) == x)),
    's_lshl_b32': lambda x, y, c: ((x << (y & 31)) & M32, int(((x << (y & 31)) & M32) != 0)),
    's_lshr_b32': lambda x, y, c: (x >> (y & 31), int((x >> (y & 31)) != 0)),
    's_ashr_i32': lambda x, y, c: ((_sh(x) >> (y & 31)) & M32, int(((_sh(x) >> (y & 31)) & M32) != 0)),
}
NOPS = ('s_delay_alu', 's_waitcnt_depctr', 's_waitcnt', 's_waitcnt_vscnt', 's_nop', 's_clause', 's_sleep', 'buffer_gl0_inv', 'buffer_gl1_inv',
        's_sendmsg', 's_set_inst_prefetch_distance', 's_code_end')


class Exec:
    def __init__(s, prog):
        s.prog = prog
        s.addr2i = {a: i for i, (a, _, _, _) in enumerate(prog)}
        s.cache = {}

    def compile(s, i):
        c = s.cache.get(i)
        if c is None:
            c = s.cache[i] = s._compile(i)
        return c

    def _compile(s, i):
        addr, op, args, size = s.prog[i]
        try:
            return s._decode(addr, op, args, size)
        except NotImplementedError as e:
            raise NotImplementedError('%s @%x: %s %s' % (e, addr, op, args))

    def _decode(s, addr, op, args, size):
        nxt = addr + size
        if op.startswith('v_dual_'):
            a, b = args.split(' :: ')
            oa, ob = [op, a], b.split(None, 1)
            ca = s._decode(addr, oa[0].replace('v_dual_', 'v_'), oa[1] + (', vcc_lo' if 'cndmask' in oa[0] else ''), size)
            cb = s._decode(addr, ob[0].replace('v_dual_', 'v_'), ob[1] + (', vcc_lo' if 'cndmask' in ob[0] else ''), size)

            def dual(w):
                w.RV = w.V.copy()
                ca(w)
                cb(w)
                w.RV = w.V
            return dual
        o, mods = split_args(args)
        base = re.sub(r'_(e32|e64)$', '', op)
        if base in NOPS:
            return lambda w: None
        if base == 's_endpgm':
            return lambda w: setattr(w, 'state', 'done')
        if base == 's_barrier':
            return lambda w: setattr(w, 'state', 'barrier')
        branchy = base == 's_branch' or base.startswith('s_cbranch_')
        P = None if branchy else [parse_op(x) for x in o]
        if base in SALU2:
            f = SALU2[base]
            d, a, b = P

            def salu(w):
                r, scc = f(w.rs(a), w.rs(b), w.scc)
                w.ws(d, r)
                if scc is not None:
                    w.scc = scc
            return salu
        if base == 's_mulk_i32':
            return lambda w: w.ws(P[0], (_sh(w.rs(P[0])) * _sh16(w.rs(P[1]))) & M32)
        if base == 's_bfe_i64':
            def bfe64(w):
                x, y = w.rs(P[1]), w.rs(P[2])
                off, wd = y & 63, (y >> 16) & 0x7f
                r = (x >> off) & ((1 << wd) - 1) if wd else 0
                if wd and (r >> (wd - 1)) & 1:
                    r -= 1 << wd
                r &= (1 << 64) - 1
                w.ws(P[0], r)
                w.scc = int(r != 0)
            return bfe64
        if base == 's_movk_i32':
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
                src = w.rs(P[1])                 # read the source before the destination is overwritten (s6, s6 form)
                w.ws(P[0], old)
                w.S[EXEC] = (src | old) & M32
                w.scc = int(w.S[EXEC] != 0)
            return orsav
        if base in ('s_mov_b32', 's_mov_b64'):
            return lambda w: w.ws(P[0], w.rs(P[1]))
        if base == 's_abs_i32':          # D = |S0| (signed); SCC = (D != 0)
            def sabs(w):
                r = abs(_sh(w.rs(P[1]))) & M32; w.ws(P[0], r); w.scc = int(r != 0)
            return sabs
        if base == 's_bcnt1_i32_b32':    # D = popcount(S0); SCC = (D != 0)
            def sbcnt(w):
                r = bin(w.rs(P[1]) & M32).count('1'); w.ws(P[0], r); w.scc = int(r != 0)
            return sbcnt
        if base == 's_cselect_b32':
            return lambda w: w.ws(P[0], w.rs(P[1]) if w.scc else w.rs(P[2]))
        if (m := re.fullmatch(r's_cmpk?_(eq|lg|gt|ge|lt|le)_(i|u)(32|64)', base)):
            c, ty, is32 = m[1], m[2], m[3] == '32'

            def scmp(w):
                x, y = w.rs(P[0]), w.rs(P[1])
                if ty == 'i' and is32:
                    x, y = _sh(x), _sh(y)
                w.scc = int({'eq': x == y, 'lg': x != y, 'gt': x > y, 'ge': x >= y, 'lt': x < y, 'le': x <= y}[c])
            return scmp
        if base in ('s_and_saveexec_b32', 's_and_not1_saveexec_b32'):
            inv = 'not1' in base

            def sav(w):
                old = w.S[EXEC]
                src = w.rs(P[1])
                w.ws(P[0], old)
                new = ((src & ~old) if inv else (src & old)) & M32
                w.S[EXEC] = new
                w.scc = int(new != 0)
            return sav
        if base == 's_getpc_b64':
            return lambda w: w.ws(P[0], nxt)
        if base == 's_swappc_b64':
            def swap(w):
                tgt = w.rs(P[1])
                w.ws(P[0], nxt)
                w.next = s.addr2i[tgt]
            return swap
        if base == 's_setpc_b64':
            def setpc(w):
                w.next = s.addr2i[w.rs(P[0])]
            return setpc
        if branchy:
            imm = int(o[0], 0)
            imm -= 65536 if imm >= 32768 else 0
            tgt = s.addr2i[addr + 4 + 4 * imm]
            cond = {'s_branch': lambda w: True, 's_cbranch_execz': lambda w: w.S[EXEC] == 0,
                    's_cbranch_execnz': lambda w: w.S[EXEC] != 0, 's_cbranch_scc0': lambda w: not w.scc,
                    's_cbranch_scc1': lambda w: bool(w.scc), 's_cbranch_vccz': lambda w: w.S[VCC] == 0,
                    's_cbranch_vccnz': lambda w: w.S[VCC] != 0}[base]

            def br(w):
                if cond(w):
                    w.next = tgt
            return br
        if base.startswith('s_load_b'):
            n = int(base[8:]) // 8

            def sld(w):
                a = w.rs(P[1])
                off = 0 if P[2]['k'] == 'null' else w.rs(P[2])
                raw = w.g.read(np.array([a + off], np.uint64), n)[0]
                words = np.ascontiguousarray(raw).view(np.uint32)
                for j in range(n // 4):
                    w.S[P[0]['i'] + j] = int(words[j])
            return sld
        if base.startswith(('global_', 'flat_', 'scratch_')):
            return s._mem(base, P, mods)
        if base.startswith('ds_'):
            return s._ds(base[3:], P, mods)
        if base == 'v_wmma_f32_16x16x16_f16':
            def wmma(w):
                def frag(o):
                    regs = np.ascontiguousarray(np.stack([w.RV[o['i'] + j] for j in range(8)], 1)[:16])
                    return regs.view(np.float16).reshape(16, 16).astype(np.float32)
                A, B = frag(P[1]), frag(P[2])
                C = np.zeros((16, 16), np.float32)
                for j in range(8):
                    reg = w.RV[P[3]['i'] + j].view(np.float32)
                    C[2 * j, :] = reg[:16]
                    C[2 * j + 1, :] = reg[16:]
                D = (A.astype(np.float64) @ B.astype(np.float64).T + C).astype(np.float32)
                for j in range(8):
                    w.V[P[0]['i'] + j] = np.ascontiguousarray(np.concatenate([D[2 * j, :], D[2 * j + 1, :]])).view(np.uint32)
            return wmma
        return s._valu(base, P, mods)

    def _mem(s, base, P, mods):
        kind, _, rest = base.partition('_')
        off = mods.get('offset', 0)
        if rest == 'atomic_cmpswap_b32':
            # ISA: tmp = MEM[ADDR]; src = DATA[31:0]; cmp = DATA[63:32];
            #      MEM[ADDR] = (tmp == cmp) ? src : tmp; RETURN_DATA = tmp   (returned only under glc)
            # With glc the destination VGPR is present and shifts the other operands along.
            glc = bool(mods.get('glc'))
            vd, va, data = (P[0], P[1], P[2]) if glc else (None, P[0], P[1])
            sa = P[3] if glc else P[2]

            def atomic(w):
                idx = np.nonzero(w.em)[0]
                if not len(idx):
                    return
                if sa is not None and sa['k'] == 's':
                    a = np.int64(w.rs(sa)) + w.RV[va['i']].astype(np.int64) + off
                else:
                    a = w.r64(va).astype(np.int64) + off
                # Copy both sources up front: the destination commonly aliases the data pair
                # (global_atomic_cmpswap_b32 v0, v2, v[0:1]), and writing the per-lane result below
                # would otherwise clobber src for the lanes not yet processed.
                src, cmp_ = w.RV[data['i']].copy(), w.RV[data['i'] + 1].copy()
                # Lanes are serialized in ascending order: two lanes hitting one address must not
                # both see the original value, or a lock built on this would hand out two winners.
                for l in idx:
                    au = np.array([a[l]], np.uint64)
                    tmp = ld_val(w.g.read(au, 4), 4, False).reshape(-1)[0]
                    new = src[l] if tmp == cmp_[l] else tmp
                    w.g.write(au, np.array([new], np.uint32).view(np.uint8).reshape(1, 4))
                    if vd is not None:
                        w.V[vd['i'], l] = tmp
            return atomic
        m = re.fullmatch(r'(load|store)_(\w+)', rest)
        if not m:
            raise NotImplementedError('mem ' + rest)
        ld, t = m[1] == 'load', m[2]
        d16 = None
        if t.startswith('d16_hi_'):
            d16, t = 'hi', t[7:]
        elif t.startswith('d16_'):
            d16, t = 'lo', t[4:]
        n = WIDTH[t]
        sign = t.startswith('i')
        if ld:
            vd, va = P[0], P[1]
        elif kind == 'scratch':
            va, vd = P[0], P[1]
        else:
            va, vd = P[0], P[1]
        sa = P[2] if kind == 'global' else None

        def ea(w):
            if kind == 'scratch':
                return (w.RV[va['i']].astype(np.int64) if va['k'] == 'v' else np.zeros(32, np.int64)) + off
            if kind == 'global' and sa['k'] == 's':
                return np.int64(w.rs(sa)) + w.RV[va['i']].astype(np.int64) + off
            return w.r64(va).astype(np.int64) + off

        def run(w):
            idx = np.nonzero(w.em)[0]
            if not len(idx):
                return
            a = ea(w)
            if kind == 'scratch':
                for l in idx:
                    p = int(a[l])
                    assert 0 <= p and p + n <= 512, 'scratch oob %d' % p
                    w.privmax = max(w.privmax, p + n)
                    if ld:
                        v = ld_val(w.priv[l, p:p + n][None, :], n, sign).reshape(-1)
                        for j in range(len(v)):
                            w.V[vd['i'] + j, l] = v[j]
                    else:
                        w.priv[l, p:p + n] = stbytes(w, vd, n, [l])[0]
                return
            au = a[idx].astype(np.uint64)
            if kind == 'flat':
                hi = au >> np.uint64(32)
                lo = (au & np.uint64(M32)).astype(np.int64)
                raw = np.zeros((len(idx), n), np.uint8)
                sels = {'lds': hi == SH_HI, 'priv': hi == PR_HI}
                sels['glob'] = ~(sels['lds'] | sels['priv'])
                for name, sel in sels.items():
                    if not sel.any():
                        continue
                    if name == 'lds':
                        assert (lo[sel] + n <= len(w.lds)).all(), 'flat lds oob'
                        if ld:
                            raw[sel] = w.lds[lo[sel][:, None] + np.arange(n)]
                        else:
                            w.lds[lo[sel][:, None] + np.arange(n)] = stbytes(w, vd, n, idx[sel])
                    elif name == 'priv':
                        for q, l in zip(np.nonzero(sel)[0], idx[sel]):
                            if ld:
                                raw[q] = w.priv[l, lo[q]:lo[q] + n]
                            else:
                                w.priv[l, lo[q]:lo[q] + n] = stbytes(w, vd, n, [l])[0]
                    else:
                        if ld:
                            raw[sel] = w.g.read(au[sel], n)
                        else:
                            w.g.write(au[sel], stbytes(w, vd, n, idx[sel]))
            else:
                if ld:
                    raw = w.g.read(au, n)
                else:
                    w.g.write(au, stbytes(w, vd, n, idx))
            if ld:
                v = ld_val(raw, n, sign)
                if d16:
                    cur = w.V[vd['i'], idx]
                    w.V[vd['i'], idx] = ((cur & np.uint32(0xffff)) | (v << np.uint32(16))) if d16 == 'hi' else ((cur & np.uint32(0xffff0000)) | (v & np.uint32(0xffff)))
                    return
                if v.ndim == 1:
                    v = v[:, None]
                for j in range(v.shape[1]):
                    w.V[vd['i'] + j, idx] = v[:, j]
        return run

    def _ds(s, rest, P, mods):
        off = mods.get('offset', 0)
        if rest == 'bpermute_b32':
            def bperm(w):
                src = ((w.RV[P[1]['i']].astype(np.int64) + off) >> 2) & 31
                data = w.RV[P[2]['i']].copy()
                w.wv(P[0], data[src])
            return bperm
        o0, o1 = mods.get('offset0', 0), mods.get('offset1', 0)
        m = re.fullmatch(r'(load|store)_(2addr_)?(?:stride64_)?(\w+?)(_d16_hi)?', rest)
        if not m:
            raise NotImplementedError('ds ' + rest)
        ld, two, t, d16 = m[1] == 'load', bool(m[2]), m[3], bool(m[4])
        n = WIDTH[t]

        def run(w):
            idx = np.nonzero(w.em)[0]
            if not len(idx):
                return
            if ld:
                vd, va = P[0], P[1]
            else:
                va, vd = P[0], P[1]
            a = w.RV[va['i']][idx].astype(np.int64)
            if two:
                for j, oo in ((0, o0), (1, o1)):
                    p = (a + oo * n) & M32
                    assert (p + n <= len(w.lds)).all(), 'lds oob'
                    if ld:
                        v = ld_val(w.lds[p[:, None] + np.arange(n)], n, False)
                        if v.ndim == 1:
                            v = v[:, None]
                        for k in range(v.shape[1]):                      # element j occupies dwords [j*n/4, (j+1)*n/4)
                            w.V[vd['i'] + j * v.shape[1] + k, idx] = v[:, k]
                    else:
                        w.lds[p[:, None] + np.arange(n)] = stbytes(w, P[1 + j], n, idx)
                return
            p = (a + off) & M32          # DS addresses wrap in 32 bits
            assert (p >= 0).all() and (p + n <= len(w.lds)).all(), 'lds oob %s' % p[:4]
            if ld:
                v = ld_val(w.lds[p[:, None] + np.arange(n)], n, False)
                if d16:
                    w.V[vd['i'], idx] = (w.V[vd['i'], idx] & np.uint32(0xffff)) | (v << np.uint32(16))
                elif v.ndim == 1:
                    w.V[vd['i'], idx] = v
                else:
                    for j in range(v.shape[1]):
                        w.V[vd['i'] + j, idx] = v[:, j]
            else:
                w.lds[p[:, None] + np.arange(n)] = stbytes(w, vd, n, idx, hi=d16)
        return run

    def _valu(s, b, P, mods):
        def finish_f32(x):
            # Saturation of finite values. NaN handling remains NumPy's
            # propagation behavior pending explicit DX10-mode modeling.
            if mods.get('clamp'):
                x = np.clip(x, np.float32(0), np.float32(1))
                x = np.where(x == 0, np.float32(0), x)
            return fbits(x)
        BIN = {'v_add_nc_u32': lambda x, y: x + y, 'v_sub_nc_u32': lambda x, y: x - y, 'v_and_b32': lambda x, y: x & y,
               'v_or_b32': lambda x, y: x | y, 'v_xor_b32': lambda x, y: x ^ y,
               'v_mul_lo_u32': lambda x, y: (x.astype(np.uint64) * y.astype(np.uint64)) & np.uint64(M32),
               'v_mul_hi_u32': lambda x, y: (x.astype(np.uint64) * y.astype(np.uint64)) >> np.uint64(32),
               'v_mul_u32_u24': lambda x, y: ((x & np.uint32(0xffffff)).astype(np.uint64) * (y & np.uint32(0xffffff)).astype(np.uint64)) & np.uint64(M32),
               'v_max_u32': np.maximum, 'v_min_u32': np.minimum,
               'v_lshlrev_b32': lambda x, y: y << (x & np.uint32(31)), 'v_lshrrev_b32': lambda x, y: y >> (x & np.uint32(31)),
               'v_ashrrev_i32': lambda x, y: (y.view(np.int32) >> (x & np.uint32(31)).astype(np.int32)).view(np.uint32),
               'v_min_i32': lambda x, y: np.minimum(x.view(np.int32), y.view(np.int32)).view(np.uint32),
               'v_max_i32': lambda x, y: np.maximum(x.view(np.int32), y.view(np.int32)).view(np.uint32)}
        if b in BIN:
            f = BIN[b]
            return lambda w: w.wv(P[0], u32(f(w.r32(P[1]), w.r32(P[2]))))
        BIN16 = {'v_add_nc_u16': lambda x, y: x + y, 'v_sub_nc_u16': lambda x, y: x - y, 'v_mul_lo_u16': lambda x, y: x * y,
                 'v_min_u16': np.minimum, 'v_lshlrev_b16': lambda x, y: y << (x & np.uint16(15)),
                 'v_lshrrev_b16': lambda x, y: y >> (x & np.uint16(15))}
        if b in BIN16:
            f = BIN16[b]
            return lambda w: w.wv(P[0], u32(f(h(w.r32(P[1])), h(w.r32(P[2])))) & np.uint32(0xffff))
        if b == 'v_ashrrev_i16':
            return lambda w: w.wv(P[0], u32((sh16(w.r32(P[2])) >> (h(w.r32(P[1])) & np.uint16(15)).astype(np.int16)).view(np.uint16)))
        F = {'v_add_f32': lambda x, y: x + y, 'v_sub_f32': lambda x, y: x - y, 'v_mul_f32': lambda x, y: x * y,
             'v_max_f32': np.fmax, 'v_min_f32': np.fmin}
        if b in F:
            f = F[b]
            return lambda w: w.wv(P[0], finish_f32(f(w.rf(P[1]), w.rf(P[2]))))
        if b == 'v_mul_f16':
            return lambda w: w.wv(P[0], hbits(w.rh(P[1]) * w.rh(P[2])))
        if b == 'v_fma_f32':
            return lambda w: w.wv(P[0], finish_f32(fma32(w.rf(P[1]), w.rf(P[2]), w.rf(P[3]))))
        if b == 'v_fmac_f32':
            return lambda w: w.wv(P[0], fbits(fma32(w.rf(P[1]), w.rf(P[2]), w.RV[P[0]['i']].view(np.float32))))
        if b == 'v_med3_f32':
            return lambda w: w.wv(P[0], fbits(np.sort(np.stack([w.rf(P[1]), w.rf(P[2]), w.rf(P[3])]), axis=0)[1]))
        if b == 'v_mov_b32':
            return lambda w: w.wv(P[0], w.r32(P[1]))
        if b == 'v_readfirstlane_b32':
            def rfl(w):
                em = w.em
                l = int(np.argmax(em)) if em.any() else 0
                w.S[P[0]['i']] = int(w.RV[P[1]['i']][l])
            return rfl
        if b == 'v_cndmask_b32':
            return lambda w: w.wv(P[0], np.where(w.lanes(w.mask(P[3])), w.r32(P[2]), w.r32(P[1])))
        U = {'v_rcp_f32': lambda x: np.float32(1) / x, 'v_rcp_iflag_f32': lambda x: np.float32(1) / x,
             'v_rsq_f32': lambda x: np.float32(1) / np.sqrt(x), 'v_log_f32': np.log2, 'v_rndne_f32': np.rint, 'v_floor_f32': np.floor}
        if b in U:
            f = U[b]
            return lambda w: w.wv(P[0], fbits(f(w.rf(P[1])).astype(np.float32)))
        if b == 'v_ldexp_f32':
            return lambda w: w.wv(P[0], fbits(np.ldexp(w.rf(P[1]), np.clip(w.r32(P[2]).view(np.int32), -400, 400)).astype(np.float32)))
        if b == 'v_cvt_i32_f32':
            return lambda w: w.wv(P[0], cvt_i32(w.rf(P[1])))
        if b == 'v_cvt_u32_f32':
            return lambda w: w.wv(P[0], cvt_u32(w.rf(P[1])))
        if b == 'v_cvt_f32_u32':
            return lambda w: w.wv(P[0], fbits(w.r32(P[1]).astype(np.float32)))
        if b == 'v_cvt_f32_ubyte0':
            return lambda w: w.wv(P[0], fbits((w.r32(P[1]) & np.uint32(0xff)).astype(np.float32)))
        if b == 'v_cvt_f32_f16':
            return lambda w: w.wv(P[0], fbits(w.rh(P[1]).astype(np.float32)))
        if b == 'v_cvt_f16_f32':
            return lambda w: w.wv(P[0], hbits(w.rf(P[1]).astype(np.float16)))
        if b in ('v_fma_mix_f32', 'v_fma_mixlo_f16'):
            osel, ohi = mods.get('op_sel', [0, 0, 0]), mods.get('op_sel_hi', [0, 0, 0])

            def mix(w):
                vals = []
                for k in range(3):
                    raw = w.r32(dict(P[1 + k], neg=False, abs=False))
                    if ohi[k]:
                        hv = (raw >> np.uint32(16)) if osel[k] else raw
                        if MIX_F16_INPUT_FLUSH:
                            hv = np.where((hv & np.uint32(0x7c00)) == 0,
                                          hv & np.uint32(0x8000), hv)
                        x = (hv & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float32)
                    else:
                        x = raw.view(np.float32)
                    if P[1 + k]['abs']:
                        x = np.abs(x)
                    if P[1 + k]['neg']:
                        x = -x
                    vals.append(x)
                r = fma32(*vals)
                if b == 'v_fma_mix_f32':
                    w.wv(P[0], fbits(r))
                else:
                    w.wv(P[0], (w.V[P[0]['i']] & np.uint32(0xffff0000)) | hbits(r.astype(np.float16)))
            return mix
        if b == 'v_perm_b32':
            def perm(w):
                s0, s1, sel = w.r32(P[1]), w.r32(P[2]), w.r32(P[3])
                out = np.zeros(32, np.uint32)
                for k in range(4):
                    sk = (sel >> np.uint32(8 * k)) & np.uint32(0xff)
                    byte = np.zeros(32, np.uint32)
                    for q in range(8):
                        src = ((s1 if q < 4 else s0) >> np.uint32(8 * (q % 4))) & np.uint32(0xff)
                        byte = np.where(sk == q, src, byte)
                    byte = np.where(sk >= 13, np.uint32(0xff), byte)
                    assert not ((sk >= 8) & (sk <= 11)).any(), 'perm sign-extend selector'
                    out |= byte << np.uint32(8 * k)
                w.wv(P[0], out)
            return perm
        if b == 'v_lshl_add_u32':
            return lambda w: w.wv(P[0], (w.r32(P[1]) << (w.r32(P[2]) & np.uint32(31))) + w.r32(P[3]))
        if b == 'v_lshl_or_b32':
            return lambda w: w.wv(P[0], (w.r32(P[1]) << (w.r32(P[2]) & np.uint32(31))) | w.r32(P[3]))
        if b == 'v_add3_u32':
            return lambda w: w.wv(P[0], w.r32(P[1]) + w.r32(P[2]) + w.r32(P[3]))
        if b == 'v_or3_b32':
            return lambda w: w.wv(P[0], w.r32(P[1]) | w.r32(P[2]) | w.r32(P[3]))
        if b == 'v_and_or_b32':
            return lambda w: w.wv(P[0], (w.r32(P[1]) & w.r32(P[2])) | w.r32(P[3]))
        if b == 'v_mad_u32_u24':
            return lambda w: w.wv(P[0], ((w.r32(P[1]) & np.uint32(0xffffff)) * (w.r32(P[2]) & np.uint32(0xffffff))) + w.r32(P[3]))
        if b == 'v_bfe_u32':
            def bfe(w):
                x, off, wd = w.r32(P[1]), w.r32(P[2]) & np.uint32(31), w.r32(P[3]) & np.uint32(31)
                w.wv(P[0], (x >> off) & ((np.uint32(1) << wd) - np.uint32(1)))
            return bfe
        if b == 'v_bfe_i32':
            def bfei(w):
                x, off, wd = w.r32(P[1]), w.r32(P[2]) & np.uint32(31), w.r32(P[3]) & np.uint32(31)
                v = ((x >> off) & ((np.uint32(1) << wd) - np.uint32(1))).astype(np.int64)
                sb = np.int64(1) << np.maximum(wd.astype(np.int64) - 1, 0)
                v = np.where((wd > 0) & ((v & sb) != 0), v - (sb << 1), v)
                w.wv(P[0], v.astype(np.uint32))
            return bfei
        if b == 'v_lshlrev_b64':
            return lambda w: w.wv64(P[0], w.r64(P[2]) << (w.r32(P[1]).astype(np.uint64) & np.uint64(63)))
        if b == 'v_mad_u64_u32':
            return lambda w: w.wv64(P[0], w.r32(P[2]).astype(np.uint64) * w.r32(P[3]).astype(np.uint64) + w.r64(P[4]))
        if b in ('v_mul_i32_i24', 'v_mad_i32_i24'):      # signed 24-bit multiply (low 32 bits), optional add
            def mul24(w):
                s24 = lambda x: (((x.astype(np.int64) & 0xffffff) ^ 0x800000) - 0x800000)
                r = s24(w.r32(P[1])) * s24(w.r32(P[2]))
                if b == 'v_mad_i32_i24': r = r + w.r32(P[3]).view(np.int32).astype(np.int64)
                w.wv(P[0], (r & M32).astype(np.uint32))
            return mul24
        if b == 'v_fmac_f16':          # D.f16 = fma(S0, S1, D); upper half written as zero (same convention as v_mul_f16 here)
            return lambda w: w.wv(P[0], hbits((w.rh(P[1]).astype(np.float64) * w.rh(P[2]).astype(np.float64) + w.V[P[0]['i']].astype(np.uint16).view(np.float16).astype(np.float64)).astype(np.float16)))
        if b == 'v_max3_f32':
            return lambda w: w.wv(P[0], fbits(np.maximum(np.maximum(w.rf(P[1]), w.rf(P[2])), w.rf(P[3]))))
        if b == 'v_med3_i32':
            return lambda w: w.wv(P[0], np.sort(np.stack([w.r32(P[1]).view(np.int32), w.r32(P[2]).view(np.int32), w.r32(P[3]).view(np.int32)]), axis=0)[1].view(np.uint32))
        if b == 'v_frexp_exp_i32_f32':
            def frexp32e(w):
                x = w.rf(P[1]); e = np.frexp(x)[1].astype(np.int32)
                w.wv(P[0], np.where(np.isfinite(x), e, 0).astype(np.int32).view(np.uint32))
            return frexp32e
        if b == 'v_readlane_b32':      # SGPR D = VGPR S0[lane S1]; ignores EXEC
            return lambda w: w.ws(P[0], int(w.RV[P[1]['i']][w.rs(P[2]) & 31]))
        if b == 'v_writelane_b32':     # VGPR D[lane S1] = S0 (scalar); ignores EXEC
            def wl(w):
                w.V[P[0]['i']][w.rs(P[2]) & 31] = np.uint32(w.rs(P[1]) & M32)
            return wl
        if b == 'v_lshrrev_b64':
            return lambda w: w.wv64(P[0], w.r64(P[2]) >> (w.r32(P[1]).astype(np.uint64) & np.uint64(63)))
        if b == 'v_add_f64':
            return lambda w: w.wv64(P[0], (w.r64(P[1]).view(np.float64) + w.r64(P[2]).view(np.float64)).view(np.uint64))
        if b == 'v_cvt_f32_f64':
            return lambda w: w.wv(P[0], fbits(w.r64(P[1]).view(np.float64).astype(np.float32)))
        if (m64 := re.fullmatch(r'v_cmp(x?)_(eq|ne|lt|le|gt|ge)_(u64|i64)', b)):
            def cmp64(w):
                dt = np.uint64 if m64[3] == 'u64' else np.int64
                bits = w.bits(CMP[m64[2]](w.r64(P[-2]).view(dt), w.r64(P[-1]).view(dt)))
                if m64[1]: w.S[EXEC] = bits
                else: w.wmask(P[0], bits)
            return cmp64
        if b == 'v_dot2acc_f32_f16':   # D.f32 += S0.f16[0]*S1.f16[0] + S0.f16[1]*S1.f16[1]; modeled as one rounding (hardware internals unverified)
            def dot2acc(w):
                a, c = w.r32(P[1]), w.r32(P[2])
                h = lambda x, sh: ((x >> np.uint32(sh)) & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float64)
                r = h(a, 0) * h(c, 0) + h(a, 16) * h(c, 16) + w.rf(P[0]).astype(np.float64)
                w.wv(P[0], fbits(r.astype(np.float32)))
            return dot2acc
        if b == 'v_minmax_i32':      # LLVM/ISA: max(min(S0, S1), S2), signed
            return lambda w: w.wv(P[0], np.maximum(np.minimum(w.r32(P[1]).view(np.int32), w.r32(P[2]).view(np.int32)),
                                                   w.r32(P[3]).view(np.int32)).view(np.uint32))
        if b == 'v_cvt_f32_i32':
            return lambda w: w.wv(P[0], fbits(w.r32(P[1]).view(np.int32).astype(np.float32)))
        if b == 'v_cvt_f32_ubyte1':
            return lambda w: w.wv(P[0], fbits(((w.r32(P[1]) >> np.uint32(8)) & np.uint32(0xff)).astype(np.float32)))
        if b == 'v_cvt_f64_f32':
            return lambda w: w.wv64(P[0], w.rf(P[1]).astype(np.float64).view(np.uint64))
        if b == 'v_frexp_mant_f32':  # mantissa in [0.5, 1) with the input's sign; zero/inf/nan pass through
            def frexpm(w):
                x = w.rf(P[1]); m = np.frexp(x)[0].astype(np.float32)
                w.wv(P[0], fbits(np.where(np.isfinite(x), m, x)))
            return frexpm
        if b == 'v_frexp_exp_i32_f64':   # unbiased exponent for a [0.5, 1) mantissa; 0 for zero/inf/nan
            def frexpe(w):
                x = w.r64(P[1]).view(np.float64); e = np.frexp(x)[1].astype(np.int32)
                w.wv(P[0], np.where(np.isfinite(x), e, 0).astype(np.int32).view(np.uint32))
            return frexpe
        if b in ('v_sub_co_ci_u32', 'v_subrev_co_ci_u32'):
            def subci(w):
                a, c = w.r32(P[2]).astype(np.int64), w.r32(P[3]).astype(np.int64)
                if b == 'v_subrev_co_ci_u32': a, c = c, a
                r = a - c - w.lanes(w.mask(P[4])).astype(np.int64)
                w.wv(P[0], (r & M32).astype(np.uint32))
                w.wmask(P[1], w.bits(r < 0))
            return subci
        if b == 'v_add_co_u32':
            def addco(w):
                r = w.r32(P[2]).astype(np.uint64) + w.r32(P[3]).astype(np.uint64)
                w.wv(P[0], u32(r & np.uint64(M32)))
                w.wmask(P[1], w.bits((r >> np.uint64(32)) != 0))
            return addco
        if b == 'v_add_co_ci_u32':
            def addci(w):
                r = w.r32(P[2]).astype(np.uint64) + w.r32(P[3]).astype(np.uint64) + w.lanes(w.mask(P[4])).astype(np.uint64)
                w.wv(P[0], u32(r & np.uint64(M32)))
                w.wmask(P[1], w.bits((r >> np.uint64(32)) != 0))
            return addci
        if b == 'v_div_scale_f32':
            def dsc(w):
                d, vcc = div_scale(w.rf(P[2]), w.rf(P[3]), w.rf(P[4]))
                w.wv(P[0], fbits(d))
                w.wmask(P[1], w.bits(vcc))
            return dsc
        if b == 'v_div_fmas_f32':
            def fmas(w):
                r = fma32(w.rf(P[1]), w.rf(P[2]), w.rf(P[3]))
                w.wv(P[0], fbits(np.where(w.lanes(w.S[VCC]), np.ldexp(r.astype(np.float64), 32).astype(np.float32), r)))
            return fmas
        if b == 'v_div_fixup_f32':
            return lambda w: w.wv(P[0], fbits(div_fixup(w.rf(P[1]), w.rf(P[2]), w.rf(P[3]))))
        if (m := re.fullmatch(r'v_cmpx?_class_f32', b)):
            def cls(w):
                bits = w.bits(((f32_class(w.r32(P[-2]).view(np.float32)) & w.r32(P[-1])) != 0))
                if b.startswith('v_cmpx'):
                    w.S[EXEC] = bits
                else:
                    w.wmask(P[0], bits)
            return cls
        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(f32|f16|i32|u32|i16|u16)', b)
        if m:
            x, cnd, ty = m[1], m[2], m[3]
            f = CMP[cnd]

            def cmpv(w):
                bits = w.bits(f(typed(w, P[-2], ty), typed(w, P[-1], ty)))
                if x:
                    w.S[EXEC] = bits
                else:
                    w.wmask(P[0], bits)
            return cmpv
        # ---- packed f16 math (op_sel selects the source half for the low result, op_sel_hi for the high result)
        def pk(f, nsrc):
            osel, ohi = mods.get('op_sel', [0] * 3), mods.get('op_sel_hi', [1] * 3)

            def run(w):
                halves = []
                for k in range(nsrc):
                    raw = w.r32(dict(P[1 + k], neg=False, abs=False), True)     # float inline constants are f16 here
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
        raise NotImplementedError('valu ' + b)


def run_workgroup(prog, gmem, lds_size, nthreads, init_sgprs, max_steps=200_000_000, trace=None, stop_at=None, snap=None, poison=False):
    ex = Exec(prog)
    lds = np.zeros(lds_size, np.uint8)
    waves = []
    for wi in range(nthreads // 32):
        w = Wave(gmem, lds, wi, wi * 32)
        if poison:
            v0 = w.V[0].copy()
            w.V[:] = np.uint32(0xDEADBEEF)
            w.V[0] = v0
            for i in range(2, 106):
                w.S[i] = 0xDEADBEEF
            w.S[VCC] = 0xDEADBEEF
            w.scc = 2                      # 2 = unwritten (poison); real SCC is 0/1
        for k, v in init_sgprs.items():
            w.S[k] = v
        waves.append(w)
    steps = 0
    trace_seen = set()
    while any(w.state != 'done' for w in waves):
        moved = False
        for w in waves:
            while w.state == 'run':
                if stop_at is not None and ex.prog[w.pc][0] == stop_at and w.wid not in snap:
                    snap[w.wid] = (w.V.copy(), list(w.S), w.scc)
                    w.state = 'done'
                    break
                w.next = w.pc + 1
                if trace is not None and w.wid == 0 and w.pc not in trace_seen:
                    trace_seen.add(w.pc)
                    trace.append(ex.prog[w.pc][0])
                try:
                    ex.compile(w.pc)(w)
                except (AssertionError, MemFault) as err:
                    a_, op_, args_, _ = ex.prog[w.pc]
                    raise type(err)('@%x wave%d %s %s :: %s' % (a_, w.wid, op_, args_, err)) from None
                w.pc = w.next
                steps += 1
                w.n += 1
                moved = True
                if steps > max_steps:
                    raise RuntimeError('step limit')
                if steps % 1000000 == 0:
                    import sys, time
                    print('progress steps=%d pcs=%s' % (steps, [hex(ex.prog[x.pc][0]) for x in waves]), file=sys.stderr, flush=True)
        bar = [w for w in waves if w.state == 'barrier']
        if bar and all(w.state in ('barrier', 'done') for w in waves):
            for w in bar:
                w.state = 'run'
        elif not moved and any(w.state != 'done' for w in waves):
            raise RuntimeError('deadlock: ' + str([w.state for w in waves]))
    return dict(steps=steps, lds=lds, waves=waves)
