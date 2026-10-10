"""gfx1030 front end for the gfx1100 CPU emulator (E:/dlss 5/rdna2/emu/gfx11emu.py).

Runs a translated gfx1030 code object (.co) instead of the gfx1100 disassembly: llvm-objdump
(--mcpu=gfx1030) text is renamed to the gfx11 mnemonics the emulator already models, and the few
gfx10-only instructions the translator emits are added here (v_dot2c_f32_f16 with DPP row_share,
v_permlanex16_b32, v_ffbh_u32, s_setreg). Unknown opcodes still raise NotImplementedError.

Purpose: run ORIGINAL and REWRITTEN translated code under one model and compare memory byte for
byte. Absolute agreement with hardware is not claimed (dot2 rounding is the emulator's model).
"""
import re, struct, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, 'E:/dlss 5/rdna2/emu')
import gfx11emu as E

ROCM = Path('C:/Program Files/AMD/ROCm/6.4/bin')
LINE = re.compile(r'^\t(\S+)(?: (.*?))?\s*// ([0-9A-Fa-f]+): ((?:[0-9A-Fa-f]{8} ?)+)')
DPP = re.compile(r'\s*(row_share|row_mask|bank_mask|bound_ctrl|fi|row_xmask|quad_perm|row_shl|row_shr|row_ror|row_mirror|row_half_mirror)(:\S+)?')

RENAME = [
    (r'^s_load_dword$', 's_load_b32'), (r'^s_load_dwordx(\d+)$', lambda m: 's_load_b%d' % (32 * int(m[1]))),
    (r'^s_buffer_load_dword', None),
    (r'^s_andn2_', 's_and_not1_'), (r'^s_orn2_', 's_or_not1_'),
    (r'^v_ffbh_u32', 'v_clz_i32_u32'),
    (r'^ds_read2st64_', 'ds_load_2addr_stride64_'), (r'^ds_write2st64_', 'ds_store_2addr_stride64_'),
    (r'^ds_read2_', 'ds_load_2addr_'), (r'^ds_write2_', 'ds_store_2addr_'),
    (r'^ds_read_', 'ds_load_'), (r'^ds_write_', 'ds_store_'),
]
MEMT = {'dword': 'b32', 'dwordx2': 'b64', 'dwordx3': 'b96', 'dwordx4': 'b128', 'ubyte': 'u8', 'sbyte': 'i8',
        'ushort': 'u16', 'sshort': 'i16', 'byte': 'b8', 'short': 'b16',
        'short_d16_hi': 'd16_hi_b16', 'short_d16': 'd16_b16', 'ubyte_d16_hi': 'd16_hi_u8', 'ubyte_d16': 'd16_u8',
        'sbyte_d16_hi': 'd16_hi_i8', 'sbyte_d16': 'd16_i8'}


def rename(op):
    base, sfx = re.fullmatch(r'(.*?)(_e32|_e64|_dpp)?', op).groups()
    m = re.fullmatch(r'(global|scratch|flat)_(load|store)_(\w+)', base)
    if m and m[3] in MEMT:
        base = '%s_%s_%s' % (m[1], m[2], MEMT[m[3]])
    for pat, rep in RENAME:
        if re.search(pat, base):
            if rep is None:
                raise NotImplementedError(base)
            base = re.sub(pat, rep, base)
            break
    return base + (sfx or '')


def objdump(co):
    return subprocess.run([str(ROCM / 'llvm-objdump.exe'), '-d', '--mcpu=gfx1030', str(co)],
                          capture_output=True, text=True, check=True).stdout


def load(co, symbol):
    """-> (prog rotated so the kernel entry is index 0, [(vaddr, bytes) of PT_LOAD segments])."""
    prog, entry = [], None
    for line in objdump(co).splitlines():
        h = re.fullmatch(r'([0-9a-fA-F]+) <([^>]+)>:', line)
        if h and h[2] == symbol:
            entry = int(h[1], 16)
        m = LINE.match(line)
        if m and not m[1].startswith('.'):
            op, args, addr, raw = m.groups()
            prog.append((int(addr, 16), op, (args or '').strip(), 4 * len(raw.split())))
    assert entry is not None, symbol
    k = next(i for i, p in enumerate(prog) if p[0] == entry)
    return prog[k:] + prog[:k], segments(co)


def segments(co):
    b = Path(co).read_bytes()
    phoff, = struct.unpack_from('<Q', b, 0x20)
    phentsize, phnum = struct.unpack_from('<HH', b, 0x36)
    out = []
    for i in range(phnum):
        t, _, off, va, _, fs, ms = struct.unpack_from('<IIQQQQQ', b, phoff + i * phentsize)
        if t == 1 and ms:
            seg = bytearray(ms); seg[:fs] = b[off:off + fs]
            out.append((va, bytes(seg)))
    return out


def descriptor(co, symbol):
    """SGPR layout from the kernel descriptor <symbol>.kd (gfx10 code object v5)."""
    b = Path(co).read_bytes()
    shoff, = struct.unpack_from('<Q', b, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', b, 0x3a)
    secs = [struct.unpack_from('<IIQQQQIIQQ', b, shoff + i * shentsize) for i in range(shnum)]
    symtab = next(s for s in secs if s[1] == 2)
    strtab = secs[symtab[6]]
    for i in range(symtab[5] // 24):
        nm, info, oth, shn, val, sz = struct.unpack_from('<IBBHQQ', b, symtab[4] + 24 * i)
        name = b[strtab[4] + nm:b.index(b'\0', strtab[4] + nm)].decode()
        if name == symbol + '.kd':
            sec = secs[shn]
            kd = b[sec[4] + val - sec[3]:][:64]
            break
    else:
        raise KeyError(symbol + '.kd')
    lds, priv = struct.unpack_from('<II', kd, 0)
    rsrc2, = struct.unpack_from('<I', kd, 0x34)
    props, = struct.unpack_from('<H', kd, 0x38)
    return dict(lds=lds, priv=priv, scratch_en=rsrc2 & 1, user_sgpr=(rsrc2 >> 1) & 31,
                wg_x=(rsrc2 >> 7) & 1, wg_y=(rsrc2 >> 8) & 1, wg_z=(rsrc2 >> 9) & 1,
                psb=props & 1, dptr=(props >> 1) & 1, qptr=(props >> 2) & 1, kptr=(props >> 3) & 1,
                did=(props >> 4) & 1, fsi=(props >> 5) & 1)


def init_sgprs(d, ka_addr, wg):
    s, i = {}, 0
    assert not (d['psb'] or d['dptr'] or d['qptr'] or d['did']), d
    if d['kptr']:
        s[i], s[i + 1] = ka_addr & 0xffffffff, ka_addr >> 32; i += 2
    if d['fsi']:
        s[i], s[i + 1] = 0, 0; i += 2          # flat scratch base: the emulator's scratch is per-lane private
    assert i == d['user_sgpr'], (i, d)
    for en, v in zip((d['wg_x'], d['wg_y'], d['wg_z']), wg):
        if en:
            s[i] = v; i += 1
    if d['scratch_en']:
        s[i] = 0; i += 1                        # private segment wave offset
    return s


class Exec10(E.Exec):
    def _decode(s, addr, op, args, size):
        if op.startswith('s_setreg'):
            # flat-scratch base (HW_REG_FLAT_SCR_*) and mode bits; checked to be only those.
            assert 'FLAT_SCR' in args or re.search(r'hwreg\((0|1|HW_REG_MODE)\b', args), args
            if 'FLAT_SCR' in args:
                return lambda w: None
            if 'hwreg(0, 4, 24)' not in args and 'hwreg(0, 8, 24)' not in args:
                raise NotImplementedError('setreg ' + args)
            return lambda w: None
        if op == 's_setreg_imm32_b32':
            return lambda w: None
        dpp = None
        if op.endswith('_dpp'):
            mods = dict((k, v[1:] if v else True) for k, v in DPP.findall(args))
            args = DPP.sub('', args).strip()
            op = op[:-4]
            assert set(mods) <= {'row_share', 'row_mask', 'bank_mask', 'bound_ctrl'}, mods
            assert int(mods.get('row_mask', '0xf'), 0) == 15 and int(mods.get('bank_mask', '0xf'), 0) == 15, mods
            n = int(mods['row_share'], 0)
            dpp = (np.arange(32) & ~15) + n         # row_share:n -> src0 of every lane comes from lane n of its row
        if op in ('v_dot2c_f32_f16', 'v_dot2c_f32_f16_e32'):
            o, _ = E.split_args(args)
            P = [E.parse_op(x) for x in o]

            def dot2c(w, P=P, dpp=dpp):
                a = w.r32(P[1]); c = w.r32(P[2])
                if dpp is not None:
                    a = w.RV[P[1]['i']][dpp]            # DPP source is the raw VGPR (no modifiers in use)
                h = lambda x, sh: ((x >> np.uint32(sh)) & np.uint32(0xffff)).astype(np.uint16).view(np.float16).astype(np.float64)
                r = h(a, 0) * h(c, 0) + h(a, 16) * h(c, 16) + w.RV[P[0]['i']].view(np.float32).astype(np.float64)
                w.wv(P[0], E.fbits(r.astype(np.float32)))
            return dot2c
        assert dpp is None, op
        if op == 'v_permlanex16_b32':
            o, mods = E.split_args(args)
            P = [E.parse_op(x) for x in o]
            fi, bc = mods.get('op_sel', [0, 0])

            def plx(w):
                sel = w.rs(P[2]) | (w.rs(P[3]) << 32)
                src = np.array([((l & 16) ^ 16) + ((sel >> (4 * (l & 15))) & 15) for l in range(32)])
                val = w.RV[P[1]['i']][src].copy()
                if not fi:                             # inactive source lane: keep D (bound_ctrl=0) or 0
                    act = w.em[src]
                    val = np.where(act, val, 0 if bc else w.RV[P[0]['i']])
                w.wv(P[0], val)
            return plx
        return super()._decode(addr, rename(op), args, size)


E.Exec = Exec10          # run_workgroup builds its decoder through the module-global name


class HwLds(np.ndarray):
    """LDS with the out-of-range rule (reads 0, writes dropped) beyond `alloc` bytes instead of an assert.
    ponytail: alloc granularity is a model parameter (512 B used), not verified on RX 6900 XT."""
    alloc = 0

    def __len__(s):
        return 1 << 32

    def __getitem__(s, idx):
        b = s.view(np.ndarray)
        if isinstance(idx, np.ndarray) and idx.dtype.kind in 'iu':
            ok = (idx >= 0) & (idx < s.alloc)
            out = np.zeros(idx.shape, np.uint8); out[ok] = b[idx[ok]]
            return out
        return b[idx]

    def __setitem__(s, idx, val):
        b = s.view(np.ndarray)
        if isinstance(idx, np.ndarray) and idx.dtype.kind in 'iu':
            ok = (idx >= 0) & (idx < s.alloc)
            b[idx[ok]] = np.broadcast_to(np.asarray(val, np.uint8), idx.shape)[ok]
            return
        b[idx] = val


_Wave = E.Wave


class Wave10(_Wave):
    hw_lds = None

    def __init__(s, gmem, lds, wid, tid0):
        if Wave10.hw_lds:
            lds = lds.view(HwLds); lds.alloc = Wave10.hw_lds
        super().__init__(gmem, lds, wid, tid0)


E.Wave = Wave10


def run(co, symbol, gmem, ka_addr, wg, nthreads, lds=None, max_steps=50_000_000):
    """lds: None = declared size (asserts on overrun), int = enlarged LDS, 'hw' = declared size rounded to
    512 B with out-of-range reads 0 / writes dropped."""
    prog, segs = load(co, symbol)
    d = descriptor(co, symbol)
    for va, data in segs:
        if not any(r.base == va for r in gmem.regions):
            gmem.add('code@%x' % va, va, np.frombuffer(bytearray(data), np.uint8).copy())
    Wave10.hw_lds = (d['lds'] + 511) // 512 * 512 if lds == 'hw' else None
    size = Wave10.hw_lds or lds or d['lds']
    r = E.run_workgroup(prog, gmem, size, nthreads, init_sgprs(d, ka_addr, wg), max_steps=max_steps)
    return r, d
