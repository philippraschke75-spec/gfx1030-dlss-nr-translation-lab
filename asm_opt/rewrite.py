"""Bit-exact asm rewrites of the translated gfx1030 kernels (generalised from astra-work/c32-optimization).

1. direct: a translated VOPD pair is `op1 T0,..; op2 T1,..; v_mov d1,T0; v_mov d2,T1` (T0/T1 = per-kernel
   temps). Write d1/d2 directly and drop both moves when op2 does not read d1 (RAW hazard), d1 != d2,
   neither op reads T0/T1 or implicitly reads its destination, and T0/T1 are never read anywhere else
   in the kernel (so their dead values cannot matter).
2. interleave: consecutive translated-WMMA accumulator chains `[v_mov acc, src] + N x v_dot2c acc,..`
   with distinct accumulators, none of which is read by any chain of the block, are emitted
   round-robin. Every accumulator keeps its own operation order; initialisers are hoisted only under
   the same "nothing in the block reads an accumulator" condition.

usage: rewrite.py OUTDIR kernel_symbol...   -> OUTDIR/<sym>/{baseline,direct,final}.{s,co}, report.json
"""
import re, json, subprocess, sys, hashlib
from pathlib import Path

SRC = Path('E:/dlss 5/rdna2/build/kernels-hw-scratch')
DEPLOYED = Path('E:/dlss5-experimental/nr_data/kernels')
ROCM = Path('C:/Program Files/AMD/ROCm/6.4/bin')
# per-kernel source of the deployed module (attention2 was built from the combo tree)
SRC_OVERRIDE = {'_Z12k_attention212AttnParams1d': Path('E:/dlss 5/rdna2/build/combo/k')}

SAFE = re.compile(r'v_(mov|mul|add|sub|subrev|fma|and|or|xor|lshlrev|lshrrev|ashrrev|max|min|cndmask|add_nc|sub_nc|subrev_nc'
                  r'|cvt_f32_i32|cvt_f32_u32|cvt_i32_f32|cvt_u32_f32|cvt_f32_f16|not|bfe|alignbit|lshl_add|add_lshl|lshl_or|and_or|or3|add3|mul_lo|mul_hi|mul_u32_u24|perm|med3|max3|min3)'
                  r'_(f32|b32|u32|i32|u32_u24|i32_f32|u32_f32|f32_i32|f32_u32|f32_f16)(_e32|_e64)?')
REGS = re.compile(r'\bv(\d+)\b|\bv\[(\d+):(\d+)\]')


def vregs(text):
    out = set()
    for a, lo, hi in REGS.findall(text):
        out |= {int(a)} if a else set(range(int(lo), int(hi) + 1))
    return out


def body(lines):
    return [l.split('//')[0].strip() for l in lines]


def direct(lines):
    """-> new lines, records, retained hazards"""
    # find the per-kernel dual temps
    temps = set()
    blocks = []
    for i, l in enumerate(lines):
        if l.startswith('// original: v_dual'):
            b = lines[i + 1:i + 5]
            m1 = re.fullmatch(r'v_mov_b32_e32 (v\d+), (v\d+)', b[2]); m2 = re.fullmatch(r'v_mov_b32_e32 (v\d+), (v\d+)', b[3])
            assert m1 and m2 and b[0].split()[1].rstrip(',') == m1[2] and b[1].split()[1].rstrip(',') == m2[2], b
            temps |= {m1[2], m2[2]}
            blocks.append(i)
    if not blocks:
        return lines, [], []
    assert len(temps) == 2, temps
    tnum = {int(t[1:]) for t in temps}
    # T0/T1 must be dead at every label: in each label-delimited block every use of a temp is
    # preceded by a full write of it in that block, so a dropped write can never be observed.
    live_in = set()
    for j, l in enumerate(lines):
        if l.endswith(':'):
            seen = set()
            continue
        if l.startswith(('//', '.')) or not l.strip():
            continue
        op, _, rest = l.partition(' ')
        rd, wr = vregs(rest), set()
        full = (op.startswith('v_') and not re.search(r'dot2c|fmac|_mac_|_f16|_u16|_i16|_b16|mix|permlane|cmp|readlane|readfirstlane|writelane', op)) \
            or (re.match(r'(global|scratch|ds|flat)_(load|read)', op) and 'd16' not in op)
        if full:
            dst = rest.split(',')[0]
            wr = vregs(dst)
            rd = vregs(rest[len(dst):])
        if (rd & tnum) - seen:
            live_in |= (rd & tnum) - seen
        seen |= wr
    if live_in:
        raise SystemExit('dual temp read before written in a block: %s' % live_in)
    for i in blocks:
        b = lines[i + 1:i + 5]
        for k in (0, 1):
            if vregs(b[k].split(None, 1)[1].split(',', 1)[1]) & tnum:
                raise SystemExit('dual op reads a temp: %r' % b)
    out, recs, haz = list(lines), [], []
    for i in blocks:
        p1, p2, mv1, mv2 = lines[i + 1:i + 5]
        d1, d2 = mv1.split()[1].rstrip(','), mv2.split()[1].rstrip(',')
        op1, op2 = p1.split()[0], p2.split()[0]
        src2 = p2.split(None, 1)[1].split(',', 1)[1]
        why = None
        if not SAFE.fullmatch(op1) or not SAFE.fullmatch(op2):
            why = 'op not in safe list (%s, %s)' % (op1, op2)
        elif d1 == d2:
            why = 'same destination'
        elif int(d1[1:]) in vregs(src2):
            why = 'RAW hazard: second op reads first destination'
        if why:
            haz.append({'line': i + 1, 'original': lines[i], 'reason': why})
            continue
        n1 = re.sub(r'^(\S+) v\d+,', r'\1 %s,' % d1, p1)
        n2 = re.sub(r'^(\S+) v\d+,', r'\1 %s,' % d2, p2)
        out[i + 1:i + 5] = ['// asm_opt: direct destinations, moves removed', n1, n2, None]
        recs.append({'line': i + 1, 'original': lines[i], 'after': [n1, n2]})
    return [l for l in out if l is not None], recs, haz


MOV = re.compile(r'v_mov_b32_e32 (v\d+), (v\d+|s\d+|-?\d+|0x[0-9a-f]+)')
DOT = re.compile(r'v_dot2c_f32_f16_dpp (v\d+), (v\d+), (v\d+) (.+)')


def interleave(lines):
    out, recs, i = [], [], 0
    while i < len(lines):
        groups, j = [], i
        while j < len(lines):
            m = MOV.fullmatch(lines[j]); k = j + 1 if m else j
            ops, dest = [], None
            while k < len(lines):
                d = DOT.fullmatch(lines[k])
                if not d or (dest is not None and d[1] != dest):
                    break
                dest = d[1]; ops.append(lines[k]); k += 1
            if not ops or (m and m[1] != dest) or (groups and len(ops) != len(groups[0][2])):
                break
            groups.append((dest, lines[j] if m else None, ops)); j = k
        if len(groups) >= 2:
            dests = {g[0] for g in groups}; reads = set()
            for _, ini, ops in groups:
                if ini:
                    reads.add(MOV.fullmatch(ini)[2])
                for op in ops:
                    d = DOT.fullmatch(op); reads |= {d[2], d[3]}
            if len(dests) == len(groups) and not dests & reads:
                out.append('// asm_opt: independent dot2 accumulators interleaved; each sum order unchanged')
                out += [g[1] for g in groups if g[1]]
                for n in range(len(groups[0][2])):
                    out += [g[2][n] for g in groups]
                recs.append({'line': i + 1, 'accumulators': len(groups), 'dot_instructions': sum(len(g[2]) for g in groups),
                             'initialisers_hoisted': sum(1 for g in groups[1:] if g[1])})
                i = j
                continue
        out.append(lines[i]); i += 1
    return out, recs


def assemble(s_path):
    o, co = s_path.with_suffix('.o'), s_path.with_suffix('.co')
    subprocess.run([str(ROCM / 'llvm-mc.exe'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1030', '-filetype=obj', str(s_path), '-o', str(o)], check=True)
    subprocess.run([str(ROCM / 'ld.lld.exe'), '-shared', str(o), '-o', str(co)], check=True)
    return co


def text(co):
    """bytes of .text (via llvm-objcopy)"""
    out = co.with_suffix('.text')
    subprocess.run([str(ROCM / 'llvm-objcopy.exe'), '-O', 'binary', '--only-section=.text', str(co), str(out)], check=True)
    return out.read_bytes()


def count(lines):
    return sum(1 for l in lines if l and not l.startswith(('//', '.')) and not l.endswith(':'))


def main(outdir, syms):
    rep = {}
    for sym in syms:
        d = Path(outdir) / sym; d.mkdir(parents=True, exist_ok=True)
        src = SRC_OVERRIDE.get(sym, SRC) / (sym + '.s')
        s = src.read_text()
        s = re.sub(r'\.incbin "C:/Users/tes43/Documents/ChatGPT/dlss 5/', '.incbin "E:/dlss 5/', s)
        base = s.splitlines()
        dl, drec, dhaz = direct(base)
        fl, irec = interleave(dl)
        for tag, ls in (('baseline', base), ('direct', dl), ('final', fl)):
            (d / (tag + '.s')).write_text('\n'.join(ls) + '\n')
            assemble(d / (tag + '.s'))
        dep = text(DEPLOYED / (sym + '.co')) if (DEPLOYED / (sym + '.co')).exists() else None
        r = {'source': str(src), 'baseline_text_equals_deployed': dep == text(d / 'baseline.co'),
             'instructions': {'baseline': count(base), 'final': count(fl)},
             'moves_removed': 2 * len(drec), 'dual_pairs_rewritten': len(drec), 'dual_pairs_retained': dhaz,
             'interleaved_blocks': len(irec), 'interleaved_dot_instructions': sum(x['dot_instructions'] for x in irec),
             'initialisers_hoisted': sum(x['initialisers_hoisted'] for x in irec),
             'final_sha256': hashlib.sha256((d / 'final.co').read_bytes()).hexdigest()}
        (d / 'report.json').write_text(json.dumps({'summary': r, 'direct': drec, 'interleave': irec}, indent=1))
        rep[sym] = r
        print(sym, json.dumps({k: (v if k != 'dual_pairs_retained' else len(v)) for k, v in r.items()}), flush=True)
    return rep


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
