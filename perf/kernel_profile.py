"""Static profile of a translated gfx1030 kernel (.s), optionally against the original gfx1100 disassembly.

Read-only. Prints numbers only - no instruction text beyond mnemonics - so the output can be pasted
into PERF_ANALYSIS.md without carrying vendor code along. Pure stdlib, no numpy.

What it answers (see PERF_ANALYSIS.md for why each matters):

  * descriptor:  VGPR/SGPR/LDS/scratch/wave size/WGP mode as the .s declares them
  * occupancy:   waves per SIMD and workgroups per CU/WGP on gfx1030 and gfx1100, which limit binds,
                 and what VGPR/LDS budget the next step up would need
  * VGPRs:       highest index used, unused holes below the allocation, registers written but never
                 read anywhere, straight-line overwrite-before-read candidates, reg-reg copies, and
                 (with --orig) the registers that exist only in the translation
  * LDS/scratch: ds_* / scratch_* counts, static offset bounds, and how many of them sit inside
                 translator expansions of some *other* original opcode (i.e. added by the translation)
  * expansion:   per original opcode, how many gfx1030 instructions the translator emitted
                 (needs the `// original:` annotations the translator writes)
  * dynamic:     with --exec-hist (from perf/exec_hist.py), executed original instructions per wave
                 times the measured expansion = estimated executed gfx1030 instructions per wave

usage:
  python perf/kernel_profile.py rdna2/build/kernels-hw-scratch/_Z21k_pre_block_1h_32_fp89PreParams.s \\
      [--orig analysis/gfx1100-disassembly.txt [--orig-sym SYM ...]] [--exec-hist hist.json] \\
      [--wg 256] [--json out.json]
"""
import argparse, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

MNEM = re.compile(r'^(?:s|v|ds|global|flat|scratch|buffer|tbuffer|image)_[a-z0-9_]+$')
LABEL = re.compile(r'^[A-Za-z_.$][\w.$]*:$')
VREG = re.compile(r'(?<![\w\[])v(\d+)\b|(?<![\w])v\[(\d+):(\d+)\]')
OBJDUMP = re.compile(r'^\t(\S+)(?: (.*?))?\s*// ([0-9A-Fa-f]+): ((?:[0-9A-Fa-f]{8} ?)+)')
SYMHDR = re.compile(r'^([0-9a-fA-F]+) <([^>]+)>:')
ADDR_IN_NOTE = re.compile(r'(?:0x|@)([0-9A-Fa-f]{4,16})\b')
BRANCH = ('s_branch', 's_cbranch_', 's_setpc', 's_swappc', 's_endpgm', 's_getpc', 's_trap')


# ------------------------------------------------------------------ parsing
def split_comment(line):
    i = line.find('//')
    if i < 0:
        j = line.find(';')
        return (line, '') if j < 0 else (line[:j], line[j + 1:])
    return line[:i], line[i + 2:]


def parse_s(path):
    """-> (directives, items). items: ('label',name) | ('instr', mnemonic, args, note, lineno)."""
    directives, items = {}, []
    for n, raw in enumerate(Path(path).read_text(errors='replace').splitlines(), 1):
        code, note = split_comment(raw)
        code = code.strip()
        note = note.strip()
        if not code:
            if 'original:' in note:
                items.append(('header', note.split('original:', 1)[1].strip(), n))
            continue
        if code.startswith('.'):
            parts = code.split(None, 1)
            directives[parts[0]] = parts[1].strip() if len(parts) > 1 else ''
            continue
        if re.match(r'^\.?[\w.]+:\s*\S', code) and not MNEM.match(code.split()[0]):
            # YAML metadata ("  .vgpr_count: 210") or similar key: value lines
            k, v = code.split(':', 1)
            directives.setdefault(k.strip(), v.strip())
            continue
        if LABEL.match(code):
            items.append(('label', code[:-1], n))
            continue
        parts = code.split(None, 1)
        if not MNEM.match(parts[0]):
            continue
        items.append(('instr', parts[0], parts[1] if len(parts) > 1 else '', note, n))
    return directives, items


def parse_objdump(path, syms):
    """Original gfx1100 instructions for the given symbols (objdump format, as gfx11emu reads it)."""
    out, cur = [], False
    for line in Path(path).read_text(encoding='utf-8-sig', errors='replace').splitlines():
        m = SYMHDR.match(line)
        if m:
            cur = m[2] in syms
            continue
        if cur:
            m = OBJDUMP.match(line)
            if m:
                out.append((int(m[3], 16), m[1], (m[2] or '').strip()))
    return out


def dnum(d, *keys):
    for k in keys:
        if k in d:
            try:
                return int(str(d[k]).split()[0], 0)
            except ValueError:
                pass
    return None


# ------------------------------------------------------------------ operand roles
def split_ops(args):
    out, depth, cur = [], 0, ''
    for ch in args:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur.strip()); cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def vregs(tok):
    r = []
    for m in VREG.finditer(tok):
        if m.group(1) is not None:
            r.append(int(m.group(1)))
        else:
            r.extend(range(int(m.group(2)), int(m.group(3)) + 1))
    return r


def roles(mn, args):
    """-> (vgpr writes, vgpr reads, partial_write). Conservative: unknown shapes count as reads."""
    if mn.startswith('v_dual_') and '::' in args:
        a, b = args.split('::', 1)
        bm, ba = (b.strip().split(None, 1) + [''])[:2]
        w1, r1, _ = roles(mn.replace('v_dual_', 'v_'), a)
        w2, r2, _ = roles(bm.replace('v_dual_', 'v_'), ba)
        return w1 + w2, r1 + r2, False
    ops = split_ops(args)
    if not ops:
        return [], [], False
    allr = [r for o in ops for r in vregs(o)]
    if mn.startswith(('s_', 'v_cmp_', 'v_cmpx_', 'v_readlane', 'v_readfirstlane')):
        # SGPR/VCC/EXEC destinations; every VGPR operand is a source
        return [], allr, False
    store = ('_store' in mn or mn.startswith(('ds_write', 'ds_store')) or
             (('_atomic' in mn or mn.startswith('ds_')) and 'glc' not in args and '_rtn' not in mn
              and not mn.startswith(('ds_read', 'ds_load', 'ds_swizzle', 'ds_permute', 'ds_bpermute',
                                     'ds_append', 'ds_consume')) and ('_atomic' in mn or
                                     re.match(r'ds_(add|sub|inc|dec|min|max|and|or|xor|mskor|cmpst|cmpswap|wrxchg)', mn))))
    if store:
        return [], allr, False
    w = vregs(ops[0])
    r = [x for o in ops[1:] for x in vregs(o)]
    partial = bool(re.search(r'(fmac|_mac_|dot\dc|writelane|d16_hi|_d16|sdwa)', mn) or 'op_sel' in args
                   or 'dst_sel' in args)
    if partial:
        r = r + w          # read-modify-write of the destination
    return w, r, partial


def writes_exec(mn, args):
    if mn.startswith('v_cmpx_') or 'saveexec' in mn:
        return True
    ops = split_ops(args)
    return bool(ops) and ops[0].startswith('exec')


# ------------------------------------------------------------------ occupancy
ARCH = {
    # per SIMD, wave32/wave64: total VGPRs, allocation granule. max waves per SIMD. LDS per CU / WGP.
    'gfx1030': dict(vgpr={32: (1024, 8), 64: (512, 4)}, maxw=16, lds_cu=65536, lds_wgp=131072),
    'gfx1100': dict(vgpr={32: (1536, 24), 64: (768, 12)}, maxw=16, lds_cu=65536, lds_wgp=131072),
}
LDS_GRANULE = 512


def occupancy(arch, vgpr, lds, wg, wave, wgp):
    a = ARCH[arch]
    total, gran = a['vgpr'][wave]
    alloc = max(gran, math.ceil(max(vgpr, 1) / gran) * gran)
    w_simd = min(a['maxw'], total // alloc)
    simds = 4 if wgp else 2
    wpg = math.ceil(wg / wave)
    by_vgpr = (w_simd * simds) // wpg
    pool = a['lds_wgp'] if wgp else a['lds_cu']
    lds_a = math.ceil(lds / LDS_GRANULE) * LDS_GRANULE if lds else 0
    by_lds = pool // lds_a if lds_a else 10 ** 9
    n = min(by_vgpr, by_lds)
    nxt = n + 1
    need_w = math.ceil(nxt * wpg / simds)
    need_vgpr = (total // need_w) // gran * gran if need_w <= a['maxw'] else 0
    return dict(arch=arch, unit='WGP' if wgp else 'CU', vgpr_alloc=alloc, waves_per_simd_vgpr_limit=w_simd,
                wg_per_unit_by_vgpr=by_vgpr, wg_per_unit_by_lds=by_lds if lds_a else None, wg_per_unit=n,
                waves_per_simd_actual=n * wpg / simds,
                binding=('VGPR+LDS' if by_vgpr == by_lds else 'VGPR' if by_vgpr < by_lds else 'LDS'),
                next_step_needs=dict(wg_per_unit=nxt, vgpr_max=need_vgpr, lds_max=pool // nxt // LDS_GRANULE * LDS_GRANULE))


# ------------------------------------------------------------------ analysis
def norm(mn):
    """Encoding suffixes differ between objdump and the translator's annotations; join on the bare op."""
    return re.sub(r'_(e32|e64|sdwa|dpp|vop3)$', '', mn)


def group_expansions(items):
    """Attach every translated instruction to the original instruction it came from."""
    headers = sum(1 for it in items if it[0] == 'header')
    trailing = sum(1 for it in items if it[0] == 'instr' and 'original:' in it[3])
    mode = 'header' if headers >= trailing and headers else 'trailing' if trailing else 'none'
    groups, cur = [], None
    for it in items:
        if it[0] == 'label':
            cur = None
            continue
        if it[0] == 'header':
            if mode == 'header':
                cur = dict(orig=it[1], out=[]); groups.append(cur)
            continue
        mn, args, note = it[1], it[2], it[3]
        if mode == 'trailing' and 'original:' in note:
            cur = dict(orig=note.split('original:', 1)[1].strip(), out=[]); groups.append(cur)
        if cur is None:
            groups.append(dict(orig=None, out=[(mn, args)]))
        else:
            cur['out'].append((mn, args))
    for g in groups:
        o = g['orig']
        g['orig_mn'] = norm(o.split()[0]) if o else None
        m = ADDR_IN_NOTE.search(o or '')
        g['orig_addr'] = int(m[1], 16) if m else None
    return mode, groups


def analyse(spath, orig=None, orig_syms=None, exec_hist=None, wg=256):
    d, items = parse_s(spath)
    ins = [it for it in items if it[0] == 'instr']
    R = dict(file=str(spath), instructions=len(ins))

    # descriptor
    desc = dict(
        next_free_vgpr=dnum(d, '.amdhsa_next_free_vgpr', '.vgpr_count'),
        next_free_sgpr=dnum(d, '.amdhsa_next_free_sgpr', '.sgpr_count'),
        lds_bytes=dnum(d, '.amdhsa_group_segment_fixed_size', '.group_segment_fixed_size'),
        scratch_bytes=dnum(d, '.amdhsa_private_segment_fixed_size', '.private_segment_fixed_size'),
        vgpr_spill=dnum(d, '.vgpr_spill_count'), sgpr_spill=dnum(d, '.sgpr_spill_count'),
        wave32=dnum(d, '.amdhsa_wavefront_size32'), wgp_mode=dnum(d, '.amdhsa_workgroup_processor_mode'),
        max_wg=dnum(d, '.max_flat_workgroup_size'), dynamic_stack=dnum(d, '.amdhsa_uses_dynamic_stack'),
        wg_id_y=dnum(d, '.amdhsa_system_sgpr_workgroup_id_y'),
        private_seg=dnum(d, '.amdhsa_enable_private_segment',
                         '.amdhsa_system_sgpr_private_segment_wavefront_offset'))
    R['descriptor'] = desc

    # VGPR usage
    wr, rd = Counter(), Counter()
    copies = selfmov = 0
    dead_cand = Counter()
    last_w = {}
    for it in items:
        if it[0] != 'instr':
            last_w.clear(); continue
        mn, args = it[1], it[2]
        w, r, partial = roles(mn, args)
        for x in r:
            rd[x] += 1; last_w.pop(x, None)
        for x in w:
            wr[x] += 1
            if not partial:
                if x in last_w:
                    dead_cand[last_w[x]] += 1
                last_w[x] = mn
        if mn.startswith('v_mov_b32'):
            ops = split_ops(args)
            if len(ops) == 2 and re.fullmatch(r'v\d+', ops[1]):
                copies += 1
                selfmov += ops[0] == ops[1]
        if mn.startswith(BRANCH) or writes_exec(mn, args):
            last_w.clear()
    used = set(wr) | set(rd)
    top = max(used) if used else -1
    alloc = desc['next_free_vgpr'] or top + 1
    R['vgpr'] = dict(max_index_used=top, declared=desc['next_free_vgpr'], distinct_used=len(used),
                     holes_below_alloc=sorted(set(range(alloc)) - used),
                     written_never_read=sorted(set(wr) - set(rd)),
                     read_never_written=len(set(rd) - set(wr)),     # v0 (thread id) and inputs
                     overwrite_before_read_same_bb=sum(dead_cand.values()),
                     overwrite_before_read_by_opcode=dead_cand.most_common(15),
                     reg_to_reg_v_mov_b32=copies, self_moves=selfmov)

    # memory / sync
    mc = Counter(it[1] for it in ins)
    def cnt(pred): return sum(v for k, v in mc.items() if pred(k))
    offs = [int(m) for it in ins if it[1].startswith('ds_') for m in re.findall(r'offset\d?:(\d+)', it[2])]
    R['memory'] = dict(
        ds_ops=cnt(lambda k: k.startswith('ds_')), ds_max_static_offset=max(offs) if offs else None,
        scratch_ops=cnt(lambda k: k.startswith('scratch_')) + sum(1 for it in ins if it[1].startswith('buffer_') and 'off' in it[2]),
        global_ops=cnt(lambda k: k.startswith(('global_', 'flat_', 'buffer_'))),
        s_barrier=mc['s_barrier'], s_waitcnt=cnt(lambda k: k.startswith('s_waitcnt')),
        cross_lane=cnt(lambda k: k.startswith(('v_readlane', 'v_writelane', 'v_readfirstlane', 'ds_swizzle',
                                               'ds_permute', 'ds_bpermute', 'v_permlane')) or '_dpp' in k))
    R['top_opcodes'] = mc.most_common(40)

    # expansions
    mode, groups = group_expansions(items)
    R['annotation_mode'] = mode
    if mode != 'none':
        per = defaultdict(lambda: [0, 0, Counter()])   # orig_mn -> [n_orig, n_out, out mnemonics]
        inside = Counter()
        for g in groups:
            if g['orig_mn'] is None:
                continue
            p = per[g['orig_mn']]
            p[0] += 1; p[1] += len(g['out'])
            for mn, _ in g['out']:
                p[2][mn] += 1
                if mn.split('_')[0] != g['orig_mn'].split('_')[0]:
                    inside[(mn.split('_')[0], g['orig_mn'])] += 1
        unann = sum(len(g['out']) for g in groups if g['orig_mn'] is None)
        rows = sorted(((k, v[0], v[1], v[1] / max(v[0], 1), v[2].most_common(6)) for k, v in per.items()),
                      key=lambda r: -r[2])
        R['expansion'] = dict(original_instructions=sum(r[1] for r in rows), emitted=sum(r[2] for r in rows),
                              unannotated_emitted=unann,
                              by_original_opcode=[dict(op=r[0], n=r[1], emitted=r[2], ratio=round(r[3], 2),
                                                       emitted_mix=r[4]) for r in rows[:40]],
                              foreign_units_inside_expansions={'%s in %s' % k: v for k, v in inside.most_common(20)})
        R['expansion']['ratio'] = round(R['expansion']['emitted'] / max(R['expansion']['original_instructions'], 1), 3)

        if exec_hist:
            h = json.loads(Path(exec_hist).read_text())
            waves = max(h.get('waves', 1), 1)
            ex_op = h.get('by_op', {})
            est = orig_dyn = 0.0
            contrib = Counter()
            for op, n in ex_op.items():
                op = norm(op)
                ratio = per[op][1] / per[op][0] if op in per and per[op][0] else 1.0
                orig_dyn += n
                est += n * ratio
                contrib[op] += n * ratio / waves
            R['dynamic'] = dict(waves=waves, orig_executed_per_wave=round(orig_dyn / waves),
                                est_gfx1030_executed_per_wave=round(est / waves),
                                top_contributors_per_wave=[(k, round(v)) for k, v in contrib.most_common(20)])

    # against the original
    if orig:
        syms = set(orig_syms or [Path(spath).stem])
        o = parse_objdump(orig, syms)
        if not o:
            print('WARNING: no instructions for %s in %s - pass --orig-sym' % (sorted(syms), orig), file=sys.stderr)
        ou = set()
        for _, mn, args in o:
            w, r, _ = roles(mn, args)
            ou.update(w); ou.update(r)
        omc = Counter(norm(mn) for _, mn, _ in o)
        R['original'] = dict(symbols=sorted(syms), instructions=len(o), max_vgpr_index=max(ou) if ou else None,
                             distinct_vgprs=len(ou),
                             vgprs_only_in_translation=sorted(used - ou) if o else [],
                             wmma=sum(v for k, v in omc.items() if 'wmma' in k),
                             vopd=sum(v for k, v in omc.items() if k.startswith('v_dual_')),
                             ds_ops=sum(v for k, v in omc.items() if k.startswith('ds_')),
                             scratch_ops=sum(v for k, v in omc.items() if k.startswith('scratch_')),
                             s_barrier=omc['s_barrier'], top_opcodes=omc.most_common(25))

    # occupancy
    wave = 32 if desc['wave32'] in (None, 1) else 64
    vg = desc['next_free_vgpr'] or top + 1
    lds = desc['lds_bytes'] or 0
    R['occupancy'] = [occupancy(a, vg, lds, wg, wave, m) for a in ('gfx1030', 'gfx1100') for m in (False, True)]
    if orig and R['original']['max_vgpr_index'] is not None:
        R['occupancy_original_vgprs_on_gfx1100'] = [occupancy('gfx1100', R['original']['max_vgpr_index'] + 1, lds, wg, wave, m)
                                                    for m in (False, True)]
    return R


def show(R):
    p = print
    p('=' * 78); p(R['file']); p('instructions (static): %d' % R['instructions'])
    p('\n-- descriptor'); [p('  %-22s %s' % kv) for kv in R['descriptor'].items()]
    p('\n-- occupancy (workgroups per CU or WGP; waves per SIMD)')
    for o in R['occupancy'] + R.get('occupancy_original_vgprs_on_gfx1100', []):
        p('  %-7s %-3s vgpr_alloc=%-3d wg/unit=%d (vgpr %d, lds %s) waves/SIMD=%.1f bind=%-8s next: VGPR<=%d and LDS<=%d'
          % (o['arch'], o['unit'], o['vgpr_alloc'], o['wg_per_unit'], o['wg_per_unit_by_vgpr'], o['wg_per_unit_by_lds'],
             o['waves_per_simd_actual'], o['binding'], o['next_step_needs']['vgpr_max'], o['next_step_needs']['lds_max']))
    v = R['vgpr']
    p('\n-- VGPRs')
    for k in ('declared', 'max_index_used', 'distinct_used', 'read_never_written', 'reg_to_reg_v_mov_b32', 'self_moves',
              'overwrite_before_read_same_bb'):
        p('  %-32s %s' % (k, v[k]))
    p('  %-32s %d  %s' % ('holes_below_alloc', len(v['holes_below_alloc']), v['holes_below_alloc'][:40]))
    p('  %-32s %d  %s' % ('written_never_read', len(v['written_never_read']), v['written_never_read'][:40]))
    p('  overwrite-before-read by opcode: %s' % v['overwrite_before_read_by_opcode'])
    p('\n-- memory / sync'); [p('  %-24s %s' % kv) for kv in R['memory'].items()]
    p('\n-- annotation mode: %s' % R['annotation_mode'])
    if 'expansion' in R:
        e = R['expansion']
        p('  original %d -> emitted %d  (x%.2f), unannotated %d' % (e['original_instructions'], e['emitted'], e['ratio'],
                                                                   e['unannotated_emitted']))
        p('  %-34s %6s %8s %7s  emitted mix' % ('original opcode', 'n', 'emitted', 'ratio'))
        for r in e['by_original_opcode'][:30]:
            p('  %-34s %6d %8d %7.2f  %s' % (r['op'], r['n'], r['emitted'], r['ratio'], r['emitted_mix'][:4]))
        p('  foreign units inside expansions: %s' % e['foreign_units_inside_expansions'])
    if 'dynamic' in R:
        p('\n-- dynamic (per wave)'); [p('  %-34s %s' % kv) for kv in R['dynamic'].items()]
    if 'original' in R:
        o = R['original']
        p('\n-- original gfx1100 %s' % o['symbols'])
        for k in ('instructions', 'max_vgpr_index', 'distinct_vgprs', 'wmma', 'vopd', 'ds_ops', 'scratch_ops', 's_barrier'):
            p('  %-26s %s' % (k, o[k]))
        p('  %-26s %d  %s' % ('vgprs_only_in_translation', len(o['vgprs_only_in_translation']),
                              o['vgprs_only_in_translation'][:60]))
    p('\n-- top opcodes (translated): %s' % R['top_opcodes'][:25])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('s', nargs='+')
    ap.add_argument('--orig'); ap.add_argument('--orig-sym', action='append')
    ap.add_argument('--exec-hist'); ap.add_argument('--wg', type=int, default=256)
    ap.add_argument('--json')
    a = ap.parse_args()
    res = [analyse(s, a.orig, a.orig_sym, a.exec_hist, a.wg) for s in a.s]
    for r in res:
        show(r)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=list))
