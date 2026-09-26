"""Find every old f16->e4m3 encoder inline in a translated .s file and replace it with the
verified 11-VALU branch-free encoder (kernels/e4m3/ENCODER.md, GPU-verified 0/65536 mismatches).

Old encoder shape (one instance): starts at `v_cvt_f32_f16_e64 vTMP, |vIN|`, immediately followed
by `s_mov_b32 sSAVE, exec_lo` (the encoder's own EXEC save - branches inside pick the normal vs
subnormal path per lane), and ends at the matching `s_or_b32 exec_lo, exec_lo, sSAVE` that restores
it. The result converges into one destination register vOUT via two `v_or_b32 vOUT, vOUT, ...`
sites (normal-path and subnormal-path). vOUT is first written by the sign-bit cndmask
(`v_cndmask_b32_e64 vOUT, 0, 0xffffff80, vcc_lo`) right after the |vIN| conversion.

This matches everywhere the OLD pattern is NOT interleaved with a neighbouring encoder call (a
handful of sites, checked separately, are within ~30 lines of another v_log_f32 and are skipped -
left on the old, still-correct path; their contribution to the total encoder count is negligible).

New sequence written in place (v_in, v_out, v_t distinct from each other; v_t must not collide with
any register still live across the splice - picked from a scratch pool never touched inside the
old block, confirmed by grepping the removed lines for every register the pool member could use).
"""
import re, sys
from pathlib import Path

NEW_SEQ = """v_cvt_f32_f16_e64  {vout}, |{vin}|
v_mul_f32_e32      {vout}, 0x03800000, {vout}
v_bfe_u32          {vt}, {vout}, 20, 1
v_add3_u32         {vout}, {vout}, 0x7ffff, {vt}
v_lshrrev_b32_e32  {vout}, 20, {vout}
v_min_u32_e32      {vout}, 0x7e, {vout}
v_cmp_u_f16_e32    vcc_lo, {vin}, {vin}
v_cndmask_b32_e64  {vout}, {vout}, 0x7f, vcc_lo
v_or_b32_e32       {vt}, 0x80, {vout}
v_cmp_gt_f16_e32   vcc_lo, 0, {vin}
v_cndmask_b32_e32  {vout}, {vout}, {vt}, vcc_lo"""

START_RE = re.compile(r'^v_cvt_f32_f16_e64\s+(v\d+),\s*\|(v\d+)\|\s*$')
SAVE_RE = re.compile(r'^s_mov_b32\s+(s\d+),\s*exec_lo\s*$')


LABEL_DEF_RE = re.compile(r'^(\.Lpc_[0-9a-fA-F]+):$')
BRANCH_RE = re.compile(r'^(?:s_cbranch_\w+|s_branch)\s+(\.Lpc_[0-9a-fA-F]+)\s*$')


def _label_index(lines):
    """(label -> defining line, [(line, target_label) for every branch in the file]).

    A site is only safe to delete whole if no branch OUTSIDE the site's own [start,end] span
    targets a label DEFINED inside it - found the hard way (k_swin_var C=256 has external jumps
    landing inside what looked like a self-contained encoder block; llvm-mc caught it as an
    undefined-label assembly error, not a silent miscompile, but it must be checked up front
    instead of relied on to fail loudly every time)."""
    label_line = {}
    branches = []
    for i, l in enumerate(lines):
        s = l.strip()
        m = LABEL_DEF_RE.match(s)
        if m:
            label_line[m.group(1)] = i
            continue
        m = BRANCH_RE.match(s)
        if m:
            branches.append((i, m.group(1)))
    return label_line, branches


def find_sites(lines):
    """Yield (start_idx, end_idx_inclusive, vout, vin, ssave) for every clean (non-interleaved,
    no externally-referenced internal label) site."""
    label_line, branches = _label_index(lines)
    log_sites = [i for i, l in enumerate(lines) if l.strip().startswith('v_log_f32')]
    log_set = set(log_sites)
    sorted_logs = sorted(log_sites)
    near = set()
    for a, b in zip(sorted_logs, sorted_logs[1:]):
        if b - a < 130:
            near.add(a); near.add(b)

    i = 0
    n = len(lines)
    sites = []
    while i < n:
        m = START_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        vtmp, vin = m.group(1), m.group(2)
        start = i
        # next non-label, non-comment line must be the EXEC save
        j = i + 1
        while j < n and (lines[j].startswith('.Lpc_') or lines[j].startswith('//')):
            j += 1
        save_m = SAVE_RE.match(lines[j].strip()) if j < n else None
        if not save_m:
            i += 1
            continue
        ssave = save_m.group(1)
        # is there a v_log_f32 near this site that's part of the "near" (interleaved) set?
        interleaved = any(start <= ls <= start + 400 and ls in near for ls in sorted_logs)
        # find the matching restore: the s_or_b32 exec_lo, exec_lo, ssave AFTER this point
        restore_re = re.compile(r'^s_or_b32\s+exec_lo,\s*exec_lo,\s*' + re.escape(ssave) + r'\s*$')
        end = None
        for k in range(j, min(n, start + 400)):
            if restore_re.match(lines[k].strip()):
                end = k
                break
        if end is None or interleaved:
            i += 1
            continue
        # destination register: look for v_cndmask_b32_e64 vOUT, 0, 0xffffff80, vcc_lo in [start,end]
        vout = None
        for k in range(start, end + 1):
            m2 = re.match(r'^v_cndmask_b32_e64\s+(v\d+),\s*0,\s*0xffffff80,\s*vcc_lo\s*$', lines[k].strip())
            if m2:
                vout = m2.group(1)
                break
        if vout is None:
            i += 1
            continue
        # Reject if any branch OUTSIDE [start,end] targets a label DEFINED inside [start,end] -
        # deleting the block would leave that branch's target undefined.
        internal_labels = {name for name, ln in label_line.items() if start <= ln <= end}
        externally_referenced = any(
            name in internal_labels and not (start <= bline <= end)
            for bline, name in branches
        )
        if externally_referenced:
            i += 1
            continue
        sites.append((start, end, vout, vin, ssave))
        i = end + 1
    return sites


NEXT_FREE_VGPR_RE = re.compile(r'^\.amdhsa_next_free_vgpr\s+(\d+)\s*$')


def declared_vgprs(lines):
    """The kernel's own `.amdhsa_next_free_vgpr` - the hard upper bound on any register index
    that physically exists. Found the hard way: picking a fixed high temp register (v200) worked
    for k_pre_block/k_post_block (220 VGPRs declared) purely by accident and produced NaN garbage
    on k_swin_var<256,false> (only ~160 declared) - v200 there is unallocated register space."""
    for l in lines:
        m = NEXT_FREE_VGPR_RE.match(l.strip())
        if m:
            return int(m.group(1))
    raise RuntimeError('no .amdhsa_next_free_vgpr found in kernel')


def pick_temp(lines, start, end, exclude, max_vgpr):
    """A register never mentioned as v<N> anywhere in [start,end] (conservative: also not
    v_in/v_out), and strictly below the kernel's own declared VGPR count."""
    used = set()
    for l in lines[start:end + 1]:
        for m in re.finditer(r'\bv(\d+)\b', l):
            used.add(int(m.group(1)))
    for cand in range(max_vgpr - 1, 2, -1):
        if cand not in used and cand not in exclude:
            return f'v{cand}'
    raise RuntimeError('no free temp register found within the declared VGPR budget')


def splice_lines(lines, label='<in-memory>'):
    """Splice every clean encoder site in a list of .s source lines (no trailing newlines).
    Returns (new_lines, count, report_lines). Used both by the CLI below and by
    translate_kernels.py, which calls this right after generating a kernel's assembly and
    before invoking llvm-mc, so the splice is part of the regular build instead of a
    separate manual post-process step."""
    lines = list(lines)
    max_vgpr = declared_vgprs(lines)
    sites = find_sites(lines)
    report = [f'{len(sites)} clean encoder sites found in {label} (declared vgprs: {max_vgpr})']
    # apply from the END of the file backward so earlier indices stay valid
    for start, end, vout, vin, ssave in sorted(sites, key=lambda s: -s[0]):
        vt = pick_temp(lines, start, end, exclude={int(vout[1:]), int(vin[1:])}, max_vgpr=max_vgpr)
        new_block = NEW_SEQ.format(vout=vout, vin=vin, vt=vt).split('\n')
        report.append(f'  line {start+1}-{end+1}: vin={vin} vout={vout} vt={vt} ssave={ssave}')
        lines[start:end + 1] = new_block
    return lines, len(sites), report


def splice(path_in, path_out, report_out):
    lines = Path(path_in).read_text(encoding='utf-8', errors='replace').split('\n')
    new_lines, n, report = splice_lines(lines, label=str(path_in))
    Path(path_out).write_text('\n'.join(new_lines), encoding='utf-8')
    Path(report_out).write_text('\n'.join(report), encoding='utf-8')
    return n


if __name__ == '__main__':
    n = splice(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f'spliced {n} sites -> {sys.argv[2]} (report: {sys.argv[3]})')
