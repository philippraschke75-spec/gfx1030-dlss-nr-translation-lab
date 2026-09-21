"""Static inventory of the shared SWIN helper (swin_layer) in the gfx1100 disassembly.

Reports instruction-family counts so lowering work can be tracked and the counts frozen.
Does not execute or translate anything.
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYMBOL = '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'
DISASM = ROOT/'analysis/gfx1100-disassembly.txt'
OUT = ROOT/'analysis/helper-inventory.json'

# Families the current lowering layer is known NOT to cover (see KERNEL_TRANSLATION_STATUS.md).
UNCOVERED_PREFIXES = ('v_fma_mix', 'v_perm_b32', 'v_pk_', 'v_dual_', 'v_div_scale', 'v_div_fmas',
                      'v_div_fixup', 's_delay_alu', 's_clause', 's_sendmsg_rtn')


def load_helper():
    ins, in_body = [], False
    for line in DISASM.read_text(encoding='utf-8-sig').splitlines():
        m = re.fullmatch(r'([0-9a-fA-F]+) <([^>]+)>:', line)
        if m:
            if in_body:
                break
            in_body = m[2] == SYMBOL
            continue
        if in_body and '//' in line:
            text, comment = line.split('//', 1)
            text = text.strip()
            if text:
                addr = re.match(r'\s*([0-9A-Fa-f]+):', comment)
                ins.append((int(addr[1], 16) if addr else None, text))
    return ins


def family(op):
    if op.startswith('v_dual_'):
        return 'v_dual_*'
    if op.startswith('v_wmma'):
        return 'v_wmma_*'
    base = re.sub(r'_(e32|e64|dpp|sdwa)$', '', op)
    return base


def main():
    ins = load_helper()
    if not ins:
        sys.exit(f'symbol {SYMBOL} not found in {DISASM}')
    ops = [t.split()[0] for _, t in ins]
    fam = collections.Counter(family(o) for o in ops)
    exec_branches = sum(o in ('s_cbranch_execz', 's_cbranch_execnz') for o in ops)
    uncovered = {f: n for f, n in fam.items() if f.startswith(UNCOVERED_PREFIXES)}
    report = dict(
        symbol=SYMBOL,
        instructions=len(ins),
        instructions_excluding_nop=sum(o != 's_nop' for o in ops),
        wmma_sites=[t for _, t in ins if t.startswith('v_wmma')].__len__(),
        wmma_forms=dict(collections.Counter(o for o in ops if o.startswith('v_wmma'))),
        exec_dependent_branches=exec_branches,
        indirect_control=sum(o in ('s_setpc_b64', 's_swappc_b64') for o in ops),
        scratch_ops=sum(o.startswith('scratch_') for o in ops),
        lds_ops=sum(o.startswith('ds_') for o in ops),
        uncovered_families=uncovered,
        families=dict(fam.most_common()),
    )
    OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for k in ('instructions', 'instructions_excluding_nop', 'wmma_sites', 'exec_dependent_branches',
              'indirect_control', 'scratch_ops', 'lds_ops'):
        print(f'{k}: {report[k]}')
    print('uncovered families:', uncovered)
    print('top 25 families:', list(fam.most_common(25)))


if __name__ == '__main__':
    main()
