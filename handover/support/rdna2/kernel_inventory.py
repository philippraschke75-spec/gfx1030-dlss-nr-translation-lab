"""Static census of the two saved disassemblies; does not execute kernels."""
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def parse(path):
    kernels = {}
    current = None
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        label = re.fullmatch(r'([0-9a-fA-F]+) <([^>]+)>:', line)
        if label:
            current = dict(address='0x'+label[1], instructions=[], wmma_sites=[])
            kernels[label[2]] = current
        elif current is not None and '//' in line:
            text = line.split('//')[0].strip()
            if not text:
                continue
            op = text.split()[0]
            if op != 's_nop':
                current['instructions'].append(text)
            if op.startswith('v_wmma'):
                current['wmma_sites'].append(text)
    return kernels

def main():
    reference = parse(ROOT/'analysis/gfx1100-disassembly.txt')
    generic = parse(ROOT/'analysis/gfx1030-disassembly.txt')
    rows = []
    variants = collections.Counter()
    for name, kernel in reference.items():
        other = generic.get(name, {'instructions': [], 'wmma_sites': []})
        for text in kernel['wmma_sites']:
            variants[text.split()[0]] += 1
        rows.append(dict(symbol=name, gfx1100_instructions=len(kernel['instructions']),
                         gfx1030_instructions=len(other['instructions']),
                         wmma_sites=len(kernel['wmma_sites']),
                         gfx1030_immediate_return=other['instructions'][:1] == ['s_endpgm'],
                         pc_relative=any('s_getpc' in x for x in kernel['instructions']),
                         indirect_control=any(x.split()[0] in ('s_setpc_b64','s_swappc_b64')
                                              for x in kernel['instructions'])))
    report = dict(wmma_variants=variants, kernels=rows,
                  caution='Instruction counts are not correctness evidence or a complete porting estimate.')
    (ROOT/'analysis/kernel-inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('WMMA variants:', dict(variants))
    print('Functions:', len(rows))
    print('Functions with WMMA:', sum(bool(x['wmma_sites']) for x in rows))
    print('Generic immediate-return functions:', [x['symbol'] for x in rows if x['gfx1030_immediate_return']])
    print('PC-relative functions:', [x['symbol'] for x in rows if x['pc_relative']])
    print('Indirect control functions:', [x['symbol'] for x in rows if x['indirect_control']])

if __name__ == '__main__':
    main()
