import collections, json, re, struct, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'research/src/emulator'))
import p14d_kd as kd
import p14d_dec as dec
symbol = sys.argv[1] if len(sys.argv)>1 else '_Z12k_final_head10HeadParams'
data, sections, syms = kd.parse_elf(str(ROOT/'analysis/installer-bundles/000e3200-3-hipv4-amdgcn-amd-amdhsa--gfx1100.elf'))
print('Sections:', [(s['name'],hex(s['addr']),s['size']) for s in sections])
for name, offset in kd.find_kernel_kds(data,sections,syms,{symbol}):
    dw = kd.dump_kd(data,offset)
    print('Descriptor:', json.dumps(dec.dec(dw),indent=2))
    print('Raw:', [hex(x) for x in dw])
lines = (ROOT/'analysis/gfx1100-disassembly.txt').read_text().splitlines()
active=False; selected=[]
for line in lines:
    if re.match(r'^[0-9a-f]+ <',line):
        if active: break
        active=f'<{symbol}>:' in line
    elif active and '//' in line:
        selected.append(line)
(ROOT/'analysis/selected-kernel.txt').write_text('\n'.join(selected))
print('Mnemonics:',dict(collections.Counter(x.strip().split()[0] for x in selected)))
