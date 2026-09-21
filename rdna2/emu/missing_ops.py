"""List every instruction of a kernel the emulator cannot decode (grouped by mnemonic)."""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R
syms = set(sys.argv[1:]) or {'_Z10k_swin_varILi32ELb1EEv9VarParams'}
prog = E.load_program(R.DIS, syms)
ex = E.Exec(prog); miss = collections.defaultdict(list)
for i in range(len(prog)):
    try: ex.compile(i)
    except Exception as e:
        m = prog[i][1]; miss[m].append((hex(prog[i][0]), prog[i][2][:60], (type(e).__name__+": "+str(e).split(" @")[0])[:44]))
print('instructions', len(prog), '| undecodable mnemonics', len(miss), '| undecodable instructions', sum(len(v) for v in miss.values()))
for m, v in sorted(miss.items(), key=lambda kv: -len(kv[1])):
    print('%5d  %-28s e.g. %s %s' % (len(v), m, v[0][1], '(' + v[0][2] + ')'))
