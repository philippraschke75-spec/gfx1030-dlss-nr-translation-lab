"""Host-only sweep of the kernarg flags word for a k_swin_var kernel."""
import sys, json, struct
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_var as V, gfx11emu as E
sym = sys.argv[1] if len(sys.argv) > 1 else '_Z10k_swin_varILi32ELb1EEv9VarParams'
rows = []
for flags in range(64):
    ka = V.make_kernarg(flags=flags)
    try:
        g, init, r, secs = V.run(sym, ka, max_steps=6_000_000)
        a = g.regions[1]; ch = np.nonzero(a.arr != init)[0]
        slots = sorted({hex(V.PTR_FIELDS[k]) for k in np.unique(ch // V.SLOT)})
        rows.append(dict(flags=flags, status='OK', steps=r['steps'], written=int(len(ch)), slots=slots, secs=round(secs, 1)))
    except Exception as ex:
        rows.append(dict(flags=flags, status=type(ex).__name__, detail=str(ex)[:110]))
    print(rows[-1], flush=True)
json.dump(rows, open('sweep_var.json', 'w'), indent=1)
ok = [r for r in rows if r['status'] == 'OK']
print('OK %d / %d ; with output written: %d' % (len(ok), len(rows), sum(r['written'] > 0 for r in ok)))
