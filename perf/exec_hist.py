"""Executed-instruction histogram of any gfx11emu-based script, without touching the script.

Runs the given emulator script (e.g. the pre-block difftest) with gfx11emu's per-instruction dispatch
wrapped in a counter, then writes {by_op, by_addr, workgroups, waves} as JSON. Counts are original
gfx1100 instructions summed over all waves; kernel_profile.py --exec-hist divides by `waves`.

Only the emulator side runs - pass whatever flag the script has to skip the GPU half, or just let the
GPU half fail after the emulator finished (the histogram is written in a finally block).

usage (from the directory the script is normally run from):
  python perf/exec_hist.py OUT.json rdna2/emu/difftest_preblock.py 1 8 8
"""
import json, runpy, sys
from collections import Counter
from pathlib import Path

out, script, argv = sys.argv[1], sys.argv[2], sys.argv[2:]
sys.path.insert(0, str(Path(script).resolve().parent))
import gfx11emu as E

by_op, by_addr, n = Counter(), Counter(), dict(wg=0, waves=0, lds_size=0, lds_high=-1)
_compile, _run = E.Exec.compile, E.run_workgroup


def compile(s, i):
    f = _compile(s, i)
    addr, op = s.prog[i][0], s.prog[i][1]

    def counted(w):
        by_op[op] += 1
        by_addr[addr] += 1
        return f(w)
    return counted


def run_workgroup(prog, gmem, lds_size, nthreads, *a, **k):
    n['wg'] += 1
    n['waves'] += -(-nthreads // 32)
    r = _run(prog, gmem, lds_size, nthreads, *a, **k)
    # Highest LDS byte left nonzero: a LOWER bound on the LDS the kernel really touches (zeros written
    # are invisible). Compare with lds_size / .amdhsa_group_segment_fixed_size.
    nz = r['lds'].nonzero()[0]
    n['lds_size'] = max(n['lds_size'], lds_size)
    if len(nz):
        n['lds_high'] = max(n['lds_high'], int(nz[-1]))
    return r


E.Exec.compile = compile
E.run_workgroup = run_workgroup
sys.argv = argv
try:
    runpy.run_path(script, run_name='__main__')
finally:
    Path(out).write_text(json.dumps(dict(workgroups=n['wg'], waves=n['waves'], lds_size=n['lds_size'],
                                         lds_highest_nonzero_byte=n['lds_high'], by_op=dict(by_op.most_common()),
                                         by_addr={'%x' % k: v for k, v in by_addr.items()}), indent=1))
    print('exec_hist: %d workgroups, %d waves, %d executed -> %s' % (n['wg'], n['waves'], sum(by_op.values()), out),
          file=sys.stderr)
