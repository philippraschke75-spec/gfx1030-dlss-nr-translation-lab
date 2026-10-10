"""GPU check of the asm-rewritten release kernels (NOT run by the author; needs the GPU lock - use gpu_validate.sh).

Runs the release runtime (NR_PROF build) on the release data and on the same data with build_asm\\kernels
overlaid, A/B/B/A, 8 frames each on shifted 1280x720 input. Requires every output frame byte-identical
to the baseline and prints per-kernel GPU ms (frame 5, NR_PROF) for the rewritten kernels.
"""
import os, re, shutil, subprocess, sys, tempfile, collections
from pathlib import Path
import numpy as np

DLL = os.environ.get('NR_DLL_DIR', r'E:\dlss-runtime\perf10\base')          # nr_runtime.dll with NR_PROF
BASE = Path(os.environ.get('NR_BASE', r'E:\dlss5-experimental\nr_data'))
ASM = Path(os.environ.get('NR_ASM', r'E:\dlss-runtime\build_asm'))
E2E = Path(__file__).resolve().parent.parent / 'handover/game-ready-sync/native/e2e.py'
FRAMES, CHILD_TIMEOUT = 8, 30

tmp = Path(tempfile.mkdtemp(prefix='nr_asm_'))
cand = tmp / 'nr_data'
shutil.copytree(BASE, cand)
for f in (ASM / 'kernels').glob('*.co'):
    shutil.copy2(f, cand / 'kernels' / f.name)
for f in ('manifest.txt', 'overrides.txt'):
    if (ASM / f).exists():
        shutil.copy2(ASM / f, cand / f)
rewritten = {f.stem for f in (ASM / 'kernels').glob('*.co')}

ref, prof = None, collections.defaultdict(list)
for i, (tag, data) in enumerate([('base', BASE), ('asm', cand), ('asm', cand), ('base', BASE)]):
    out = tmp / ('%d-%s.npy' % (i, tag))
    p = subprocess.run([sys.executable, '-B', str(E2E), DLL, str(data), str(out), str(FRAMES)],
                       env=dict(os.environ, NR_PROF='1'), capture_output=True, text=True, timeout=CHILD_TIMEOUT)
    print(tag, p.stdout.strip(), flush=True)
    assert p.returncode == 0, p.stderr[-2000:]
    o = np.load(out)
    if ref is None:
        ref = o
    assert o.tobytes() == ref.tobytes(), '%s run %d: output differs from baseline (%d values)' % (tag, i, int((o != ref).sum()))
    t = collections.defaultdict(float)
    for m in re.finditer(r'^prof\s+\d+\s+([\d.]+)\s+(\S+)', p.stderr, re.M):
        if m[2] in rewritten:
            t[m[2]] += float(m[1])
    prof[tag].append(t)
print('PASS: all %d frames x 4 runs byte-identical to the release kernels' % FRAMES)
for k in sorted(rewritten):
    b = [r[k] for r in prof['base']]; a = [r[k] for r in prof['asm']]
    print('%-42s base %s  asm %s ms' % (k, ' '.join('%.3f' % x for x in b), ' '.join('%.3f' % x for x in a)))
shutil.rmtree(tmp, ignore_errors=True)
