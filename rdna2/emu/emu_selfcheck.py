"""Emulator self-check: hash the emulator's output (the full post-run arena) for a fixed set of kernels and inputs,
and time it. Run before and after an emulator change; the hashes must be identical. No GPU needed.
usage: emu_selfcheck.py [--save FILE | --compare FILE]"""
import sys, time, hashlib, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernelspec as K, difftest_spec as DS
CASES = [n for n in ('ffwd2', 'post_block_const', 'post_block') if n in DS.SPECS]
CASES += [n for n in sorted(DS.SPECS) if n not in CASES][:int(os.environ.get('EXTRA', '3'))]
res, tot_steps, tot_t = {}, 0, 0.0
for name in CASES:
    spec = DS.SPECS[name]()
    t = time.time()
    try:
        ka, init, ref, steps = K.emulate(spec, 1)
        h = hashlib.sha256(ref.tobytes()).hexdigest()[:16]
    except Exception as e:
        steps, h = 0, 'ERROR ' + str(e)[:60]
    dt = time.time() - t; tot_steps += steps; tot_t += dt
    res[name] = h
    print('  %-22s %9d steps %6.1fs %7.0f instr/s  %s' % (name, steps, dt, steps / max(dt, 1e-9), h), flush=True)
print('total %d steps in %.1fs -> %.0f instr/s' % (tot_steps, tot_t, tot_steps / tot_t))
if '--save' in sys.argv: Path(sys.argv[sys.argv.index('--save') + 1]).write_text(json.dumps(res, indent=1))
if '--compare' in sys.argv:
    old = json.loads(Path(sys.argv[sys.argv.index('--compare') + 1]).read_text())
    bad = [n for n in res if old.get(n) != res[n]]
    print('IDENTICAL to baseline' if not bad else 'DIFFERS: %s' % bad); raise SystemExit(1 if bad else 0)
