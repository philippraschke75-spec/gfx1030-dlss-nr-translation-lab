"""Step 1: which release dispatches still run translated code, and their share of frame time.
Times: astra-work c32-optimization profile (per-step medians, RX 6900 XT, 1280x720, attention variant)."""
import json, collections, sys
NR = sys.argv[1] if len(sys.argv) > 1 else 'E:/dlss5-experimental/nr_data'
PROF = 'E:/dlss 5/astra-work/c32-optimization/profile/summary.json'
ov = {l.split('|')[0] for l in open(NR + '/overrides.txt') if l.strip()}
m = [l.strip().split('|') for l in open(NR + '/manifest.txt') if l.strip() and not l.startswith('#')]
base = next(s for s in json.load(open(PROF)) if s['variant'] == 'attention' and not s['moving'])['step_medians_ms']
print(f'frame total {base["-1"]:.2f} ms (profiled build), {len(m)} dispatches')
agg = collections.defaultdict(lambda: [0.0, []])
for i, r in enumerate(m):
    if r[1] not in ov:
        agg[r[1]][0] += base.get(str(i), 0.0)
        agg[r[1]][1].append(i)
for k, (t, st) in sorted(agg.items(), key=lambda x: -x[1][0]):
    print(f'{t:7.3f} ms {100 * t / base["-1"]:5.1f}%  n={len(st):3d}  {k}  steps={st}')
