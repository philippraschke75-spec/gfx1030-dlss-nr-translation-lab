"""Dynamic instruction mix of one kernel on RDNA2: run the gfx1100 original in the emulator with per-address counts,
then weight every executed original instruction by what the translation emits for it (the .s in KERNEL_DIR).
usage: profile_mix.py <difftest script> <args...>   (the script must call E.run_workgroup(... max_steps=...))"""
import sys, os, re, collections
from pathlib import Path
here = Path(__file__).resolve().parent; sys.path.insert(0, str(here))
script, args = sys.argv[1], sys.argv[2:]
sys.argv = [script] + args
src = open(here / script, encoding='utf-8').read()
CNT = {}
src = re.sub(r"max_steps=([0-9_]+)\)", r"max_steps=\1, counts=CNT)", src)
src = src.split('\nka, init, ref, steps = emulate()')[0] + '\nemulate()\n' if 'ka, init, ref, steps = emulate()' in src else src
g = {'__file__': str(here / script), '__name__': 'prof', 'CNT': CNT}
exec(compile(src, script, 'exec'), g)
sym = g['SYM']
S = (here.parent / 'build' / os.environ.get('KERNEL_DIR', 'kernels-hw-scratch') / (sym + '.s')).read_text(errors='replace').splitlines()
cat = lambda m: ('lds' if m.startswith('ds_') else 'vmem' if m.startswith(('global_', 'buffer_', 'scratch_', 'flat_')) else
                 'smem' if m.startswith(('s_load', 's_buffer')) else 'wait' if m.startswith('s_waitcnt') else
                 'branch' if m.startswith(('s_cbranch', 's_branch', 's_setpc', 's_swappc')) else 'nop' if m == 's_nop' else
                 'barrier' if m == 's_barrier' else 'salu' if m.startswith('s_') else 'valu' if m.startswith('v_') else 'other')
emit = {}
cur = None
for l in S:
    m = re.match(r'^\.Lpc_([0-9a-f]+):$', l)
    if m: cur = int(m[1], 16); emit[cur] = collections.Counter(); continue
    t = l.strip()
    if cur is None or not t or t.startswith(('//', '.')) or t.endswith(':'): continue
    emit[cur][cat(t.split()[0])] += 1
tot = collections.Counter(); orig = collections.Counter(); per_pc = {}
import gfx11emu as E, run_emu as R
ops = {a: op for a, op, _, _ in E.load_program(R.DIS, {sym, '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'})}
for a, n in CNT.items():
    e = emit.get(a, collections.Counter({'?': 1}))
    for k, v in e.items(): tot[k] += n * v
    orig[ops.get(a, '?')] += n
    per_pc[a] = n * sum(e.values())
T = sum(tot.values())
print('%s: %d original wave-instructions executed -> %d emitted wave-instructions (%.2fx)' % (sym, sum(CNT.values()), T, T / sum(CNT.values())))
for k, v in tot.most_common(): print('  %-8s %10d  %5.1f%%' % (k, v, 100 * v / T))
print('top original opcodes by executions:')
for k, v in orig.most_common(12): print('  %-34s %9d' % (k, v))
print('in swin_layer (0xbd00..) vs kernel body:', sum(v for a, v in per_pc.items() if a < 0x2de00), 'vs', sum(v for a, v in per_pc.items() if a >= 0x2de00))
# hot regions: contiguous address runs weighted by emitted instructions
if os.environ.get('HOT'):
    import itertools
    addrs = sorted(per_pc)
    buckets = collections.Counter(); mix = collections.defaultdict(collections.Counter)
    for a in addrs:
        b = a // 0x80 * 0x80; buckets[b] += per_pc[a]; mix[b][ops.get(a, '?')] += CNT[a]
    print('hottest 128-byte regions (share of emitted wave-instructions):')
    for b, v in buckets.most_common(int(os.environ['HOT'])):
        print('  %#7x %5.1f%%  %s' % (b, 100 * v / T, ', '.join('%s x%d' % (k.replace('_e32', '').replace('_e64', ''), n) for k, n in mix[b].most_common(5))))
if os.environ.get('DUAL'):
    # executed v_dual_* and how many are hazard-free (no half's destination is read by the other half)
    progl = {a: (op, args) for a, op, args, _ in E.load_program(R.DIS, {sym, '_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti'})}
    def regs(t): return set(re.findall(r'\bv(\d+)\b', t))
    n_all = n_safe = 0; kinds = collections.Counter()
    for a, n in CNT.items():
        op, args = progl.get(a, ('', ''))
        if not op.startswith('v_dual_'): continue
        text = op + ' ' + args
        halves = text.split(' :: ')
        d = []; s_ = []
        for h in halves:
            mn, rest = h.split(None, 1); dst, src = rest.split(',', 1)
            d.append(regs(dst)); s_.append(regs(src) | (regs(dst) if 'fmac' in mn or 'fmaak' in mn or 'fmamk' in mn else set()))
        safe = not (d[0] & s_[1])            # first half is emitted first; only its write can corrupt the second's read
        n_all += n; n_safe += n * safe
        kinds[(halves[0].split()[0], halves[1].split()[0].lstrip(':').strip())] += n
    print('v_dual executed %d wave-instr (%.1f%% of original); hazard-free %d (%.1f%%)' % (n_all, 100 * n_all / sum(CNT.values()), n_safe, 100 * n_safe / max(n_all, 1)))
    print('emitted now %d, would be %d with direct emission when safe -> saves %.1f%% of emitted' % (4 * n_all, 4 * n_all - 2 * n_safe, 100 * 2 * n_safe / T))
