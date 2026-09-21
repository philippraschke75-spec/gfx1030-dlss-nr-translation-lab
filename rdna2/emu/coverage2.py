"""Lane-aware coverage: a vector instruction counts only if >=1 lane was active (EXEC != 0) when it ran."""
import collections, json
import numpy as np
import gfx11emu as E, run_emu as R, difftest as D
p = E.load_program(R.DIS, {R.KERNEL, R.HELPER}); k = min(i for i, x in enumerate(p) if x[0] == 0x2cd00); prog = p[k:] + p[:k]
active = collections.Counter(); reached = set()
orig = E.Exec.compile
def spy(self, i):
    f = orig(self, i)
    def wrap(w):
        reached.add(i)
        if w.S[E.EXEC] != 0: active[i] += bin(w.S[E.EXEC]).count('1')
        f(w)
    return wrap
E.Exec.compile = lambda self, i: self.cache.setdefault(('w', i), spy(self, i))
D.PROG = prog
mode = 'small'
e = D.emulate(1, 16, 16, -4, -4, (1, 1), mode=mode)
def fam(x): return 'v_dual' if x.startswith('v_dual') else x.split('_e')[0] if x[-4:] in ('_e32', '_e64') else x
vec = [i for i, x in enumerate(prog) if x[1].startswith(('v_', 'ds_', 'flat_', 'global_', 'scratch_'))]
live = [i for i in vec if active[i] > 0]
print('vector instructions: %d; reached %d; with >=1 active lane %d (%.1f%%)' % (len(vec), len([i for i in vec if i in reached]), len(live), 100 * len(live) / len(vec)))
tot = collections.Counter(fam(prog[i][1]) for i in vec); hit = collections.Counter(fam(prog[i][1]) for i in live)
for f in ('v_wmma_f32_16x16x16_f16', 'v_div_scale_f32', 'v_div_fmas_f32', 'v_div_fixup_f32', 'v_fma_mix_f32', 'v_fma_mixlo_f16', 'v_perm_b32', 'v_dual', 'v_ldexp_f32', 'v_log_f32', 'v_rcp_f32', 'v_rsq_f32', 'v_med3_f32', 'v_pk_mul_f16', 'ds_load_2addr_b32', 'flat_load_u16', 'scratch_load_b32'):
    print('  %-26s active in %4d / %4d sites' % (f, hit.get(f, 0), tot.get(f, 0)))
json.dump(sorted(prog[i][0] for i in live), open(D.OUTDIR / 'live_addrs.json', 'w'))
