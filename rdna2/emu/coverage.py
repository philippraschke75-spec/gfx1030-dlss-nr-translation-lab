"""Which helper/caller instructions does a fixture actually execute? (emulator, all waves)"""
import collections, json
import numpy as np
import gfx11emu as E, run_emu as R, difftest as D
p = E.load_program(R.DIS, {R.KERNEL, R.HELPER}); k = min(i for i, x in enumerate(p) if x[0] == 0x2cd00); prog = p[k:] + p[:k]
ex_seen = set()
orig = E.Exec.compile
def spy(self, i):
    ex_seen.add(i); return orig(self, i)
E.Exec.compile = spy
D.PROG = prog
e = D.emulate(1, 16, 16, -4, -4, (1, 1), mode='small')
tot = collections.Counter(x[1].split('_e')[0] if not x[1].startswith('v_dual') else 'v_dual' for x in prog)
hit = collections.Counter((prog[i][1].split('_e')[0] if not prog[i][1].startswith('v_dual') else 'v_dual') for i in ex_seen)
print('instructions executed at least once: %d of %d (%.1f%%)' % (len(ex_seen), len(prog), 100 * len(ex_seen) / len(prog)))
for fam in ('v_wmma_f32_16x16x16_f16', 'v_div_scale_f32', 'v_div_fmas_f32', 'v_div_fixup_f32', 'v_fma_mix_f32', 'v_fma_mixlo_f16', 'v_perm_b32', 'v_pk_mul_f16', 'v_dual', 'v_ldexp_f32', 'v_log_f32', 'v_rcp_f32', 'v_dot2acc_f32_f16', 'ds_load_2addr_b32', 'flat_load_u16', 'flat_load_b32', 'scratch_load_b32', 'v_med3_f32', 's_swappc_b64', 's_setpc_b64'):
    print('  %-26s executed %4d / %4d sites' % (fam, hit.get(fam, 0), tot.get(fam, 0)))
json.dump(sorted(prog[i][0] for i in ex_seen), open(D.OUTDIR / 'executed_addrs.json', 'w'))
