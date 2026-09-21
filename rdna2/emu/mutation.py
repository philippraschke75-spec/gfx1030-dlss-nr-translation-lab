"""Mutation testing of the differential test: corrupt executed sites, expect GPU-vs-emulator mismatches."""
import json, re, random, subprocess, sys
import numpy as np
import difftest as D, run_emu as R, sanity as S
mut = {  # family: (regex on the translated line, replacement)
    'v_perm_b32': (r'(v_perm_b32 .*, )0x[0-9a-f]+$', r'\g<1>0x0c0c0c0c'),
    'v_ldexp_f32': (r'^v_ldexp_f32', 'v_mul_f32'),
    'v_log_f32': (r'^v_log_f32', 'v_rcp_f32'),
    'v_rcp_f32': (r'^v_rcp_f32', 'v_rsq_f32'),
    'v_fma_mix_f32': (r'op_sel_hi:\[1,1,0\]', 'op_sel_hi:[0,0,0]'),
    'v_med3_f32': (r'^v_med3_f32', 'v_max3_f32'),
    'v_cvt_f16_f32': (r'^v_cvt_f16_f32', 'v_cvt_f16_f32_e64 clamp'),   # placeholder replaced below
    'v_fma_f32': (r'^v_fma_f32 (v\d+), (.+), (.+), (.+)$', r'v_fma_f32 \1, \2, \3, 0'),
    'v_div_fixup_f32': (r'^v_div_fixup_f32 (v\d+), (.+), (.+), (.+)$', r'v_mov_b32 \1, \2'),
    'ds_bpermute_b32': (r'ds_bpermute_b32 (v\d+), (v\d+), (v\d+) offset:8$', r'ds_bpermute_b32 , ,  offset:24'),
    'v_fma_mix_f32_zero': (r'^v_fma_mix_f32 (v\d+), .*$', r'v_mov_b32 \g<1>, 0'),
    'v_fma_mixlo_f16': (r'op_sel_hi:\[0,1,0\]', 'op_sel_hi:[0,0,0]'),
}
del mut['v_cvt_f16_f32']
random.seed(11)
executed = set(json.load(open(D.OUTDIR / 'executed_addrs.json')))
lines = S.SRC.read_text().split('\n')
# map addr -> (line index of label)
lab = {int(l[5:-1], 16): i for i, l in enumerate(lines) if l.startswith('.Lpc_') and l.endswith(':')}
sites = {}
ONLY = sys.argv[1:] or list(mut)
for fam in [f for f in mut if f in ONLY]:
    cands = []
    scan = fam.replace('_zero', '')
    for a in sorted(executed):
        i = lab.get(a)
        if i is None: continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith('.Lpc_'):
            l = lines[j]
            if not l.startswith('//') and re.search(mut[fam][0], l): cands.append((a, j)); break
            j += 1
    sites[fam] = random.sample(cands, min(3, len(cands))) if fam != 'ds_bpermute_b32' else random.sample(cands, min(3, len(cands)))
    print(fam, 'executed candidate sites:', len(cands), flush=True)
grid = (2, 2)
e1 = D.emulate(1, 16, 16, -4, -4, grid, mode=__import__('os').environ.get('MODE','small')); ref = e1['g'].regions[3].arr.copy()
killed = survived = 0; rows = []
for fam, ss in sites.items():
    for a, j in ss:
        L = list(lines); pat, rep = mut[fam]; L[j] = re.sub(pat, rep, L[j])
        tag = 'm_%s_%x' % (fam, a); path = S.W / (tag + '.s'); path.write_text('\n'.join(L))
        try:
            subprocess.run([str(S.BIN / 'llvm-mc.exe'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1030', '-filetype=obj', str(path), '-o', str(S.W / (tag + '.o'))], check=True, capture_output=True)
            subprocess.run([str(S.BIN / 'ld.lld.exe'), '-shared', str(S.W / (tag + '.o')), '-o', str(S.W / (tag + '.co'))], check=True, capture_output=True)
        except subprocess.CalledProcessError as ex:
            rows.append((fam, hex(a), 'ASSEMBLE_ERROR')); print(fam, hex(a), 'assemble error', ex.stderr.decode()[:80]); continue
        rc, msg, g = S.gpu_with(S.W / (tag + '.co'), e1, 'smut', grid)
        n = int(np.sum(g != ref)) if g is not None else -1
        state = 'KILLED' if n != 0 else 'SURVIVED'
        killed += state == 'KILLED'; survived += state == 'SURVIVED'
        rows.append((fam, hex(a), state, n, rc)); print('%-18s @%s %-8s mismatching bytes=%d rc=%d' % (fam, hex(a), state, n, rc), flush=True)
print('mutation score: %d killed / %d survived (of %d executed-site mutants)' % (killed, survived, killed + survived))
json.dump(rows, open(D.OUTDIR / 'mutation_results.json', 'w'))
