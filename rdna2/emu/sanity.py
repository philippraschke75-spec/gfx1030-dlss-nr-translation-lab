"""Sanity checks that the differential test has discriminating power."""
import subprocess, os, sys
import numpy as np
import difftest as D, run_emu as R
from pathlib import Path
BIN = Path(r'C:\Program Files\AMD\ROCm\6.4\bin'); W = D.OUTDIR / 'sanity'; W.mkdir(parents=True, exist_ok=True)
SRC = D.ROOT / 'build' / 'kernels-hw-scratch' / (R.KERNEL + '.s')

def variant(subs, tag):
    txt = SRC.read_text()
    for a, b in subs:
        assert a in txt, a
        txt = txt.replace(a, b, 1)
    (W / (tag + '.s')).write_text(txt)
    subprocess.run([str(BIN / 'llvm-mc.exe'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1030', '-filetype=obj', str(W / (tag + '.s')), '-o', str(W / (tag + '.o'))], check=True, capture_output=True)
    subprocess.run([str(BIN / 'ld.lld.exe'), '-shared', str(W / (tag + '.o')), '-o', str(W / (tag + '.co'))], check=True, capture_output=True)
    return W / (tag + '.co')

def gpu_with(module, case, tag, grid):
    D.MODULE = module; return D.gpu(case, tag, grid)

if __name__ == '__main__':
    grid = (2, 2)
    e1 = D.emulate(1, 16, 16, -4, -4, grid, mode='small'); out1 = e1['g'].regions[3].arr.copy()
    print('output stats (seed 1): %d bytes, %d distinct values, %.1f%% zero, %.1f%% still 0xA5' % (len(out1[:0x1600]), len(np.unique(out1[:0x1600])), 100 * np.mean(out1[:0x1600] == 0), 100 * np.mean(out1[:0x1600] == 0xA5)))
    e2 = D.emulate(4, 16, 16, -4, -4, grid, mode='small'); out2 = e2['g'].regions[3].arr
    print('emulator: different inputs -> %d differing output bytes (must be > 0)' % int(np.sum(out1 != out2)))
    # blob sensitivity: flip one blob byte in the read range and re-emulate
    import gfx11emu as E
    # 1) good translation on GPU
    good = D.ROOT / 'build' / 'kernels-hw-scratch' / (R.KERNEL + '.co')
    rc, msg, g = gpu_with(good, e1, 'sgood', grid); print('good translation: rc=%d, mismatches vs emulator = %d' % (rc, int(np.sum(g != out1))))
    # 2) deliberately broken translations must FAIL the comparison
    muts = {'wrong_bpermute_offset': [('ds_bpermute_b32', 'ds_bpermute_b32')],
            'v_dot2c_to_add': [('v_dot2c_f32_f16', 'v_dot2c_f32_f16')]}
    txt = SRC.read_text()
    import re
    m = re.search(r'ds_bpermute_b32 (v\d+), (v\d+), (v\d+) offset:8\n', txt)
    cases = {'wrong bpermute row offset (WMMA lowering)': [(m[0], m[0].replace('offset:8', 'offset:16'))]}
    k = txt.index('v_ldexp_f32'); ln = txt[k:txt.index('\n', k)]
    cases['v_ldexp -> v_mul (transcendental path)'] = [(ln, ln.replace('v_ldexp_f32', 'v_mul_f32'))]
    k = txt.index('v_perm_b32'); ln = txt[k:txt.index('\n', k)]
    cases['v_perm selector corrupted'] = [(ln, re.sub(r'0x[0-9a-f]+$', '0x5040302', ln))]
    for name, subs in cases.items():
        co = variant(subs, 'mut%d' % (abs(hash(name)) % 1000))
        rc, msg, g = gpu_with(co, e1, 'smut', grid)
        print('mutant [%s]: rc=%d %s -> mismatches = %s' % (name, rc, msg[:40], int(np.sum(g != out1)) if g is not None else 'n/a'))
