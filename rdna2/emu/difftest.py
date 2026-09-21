"""Differential test: gfx1100 emulator (reference) vs translated gfx1030 kernel on the RX 6900 XT."""
import sys, os, struct, subprocess, time, json
from pathlib import Path
import numpy as np
import gfx11emu as E
import run_emu as R
ROOT = Path(__file__).resolve().parent.parent
GPU = ROOT / 'build' / 'swin_gpu_test.exe'
MODULE = ROOT / 'build' / 'kernels-hw-scratch' / (R.KERNEL + '.co')
OUTDIR = ROOT / 'build' / 'emu'
PROG = None

def emulate(seed, H, W, offy, offx, grid, io=1 << 18, blob=1 << 18, mode='random'):
    global PROG
    if PROG is None:
        p = E.load_program(R.DIS, {R.KERNEL, R.HELPER}); k = min(i for i, x in enumerate(p) if x[0] == 0x2cd00); PROG = p[k:] + p[:k]
    g, a = R.build(seed, H, W, offy, offx, 256, blob, io)
    rng = np.random.default_rng(seed + 1000)
    if mode.startswith('small'):                              # finite, small-magnitude e4m3: exponent field in [lo,hi]
        lo_e, hi_e = {'small': (2, 7), 'small_hi': (4, 7), 'small_big': (8, 11), 'small_tiny': (0, 3)}[mode]
        for r in g.regions:
            if r.name in ('in', 'blob'):
                e = rng.integers(lo_e, hi_e + 1, r.arr.shape, dtype=np.uint8); m = rng.integers(0, 8, r.arr.shape, dtype=np.uint8); sgn = rng.integers(0, 2, r.arr.shape, dtype=np.uint8)
                r.arr[:] = (sgn << 7) | (e << 3) | m
    if mode == 'nan_free':                                   # avoid e4m3 NaN (0x7f/0xff) in the inputs
        for r in g.regions:
            if r.name in ('in', 'blob'): r.arr[:] = np.where((r.arr & 0x7f) == 0x7f, r.arr & 0xfe, r.arr)
    init = {r.name: r.arr.copy() for r in g.regions}
    ka = g.regions[0].arr
    t0 = time.time(); steps = 0
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            sg = {0: a['DP'] & 0xffffffff, 1: a['DP'] >> 32, 2: a['KA'] & 0xffffffff, 3: a['KA'] >> 32, 14: wx, 15: wy}
            steps += E.run_workgroup(PROG, g, 62592, 256, sg)['steps']
    touched = {r.name: (r.lo, r.hi, r.rd, r.wr) for r in g.regions if r.hi}
    return dict(g=g, init=init, params=bytes(ka[:80]), steps=steps, secs=time.time() - t0, touched=touched)

def gpu(case, tag, grid):
    d = OUTDIR / tag; d.mkdir(parents=True, exist_ok=True)
    (d / 'params.bin').write_bytes(case['params']); (d / 'in.bin').write_bytes(case['init']['in'].tobytes())
    (d / 'blob.bin').write_bytes(case['init']['blob'].tobytes()); (d / 'out.bin').write_bytes(case['init']['out'].tobytes())
    r = subprocess.run([str(GPU), str(MODULE), R.KERNEL, str(d / 'params.bin'), str(d / 'in.bin'), str(d / 'blob.bin'), str(d / 'out.bin'), str(grid[0]), str(grid[1])],
                       capture_output=True, text=True, timeout=90)
    return r.returncode, r.stdout.strip(), (np.fromfile(d / 'out.bin', np.uint8) if r.returncode == 0 else None)

def compare(case, gout):
    emu = case['g'].regions[3].arr
    diff = np.nonzero(emu != gout)[0]
    written = np.nonzero(emu != 0xA5)[0]
    return len(diff), len(written), diff[:8], emu, gout

if __name__ == '__main__':
    cfgs = [dict(seed=1, H=16, W=16, offy=-4, offx=-4, grid=(2, 2), mode='small'),
            dict(seed=2, H=16, W=16, offy=0, offx=0, grid=(2, 2), mode='small_big'),
            dict(seed=3, H=24, W=16, offy=-4, offx=-4, grid=(2, 3), mode='small'),
            dict(seed=4, H=16, W=16, offy=-4, offx=-4, grid=(2, 2), mode='small_hi'),
            dict(seed=5, H=16, W=24, offy=-4, offx=-4, grid=(3, 2), mode='small_tiny')]
    if len(sys.argv) > 1: cfgs = [cfgs[int(x)] for x in sys.argv[1:]]
    results = []
    for i, c in enumerate(cfgs):
        c = dict(c); grid = c.pop('grid'); tag = 'case%d' % i
        e = emulate(grid=grid, **c)
        print(tag, c, 'emu steps=%d %.1fs' % (e['steps'], e['secs']), 'touched', {k: (hex(v[0]), hex(v[1])) for k, v in e['touched'].items()}, flush=True)
        rc, out, gout = gpu(e, tag, grid)
        print('  gpu rc=%d %s' % (rc, out), flush=True)
        if gout is None: results.append(dict(case=tag, status='GPU_FAIL', rc=rc, detail=out)); continue
        nd, nw, first, emu, g = compare(e, gout)
        print('  emu wrote %d bytes; mismatching bytes: %d; first at %s' % (nw, nd, [int(x) for x in first]), flush=True)
        if nd:
            for j in first[:4]: print('    off %d emu=%02x gpu=%02x' % (j, emu[j], g[j]))
        results.append(dict(case=tag, cfg=c, grid=grid, emu_bytes_written=int(nw), mismatches=int(nd), status='PASS' if nd == 0 and nw > 0 else 'FAIL'))
    (OUTDIR / 'difftest_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps([(r['case'], r['status']) for r in results]))
