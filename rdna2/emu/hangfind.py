"""Find the first point in execution order where the translated kernel stops finishing on the GPU."""
import sys, subprocess, os, re
from pathlib import Path
import numpy as np
import gfx11emu as E, run_emu as R, difftest as D
ROOT = D.ROOT; BIN = Path(r'C:\Program Files\AMD\ROCm\6.4\bin')
SRC = ROOT / 'build' / 'kernels-hw-scratch' / (R.KERNEL + '.s')
W = ROOT / 'build' / 'emu' / 'bisect'; W.mkdir(parents=True, exist_ok=True)

def make_case():
    p = E.load_program(R.DIS, {R.KERNEL, R.HELPER}); k = min(i for i, x in enumerate(p) if x[0] == 0x2cd00); prog = p[k:] + p[:k]
    g, a = R.build(1, 16, 16, -4, -4, 256, 1 << 18, 1 << 18)
    trace = []
    sg = {0: a['DP'] & 0xffffffff, 1: a['DP'] >> 32, 2: a['KA'] & 0xffffffff, 3: a['KA'] >> 32, 14: 0, 15: 0}
    E.run_workgroup(prog, g, 62592, 256, sg, trace=trace)
    for name, f in (('in', 'in.bin'), ('blob', 'blob.bin')): pass
    init = {r.name: r.arr for r in g.regions}
    return trace, g

def build_variant(addr, tag):
    lines = SRC.read_text().split('\n'); out = []
    for l in lines:
        out.append(l)
        if addr is not None and l == '.Lpc_%x:' % addr: out.append('s_endpgm')
    s = W / (tag + '.s'); s.write_text('\n'.join(out))
    o, co = W / (tag + '.o'), W / (tag + '.co')
    subprocess.run([str(BIN / 'llvm-mc.exe'), '-triple=amdgcn-amd-amdhsa', '-mcpu=gfx1030', '-filetype=obj', str(s), '-o', str(o)], check=True, capture_output=True)
    subprocess.run([str(BIN / 'ld.lld.exe'), '-shared', str(o), '-o', str(co)], check=True, capture_output=True)
    return co

def run(co, files):
    env = dict(os.environ, SWIN_TIMEOUT='4')
    r = subprocess.run([str(D.GPU), str(co), R.KERNEL] + files + ['1', '1'], capture_output=True, text=True, timeout=60, env=env)
    return r.returncode, r.stdout.strip()

if __name__ == '__main__':
    trace, g = make_case()
    print('first-visit trace length', len(trace), flush=True)
    files = []
    for name, data in (('params', bytes(g.regions[0].arr[:80])),):
        (W / 'params.bin').write_bytes(data)
    (W / 'in.bin').write_bytes(np.random.default_rng(1).integers(0, 256, 1 << 18, dtype=np.uint8).tobytes())
    rr = np.random.default_rng(1); 
    # use the emulator's initial buffers so the fixture is identical
    g0, _ = R.build(1, 16, 16, -4, -4, 256, 1 << 18, 1 << 18)
    (W / 'in.bin').write_bytes(g0.regions[2].arr.tobytes()); (W / 'blob.bin').write_bytes(g0.regions[4].arr.tobytes()); (W / 'out.bin').write_bytes(g0.regions[3].arr.tobytes())
    files = [str(W / f) for f in ('params.bin', 'in.bin', 'blob.bin', 'out.bin')]
    lo, hi = 0, len(trace) - 1          # invariant: variant ending at trace[lo] finishes; find first that does not
    co = build_variant(trace[0], 'v0'); rc, out = run(co, files); print('end at first instr:', rc, out, flush=True)
    if rc != 0: print('even the first instruction fails -> launch/setup problem'); sys.exit()
    co = build_variant(None, 'full'); rc, out = run(co, files); print('full kernel:', rc, out, flush=True)
    if rc == 0: print('full kernel completes with this fixture'); sys.exit()
    while hi - lo > 1:
        mid = (lo + hi) // 2
        co = build_variant(trace[mid], 'v%d' % mid); rc, out = run(co, files)
        print('endpgm at trace[%d]=%x -> rc=%d %s' % (mid, trace[mid], rc, out[:60]), flush=True)
        if rc == 0: lo = mid
        else: hi = mid
    print('LAST OK trace[%d]=%x ; FIRST BAD trace[%d]=%x' % (lo, trace[lo], hi, trace[hi]))
