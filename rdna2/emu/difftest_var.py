"""Differential test for k_swin_var kernels: gfx1100 emulator (reference) vs translated gfx1030 kernel.

Run inside the sandbox (python sandbox.py run <sb> <timeout> -- python rdna2/emu/difftest_var.py ...).
usage: difftest_var.py <symbol-suffix e.g. 32_1> <flags,flags,...> [seed] [gx gy]
"""
import sys, os, json, struct, subprocess, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V

ROOT = Path(__file__).resolve().parent.parent
GPU = ROOT / 'build' / 'var_gpu_test.exe'
SYMS = {'32_1': ('_Z10k_swin_varILi32ELb1EEv9VarParams', 15616), '32_0': ('_Z10k_swin_varILi32ELb0EEv9VarParams', None),
        '64_0': ('_Z10k_swin_varILi64ELb0EEv9VarParams', None), '128_0': ('_Z10k_swin_varILi128ELb0EEv9VarParams', None),
        '256_0': ('_Z10k_swin_varILi256ELb0EEv9VarParams', None)}
OUT = ROOT / 'build' / 'emu_var'; OUT.mkdir(parents=True, exist_ok=True)


def group_size(sym):
    sys.path.insert(0, str(ROOT))
    import translate_kernels as t, translate_final_head as base
    blob, sections, syms = base.kd.parse_elf(str(base.INPUT))
    for n, off in base.kd.find_kernel_kds(blob, sections, syms):
        if n == sym:
            return t.dec.dec(base.kd.dump_kd(blob, off))['group']
    raise KeyError(sym)


def emulate(sym, lds, flags, seed, grid, H=16, W=16):
    ka = V.make_kernarg(H=H, W=W, flags=flags, grid=grid)
    prog = E.load_program(R.DIS, {sym})
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    init = g.regions[1].arr.copy()
    steps = 0; t0 = time.time()
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}
            steps += E.run_workgroup(prog, g, lds, 256, sg, max_steps=8_000_000)['steps']
    return bytes(ka), init, g.regions[1].arr.copy(), steps, time.time() - t0


def gpu(module, sym, tag, ka, init, grid):
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
    (d / 'kernarg.bin').write_bytes(ka); (d / 'arena.bin').write_bytes(init.tobytes())
    r = subprocess.run([str(GPU), str(module), sym, str(d / 'kernarg.bin'), str(d / 'arena.bin'), hex(V.ARENA), str(grid[0]), str(grid[1]), '256'],
                       capture_output=True, text=True, timeout=60, env=dict(os.environ, SWIN_TIMEOUT=os.environ.get('SWIN_TIMEOUT', '8')))
    return r.returncode, r.stdout.strip(), (np.fromfile(d / 'arena.bin', np.uint8) if r.returncode == 0 else None)


def where(idx):
    slot = int(idx // V.SLOT)
    return 'slot%d(+%s)+0x%x' % (slot, hex(V.PTR_FIELDS[slot]) if slot < len(V.PTR_FIELDS) else '?', idx % V.SLOT)


if __name__ == '__main__':
    key = sys.argv[1]; flag_list = [int(x, 0) for x in sys.argv[2].split(',')]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    grid = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (2, 2)
    sym, lds = SYMS[key]; lds = lds or group_size(sym)
    module = ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.co')
    rows = []
    for flags in flag_list:
        ka, init, ref, steps, secs = emulate(sym, lds, flags, seed, grid)
        changed = np.nonzero(ref != init)[0]
        tag = 'v%s_f%d_s%d' % (key, flags, seed)
        rc, msg, out = gpu(module, sym, tag, ka, init, grid)
        row = dict(kernel=sym, flags=flags, seed=seed, grid=grid, emu_steps=steps, emu_bytes_written=int(len(changed)),
                   emu_distinct=int(len(np.unique(ref[changed]))) if len(changed) else 0, gpu_rc=rc, gpu=msg)
        if out is None:
            row['status'] = 'GPU_FAIL'
        else:
            diff = np.nonzero(out != ref)[0]
            row['mismatches'] = int(len(diff)); row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
            if len(diff):
                row['first_diffs'] = [where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:5]]
        rows.append(row); print(json.dumps(row), flush=True)
    (OUT / ('results_%s_s%d.json' % (key, seed))).write_text(json.dumps(rows, indent=1))
    print('SUMMARY', [(r['flags'], r['status']) for r in rows])
