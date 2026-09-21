"""Differential test for k_swin_var kernels: gfx1100 emulator (reference) vs translated gfx1030 kernel.

Run inside the sandbox (python sandbox.py run <sb> <timeout> -- python rdna2/emu/difftest_var.py ...).
usage: difftest_var.py <symbol-suffix e.g. 32_1> <flags,flags,...> [seed] [gx gy]
                      [--height H] [--width W]
"""
import sys, os, json, struct, subprocess, time, re
from pathlib import Path
import numpy as np
import queue
import threading
import hashlib
import argparse
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


def emulate(sym, lds, flags, seed, grid, H=16, W=16, offset_x=-4, offset_y=-4):
    ka = V.make_kernarg(H=H, W=W, offy=offset_x, offx=offset_y, flags=flags, grid=grid)
    prog = E.load_program(R.DIS, {sym})
    g, KA = V.build(seed, ka, len(V.PTR_FIELDS))
    init = g.regions[1].arr.copy()
    steps = 0; t0 = time.time()
    for wy in range(grid[1]):
        for wx in range(grid[0]):
            sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy}
            steps += E.run_workgroup(prog, g, lds, 256, sg, max_steps=8_000_000)['steps']
    return bytes(ka), init, g.regions[1].arr.copy(), steps, time.time() - t0


def gpu(module, sym, tag, ka, init, grid, preflight=None):
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
    (d / 'kernarg.bin').write_bytes(ka); (d / 'arena.bin').write_bytes(init.tobytes())
    cmd = [str(GPU), str(module), sym, str(d / 'kernarg.bin'), str(d / 'arena.bin'), hex(V.ARENA), str(grid[0]), str(grid[1]), '256']
    if preflight is not None:
        p = subprocess.Popen(cmd + ['--preflight'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True,
                             env=dict(os.environ, SWIN_TIMEOUT=os.environ.get('SWIN_TIMEOUT', '8')))
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(p.stdout.readline()), daemon=True).start()
        try:
            line = ready.get(timeout=30)
            match = re.fullmatch(r'READY arena_dev=0x([0-9a-fA-F]+)\s*', line)
            if not match:
                raise RuntimeError('allocation handshake failed: ' + line)
            preflight(int(match[1], 16), (d / 'kernarg.bin.rebased').read_bytes())
            stdout, stderr = p.communicate('GO\n', timeout=60)
            return p.returncode, (line + stdout + stderr).strip(), (np.fromfile(d / 'arena.bin', np.uint8) if p.returncode == 0 else None)
        finally:
            if p.poll() is None:
                p.kill()
            p.wait()
    r = subprocess.run(cmd,
                       capture_output=True, text=True, timeout=60, env=dict(os.environ, SWIN_TIMEOUT=os.environ.get('SWIN_TIMEOUT', '8')))
    return r.returncode, r.stdout.strip(), (np.fromfile(d / 'arena.bin', np.uint8) if r.returncode == 0 else None)


def where(idx):
    slot = int(idx // V.SLOT)
    return 'slot%d(+%s)+0x%x' % (slot, hex(V.PTR_FIELDS[slot]) if slot < len(V.PTR_FIELDS) else '?', idx % V.SLOT)


def parse_case(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('key', choices=SYMS)
    p.add_argument('flags', help='comma-separated flag values')
    p.add_argument('seed', type=int, nargs='?', default=1)
    p.add_argument('gx', type=int, nargs='?', default=2)
    p.add_argument('gy', type=int, nargs='?', default=2)
    p.add_argument('--height', type=int, default=16)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--offset-x', type=int, default=-4)
    p.add_argument('--offset-y', type=int, default=-4)
    a = p.parse_args(argv)
    if min(a.gx, a.gy, a.height, a.width) <= 0:
        p.error('dimensions and grid counts must be positive')
    a.flag_list = [int(x, 0) for x in a.flags.split(',')]
    if any(x < 0 or x > 0xffffffff for x in a.flag_list):
        p.error('flags must fit uint32')
    return a


def case_tag(key, flags, seed, grid, height, width, offset_x=-4, offset_y=-4):
    return 'v%s_f%d_s%d_g%dx%d_h%d_w%d_x%d_y%d' % (key, flags, seed, *grid, height, width, offset_x, offset_y)


if __name__ == '__main__':
    args = parse_case(sys.argv[1:])
    key, flag_list, seed = args.key, args.flag_list, args.seed
    grid = (args.gx, args.gy)
    sym, lds = SYMS[key]; lds = lds or group_size(sym)
    module = ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.co')
    rows = []
    for flags in flag_list:
        ka, init, ref, steps, secs = emulate(sym, lds, flags, seed, grid, args.height, args.width, args.offset_x, args.offset_y)
        changed = np.nonzero(ref != init)[0]
        tag = case_tag(key, flags, seed, grid, args.height, args.width, args.offset_x, args.offset_y)
        def validate_device_fixture(dev, rebased):
            V.ARENA = dev
            actual_ka, actual_init, actual_ref, exact_steps, _ = emulate(sym, lds, flags, seed, grid, args.height, args.width, args.offset_x, args.offset_y)
            if actual_ka != rebased or not np.array_equal(actual_init, init):
                raise RuntimeError('device fixture differs from emulator preflight')
            evidence = OUT / tag
            (evidence / 'reference.bin').write_bytes(actual_ref.tobytes())
            (evidence / 'preflight.json').write_text(json.dumps(dict(
                device_base=hex(dev), steps=exact_steps,
                module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
                kernarg_sha256=hashlib.sha256(rebased).hexdigest(),
                input_sha256=hashlib.sha256(actual_init.tobytes()).hexdigest(),
                reference_sha256=hashlib.sha256(actual_ref.tobytes()).hexdigest()), indent=2))
        rc, msg, out = gpu(module, sym, tag, ka, init, grid, preflight=validate_device_fixture)
        # The exploratory fixture has fields whose pointer-vs-scalar role is
        # unresolved. Rebasing changes their numeric bits as seen by the GPU.
        # Compare against a reference using the actual device base, retaining
        # the original preflight before dispatch.
        if out is not None:
            match = re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)
            if not match:
                raise RuntimeError('missing device base; cannot compare rebased fixture')
            V.ARENA = int(match[1], 16)
            ka, init, ref, steps, secs = emulate(sym, lds, flags, seed, grid, args.height, args.width, args.offset_x, args.offset_y)
            changed = np.nonzero(ref != init)[0]
        row = dict(kernel=sym, flags=flags, seed=seed, grid=grid, emu_steps=steps, emu_bytes_written=int(len(changed)),
                   height=args.height, width=args.width, offset_x=args.offset_x, offset_y=args.offset_y,
                   emu_distinct=int(len(np.unique(ref[changed]))) if len(changed) else 0, gpu_rc=rc, gpu=msg)
        if out is None:
            row['status'] = 'GPU_FAIL'
        else:
            diff = np.nonzero(out != ref)[0]
            row['mismatches'] = int(len(diff)); row['status'] = 'PASS' if len(diff) == 0 and len(changed) > 0 else 'FAIL'
            if len(diff):
                row['first_diffs'] = [where(int(i)) + ' emu=%02x gpu=%02x' % (ref[i], out[i]) for i in diff[:5]]
        if out is not None:
            row['output_sha256'] = hashlib.sha256(out.tobytes()).hexdigest()
        (OUT / tag / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row); print(json.dumps(row), flush=True)
    summary = case_tag(key, flag_list[0], seed, grid, args.height, args.width, args.offset_x, args.offset_y)
    summary += '_flags_' + '_'.join(map(str, flag_list))
    (OUT / ('results_' + summary + '.json')).write_text(json.dumps(rows, indent=1))
    print('SUMMARY', [(r['flags'], r['status']) for r in rows])
    # A printed FAIL must also fail the enclosing sandbox/CI command.
    raise SystemExit(0 if rows and all(r['status'] == 'PASS' for r in rows) else 1)
