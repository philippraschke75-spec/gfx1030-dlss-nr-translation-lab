"""Smoke test for net_run.exe, the multi-dispatch runner.

Checks two things the single-dispatch harness cannot: that net_run reproduces a known-good result
(so it is equivalent to var_gpu_test for one dispatch), and that two dispatches against one shared
arena chain correctly - the second kernel must see the first one's writes, which is the whole point
of the runner.

usage: net_smoke.py    (inside sandbox.py)
"""
import sys, re, subprocess, struct
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernelspec as K, run_var as V, difftest_var as D
import difftest_spec as S

RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_smoke'; OUT.mkdir(parents=True, exist_ok=True)


def net_run(steps, arena, base):
    """steps: [(spec, kernarg_bytes)] -> runs them in order over one arena, returns final arena."""
    ka_blob = b''
    lines = []
    for spec, ka in steps:
        off = len(ka_blob); ka_blob += bytes(ka)
        mod = D.ROOT / 'build' / 'kernels-hw-scratch' / (spec.sym + '.co')
        gx, gy = spec.grid
        lines.append('%s|%s|%d|%d|%d|%d|%d' % (mod, spec.sym, off, len(ka), gx, gy, spec.threads))
    (OUT / 'manifest.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'kernargs.bin').write_bytes(ka_blob)
    (OUT / 'arena.bin').write_bytes(arena.tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'manifest.txt'), str(OUT / 'kernargs.bin'),
                        str(OUT / 'arena.bin'), '%x' % base],
                       capture_output=True, text=True, timeout=900)
    msg = (r.stdout or '') + (r.stderr or '')
    out = np.frombuffer((OUT / 'arena.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg.strip(), out


print('=== 1. single dispatch: net_run vs the emulator (k_final_head) ===')
spec = S.SPECS['final_head']()
ka, init, ref, steps = K.emulate(spec, 1)
rc, msg, out = net_run([(spec, ka)], init, V.ARENA)
print('   ', msg.splitlines()[-1] if msg else 'no output')
if out is None:
    print('    FAIL rc=%d' % rc); raise SystemExit(1)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
ka, init, ref, steps = K.emulate(spec, 1)
d = np.nonzero(out != ref)[0]
print('    written=%d  mismatches=%d  -> %s' % (int((ref != init).sum()), len(d), 'PASS' if not len(d) else 'FAIL'))
single_ok = len(d) == 0

print()
print('=== 2. chained dispatches: second kernel consumes the first one\'s output ===')
# k_final_head reads slots 0 and 2 and writes slot 1, so running it twice unchanged is legitimately
# idempotent and proves nothing. Instead, point a second dispatch's INPUT at slot 1 - the first
# one's output - and its output at slot 3, then check the whole thing against the emulator running
# the same two steps over one arena. That validates the chaining mechanism, not just the dispatch.
import struct, gfx11emu as E, run_emu as R


def chained_kernargs():
    ka1 = spec.kernarg()
    ka2 = bytearray(spec.kernarg())
    struct.pack_into('<Q', ka2, 0x00, V.ARENA + 1 * V.SLOT)   # input  <- step 1's output
    struct.pack_into('<Q', ka2, 0x08, V.ARENA + 3 * V.SLOT)   # output -> a fresh slot
    return bytes(ka1), bytes(ka2)


def emulate_chain(seed):
    """Both dispatches over one emulator arena, mirroring what net_run does on the GPU."""
    ka1, ka2 = chained_kernargs()
    prog = E.load_program(R.DIS, {spec.sym})
    g, KA1 = V.build(seed, ka1, spec.nslot)
    a = g.regions[1].arr
    a[:] = spec.arena(seed)[:len(a)]
    init = a.copy()
    E.run_workgroup(prog, g, spec.lds, spec.threads, {0: KA1 & 0xffffffff, 1: KA1 >> 32, 14: 0, 15: 0},
                    max_steps=30_000_000)
    g2, KA2 = V.build(seed, ka2, spec.nslot)     # second kernarg, same arena contents carried over
    g2.regions[1].arr[:] = a
    E.run_workgroup(prog, g2, spec.lds, spec.threads, {0: KA2 & 0xffffffff, 1: KA2 >> 32, 14: 0, 15: 0},
                    max_steps=30_000_000)
    return init, g2.regions[1].arr.copy()


init2, ref2 = emulate_chain(1)
ka1, ka2 = chained_kernargs()
rc2, msg2, out2 = net_run([(spec, ka1), (spec, ka2)], init2, V.ARENA)
print('   ', msg2.splitlines()[-1] if msg2 else 'no output')
if out2 is None:
    print('    FAIL rc=%d' % rc2); raise SystemExit(1)
slot3 = slice(3 * V.SLOT, 4 * V.SLOT)
wrote_slot3 = int((ref2[slot3] != init2[slot3]).sum())
d2 = np.nonzero(out2 != ref2)[0]
print('    step 2 wrote %d bytes into slot 3 (reading slot 1 = step 1 output)' % wrote_slot3)
print('    net_run vs emulator over the chain: %d mismatches -> %s'
      % (len(d2), 'PASS' if not len(d2) and wrote_slot3 else 'FAIL'))
chain_ok = len(d2) == 0 and wrote_slot3 > 0
raise SystemExit(0 if (single_ok and chain_ok) else 1)
