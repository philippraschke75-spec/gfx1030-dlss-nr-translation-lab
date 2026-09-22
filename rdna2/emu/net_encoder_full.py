"""Run the whole encoder - all 22 blocks, four stages, C=32/64/128/256 - as ONE net_run sequence.

Everything here is the documented schedule:
  * stages are blocks 1-4 / 5-8 / 9-14 / 15-22 at C=32/64/128/256 (4/4/6/8), launcher A each time
  * per-block shifted-window modes cycle 0,1,2,3 through the origin table
  * flags = (first_in_stage) | 4*(last_in_stage)
  * spatial size halves per stage
  * the stage transition is fused into the last block of each stage, which writes a pooled output
    at VarParams +0x38; that buffer is the next stage's input

The interesting part is the transitions. Within a stage the blocks ping-pong between two buffers -
already proven bit-exact for stage 1 and for a decoder stage - so what this adds is the hand-off
between stages, where a mistake shows up as a mismatch at the first block of the next stage.

usage: net_encoder_full.py [H]    (inside sandbox.py)
"""
import sys, re, struct, subprocess, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D

H0 = int(sys.argv[1]) if len(sys.argv) > 1 else 16
STAGES = [('32_0', [1, 2, 3, 4]), ('64_0', [5, 6, 7, 8]),
          ('128_0', [9, 10, 11, 12, 13, 14]), ('256_0', [15, 16, 17, 18, 19, 20, 21, 22])]
MODES = [(0, 0), (-4, -4), (-4, 0), (0, -4)]
POOL = 0x38                                    # VarParams pooled-output pointer

# Everything must sit BEYOND the VarParams pointer slots. make_kernarg maps each of the 18
# PTR_FIELDS to slots 0..17, and the fields this test does not override still point there - notably
# the scratch pointer at +0xA0, which is slot 17. An earlier version put weights at slot 6 and had
# scratch overwrite block 12's weight record mid-chain.
BASE0 = len(V.PTR_FIELDS)                      # 18
PP = [BASE0, BASE0 + 1]                        # per-stage ping-pong
STAGE_IN = [BASE0 + 2, BASE0 + 3, BASE0 + 4, BASE0 + 5]
WSLOT0 = BASE0 + 6
ALLB = [b for _, bs in STAGES for b in bs]
NSLOT = WSLOT0 + len(ALLB)
S = lambda i: V.ARENA + i * V.SLOT
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_enc_full'; OUT.mkdir(parents=True, exist_ok=True)


def plan():
    out = []
    wi = 0
    for si, (key, blocks) in enumerate(STAGES):
        sym, lds = D.SYMS[key]
        lds = lds or D.group_size(sym)
        h = w = max(H0 >> si, 8)               # halve per stage, but stay off the degenerate sizes
        for i, blk in enumerate(blocks):
            ox, oy = MODES[i % 4]
            last = (i == len(blocks) - 1)
            flags = (1 if i == 0 else 0) | (4 if last else 0)
            grid = ((w - ox + 7) // 8, (h - oy + 7) // 8)
            ka = V.make_kernarg(H=h, W=w, offy=oy, offx=ox, flags=flags, grid=grid)
            src = STAGE_IN[si] if i == 0 else PP[(i + 1) % 2]
            struct.pack_into('<Q', ka, 0x00, S(src))
            struct.pack_into('<Q', ka, 0x08, S(PP[i % 2]))
            struct.pack_into('<Q', ka, 0x10, S(WSLOT0 + wi))
            if last and si + 1 < len(STAGES):
                # the fused downsample: this block also emits the next stage's input
                struct.pack_into('<Q', ka, POOL, S(STAGE_IN[si + 1]))
            out.append((sym, lds, bytes(ka), grid, blk, si, h))
            wi += 1
    return out


def arena(seed):
    g, _ = V.build(seed, plan()[0][2], NSLOT)
    a = g.regions[1].arr.copy()
    for i, blk in enumerate(ALLB):
        w = np.frombuffer((D.ROOT / 'build' / 'weights' / ('block%d.bin' % blk)).read_bytes(), np.uint8)
        a[(WSLOT0 + i) * V.SLOT:(WSLOT0 + i) * V.SLOT + len(w)] = w
    return a


def net_run(seed):
    blob = b''; lines = []
    for sym, lds, ka, grid, blk, si, h in plan():
        o = len(blob); blob += ka
        mod = D.ROOT / 'build' / 'kernels-hw-scratch' / (sym + '.co')
        lines.append('%s|%s|%d|%d|%d|%d|256' % (mod, sym, o, len(ka), grid[0], grid[1]))
    (OUT / 'm.txt').write_text('\n'.join(lines) + '\n')
    (OUT / 'k.bin').write_bytes(blob)
    (OUT / 'a.bin').write_bytes(arena(seed).tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'm.txt'), str(OUT / 'k.bin'), str(OUT / 'a.bin'), '%x' % V.ARENA],
                       capture_output=True, text=True, timeout=3600)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'a.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out


def emulate(seed):
    base = arena(seed); cur = base.copy()
    t0 = time.time()
    for n, (sym, lds, ka, grid, blk, si, h) in enumerate(plan()):
        prog = E.load_program(R.DIS, {sym})
        g, KA = V.build(seed, ka, NSLOT)
        g.regions[1].arr[:] = cur
        for wy in range(grid[1]):
            for wx in range(grid[0]):
                E.run_workgroup(prog, g, lds, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                                max_steps=8_000_000)
        cur = g.regions[1].arr.copy()
        print('    emulated block%-3d (stage %d, %dx%d, grid %s)  %.0fs'
              % (blk, si + 1, h, h, grid, time.time() - t0), flush=True)
    return base, cur


steps = plan()
print('full encoder: %d blocks, %d dispatches, H0=%d' % (len(ALLB), len(steps), H0))
for si, (key, blocks) in enumerate(STAGES):
    print('   stage %d: C=%-4s blocks %s' % (si + 1, key.split('_')[0], blocks))
rc, msg, out = net_run(1)
print('\n', msg.splitlines()[-1] if msg else 'no output')
if out is None:
    print('GPU FAIL rc=%d\n%s' % (rc, msg)); raise SystemExit(1)
V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
print('emulating the same 22-block chain for comparison (slow):', flush=True)
base, ref = emulate(1)
d = np.nonzero(out != ref)[0]
touched = sorted({int(i // V.SLOT) for i in np.nonzero(ref != base)[0]})
print('emulator chain wrote %d bytes into slots %s' % (int((ref != base).sum()), touched))
print('net_run vs emulator over the full encoder: %d mismatches -> %s'
      % (len(d), 'PASS' if len(d) == 0 and touched else 'FAIL'))
raise SystemExit(0 if (len(d) == 0 and touched) else 1)
