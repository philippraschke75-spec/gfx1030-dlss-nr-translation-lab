"""Does the GPU's pre-block respond sensibly to its input? Emulator not involved.

Every block-0 result so far is "disagrees with gfx11emu", and this project has already had three
cases where that meant the emulator was wrong rather than the translation. This asks a question the
emulator cannot answer: feed the real kernel several different images and see whether its output
varies the way a working convolution must.

A working kernel fed a FLAT image produces low-variance output; fed a detailed frame it produces
high-variance output; fed the same image twice it produces identical output. A kernel whose output
is unrelated to its input gives roughly the same statistics for all of them.

usage: preblock_response.py <color.bin> <src_w> <src_h>   (inside sandbox.py)
"""
import sys, struct, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_var as V, difftest_var as D

import os as _o
VARIANT = _o.environ.get('PRE_VARIANT', '32_1')   # '32_0' = an encoder block
FLAGS = int(_o.environ.get('PRE_FLAGS', '0x14'), 0)
PRE_SYM, PRE_LDS = D.SYMS[VARIANT]
PRE_LDS = PRE_LDS or D.group_size(PRE_SYM)
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'preblock_response'; OUT.mkdir(parents=True, exist_ok=True)
MOD = lambda s: D.ROOT / 'build' / 'kernels-hw-scratch' / (s + '.co')
WEIGHTS = D.ROOT / 'build' / 'weights'

color_path, SRC_W, SRC_H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
import os
H = W = int(os.environ.get('TILE', '64'))   # TILE=8 gives a single workgroup
full = np.fromfile(color_path, np.uint8).reshape(SRC_H, SRC_W, 8)


def act_bytes(C, h, w):
    return C * (-(-h // 4)) * (-(-w // 4)) * 16 * 2


A = 0


def place(n, align=1 << 16):
    global A
    o = (A + align - 1) // align * align
    A = o + n
    return o


NACT = act_bytes(32, H, W)
off_rgb = place(H * W * 12)
off_out = place(NACT)
off_p38 = place(NACT)
off_p48 = place(NACT)
off_scratch = place(NACT)
w0 = (WEIGHTS / 'block0.bin').read_bytes()
off_w0 = place(len(w0))
ARENA_SZ = (A + (1 << 20) - 1) // (1 << 20) * (1 << 20)
BASE = V.ARENA
grid = ((W + 7) // 8, (H + 7) // 8)

# Each pointer field needs its OWN buffer. Pointing them all at one shared scratch made 64
# workgroups collide on it and produced run-to-run differences that looked like a hardware race and
# were entirely this harness's doing. difftest_preblock gives every field a separate 1 MiB slot and
# is deterministic at the same workgroup count.
ka = bytearray(424)
# Sized like difftest_preblock's slots (1 MiB), not by act_bytes. The activation formula gives
# 262144 B here, and at that size the run is not reproducible while difftest_preblock at the
# same 64 workgroups is - so something writes past it. BUF is overridable to test that.
BUF = int(os.environ.get('BUF', str(V.SLOT)))
field_buf = {off: place(BUF) for off in V.PTR_FIELDS}
ARENA_SZ = (A + (1 << 20) - 1) // (1 << 20) * (1 << 20)
for off in V.PTR_FIELDS:
    struct.pack_into('<Q', ka, off, BASE + field_buf[off])
struct.pack_into('<Q', ka, 0x00, 0 if FLAGS == 0x14 else BASE + off_rgb)
struct.pack_into('<Q', ka, 0x30, 0 if FLAGS == 0x14 else BASE + off_scratch)
struct.pack_into('<Q', ka, 0x08, BASE + off_out)
struct.pack_into('<Q', ka, 0x10, BASE + off_w0)
struct.pack_into('<Q', ka, 0x38, BASE + off_p38)
struct.pack_into('<Q', ka, 0x40, BASE + off_rgb)
struct.pack_into('<Q', ka, 0x48, BASE + off_p48)
struct.pack_into('<Q', ka, 0xa0, BASE + off_scratch)
struct.pack_into('<iiii', ka, 0x18, H, W, 0, 0)
struct.pack_into('<I', ka, 0x28, FLAGS)
# +0x50..+0x68 are SCALARS, not pointers: the loop above filled them with buffer addresses.
# The verified pre-block layout zeroes the whole span and then writes 1.0 at +0x50.
for off in (0x50, 0x58, 0x60, 0x68):
    struct.pack_into('<Q', ka, off, 0)
for off in (0x54, 0x5c, 0x64):
    struct.pack_into('<I', ka, off, 0)
for off in range(0x70, 0xa0, 8):
    struct.pack_into('<Q', ka, off, 0)
struct.pack_into('<f', ka, 0x50, 1.0)
struct.pack_into('<III', ka, 0xA8, grid[0], grid[1], 1)
struct.pack_into('<HHH', ka, 0xB4, 256, 1, 1)
(OUT / 'm.txt').write_text('%s|%s|0|%d|%d|%d|256\n' % (MOD(PRE_SYM), PRE_SYM, len(ka), grid[0], grid[1]))
(OUT / 'k.bin').write_bytes(bytes(ka))


def run(rgb, tag):
    # V.build fills the arena with e4m3-shaped random bytes, not zeros. A zeroed arena is a
    # different fixture, and ARENA_FILL exists to tell whether that is what differs.
    if os.environ.get('ARENA_FILL', 'zero') == 'rand':
        _r = np.random.default_rng(7)
        _e = _r.integers(2, 8, ARENA_SZ, dtype=np.uint8)
        _m = _r.integers(0, 8, ARENA_SZ, dtype=np.uint8)
        _s = _r.integers(0, 2, ARENA_SZ, dtype=np.uint8)
        arena = ((_s << 7) | (_e << 3) | _m).astype(np.uint8)
    else:
        arena = np.zeros(ARENA_SZ, np.uint8)
    arena[off_rgb:off_rgb + rgb.nbytes] = rgb.astype(np.float32).view(np.uint8).reshape(-1)
    arena[off_w0:off_w0 + len(w0)] = np.frombuffer(w0, np.uint8)
    (OUT / 'a.bin').write_bytes(arena.tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'm.txt'), str(OUT / 'k.bin'), str(OUT / 'a.bin'),
                        '%x' % BASE], capture_output=True, text=True, timeout=600)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    if r.returncode != 0:
        print('%-12s GPU FAIL: %s' % (tag, msg.splitlines()[-1] if msg else '?'))
        return None
    res = np.fromfile(OUT / 'a.bin', np.uint8)[off_out:off_out + NACT]
    v = res.view(np.float16).astype(np.float32)
    v = v[np.isfinite(v)]
    print('%-12s  nonzero=%5.1f%%  std=%-10.4g  absmean=%-10.4g  distinct bytes=%d'
          % (tag, 100.0 * float((res != 0).mean()), float(v.std()) if v.size else 0,
             float(np.abs(v).mean()) if v.size else 0, len(np.unique(res))))
    return res


tile = full[:H, :W].astype(np.float32)
real = (tile[:, :, :3].view(np.float16)[:, :, :3] if False else
        np.repeat(tile[:, :, :1], 3, axis=2) * 0)    # placeholder, replaced below
# decode the captured RGBA16F tile to float RGB the way k_import does
real = full[:H, :W].view(np.float16).reshape(H, W, 4)[:, :, :3].astype(np.float32)

cases = [
    ('flat 0.5', np.full((H, W, 3), 0.5, np.float32)),
    ('flat 0.0', np.zeros((H, W, 3), np.float32)),
    ('real frame', np.ascontiguousarray(real)),
    ('real frame#2', np.ascontiguousarray(real)),          # identical -> must match exactly
    ('real x0.5', np.ascontiguousarray(real * 0.5)),
    ('noise', np.random.default_rng(3).random((H, W, 3), dtype=np.float32)),
]
print('pre-block output statistics, GPU only (%dx%d tile)\n' % (H, W))
outs = {}
for tag, img in cases:
    outs[tag] = run(np.ascontiguousarray(img), tag)

print()
a, b = outs.get('real frame'), outs.get('real frame#2')
if a is not None and b is not None:
    print('determinism  : identical inputs -> %s'
          % ('IDENTICAL output' if np.array_equal(a, b) else 'DIFFERENT output (!)'))
f, r = outs.get('flat 0.5'), outs.get('real frame')
if f is not None and r is not None:
    print('responds to input: flat vs real differ in %.1f%% of bytes'
          % (100.0 * float((f != r).mean())))
h, s = outs.get('real frame'), outs.get('real x0.5')
if h is not None and s is not None:
    print('responds to scale: real vs real*0.5 differ in %.1f%% of bytes'
          % (100.0 * float((h != s).mean())))

# Determinism is the precondition for every other comparison: if the kernel does not give the same
# answer twice on the same bytes, no difftest against a sequential emulator can ever pass.
print()
print('=== determinism: same input, %d consecutive runs ===' % 5)
img = np.ascontiguousarray(real)
runs = [run(img, 'run %d' % i) for i in range(5)]
ok = [r for r in runs if r is not None]
if len(ok) > 1:
    base = ok[0]
    same = sum(1 for r in ok[1:] if np.array_equal(base, r))
    print('   %d of %d later runs identical to the first' % (same, len(ok) - 1))
    if same != len(ok) - 1:
        d = [int((base != r).sum()) for r in ok[1:]]
        print('   differing bytes vs run 0: %s  (of %d)' % (d, base.size))
        print('   NON-DETERMINISTIC on hardware')
    else:
        print('   deterministic')
