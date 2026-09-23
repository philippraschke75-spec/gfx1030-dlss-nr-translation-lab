"""Run the real captured frame through the network on the gfx1030 at full resolution, GPU only.

This is the frame path, not a difftest. The emulator runs at ~31k instructions/second, so checking a
1707x960 chain against it would take days; every kernel here is instead covered by its own small
fixture, with one loud exception recorded below.

Buffers are allocated by the documented activation formula C * ceil(H/4) * ceil(W/4) * 16 with
(C,H,W) from the ctx+0x190 stage tuple, so nothing here reproduces the host's allocator.

HONEST STATUS, kept in the output so a passing run cannot imply more than it shows:
  * k_import (format 0) is GPU-verified at this exact resolution.
  * block 0, the pre-block k_swin_var<32,true>, FAILS its own difftest by 28,825 bytes on valid
    float input. It is the network's entry point, so anything downstream inherits that.
  * k_export has never been validated - it needs the network's own tiled output at +0x00.

usage: net_frame_full.py <color.bin> <src_w> <src_h> [--stop=<stage>]   (inside sandbox.py)
"""
import sys, struct, subprocess, zlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_var as V, difftest_var as D

IMPORT_SYM = '_Z8k_import12ImportParams'
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_frame_full'; OUT.mkdir(parents=True, exist_ok=True)
MOD = lambda s: D.ROOT / 'build' / 'kernels-hw-scratch' / (s + '.co')
WEIGHTS = D.ROOT / 'build' / 'weights'

color_path, SRC_W, SRC_H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
stop = 'preblock'
for a in sys.argv[4:]:
    if a.startswith('--stop='):
        stop = a.split('=', 1)[1]

src = np.fromfile(color_path, np.uint8)
assert src.size == SRC_H * SRC_W * 8, (src.size, SRC_H * SRC_W * 8)

# ---------------------------------------------------------------- arena, in real bytes
A = 0
placements = []


def place(name, n, align=1 << 16):
    global A
    o = (A + align - 1) // align * align
    A = o + n
    placements.append((name, o, n))
    return o


ELEM = int(__import__('os').environ.get('ACT_ELEM', '2'))     # activations are f16


def act_bytes(C, h, w):
    """C * ceil(H/4) * ceil(W/4) * 16, which counts ELEMENTS - 16 per 4x4 tile per channel.

    The contract states the formula without a unit. Sized as bytes the pre-block writes past the end
    of the arena and trips the guards, so the 16 is an element count and the buffer is that times the
    element width. ACT_ELEM overrides it for bisecting.
    """
    return C * (-(-h // 4)) * (-(-w // 4)) * 16 * ELEM


off_src = place('src RGBA16F', src.size)
off_rgb = place('import RGB32F', SRC_H * SRC_W * 12)

# stage 1 is at render resolution; each later stage halves spatially and doubles channels
S1 = act_bytes(32, SRC_H, SRC_W)
off_a = place('act A (C=32)', S1)
off_b = place('act B (C=32)', S1)
off_scratch = place('pre-block scratch', S1)
off_p38 = place('pre-block +0x38', S1)

off_p48 = place('pre-block +0x48', S1)          # the "optional 3rd buffer"; difftest_preblock
                                                # leaves make_kernarg's pointer here rather than
                                                # nulling it, so it is not optional in practice
w_block0 = (WEIGHTS / 'block0.bin').read_bytes()
off_w0 = place('block0 weights', len(w_block0))

ARENA_SZ = (A + (1 << 20) - 1) // (1 << 20) * (1 << 20)
BASE = V.ARENA

print('=== arena map ===')
for nm, o, n in placements:
    print('  %-22s @0x%08x  %10d B  %7.1f MB' % (nm, o, n, n / 1e6))
print('  total %.1f MB\n' % (ARENA_SZ / 1e6))

arena = np.zeros(ARENA_SZ, np.uint8)
arena[off_src:off_src + src.size] = src
arena[off_w0:off_w0 + len(w_block0)] = np.frombuffer(w_block0, np.uint8)

steps = []

# ---------------------------------------------------------------- k_import (verified at this size)
ka = bytearray(0x130)
struct.pack_into('<Q', ka, 0x00, BASE + off_src)
struct.pack_into('<iiiiii', ka, 0x08, SRC_W * 8, 0, SRC_H, SRC_W, SRC_H, SRC_W)
struct.pack_into('<Q', ka, 0x20, BASE + off_rgb)
struct.pack_into('<if', ka, 0x28, 0, 1.0)
g_imp = ((SRC_W + 255) // 256, SRC_H)
struct.pack_into('<III', ka, 0x30, g_imp[0], g_imp[1], 1)
struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
steps.append((IMPORT_SYM, bytes(ka), g_imp, 256))

# ---------------------------------------------------------------- block 0, the pre-block
# Convention from chain_import_preblock_real.py, NOT the encoder one: flags 0x14, +0x00 and +0x30
# null, float-RGB input at +0x40. Built by hand rather than via make_kernarg because that fills the
# pointer fields with 1 MiB difftest slots, which are far too small at this resolution.
PRE_SYM, PRE_LDS = D.SYMS['32_1']
g_pre = ((SRC_W + 7) // 8, (SRC_H + 7) // 8)
ka = bytearray(424)
struct.pack_into('<Q', ka, 0x00, 0)
struct.pack_into('<Q', ka, 0x08, BASE + off_a)              # output
struct.pack_into('<Q', ka, 0x10, BASE + off_w0)             # block0 weights
struct.pack_into('<Q', ka, 0x30, 0)
struct.pack_into('<Q', ka, 0x38, BASE + off_p38)
struct.pack_into('<Q', ka, 0x40, BASE + off_rgb)            # float RGB from k_import
struct.pack_into('<Q', ka, 0x48, BASE + off_p48)
struct.pack_into('<iiii', ka, 0x18, SRC_H, SRC_W, 0, 0)
struct.pack_into('<I', ka, 0x28, 0x14)
struct.pack_into('<f', ka, 0x50, 1.0)
struct.pack_into('<Q', ka, 0xa0, BASE + off_scratch)
struct.pack_into('<III', ka, 0xA8, g_pre[0], g_pre[1], 1)
struct.pack_into('<HHH', ka, 0xB4, 256, 1, 1)
steps.append((PRE_SYM, bytes(ka), g_pre, 256))

# ---------------------------------------------------------------- dispatch
blob = b''; lines = []
for sym, k, grid, thr in steps:
    o = len(blob); blob += k
    lines.append('%s|%s|%d|%d|%d|%d|%d' % (MOD(sym), sym, o, len(k), grid[0], grid[1], thr))
(OUT / 'manifest.txt').write_text('\n'.join(lines) + '\n')
(OUT / 'kernargs.bin').write_bytes(blob)
(OUT / 'arena.bin').write_bytes(arena.tobytes())

print('=== dispatches ===')
for l in lines:
    f = l.split('|')
    print('  %-44s grid %sx%s' % (f[1], f[4], f[5]))

r = subprocess.run([str(RUN), str(OUT / 'manifest.txt'), str(OUT / 'kernargs.bin'),
                    str(OUT / 'arena.bin'), '%x' % BASE], capture_output=True, text=True, timeout=1800)
msg = ((r.stdout or '') + (r.stderr or '')).strip()
print('\n' + msg)
if r.returncode != 0:
    raise SystemExit(1)

res = np.fromfile(OUT / 'arena.bin', np.uint8)


def report(name, buf):
    f = buf.view(np.float16).astype(np.float64)      # activations are f16, not f32
    finite = bool(np.isfinite(f).all())
    nz = 100.0 * float((buf != 0).mean())
    print('  %-18s finite=%-5s min=%-12.4g max=%-12.4g mean=%-12.4g nonzero=%.1f%%'
          % (name, finite, float(np.nanmin(f)), float(np.nanmax(f)), float(np.nanmean(f)), nz))
    return finite, nz


print('\n=== buffers after the run ===')
rgb = res[off_rgb:off_rgb + SRC_H * SRC_W * 12].view(np.float32).reshape(SRC_H, SRC_W, 3)
report('k_import RGB', res[off_rgb:off_rgb + SRC_H * SRC_W * 12])
a_fin, a_nz = report('block0 out', res[off_a:off_a + S1])
report('block0 +0x38', res[off_p38:off_p38 + S1])

if a_nz < 1.0:
    print('\n  block0 wrote essentially nothing - the pre-block did not run as intended.')


def write_png(path, img8):
    h, w, _ = img8.shape
    raw = b''.join(b'\x00' + img8[y].tobytes() for y in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'
                           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def tonemap(lin):
    lin = np.clip(np.nan_to_num(lin, nan=0.0, posinf=0.0, neginf=0.0), 0, None)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


write_png(OUT / 'imported.png', tonemap(rgb))
print('\nwrote', OUT / 'imported.png')

# The pre-block output is C=32 in the network's tiled layout, not an image. Showing the first three
# channels of each 4x4 tile is not a picture of the frame - it is a sanity view: structure here means
# the kernel saw the image, noise means it did not.
tiles_y, tiles_x = -(-SRC_H // 4), -(-SRC_W // 4)
a16 = res[off_a:off_a + S1].view(np.float16)   # f16 confirmed: f32 gives 1e38 nonsense
if a16.size >= tiles_y * tiles_x * 8:
    v = a16[:tiles_y * tiles_x * 8].reshape(tiles_y, tiles_x, 8)[:, :, :3].astype(np.float32)
    if np.isfinite(v).any():
        lo, hi = float(np.nanpercentile(v, 1)), float(np.nanpercentile(v, 99))
        if hi > lo:
            write_png(OUT / 'block0_preview.png',
                      tonemap((np.nan_to_num(v) - lo) / (hi - lo)))
            print('wrote', OUT / 'block0_preview.png', '(channel view, NOT the frame)')
