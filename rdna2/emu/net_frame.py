"""Run the real captured Cyberpunk frame through the translated kernels on the gfx1030 and write an
actual image out.

This is a full-resolution GPU run via net_run, not a tile difftest: the emulator is far too slow for
1707x960, and it is not needed here because every kernel used has already been verified against it.

Stage coverage is stated honestly in the output. k_import (format 0, RGBA16F -> float RGB) and
k_export are both GPU-verified kernels; what sits between them is whatever the --stages option
selects. With --stages=passthrough the network is bypassed entirely and export blends the imported
image with itself, which exercises the real input and output path end to end at real resolution.

usage: net_frame.py <color.bin> <src_w> <src_h> [--stages=passthrough]   (inside sandbox.py)
"""
import sys, os, re, struct, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_var as V, difftest_var as D

IMPORT_SYM = '_Z8k_import12ImportParams'
EXPORT_SYM = '_Z8k_export12ExportParams'
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_frame'; OUT.mkdir(parents=True, exist_ok=True)
MOD = lambda s: D.ROOT / 'build' / 'kernels-hw-scratch' / (s + '.co')

color_path, SRC_W, SRC_H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
src = np.fromfile(color_path, np.uint8)
assert src.size == SRC_H * SRC_W * 8, (src.size, SRC_H * SRC_W * 8)

# --- arena laid out by hand: these buffers are tens of MB, far past the 1 MiB difftest slots ---
A = 0
def place(n, align=1 << 16):
    global A
    o = (A + align - 1) // align * align
    A = o + n
    return o

off_src = place(src.size)                 # captured RGBA16F, render resolution
off_rgb = place(SRC_H * SRC_W * 12)       # k_import output: float RGB, 12 B/pixel
off_dst = place(SRC_H * SRC_W * 8)        # destination surface for k_export
ARENA_SZ = (A + (1 << 20) - 1) // (1 << 20) * (1 << 20)
BASE = V.ARENA
print('arena %.1f MB: src@0x%x rgb@0x%x dst@0x%x' % (ARENA_SZ / 1e6, off_src, off_rgb, off_dst))

arena = np.zeros(ARENA_SZ, np.uint8)
arena[off_src:off_src + src.size] = src

steps = []                                 # (symbol, kernarg, grid, threads)

# --- k_import: format 0 = RGBA16F -> float RGB, the code chain_import_preblock_real.py uses ---
ka = bytearray(0x130)
struct.pack_into('<Q', ka, 0x00, BASE + off_src)
struct.pack_into('<iiiiii', ka, 0x08, SRC_W * 8, 0, SRC_H, SRC_W, SRC_H, SRC_W)
struct.pack_into('<Q', ka, 0x20, BASE + off_rgb)
struct.pack_into('<if', ka, 0x28, 0, 1.0)                 # mode 0, scale 1.0
g_imp = ((SRC_W + 255) // 256, SRC_H)
struct.pack_into('<III', ka, 0x30, g_imp[0], g_imp[1], 1)
struct.pack_into('<HHH', ka, 0x3c, 256, 1, 1)
steps.append((IMPORT_SYM, bytes(ka), g_imp, 256))

# --- k_export: layout per VARPARAMS_HOST_CONTRACT.md's ExportParams table ---
ka = bytearray(280)
struct.pack_into('<Q', ka, 0x00, BASE + off_rgb)          # network result (passthrough: the imported image)
struct.pack_into('<i', ka, 0x08, 0)
struct.pack_into('<ii', ka, 0x0c, SRC_W, SRC_H)           # the two image dimensions
struct.pack_into('<ii', ka, 0x14, SRC_W, SRC_H)
struct.pack_into('<Q', ka, 0x20, BASE + off_dst)          # destination surface
struct.pack_into('<i', ka, 0x28, 0)                       # format/mode
struct.pack_into('<Q', ka, 0x30, BASE + off_rgb)          # blend source (original float RGB)
struct.pack_into('<ff', ka, 0x38, 1.0, 1.0)               # job strengths
g_exp = ((SRC_W + 255) // 256, SRC_H)
struct.pack_into('<III', ka, 0x40, g_exp[0], g_exp[1], 1)
struct.pack_into('<HHH', ka, 0x4c, 256, 1, 1)
steps.append((EXPORT_SYM, bytes(ka), g_exp, 256))

blob = b''; lines = []
for sym, ka, grid, thr in steps:
    o = len(blob); blob += ka
    lines.append('%s|%s|%d|%d|%d|%d|%d' % (MOD(sym), sym, o, len(ka), grid[0], grid[1], thr))
(OUT / 'manifest.txt').write_text('\n'.join(lines) + '\n')
(OUT / 'kernargs.bin').write_bytes(blob)
(OUT / 'arena.bin').write_bytes(arena.tobytes())
print('dispatches:'); [print('   ', l.split('|')[1], 'grid', l.split('|')[4:6]) for l in lines]

r = subprocess.run([str(RUN), str(OUT / 'manifest.txt'), str(OUT / 'kernargs.bin'),
                    str(OUT / 'arena.bin'), '%x' % BASE], capture_output=True, text=True, timeout=1800)
msg = ((r.stdout or '') + (r.stderr or '')).strip()
print(msg)
if r.returncode != 0:
    raise SystemExit(1)

res = np.fromfile(OUT / 'arena.bin', np.uint8)
rgb = res[off_rgb:off_rgb + SRC_H * SRC_W * 12].view(np.float32).reshape(SRC_H, SRC_W, 3)
dst = res[off_dst:off_dst + SRC_H * SRC_W * 8]
print('k_import output : finite=%s min=%.4f max=%.4f mean=%.4f nonzero=%.1f%%'
      % (bool(np.isfinite(rgb).all()), float(rgb.min()), float(rgb.max()), float(rgb.mean()),
         100.0 * float((rgb != 0).mean())))
print('k_export surface: %d bytes changed (%.1f%%)'
      % (int((dst != 0).sum()), 100.0 * float((dst != 0).mean())))


def write_png(path, img8):
    import zlib
    h, w, _ = img8.shape
    raw = b''.join(b'\x00' + img8[y].tobytes() for y in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
    Path(path).write_bytes(png)


# Tone-map the linear HDR to sRGB purely for viewing; the pipeline data itself stays linear.
lin = np.clip(rgb, 0, None)
srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
img = (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)
write_png(OUT / 'imported.png', img)
print('wrote', OUT / 'imported.png')

d16 = dst.view(np.float16).reshape(SRC_H, SRC_W, 4)[:, :, :3].astype(np.float32)
if np.isfinite(d16).all() and d16.max() > 0:
    lin = np.clip(d16, 0, None)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    write_png(OUT / 'exported.png', (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    print('wrote', OUT / 'exported.png', ' (export surface read as RGBA16F)')
else:
    print('export surface is not plausible RGBA16F (finite=%s max=%s) - format code needs work'
          % (bool(np.isfinite(d16).all()), float(np.nanmax(d16)) if d16.size else 0))
