"""Difftest k_export (_Z8k_export12ExportParams) on the gfx1030 against the gfx1100 original in the emulator.

Kernarg fields as the gfx1100 code reads them (analysis/gfx1100-disassembly.txt, k_export at 0xab200):
    +0x00 ptr  input, 16 B/pixel, RGB f32 read by global_load_b96 at (+0x08*row + x)*16   0xab258, 0xab294
    +0x08 i32  input row stride in elements                                               0xab250, 0xab270
    +0x0c H, +0x10 W   guards row < H, x < W   (s_load_b128 s[4:7], 0xc)                  0xab20c, 0xab220
    +0x14 i32  output row pitch in bytes (s6)                                             0xab2b8
    +0x18 i32  format/mode (s7), compared against 3, 5, 6, 4                              0xab2d4-0xab318
    +0x20 ptr  output
    +0x30 ptr  second input, RGB f32 at 12 B/pixel, subtracted from +0x00                 0xab2c4-0xab300
    +0x38 i32  history flag: 0 branches to the path that reads +0x28 and +0x3c            0xab2a0, 0xab2b4
    +0x28 i32  read only on that path (0 skips a further branch)                          0xab76c
    +0x3c f32  read only on that path, used as 1/+0x3c                                    0xab780
    +0x4c u16  group_size_x                                                               0xab204

Every combination below is packed the way net_frame_full.py packs k_export, except that +0x18 (mode),
+0x28, +0x38 and +0x3c are swept. Input is finite f32 (plus a few specials in SPECIAL=1), and the output
slot is filled with 0xA5 so "written" is measured. The output pitch is 16*W so no path overlaps rows.

env: BH, BW     geometry (default 32, 56)
     PADW       input row stride in elements (default BW + 8, so +0x08 != W is exercised)
     MODES      comma list for +0x18 (default 0..8)
     F28S       comma list for +0x28 (default 0,1)
     HISTS      comma list for +0x38 (default 0,1)
     S1         f32 for +0x3c (default 1.0, the host value)
     SPECIAL    1 = sprinkle inf/nan/-0/denormals into the input
     ULP_TOL    max f32 ULP distance accepted as 'ULP' instead of FAIL (default 16). The +0x38 == 0,
                +0x28 != 0 path runs v_exp_f32 (0xabdf8, 0xac2b4, 0xac73c): the hardware result is ~1 ULP,
                the emulator's exact (gfx11emu.py, np.exp2 in f64), and 16 B/px modes store that f32 as is.
                0 makes every mismatch a FAIL.

usage: net_export.py
"""
import sys, re, struct, subprocess, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gfx11emu as E, run_emu as R, run_var as V, difftest_var as D, kernelspec as K

H = int(os.environ.get('BH', '32'))
W = int(os.environ.get('BW', '56'))
PADW = int(os.environ.get('PADW', str(W + 8)))
MODES = [int(x) for x in os.environ.get('MODES', '0,1,2,3,4,5,6,7,8').split(',')]
F28S = [int(x) for x in os.environ.get('F28S', '0,1').split(',')]
HISTS = [int(x) for x in os.environ.get('HISTS', '0,1').split(',')]
S1 = float(os.environ.get('S1', '1.0'))
ULP_TOL = int(os.environ.get('ULP_TOL', '16'))
PITCH = W * 16

SIN, SOUT, SRGB = 0, 1, 2
NSLOT = 3
S = lambda i: V.ARENA + i * V.SLOT
RUN = D.ROOT / 'build' / 'net_run.exe'
OUT = D.ROOT / 'build' / 'net_export'; OUT.mkdir(parents=True, exist_ok=True)
EXPORT = '_Z8k_export12ExportParams'
MOD = D.ROOT / 'build' / 'kernels-hw-scratch' / (EXPORT + '.co')
EXPLICIT, KSIZE, LDS, HID = K.kernel_meta(EXPORT)
GRID = ((W + 255) // 256, H)
assert H * PADW * 16 <= V.SLOT and H * W * 12 <= V.SLOT and H * PITCH <= V.SLOT


def pack(mode, f28, hist):
    ka = bytearray(KSIZE)
    struct.pack_into('<Q', ka, 0x00, S(SIN))
    struct.pack_into('<i', ka, 0x08, PADW)
    struct.pack_into('<ii', ka, 0x0c, H, W)
    struct.pack_into('<ii', ka, 0x14, PITCH, mode)
    struct.pack_into('<Q', ka, 0x20, S(SOUT))
    struct.pack_into('<i', ka, 0x28, f28)
    struct.pack_into('<Q', ka, 0x30, S(SRGB))
    struct.pack_into('<i', ka, 0x38, hist)
    struct.pack_into('<f', ka, 0x3c, S1)
    for name, val in (('block_count_x', GRID[0]), ('block_count_y', GRID[1]), ('block_count_z', 1),
                      ('group_size_x', 256), ('group_size_y', 1), ('group_size_z', 1), ('grid_dims', 2)):
        if name in HID:
            o, sz = HID[name]
            struct.pack_into('<' + {2: 'H', 4: 'I', 8: 'Q'}[sz], ka, o, val)
    return bytes(ka)


def arena(seed):
    rng = np.random.default_rng(seed + 77)
    a = np.full(V.SLOT * NSLOT, 0xA5, np.uint8)
    for slot, n in ((SIN, V.SLOT // 4), (SRGB, V.SLOT // 4)):
        f = rng.uniform(-2.0, 4.0, n).astype(np.float32)
        if os.environ.get('SPECIAL') == '1':
            idx = rng.integers(0, n, 64)
            f[idx] = np.array([np.inf, -np.inf, np.nan, -0.0, 1e-40, -1e-40, 65504.0, 1e6] * 8, np.float32)
        a[slot * V.SLOT:(slot + 1) * V.SLOT] = f.view(np.uint8)
    return a


def emulate(seed, ka):
    prog = E.load_program(R.DIS, {EXPORT})
    g, KA = V.build(seed, ka, NSLOT)
    g.regions[1].arr[:] = arena(seed)
    for wy in range(GRID[1]):
        for wx in range(GRID[0]):
            E.run_workgroup(prog, g, LDS, 256, {0: KA & 0xffffffff, 1: KA >> 32, 14: wx, 15: wy},
                            max_steps=10_000_000)
    return g.regions[1].arr.copy()


def net_run(seed, ka):
    (OUT / 'm.txt').write_text('%s|%s|0|%d|%d|%d|256|1\n' % (MOD, EXPORT, len(ka), GRID[0], GRID[1]))
    (OUT / 'k.bin').write_bytes(ka)
    (OUT / 'a.bin').write_bytes(arena(seed).tobytes())
    r = subprocess.run([str(RUN), str(OUT / 'm.txt'), str(OUT / 'k.bin'), str(OUT / 'a.bin'), '%x' % V.ARENA],
                       capture_output=True, text=True, timeout=600)
    msg = ((r.stdout or '') + (r.stderr or '')).strip()
    out = np.frombuffer((OUT / 'a.bin').read_bytes(), np.uint8) if r.returncode == 0 else None
    return r.returncode, msg, out


def max_ulp(out, ref, d):
    """Largest f32 ULP distance over the 4-byte words containing a mismatch (both finite), else None."""
    w = np.unique(d // 4)
    a = out[:len(out) // 4 * 4].view(np.int32)[w].astype(np.int64)
    b = ref[:len(ref) // 4 * 4].view(np.int32)[w].astype(np.int64)
    fa, fb = a.astype(np.int32).view(np.float32), b.astype(np.int32).view(np.float32)
    if not (np.isfinite(fa).all() and np.isfinite(fb).all()) or ((a < 0) != (b < 0)).any():
        return None
    return int(np.abs(a - b).max())


def row_bpp(buf):
    """Bytes per pixel actually written in output row 0 (sentinel-measured)."""
    row = buf[SOUT * V.SLOT:SOUT * V.SLOT + PITCH]
    w = np.nonzero(row != 0xA5)[0]
    return 0 if len(w) == 0 else round((int(w[-1]) + 1) / W)


print('k_export difftest: H=%d W=%d padw=%d pitch=%d grid=%s kernarg=%d B S1=%g special=%s'
      % (H, W, PADW, PITCH, GRID, KSIZE, S1, os.environ.get('SPECIAL', '0')))
print('%-5s %-5s %-5s %-6s %-10s %-10s %-8s %s' % ('mode', '+28', 'hist', 'B/px', 'written', 'mismatch', 'max ulp', ''))
ok = True
for hist in HISTS:
    for f28 in F28S:
        for mode in MODES:
            ka = pack(mode, f28, hist)
            rc, msg, out = net_run(1, ka)
            if out is None:
                print('GPU FAIL mode=%d f28=%d hist=%d rc=%d\n%s' % (mode, f28, hist, rc, msg)); raise SystemExit(1)
            V.ARENA = int(re.search(r'arena_dev=0x([0-9a-fA-F]+)', msg)[1], 16)
            ka = pack(mode, f28, hist)          # re-pack with the device arena base the emulator mirrors
            ref = emulate(1, ka)
            base = arena(1)
            d = np.nonzero(out != ref)[0]
            wrote = np.nonzero(ref != base)[0]
            mu = max_ulp(out, ref, d) if len(d) else 0
            good = len(wrote) > 0 and mu is not None and mu <= ULP_TOL
            ok &= good
            verdict = 'PASS' if len(d) == 0 and len(wrote) else ('ULP' if good else 'FAIL')
            print('%-5d %-5d %-5d %-6d %-10d %-10d %-8s %s' % (mode, f28, hist, row_bpp(ref), len(wrote), len(d),
                                                              '-' if mu is None else mu, verdict), flush=True)
raise SystemExit(0 if ok else 1)
