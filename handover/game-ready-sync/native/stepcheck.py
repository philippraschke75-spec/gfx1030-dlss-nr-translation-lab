"""stepcheck.py: run a data dir's manifest on the GPU step by step (translated kernels), and at chosen steps compare a
native override kernel against the translated one on the same inputs: written-byte set, exact %, PSNR, time (median of
10). The translated result feeds the next step. The override is given in overrides.txt syntax (see runtime/OVERRIDES.md).
usage: stepcheck.py <data dir> <overrides.txt> [first last]   env: ONLY=<symbol substring>, DLL=<varrun.dll>
"""
import math, os, re, struct, sys, ctypes, numpy as np

G, OVR = sys.argv[1].replace("\\", "/"), sys.argv[2]
BASE = int(re.search(r"^base=(\d+)", open(f"{G}/meta.txt").read(), re.M).group(1))
lib = ctypes.CDLL(os.environ.get("DLL", "E:/dlss 5/native-mid/varrun.dll"))
P, I, U = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
lib.vr_init.argtypes = [ctypes.c_char_p, U]; lib.vr_write.argtypes = [U, P, U]; lib.vr_read.argtypes = [U, P, U]
lib.vr_launch.argtypes = [ctypes.c_char_p, ctypes.c_char_p, P, I, I, I, I, I, I, P]
KA = open(f"{G}/kernargs.bin", "rb").read()
MAN = [l.split("|") for l in open(f"{G}/manifest.txt").read().splitlines()]
b = np.arange(256); e = (b >> 3) & 15; m = b & 7
LUT = np.where(e == 0, m * 2.0 ** -9, (8 + m) * 2.0 ** (e - 10)) * np.where(b & 128, -1, 1)


def rd(off, n): a = np.empty(n, np.uint8); assert lib.vr_read(off, a.ctypes.data, n) == 0; return a
def wr(off, a): a = np.ascontiguousarray(a); assert lib.vr_write(off, a.ctypes.data, a.nbytes) == 0
def launch(mod, sym, ka, grid, thr, iters=0):
    ms = np.zeros(max(iters, 1), np.float32); ka = bytes(ka)
    rc = lib.vr_launch(mod.encode(), sym.encode(), ka, len(ka), grid[0], grid[1], grid[2], thr, iters, ms.ctypes.data)
    assert rc == 0, (rc, mod, sym); return float(np.median(ms[:iters])) if iters else 0.0


def parse_ovr(path):   # orig_symbol -> (module, symbol, gx, gy, gz, threads, ka_len, recipe)
    d = {}
    for l in open(path):
        l = l.split("#", 1)[0].strip()
        if not l: continue
        f = l.split("|"); d[f[0]] = (f[1].replace("\\", "/"), f[2], f[3:7], int(f[7]), f[8])
    return d


def apply_recipe(ka, n, recipe):
    out = bytearray(n)
    for it in recipe.split():
        if "<-" in it:
            dst, rest = it.split("<-"); src, cnt = rest.split(":"); dst, src, cnt = int(dst, 0), int(src, 0), int(cnt, 0)
            out[dst:dst + cnt] = ka[src:src + cnt]
        else:
            dst, rest = it.split("="); kind, v = rest.split(":"); dst = int(dst, 0)
            if kind == "u32": out[dst:dst + 4] = struct.pack("<I", int(v, 0))
            elif kind == "u64": out[dst:dst + 8] = struct.pack("<Q", int(v, 0))
            elif kind == "f32": out[dst:dst + 4] = struct.pack("<f", float(v))
    return out


Pq = lambda k, o: struct.unpack_from("<Q", k, o)[0] - BASE
Ii = lambda k, o: struct.unpack_from("<i", k, o)[0]
# translated symbol -> output regions (offset, bytes) from the original kernarg and manifest grid
def outs_of(sym, k, gx, gy):
    T = gx * 4
    if sym.startswith("_Z9k_expand2"): return [(Pq(k, 8), T * 65536)]
    if sym.startswith("_Z11k_contract2"): return [(Pq(k, 0x10), T * 16384)]
    if sym.startswith("_Z6k_qkv2"): return [(Pq(k, o), T * 16384) for o in (8, 0x10, 0x18)]
    if sym.startswith("_Z12k_attention2"): return [(Pq(k, 0x18), T * 16384)]
    if sym.startswith("_Z14k_dec_upsample"): return [(Pq(k, 0x10), gx * 8192)]
    if sym.startswith("_Z11k_qkv_attn2"): return [(Pq(k, 8), Ii(k, 0x18) * Ii(k, 0x1c) * 512)]
    if sym.startswith("_Z7k_ffwd2"): return [(Pq(k, 0x10), Ii(k, 0x28) * 8192)]
    if sym.startswith("_Z11k_conv_res2"):
        H, W, T2 = Ii(k, 0x30), Ii(k, 0x34), Ii(k, 0x38); o = [(Pq(k, 0x18), T2 * 8192)]
        if Pq(k, 0x20) + BASE: o.append((Pq(k, 0x20), 512 * H * W))
        return o
    raise KeyError(sym)


def psnr(rb, gb):
    r, g = LUT[rb], LUT[gb]; d = r - g; mse = (d ** 2).mean(); peak = max(r.max() - r.min(), 1e-9)
    return (rb == gb).mean() * 100, (10 * math.log10(peak ** 2 / mse) if mse else float("inf"))


assert lib.vr_init(f"{G}/arena.bin".encode(), BASE) == 0
OV = parse_ovr(OVR); ONLY = os.environ.get("ONLY")
args = [int(a) for a in sys.argv[3:]]
first, last = (args[0], args[1]) if args else (0, len(MAN) - 1)
tot = {}; worst = 1e9
for i in range(0, last + 1):
    mod, sym, ko, kl, gx, gy, thr = MAN[i][0].replace("\\", "/"), MAN[i][1], int(MAN[i][2]), int(MAN[i][3]), int(MAN[i][4]), int(MAN[i][5]), int(MAN[i][6])
    ka = KA[ko:ko + kl]; grid = (gx, gy, 1)
    ov = OV.get(sym) if i >= first and (not ONLY or ONLY in sym) else None
    if not ov: launch(mod, sym, ka, grid, thr); continue
    nmod, nsym, g, nlen, recipe = ov
    ngrid = tuple(int(v) if v != "=" else d for v, d in zip(g[:3], grid)); nthr = int(g[3]) if g[3] != "=" else thr
    nka = apply_recipe(ka, nlen, recipe)
    bufs = outs_of(sym, ka, gx, gy)
    keep = [rd(o, n) for o, n in bufs]
    for o, n in bufs: wr(o, np.full(n, 0x7F, np.uint8))
    launch(mod, sym, ka, grid, thr); refs = [rd(o, n) for o, n in bufs]
    ms_t = launch(mod, sym, ka, grid, thr, iters=10)
    for o, n in bufs: wr(o, np.full(n, 0x7F, np.uint8))
    ms_n = launch(nmod, nsym, nka, ngrid, nthr, iters=10)
    gots = [rd(o, n) for o, n in bufs]
    msg = []
    for (o, n), kp, rb, gb in zip(bufs, keep, refs, gots):
        w = rb != 0x7F; same = bool(((gb != 0x7F) == w).all())
        rb = np.where(w, rb, kp); gb = np.where(w, gb, kp); wr(o, rb)
        ex, p = psnr(rb, gb); worst = min(worst, p)
        msg.append(f"writes {same} exact {ex:7.3f}% {p:5.1f} dB")
    t = tot.setdefault(nsym, [0.0, 0.0, 0]); t[0] += ms_t; t[1] += ms_n; t[2] += 1
    print(f"step {i:3d} {nsym:18s} translated {ms_t:.3f} native {ms_n:.3f} ms | " + " | ".join(msg), flush=True)
for s, (a, b_, n) in tot.items(): print(f"{s}: {n} steps, translated {a:.3f} ms, native {b_:.3f} ms, saved {a - b_:.3f} ms")
print(f"worst PSNR {worst:.1f} dB")
