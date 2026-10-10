"""Translated k_swin_var<32> vs native k_swin_c32 on the g_A frame (native-mid/vr.py), per step: write set, exact %, PSNR, time."""
import math, sys, numpy as np
sys.path.insert(0, "E:/dlss 5/native-mid")
import vr
MOD = sys.argv[1]
b = np.arange(256); e = (b >> 3) & 15; m = b & 7
LUT = np.where(e == 0, m * 2.0 ** -9, (8 + m) * 2.0 ** (e - 10)) * np.where(b & 128, -1, 1)
def psnr(rb, gb):
    r, g = LUT[rb], LUT[gb]; d = r - g; mse = (d ** 2).mean(); peak = r.max() - r.min()
    return (rb == gb).mean() * 100, (10 * math.log10(peak ** 2 / mse) if mse else float("inf"))
vr.init()
SYM = "_Z10k_swin_varILi32ELb0EEv9VarParams"
steps = [i for i, m in enumerate(vr.MAN) if m[1] == SYM]
tot = [0.0, 0.0]; worst = 1e9
for i in steps:
    ka = vr.kernarg(i); m = vr.MAN[i]
    H, W = np.frombuffer(bytes(ka[0x18:0x20]), np.int32); flags = ka[0x28]; C = 32
    bufs = [(int.from_bytes(ka[8:16], "little") - vr.BASE, C * H * W)]
    if flags & 4: bufs.append((int.from_bytes(ka[0x38:0x40], "little") - vr.BASE, 2 * C * (H // 2) * (W // 2)))
    keep = [vr.read(o, n).copy() for o, n in bufs]
    for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
    vr.step(i); refs = [vr.read(o, n).copy() for o, n in bufs]
    ms_t = vr.step(i, iters=10)
    for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
    ms_n = vr.launch(MOD, "k_swin_c32", ka, (int(m[4]), int(m[5])), 256, iters=10)
    gots = [vr.read(o, n).copy() for o, n in bufs]
    msg = []
    for (o, n), k, rb, gb in zip(bufs, keep, refs, gots):
        wr = rb != 0x7F; same = bool(((gb != 0x7F) == wr).all())
        rb = np.where(wr, rb, k); gb = np.where(wr, gb, k)
        vr.write(o, rb)
        ex, p = psnr(rb, gb); worst = min(worst, p)
        msg.append(f"same writes {same}  exact {ex:6.2f}%  PSNR {p:5.1f} dB")
    print(f"step {i:3d} flags {flags:2d}  translated {ms_t:.3f} ms  native {ms_n:.3f} ms  | " + " | +0x38: ".join(msg), flush=True)
    tot[0] += ms_t; tot[1] += ms_n
print(f"sum: translated {tot[0]:.2f} ms, native {tot[1]:.2f} ms; worst PSNR {worst:.1f} dB")
