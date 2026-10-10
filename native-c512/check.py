"""Run the translated steps of a range on g_A in order; for the overridden symbols also run the native kernel on the
same input and compare (0x7F sentinel marks unwritten bytes). The translated result feeds the next step.
usage: check.py [first last]   (default: 24 54, then 100 131)"""
import math, os, sys, struct, numpy as np, torch
sys.path.insert(0, "E:/dlss 5/native-mid"); sys.path.insert(0, "E:/dlss 5/dlssnr-pytorch")
import vr
from dlssnr.quant import e4m3_decode
HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.environ.get("C512_CO", os.path.join(HERE, "c512.co"))
P = lambda k, o: struct.unpack_from("<Q", k, o)[0]
I = lambda k, o: struct.unpack_from("<i", k, o)[0]


def conv2_outs(k):
    H, W, T = I(k, 0x30), I(k, 0x34), I(k, 0x38)
    o = [(P(k, 0x18) - vr.BASE, T * 8192)]
    if P(k, 0x20): o.append((P(k, 0x20) - vr.BASE, 512 * H * W))
    return o


NATIVE = {  # translated symbol -> (native symbol, grid(manifest gx, gy), outputs(kernarg))
    "_Z11k_conv_res211Conv2Params": ("k_conv2_n", lambda gx, gy: (gx, 8), conv2_outs),
    "_Z9k_expand212ExpandParams": ("k_vit_expand_n", lambda gx, gy: (gx, 64), lambda k: [(P(k, 8) - vr.BASE, 448 * 4096)]),
    "_Z11k_contract212ConvParams1d": ("k_vit_contract_n", lambda gx, gy: (gx, 16), lambda k: [(P(k, 0x10) - vr.BASE, 448 * 1024)]),
    "_Z11k_qkv_attn210AttnParams": ("k_qkv_attn_n", lambda gx, gy: (gx, gy, 16), lambda k: [(P(k, 8) - vr.BASE, I(k, 0x18) * I(k, 0x1c) * 512)]),
    "_Z7k_ffwd211Ffwd2Params": ("k_ffwd2_n", lambda gx, gy: (gx, gy), lambda k: [(P(k, 0x10) - vr.BASE, I(k, 0x28) * 8192)]),
    "_Z6k_qkv29QkvParams": ("k_vit_qkv_n", lambda gx, gy: (gx, 48), lambda k: [(P(k, o) - vr.BASE, 448 * 1024) for o in (8, 0x10, 0x18)]),
}
ONLY = os.environ.get("ONLY")   # restrict to one translated symbol substring


def psnr(rb, gb):
    ref = e4m3_decode(torch.from_numpy(rb)).double(); got = e4m3_decode(torch.from_numpy(gb)).double()
    d = ref - got; mse = (d ** 2).mean().item(); peak = (ref.max() - ref.min()).item()
    return (d == 0).double().mean().item() * 100, (10 * math.log10(peak ** 2 / mse) if mse else float("inf"))


vr.init()
args = [int(a) for a in sys.argv[1:]]
ranges = [range(args[0], args[1] + 1)] if args else [range(24, 55), range(58, 98), range(100, 132)]
tot = {}; worst = 1e9
for rg in ranges:
    for i in rg:
        m = vr.MAN[i]; ka = vr.kernarg(i); gx, gy = int(m[4]), int(m[5])
        nat = NATIVE.get(m[1]) if (not ONLY or ONLY in m[1]) else None
        grid = (gx, gy, 16) if "qkv_attn2" in m[1] else (gx, gy)
        mod = os.path.join("E:/dlss 5", m[0].split("dlss 5\\", 1)[1])
        if not nat:
            vr.launch(mod, m[1], ka, grid, int(m[6])); continue
        nsym, ngrid, outs = nat
        bufs = outs(ka)
        keep = [vr.read(o, n).copy() for o, n in bufs]
        for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
        vr.launch(mod, m[1], ka, grid, int(m[6]))
        refs = [vr.read(o, n).copy() for o, n in bufs]
        ms_t = vr.launch(mod, m[1], ka, grid, int(m[6]), iters=10)
        for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
        ms_n = vr.launch(MOD, nsym, ka, ngrid(gx, gy), 256, iters=10)
        gots = [vr.read(o, n).copy() for o, n in bufs]
        msg = []
        for (o, n), kp, rb, gb in zip(bufs, keep, refs, gots):
            wr = rb != 0x7F
            same = bool(((gb != 0x7F) == wr).all())
            rb = np.where(wr, rb, kp); gb = np.where(wr, gb, kp)
            vr.write(o, rb)
            ex, p = psnr(rb, gb); worst = min(worst, p)
            msg.append(f"writes {same} exact {ex:6.2f}% {p:5.1f} dB")
        t = tot.setdefault(nsym, [0.0, 0.0]); t[0] += ms_t; t[1] += ms_n
        print(f"step {i:3d} {nsym:10s} translated {ms_t:.3f} native {ms_n:.3f} ms | " + " | ".join(msg), flush=True)
for s, (a, b) in tot.items():
    print(f"{s}: translated {a:.2f} ms, native {b:.2f} ms, saved {a - b:.2f} ms")
print(f"worst PSNR {worst:.1f} dB")
