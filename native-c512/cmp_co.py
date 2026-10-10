"""cmp_co.py BASE.co NEW.co [first last]: on g_A, for every overridden step run BASE's native kernel (its result feeds
on, as in the runtime) and NEW's on the same input; report byte-exactness and times. Default ranges: 24-54, 100-131.
Non-overridden steps run the translated kernels (same chain as check.py)."""
import os, struct, sys, numpy as np
sys.path.insert(0, "E:/dlss 5/native-mid")
import vr
P = lambda k, o: struct.unpack_from("<Q", k, o)[0]
I = lambda k, o: struct.unpack_from("<i", k, o)[0]


def conv2_outs(k):
    H, W, T = I(k, 0x30), I(k, 0x34), I(k, 0x38)
    o = [(P(k, 0x18) - vr.BASE, T * 8192)]
    if P(k, 0x20): o.append((P(k, 0x20) - vr.BASE, 512 * H * W))
    return o


NATIVE = {
    "_Z11k_conv_res211Conv2Params": ("k_conv2_n", lambda gx, gy: (gx, 8), conv2_outs),
    "_Z11k_qkv_attn210AttnParams": ("k_qkv_attn_n", lambda gx, gy: (gx, gy, 16), lambda k: [(P(k, 8) - vr.BASE, I(k, 0x18) * I(k, 0x1c) * 512)]),
    "_Z7k_ffwd211Ffwd2Params": ("k_ffwd2_n", lambda gx, gy: (gx, gy), lambda k: [(P(k, 0x10) - vr.BASE, I(k, 0x28) * 8192)]),
}
BASE, NEW = sys.argv[1], sys.argv[2]
args = [int(a) for a in sys.argv[3:]]
ranges = [range(args[0], args[1] + 1)] if args else [range(24, 55), range(100, 132)]
ITERS = int(os.environ.get("ITERS", "20"))
ONLY = os.environ.get("ONLY", "")   # translated-symbol substring; other overridden steps run BASE's native kernel untimed
vr.init()
tot = {}; allsame = True
for rg in ranges:
    for i in rg:
        m = vr.MAN[i]; ka = vr.kernarg(i); gx, gy = int(m[4]), int(m[5])
        nat = NATIVE.get(m[1])
        grid = (gx, gy, 16) if "qkv_attn2" in m[1] else (gx, gy)
        mod = os.path.join("E:/dlss 5", m[0].split("dlss 5\\", 1)[1])
        if not nat:
            vr.launch(mod, m[1], ka, grid, int(m[6])); continue
        nsym, ngrid, outs = nat
        if ONLY not in m[1]:
            vr.launch(BASE, nsym, ka, ngrid(gx, gy), 256); continue
        bufs = outs(ka)
        for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
        vr.launch(NEW, nsym, ka, ngrid(gx, gy), 256)
        gots = [vr.read(o, n).copy() for o, n in bufs]
        ms_n = vr.launch(NEW, nsym, ka, ngrid(gx, gy), 256, iters=ITERS)
        for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
        vr.launch(BASE, nsym, ka, ngrid(gx, gy), 256)
        refs = [vr.read(o, n).copy() for o, n in bufs]
        ms_b = vr.launch(BASE, nsym, ka, ngrid(gx, gy), 256, iters=ITERS)
        msg = []
        for (o, n), rb, gb in zip(bufs, refs, gots):
            vr.write(o, rb)
            same = bool((rb == gb).all()); allsame &= same
            msg.append("SAME" if same else f"DIFF {int((rb != gb).sum())} bytes")
        t = tot.setdefault(nsym, [0.0, 0.0]); t[0] += ms_b; t[1] += ms_n
        print(f"step {i:3d} {nsym:12s} base {ms_b:.3f} new {ms_n:.3f} ms | " + " | ".join(msg), flush=True)
for s, (a, b) in tot.items():
    print(f"{s}: base {a:.3f} ms, new {b:.3f} ms, {100 * (1 - b / a):+.1f}%")
print("ALL SAME" if allsame else "MISMATCH")
