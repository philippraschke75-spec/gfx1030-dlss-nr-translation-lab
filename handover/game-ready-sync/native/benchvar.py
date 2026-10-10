"""benchvar.py mod.co[@nt64,nt128,nt256] ...: time k_swin_c64/c128/c256 of each module on one plain step per width
(g_A frame, median of 20 launches). Timing only; use cmpvar.py for exactness."""
import sys
sys.path.insert(0, "E:/dlss 5/native-mid")
import vr
vr.init()
for arg in sys.argv[1:]:
    mod, _, nts = arg.partition("@")
    nt = dict(zip((64, 128, 256), (int(x) for x in nts.split(",")))) if nts else {}
    t = {}
    for i, c in ((7, 64), (11, 128), (17, 256)):
        m = vr.MAN[i]
        t[c] = vr.launch(mod, f"k_swin_c{c}", vr.kernarg(i), (int(m[4]), int(m[5])), nt.get(c, 256), iters=20)
    print(f"{mod.split('/')[-1]:14s} " + "  ".join(f"C{c} {v:.3f}" for c, v in t.items()), flush=True)
