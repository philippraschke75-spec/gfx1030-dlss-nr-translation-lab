import sys, numpy as np
sys.path.insert(0, "E:/dlss 5/native-mid")
import vr
vr.init()
st = [i for i, m in enumerate(vr.MAN) if m[1] == "_Z10k_swin_varILi32ELb0EEv9VarParams"]
for mod in sys.argv[1:]:
    t = [vr.launch(mod, "k_swin_c32", vr.kernarg(i), (int(vr.MAN[i][4]), int(vr.MAN[i][5])), 256, iters=10) for i in st]
    print(f"{mod}: sum {sum(t):.2f} ms  " + " ".join(f"{x:.3f}" for x in t), flush=True)
