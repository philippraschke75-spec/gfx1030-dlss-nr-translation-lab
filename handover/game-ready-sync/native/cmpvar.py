"""cmpvar.py ref.co test.co [steps...]: byte-compare k_swin_c64/c128/c256 of test.co against ref.co on every swin_var step
of the g_A frame (native-mid/vr.py) and time both (median of 10). Outputs are pre-filled with 0x7F so the write set is checked too.
Env SWIN_NT_REF / SWIN_NT_TEST = "64:512,128:512,256:1024" set the block size per width (default 256).
Env SWIN_SPLIT_TEST = "256,128" times the split kernels k_swin_c{C}s (grid z = 2) of test.co against k_swin_c{C} of ref.co."""
import os, sys, numpy as np
sys.path.insert(0, "E:/dlss 5/native-mid")
import vr
REF, TEST = sys.argv[1], sys.argv[2]
nt = lambda k: {int(a): int(b) for a, b in (x.split(":") for x in os.environ.get(k, "").split(",") if x)}
TR, TT = nt("SWIN_NT_REF"), nt("SWIN_NT_TEST")
SPL = [int(a) for a in os.environ.get("SWIN_SPLIT_TEST", "").split(",") if a]   # e.g. "256,128": test k_swin_c{C}s, grid z = 2
SYMS = {f"_Z10k_swin_varILi{c}ELb0EEv9VarParams": (c, f"k_swin_c{c}") for c in (64, 128, 256)}
vr.init()
steps = [int(a) for a in sys.argv[3:]] or [i for i, m in enumerate(vr.MAN) if m[1] in SYMS]
tot = {}; ok = True
for i in steps:
    ka = vr.kernarg(i); m = vr.MAN[i]; grid = (int(m[4]), int(m[5]))
    H, W = np.frombuffer(bytes(ka[0x18:0x20]), np.int32); flags = ka[0x28]; C, SYM = SYMS[m[1]]
    bufs = [(int.from_bytes(ka[8:16], "little") - vr.BASE, C * H * W)]
    if flags & 4: bufs.append((int.from_bytes(ka[0x38:0x40], "little") - vr.BASE, 2 * C * (H // 2) * (W // 2)))
    keep = [vr.read(o, n).copy() for o, n in bufs]
    for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
    ms_r = vr.launch(REF, SYM, ka, grid, TR.get(C, 256), iters=10); refs = [vr.read(o, n).copy() for o, n in bufs]
    for o, n in bufs: vr.write(o, np.full(n, 0x7F, np.uint8))
    ms_t = vr.launch(TEST, SYM + "s" if C in SPL else SYM, ka, grid + (2,) if C in SPL else grid, TT.get(C, 256), iters=10); gots = [vr.read(o, n).copy() for o, n in bufs]
    msg = []
    for (o, n), k, rb, gb in zip(bufs, keep, refs, gots):
        same = bool((rb == gb).all()); ok &= same
        vr.write(o, np.where(rb != 0x7F, rb, k))   # the reference result feeds the next step
        msg.append(f"{'EXACT' if same else 'DIFF ' + str(int((rb != gb).sum())) + ' bytes'}")
    t = tot.setdefault(C, [0.0, 0.0]); t[0] += ms_r; t[1] += ms_t
    print(f"step {i:3d} C{C:3d} flags {flags:2d} grid {grid[0]:3d}x{grid[1]:<3d} ref {ms_r:.3f} ms  test {ms_t:.3f} ms  " + " | +0x38: ".join(msg), flush=True)
for C, (a, b) in sorted(tot.items()): print(f"C{C}: ref {a:.3f} ms  test {b:.3f} ms  ({(a - b) / a * 100:+.1f}% faster)")
print("ALL EXACT" if ok else "MISMATCH")
