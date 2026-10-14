"""stress.py <native .co> [reps]: per k_swin_var<32> step of g_A, run native k_swin_c32 `reps` times with
alternating output prefills (0x00 / 0x7F) and a translated dummy launch every third rep; report differing bytes."""
import sys, numpy as np
sys.path.insert(0, "E:/dlss 5/native-mid")
import vr
MOD = sys.argv[1]; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
vr.init()
st = [i for i, m in enumerate(vr.MAN) if m[1] == "_Z10k_swin_varILi32ELb0EEv9VarParams"]
bad = 0
for i in st:
    ka = vr.kernarg(i); m = vr.MAN[i]
    H, W = np.frombuffer(bytes(ka[0x18:0x20]), np.int32); fl = ka[0x28]
    bufs = [(int.from_bytes(ka[8:16], "little") - vr.BASE, 32 * H * W)]
    if fl & 4: bufs.append((int.from_bytes(ka[0x38:0x40], "little") - vr.BASE, 64 * (H // 2) * (W // 2)))
    ino = int.from_bytes(ka[0:8], "little") - vr.BASE
    alias = any(o <= ino < o + n for o, n in bufs)
    keep = [vr.read(o, n).copy() for o, n in bufs]
    res = []
    for r in range(REPS):
        if r % 3 == 2: vr.step(i)   # dummy: translated kernel on the same buffers
        for (o, n), k in zip(bufs, keep): vr.write(o, k)
        if not alias:
            for o, n in bufs: vr.write(o, np.full(n, 0x7F if r & 1 else 0x00, np.uint8))
        vr.launch("E:/dlss-runtime/c32fix/scribble.co", "scribble", (r * 7919 + 1).to_bytes(4, "little"), (4096, 1), 256)  # random leftover LDS
        vr.launch(MOD, "k_swin_c32", ka, (int(m[4]), int(m[5])), 256)
        res.append(np.concatenate([vr.read(o, n) for o, n in bufs]))
    for (o, n), k in zip(bufs, keep): vr.write(o, k)
    a = np.stack(res); ev, od = a[0::2], a[1::2]
    nd_e = int((ev != ev[0]).any(0).sum()); nd_o = int((od != od[0]).any(0).sum()); unw = int((ev[0] != od[0]).sum())
    if nd_e or nd_o: bad += 1
    print(f"step {i} flags {fl} {W}x{H} alias {alias}: nondet bytes {nd_e}/{nd_o}  prefill-dependent {unw}", flush=True)
print("RESULT", "DETERMINISTIC" if not bad else f"{bad} NONDETERMINISTIC STEPS")
