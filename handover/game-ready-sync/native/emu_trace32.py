"""Emulate one workgroup of a k_swin_var<32> step on the real g_A arena and capture every WMMA visit
(A, B, C before; D after) per wave. WMMA is modelled as the translator's lowering: sequential v_dot2c over k pairs.
usage: emu_trace32.py STEP WGX WGY OUT.npz"""
import sys, struct, time, numpy as np
sys.path.insert(0, "E:/dlss 5/rdna2/emu"); sys.path.insert(0, "E:/dlss 5/rdna2"); sys.path.insert(0, "E:/dlss 5/native-mid")
import gfx11emu as E, run_emu as R, trace_wmma_var as TW, vr
step, wgx, wgy, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sym = TW.SYMS["32_0"]
prog = E.load_program(R.DIS, {sym, TW.HELPER})
entry = E.load_program(R.DIS, {sym})[0][0]
start = min(i for i, ins in enumerate(prog) if ins[0] == entry)
prog = prog[start:] + prog[:start]
ka = vr.kernarg(step)
KA = 0x7000_0000_0000
g = E.GMem()
g.add("kernarg", KA, np.frombuffer(bytes(ka), np.uint8).copy())
g.add("arena", vr.BASE, np.memmap(f"{vr.G}/arena.bin", np.uint8, mode="c"))
import translate_final_head as base
_, sections, _ = base.kd.parse_elf(str(base.INPUT))
ro = next(x for x in sections if x["name"] == ".rodata")
g.add("rodata", ro["addr"], np.frombuffer(bytearray(open("E:/dlss 5/rdna2/build/kernels-hw-scratch/initialized-rodata.bin", "rb").read()), np.uint8).copy())
lds = TW.lds_size(sym)
sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: wgx, 15: wgy}

caps = []   # (addr, wid, visit, A, B, C, D)
visits = {}
old_compile = E.Exec._compile
def frag(w, o):
    regs = np.ascontiguousarray(np.stack([w.RV[o + j] for j in range(8)], 1)[:16])
    return regs.view(np.float16).reshape(16, 16).copy()
def accum(w, o):
    C = np.zeros((16, 16), np.float32)
    for j in range(8):
        reg = w.RV[o + j].view(np.float32); C[2 * j, :] = reg[:16]; C[2 * j + 1, :] = reg[16:]
    return C
def compile_hook(ex, pc):
    fn = old_compile(ex, pc)
    addr, op, args, _ = ex.prog[pc]
    if op != "v_wmma_f32_16x16x16_f16": return fn
    P = E.split_args(args) if hasattr(E, "split_args") else None
    regs = [int(m) for m in __import__("re").findall(r"v\[(\d+):\d+\]", args)]
    d, a, b, c = regs
    def seq(w):
        A, B, C = frag(w, a), frag(w, b), accum(w, c)
        Af, Bf = A.astype(np.float64), B.astype(np.float64)
        D = C.copy()
        for k in range(8):   # v_dot2c order: pair k = (2k, 2k+1), one rounding per dot2
            D = (D.astype(np.float64) + Af[:, 2 * k, None] * Bf[None, :, 2 * k] + Af[:, 2 * k + 1, None] * Bf[None, :, 2 * k + 1]).astype(np.float32)
        for j in range(8):
            w.V[d + j] = np.ascontiguousarray(np.concatenate([D[2 * j, :], D[2 * j + 1, :]])).view(np.uint32)
        key = (addr, w.wid); visits[key] = visits.get(key, 0) + 1
        caps.append((addr, w.wid, visits[key], A, B, C, D))
    return seq
E.Exec._compile = compile_hook
t = time.time()
r = E.run_workgroup(prog, g, lds, 256, sg, max_steps=50_000_000)
print("steps", r["steps"], "secs %.0f" % (time.time() - t), "captures", len(caps))
np.savez_compressed(out, addr=np.array([c[0] for c in caps]), wid=np.array([c[1] for c in caps]), visit=np.array([c[2] for c in caps]),
                    A=np.stack([c[3] for c in caps]), B=np.stack([c[4] for c in caps]), C=np.stack([c[5] for c in caps]), D=np.stack([c[6] for c in caps]))
import ana32
o = ana32.load(step, wgx, wgy, g.regions[1].arr)["out"]
np.save(out.replace(".npz", "_out.npy"), o)
sites = sorted(set(c[0] for c in caps))
print("sites:", [hex(s) for s in sites])
print("wave0 order:", [(sites.index(c[0]) + 1, c[2]) for c in caps if c[1] == 0][:80])
