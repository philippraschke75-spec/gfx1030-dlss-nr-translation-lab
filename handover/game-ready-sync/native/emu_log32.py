"""Log the arithmetic executed by wave 0 between two WMMA site visits (lane values after each instruction).
usage: emu_log32.py STEP WGX WGY FROM_SITE FROM_VISIT TO_SITE TO_VISIT OUT.txt [LANES]"""
import sys, re, time, numpy as np
sys.path.insert(0, "E:/dlss 5/rdna2/emu"); sys.path.insert(0, "E:/dlss 5/rdna2"); sys.path.insert(0, "E:/dlss 5/native-mid")
import gfx11emu as E, run_emu as R, trace_wmma_var as TW, vr
step, wgx, wgy, fs, fv, ts, tv, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4], 16), int(sys.argv[5]), int(sys.argv[6], 16), int(sys.argv[7]), sys.argv[8]
lanes = [int(x) for x in sys.argv[9].split(",")] if len(sys.argv) > 9 else [0, 16]
sym = TW.SYMS["32_0"]
prog = E.load_program(R.DIS, {sym, TW.HELPER})
entry = E.load_program(R.DIS, {sym})[0][0]
start = min(i for i, ins in enumerate(prog) if ins[0] == entry)
prog = prog[start:] + prog[:start]
ka = vr.kernarg(step); KA = 0x7000_0000_0000
g = E.GMem(); g.add("kernarg", KA, np.frombuffer(bytes(ka), np.uint8).copy())
g.add("arena", vr.BASE, np.memmap(f"{vr.G}/arena.bin", np.uint8, mode="c"))
import translate_final_head as base
_, sections, _ = base.kd.parse_elf(str(base.INPUT)); ro = next(x for x in sections if x["name"] == ".rodata")
g.add("rodata", ro["addr"], np.frombuffer(bytearray(open("E:/dlss 5/rdna2/build/kernels-hw-scratch/initialized-rodata.bin", "rb").read()), np.uint8).copy())
lds = TW.lds_size(sym); sg = {0: KA & 0xffffffff, 1: KA >> 32, 14: wgx, 15: wgy}
sites = sorted(set(a for a, op, _, _ in prog if op == "v_wmma_f32_16x16x16_f16" and a >= entry))
visits = {}; state = {"on": False}; log = open(out, "w")
old = E.Exec._compile
def hook(ex, pc):
    fn = old(ex, pc); addr, op, args, _ = ex.prog[pc]
    if op == "v_wmma_f32_16x16x16_f16":
        regs = [int(m) for m in re.findall(r"v\[(\d+):\d+\]", args)]; d, a, b, c = regs
        def seq(w):
            A = np.ascontiguousarray(np.stack([w.RV[a + j] for j in range(8)], 1)[:16]).view(np.float16).reshape(16, 16).astype(np.float64)
            B = np.ascontiguousarray(np.stack([w.RV[b + j] for j in range(8)], 1)[:16]).view(np.float16).reshape(16, 16).astype(np.float64)
            D = np.zeros((16, 16), np.float32)
            for j in range(8):
                reg = w.RV[c + j].view(np.float32); D[2 * j, :] = reg[:16]; D[2 * j + 1, :] = reg[16:]
            for k in range(8): D = (D.astype(np.float64) + A[:, 2 * k, None] * B[None, :, 2 * k] + A[:, 2 * k + 1, None] * B[None, :, 2 * k + 1]).astype(np.float32)
            for j in range(8): w.V[d + j] = np.ascontiguousarray(np.concatenate([D[2 * j, :], D[2 * j + 1, :]])).view(np.uint32)
            if w.wid == 0:
                s = addr; visits[s] = visits.get(s, 0) + 1
                if (s, visits[s]) == (fs, fv): state["on"] = True
                if (s, visits[s]) == (ts, tv): state["on"] = False
                if state["on"] or (s, visits[s]) == (ts, tv):
                    log.write(f"{addr:x} WMMA site {s} visit {visits[s]}: {args}\n")
        return seq
    if not re.search(r"f16|f32|^ds_|pk_|v_dot|exec|cbranch|cndmask|perm|alignbit|pack|cvt|lshl_or|v_and_b32|v_or_b32|v_bfe|v_lsh|v_mov|dual", op): return fn
    def logged(w):
        fn(w)
        if w.wid != 0 or not state["on"]: return
        m = re.match(r"v(\d+)|v\[(\d+):(\d+)\]", args)
        vals = ""
        if m:
            r = int(m.group(1) or m.group(2))
            vals = " | ".join(f"L{l}: {w.V[r][l]:08x} f{w.V[r][l:l+1].view(np.float32)[0]:.6g} h{w.V[r][l:l+1].view(np.float16)[0]:.5g}/{w.V[r][l:l+1].view(np.float16)[1]:.5g}" for l in lanes)
        log.write(f"{addr:x} {op} {args}  exec={int(w.S[E.EXEC]):08x}  {vals}\n")
    return logged
E.Exec._compile = hook
t = time.time(); r = E.run_workgroup(prog, g, lds, 256, sg, max_steps=50_000_000); log.close()
print("done", r["steps"], "secs %.0f" % (time.time() - t))
