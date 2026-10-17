r"""host.py pre|post <module.co> <out.npy> [iters]: run the native pre/post kernel on the g_A frame (1792x1024 padded)
with g_A's kernarg scalars, save the outputs (pre: skip+pool bytes, post: f32 RGBA) and print the median launch time.
Fresh process per run so the pre weight cache is cold once (first launch excluded from timing)."""
import ctypes, os, struct, sys
import numpy as np

kind, co, outp = sys.argv[1], sys.argv[2], sys.argv[3]
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 21
NW = int(sys.argv[5]) if len(sys.argv) > 5 else 1   # windows per workgroup (64 threads each)
ARENA, KA = r"E:\dlss-vit\g_A\arena.bin", r"E:\dlss-vit\g_A\kernargs.bin"
H, W = 1024, 1792
hip = ctypes.CDLL(r"C:\Program Files\AMD\ROCm\6.4\bin\amdhip64_6.dll")
V = ctypes.c_void_p
def ck(e): assert e == 0, f"hip error {e}"
a = np.memmap(ARENA, np.uint8, "r")
def up(off, n):
    p = V(); ck(hip.hipMalloc(ctypes.byref(p), ctypes.c_size_t(n)))
    if off is not None:
        buf = np.ascontiguousarray(a[off:off + n]); ck(hip.hipMemcpy(p, buf.ctypes.data_as(V), ctypes.c_size_t(n), 1))
    else:
        ck(hip.hipMemset(p, 0, ctypes.c_size_t(n)))
    return p
if kind == "pre":
    ka = bytearray(open(KA, "rb").read()[304:304 + 0x50]); sym = b"k_pre_native"; gy = H // 8
    ptr = {0x00: up(0xC90000, H * W * 12), 0x08: up(None, H * W * 32), 0x10: up(0x571F0000, 21696), 0x38: up(None, H * W * 8)}
    ka[0x40:0x48] = bytes(8)
    outs = [(0x08, H * W * 32), (0x38, H * W * 8)]
else:
    ka = bytearray(open(KA, "rb").read()[65088:65088 + 0x50]); sym = b"k_post_native"; gy = H // 8
    ptr = {0x00: up(0x38280000, H * W * 8), 0x08: up(0x2190000, H * W * 32), 0x10: up(None, H * W * 16),
           0x38: up(0xC90000, H * W * 12), 0x40: up(0x4D280000, H * W * 12)}
    outs = [(0x10, H * W * 16)]
for o, p in ptr.items(): ka[o:o + 8] = struct.pack("<Q", p.value)
mod, fn = V(), V()
ck(hip.hipModuleLoad(ctypes.byref(mod), co.encode())); ck(hip.hipModuleGetFunction(ctypes.byref(fn), mod, sym))
kbuf = (ctypes.c_char * len(ka)).from_buffer(ka); ksz = ctypes.c_size_t(len(ka))
cfg = (V * 5)(V(1), ctypes.cast(kbuf, V), V(2), ctypes.cast(ctypes.byref(ksz), V), V(3))
def launch(): ck(hip.hipModuleLaunchKernel(fn, (W // 8 + NW - 1) // NW, gy, 1, 64 * NW, 1, 1, 0, None, None, cfg))
launch(); ck(hip.hipDeviceSynchronize())
ev = [V(), V()]
for x in ev: ck(hip.hipEventCreate(ctypes.byref(x)))
ms, t = [], ctypes.c_float()
for _ in range(iters):
    hip.hipEventRecord(ev[0], None); launch(); hip.hipEventRecord(ev[1], None); ck(hip.hipEventSynchronize(ev[1]))
    hip.hipEventElapsedTime(ctypes.byref(t), ev[0], ev[1]); ms.append(t.value)
res = []
for o, n in outs:
    got = np.empty(n, np.uint8); ck(hip.hipMemcpy(got.ctypes.data_as(V), ptr[o], ctypes.c_size_t(n), 2)); res.append(got)
np.save(outp, np.concatenate(res))
print(f"{kind} {os.path.basename(co)}: median {np.median(ms):.3f} ms  min {min(ms):.3f}  max {max(ms):.3f}")
