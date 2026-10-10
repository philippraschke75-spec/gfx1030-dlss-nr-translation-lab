"""e2e.py <runtime dll dir> <data dir> <out.npy> [frames]: run nr_run on shifted copies of color_1280x720.bin, save outputs (f16)."""
import ctypes, os, sys, time, numpy as np
d, data, outp = sys.argv[1], sys.argv[2], sys.argv[3]
frames = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.add_dll_directory(d)
lib = ctypes.CDLL(os.path.join(d, "nr_runtime.dll"))
lib.nr_create.restype = ctypes.c_void_p
lib.nr_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.nr_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
W, H = 1280, 720
base = np.fromfile("E:/dlss-runtime/color_1280x720.bin", np.uint16).reshape(H, W, 4)
err = ctypes.create_string_buffer(512)
c = lib.nr_create(data.encode(), W, H, 1, err, 512)
assert c, err.value
outs, ms = [], []
for k in range(frames):
    x = np.ascontiguousarray(np.roll(base, -13 * k, axis=1)); o = np.empty_like(x); g = ctypes.c_float()
    assert lib.nr_run(c, x.ctypes.data, o.ctypes.data, ctypes.byref(g)) == 0
    outs.append(o); ms.append(g.value)
np.save(outp, np.stack(outs))
print("gpu ms per frame:", " ".join(f"{m:.2f}" for m in ms))
