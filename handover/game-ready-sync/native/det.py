"""det.py <data dir> <W> <H> <frames> <reps>: nr_run the same shifted-frame sequence `reps` times in one context
(nr_reset between), report frames whose output differs bitwise from rep 0. Honors NR_PROF from the environment."""
import ctypes, os, sys, numpy as np
d = "E:/dlss-runtime/quick/b"; data, W, H, F, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
os.add_dll_directory(d); lib = ctypes.CDLL(os.path.join(d, "nr_runtime.dll"))
lib.nr_create.restype = ctypes.c_void_p
lib.nr_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
lib.nr_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
lib.nr_reset.argtypes = [ctypes.c_void_p]
base = np.fromfile(f"E:/dlss-runtime/color_{W}x{H}.bin", np.uint16).reshape(H, W, 4)
err = ctypes.create_string_buffer(512); c = lib.nr_create(data.encode(), W, H, 1, err, 512); assert c, err.value
ref, bad = None, 0
for r in range(R):
    if r: assert lib.nr_reset(c) == 0
    outs = []
    for k in range(F):
        x = np.ascontiguousarray(np.roll(base, -13 * k, axis=1)); o = np.empty_like(x); g = ctypes.c_float()
        assert lib.nr_run(c, x.ctypes.data, o.ctypes.data, ctypes.byref(g)) == 0; outs.append(o)
    if ref is None: ref = outs; continue
    diff = [k for k in range(F) if not np.array_equal(outs[k], ref[k])]; bad += len(diff)
    print(f"rep {r}: differing frames {diff}", flush=True)
print("RESULT", "DETERMINISTIC" if not bad else f"{bad} differing frames")
