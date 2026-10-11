"""hist_test.py <runtime dll dir> <data dir> W H: (A) history off == saved combo output (out_cmb_WxH.npy, e2e_wh shifts of 13 px);
(B) history on, static frame: |o_t - o_{t-1}| per frame; (C) +-1 LSB input noise: flicker without vs with history."""
import ctypes as C, os, sys, numpy as np
d, data, W, H = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.add_dll_directory(d); lib = C.CDLL(os.path.join(d, 'nr_runtime.dll'))
lib.nr_create.restype = C.c_void_p; lib.nr_create.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int]
lib.nr_run2.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
base = np.fromfile(f'E:/dlss-runtime/color_{W}x{H}.bin', np.uint16).reshape(H, W, 4)
err = C.create_string_buffer(512); c = lib.nr_create(data.encode(), W, H, 1, err, 512); assert c, err.value
def run(x, hv):
    x = np.ascontiguousarray(x); o = np.empty_like(x); assert lib.nr_run2(c, x.ctypes.data, o.ctypes.data, None, int(hv)) == 0; return o
f = lambda o: o.view(np.float16)[..., :3].astype(np.float32)
ref = np.load(f'E:/dlss-runtime/combo/out_cmb_{W}x{H}.npy')
A = [np.array_equal(run(np.roll(base, -13 * k, axis=1), 0), ref[k]) for k in range(4)]
print('A history off bit-exact vs combo:', A, 'PASS' if all(A) else 'FAIL')
prev = None; dd = []
for t in range(12):
    o = f(run(base, t > 0))
    if prev is not None: dd.append((np.abs(o - prev).max(), np.abs(o - prev).mean()))
    prev = o
print('B static, history on: max|o_t-o_t-1| =', ' '.join('%.1e/%.1e' % v for v in dd))
A2 = np.array_equal(run(base, 0), ref[0]); print('A2 history off again after history frames == combo frame 0:', A2)
rng = np.random.default_rng(0)
for hv in (0, 1):
    prev = None; fl = []
    for t in range(12):
        x = (base.astype(np.int32) + rng.integers(-1, 2, base.shape) * (np.arange(4) < 3)).astype(np.uint16)
        o = f(run(x, hv and t > 0))
        if prev is not None and t > 3: fl.append(np.abs(o - prev).mean())
        prev = o
    print('C noise +-1 LSB, history %d: mean|o_t-o_t-1| = %.3e, mean(o) = %.4f' % (hv, np.mean(fl), o.mean()))
