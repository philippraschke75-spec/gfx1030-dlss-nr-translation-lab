"""amp_test.py <dll dir> <data dir> W H [trace]: noise amplification, history off. Input +-1 LSB (f16) noise on rgb.
Amp = mean|dP_out| / mean|dP_in| with P = sRGB(x/(1+x)) (the network's proxy). With 'trace': after the first n dispatches
(checkpoints), fraction of changed bytes per arena placement between a clean and a noisy run -> where the change grows.
Also runs the history loop (nr_run2 1/2) on a static frame and with noise."""
import ctypes as C, os, sys, json, numpy as np
d, data, W, H = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.add_dll_directory(d); lib = C.CDLL(os.path.join(d, 'nr_runtime.dll'))
lib.nr_create.restype = C.c_void_p; lib.nr_create.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_char_p, C.c_int]
lib.nr_run2.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
lib.nr_dbg_steps.argtypes = [C.c_void_p, C.c_int]; lib.nr_dbg_read.argtypes = [C.c_void_p, C.c_ulonglong, C.c_ulonglong, C.c_void_p]
base = np.fromfile(f'E:/dlss-runtime/color_{W}x{H}.bin', np.uint16).reshape(H, W, 4)
err = C.create_string_buffer(512); c = lib.nr_create(data.encode(), W, H, 1, err, 512); assert c, err.value
def run(x, hv=0):
    x = np.ascontiguousarray(x); o = np.empty_like(x); assert lib.nr_run2(c, x.ctypes.data, o.ctypes.data, None, int(hv)) == 0; return o
def P(o):
    x = np.maximum(o.view(np.float16)[..., :3].astype(np.float32), 0); x = x / (1 + x)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
rng = np.random.default_rng(0)
noisy = lambda: (base.astype(np.int32) + rng.integers(-1, 2, base.shape) * (np.arange(4) < 3)).astype(np.uint16)
x0 = base; o0 = run(x0); amps = []; dos = []
for k in range(4):
    x1 = noisy(); o1 = run(x1); din = np.abs(P(x1) - P(x0)).mean(); do = np.abs(P(o1) - P(o0)).mean()
    amps.append(do / din); dos.append(do)
print('noise amp (history off): mean|dP_in| = %.2e, mean|dP_out| = %.2e, amp = %.0fx, out pixels changed >1/255: %.1f%%' %
      (din, np.mean(dos), np.mean(amps), 100 * (np.abs(P(o1) - P(o0)).max(-1) > 1 / 255).mean()))
print('repeat same input bit-exact:', np.array_equal(run(x0), o0))
for hv in (1, 2):
    prev = None; dd = []
    for t in range(12):
        o = P(run(x0, hv if t else 0))
        if prev is not None: dd.append(np.abs(o - prev).mean())
        prev = o
    fl = []
    for t in range(12):
        o = P(run(noisy(), hv if t else 0))
        if t > 3: fl.append(np.abs(o - prev).mean())
        prev = o
    print('history %d: static mean|o_t-o_t-1| (P) %s | noise flicker %.2e' % (hv, ' '.join('%.1e' % v for v in dd), np.mean(fl)))
fl = []; prev = None
for t in range(12):
    o = P(run(noisy(), 0))
    if t > 3: fl.append(np.abs(o - prev).mean())
    prev = o
print('history 0: noise flicker %.2e' % np.mean(fl))
if len(sys.argv) > 5:
    meta = json.load(open(data + '/meta.json')); pl = [(n, o, s) for n, o, s in meta['placements'] if not (n.startswith('w:') or n.endswith(' w') or n.endswith('weights'))]
    names = [l.split('|')[1] for l in open(data + '/manifest.txt') if l.strip()]
    x1 = noisy()
    for n in [1, 2, 3, 7, 11, 15, 19, 23, 24, 40, 57, 80, 99, 116, 131, 135, 140, 147, 154, 155, 156]:
        lib.nr_dbg_steps(c, n); bufs = []
        for x in (x0, x1):
            run(x); bufs.append([np.empty(s, np.uint8) for _, _, s in pl])
            for b, (_, o, s) in zip(bufs[-1], pl): lib.nr_dbg_read(c, o, s, b.ctypes.data)
        fr = sorted(((np.count_nonzero(a != b) / a.size, nm) for (nm, _, _), a, b in zip(pl, *bufs)), reverse=True)
        print('after %3d (%s): ' % (n, names[n - 1][:28]) + ', '.join('%s %.1f%%' % (nm, 100 * f) for f, nm in fr[:3] if f > 0))
    lib.nr_dbg_steps(c, 0)
