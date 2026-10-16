import numpy as np, sys
x = np.fromfile('E:/dlss-runtime/color_1920x1080.bin', np.uint16).reshape(1080, 1920, 4).view(np.float16).astype(np.float32)
def rs(a, n, ax):
    m = a.shape[ax]; t = (np.arange(n) + 0.5) * m / n - 0.5; i = np.clip(np.floor(t).astype(int), 0, m - 1); j = np.clip(i + 1, 0, m - 1); f = np.clip(t - i, 0, 1)
    sh = [1] * a.ndim; sh[ax] = n; f = f.reshape(sh)
    return np.take(a, i, ax) * (1 - f) + np.take(a, j, ax) * f
for s in sys.argv[1:]:
    W, H = map(int, s.split('x')); rs(rs(x, H, 0), W, 1).astype(np.float16).view(np.uint16).tofile('E:/dlss-runtime/sweep/color_%s.bin' % s)
