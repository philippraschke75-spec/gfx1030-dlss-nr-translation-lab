import numpy as np, sys
from PIL import Image
D = r'rdna2/build/import-real-20260922/rdna2/build/net_frame_full'

def gauss(img, s):
    if s <= 0: return img.copy()
    r = int(3 * s + 0.5); x = np.arange(-r, r + 1)
    k = np.exp(-(x * x) / (2 * s * s)); k /= k.sum()
    o = np.apply_along_axis(lambda m: np.convolve(m, k, 'same'), 1, img)
    return np.apply_along_axis(lambda m: np.convolve(m, k, 'same'), 0, o)

def Y(p):
    a = np.asarray(Image.open(p).convert('RGB')).astype(np.float64)
    return a @ np.array([0.2126, 0.7152, 0.0722]), a

def z(a): return (a - a.mean()) / max(a.std(), 1e-12)

yo, ao = Y(D + '/rendered.png')
yi, ai = Y(D + '/imported.png')
yo, yi = z(yo), z(yi)
out = []
for nm, s1, s2 in (('mid', 1.5, 3.0), ('fine', 0.7, 1.5)):
    bo = gauss(yo, s1) - gauss(yo, s2)
    bi = gauss(yi, s1) - gauss(yi, s2)
    S = np.corrcoef(bo.ravel(), bi.ravel())[0, 1]
    out.append((nm, S, 1 - S * S))
L = ao.mean(2); cm = L.mean(0); d = cm - cm.mean()
ac = np.correlate(d, d, 'full')[len(d) - 1:]; ac /= ac[0]
print('  %-26s S_mid %+.4f (resid %.3f)   S_fine %+.4f (resid %.3f)   lag8 %+.2f'
      % (sys.argv[1], out[0][1], out[0][2], out[1][1], out[1][2], ac[8]))
