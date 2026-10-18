"""Per-frame metrics + PNGs for temporal_ref.py runs.   py -I analyse.py <run dir> <png prefix> [modes]

Interior only (12 px margin). Per mode and frame: edit mean|out-in|, edit HF (rms of the edit minus its 5x5 box blur),
detail = mean|laplacian(out)| / mean|laplacian(in)|, corr of the edit with the single-frame edit, PSNR vs 'nv'
(NVIDIA's temporal rules), temporal instability mean|O_k - warp O_{k-1}| (exact motion) minus the input's.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = 12


def box(x, r=2):
    k = 2 * r + 1; p = np.pad(x, ((r, r), (r, r), (0, 0)), mode='edge'); c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def lap(x):
    return np.abs(4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]).mean()


def motion(k, VX=1.3, VY=0.45, JIT=0.25):
    def halton(i, b):
        f, r, i = 1.0, 0.0, i + 1
        while i > 0:
            f /= b; r += f * (i % b); i //= b
        return r
    s = lambda k: (k * VX + JIT * 2 * (halton(k % 8, 2) - .5), k * VY + JIT * 2 * (halton(k % 8, 3) - .5))
    a, b = s(k), s(k - 1); return a[0] - b[0], a[1] - b[1]


def warp(img, dx, dy):
    h, w, _ = img.shape; ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = np.clip(xs + dx, 0, w - 1.001), np.clip(ys + dy, 0, h - 1.001); x0, y0 = u.astype(int), v.astype(int)
    a, b = (u - x0)[..., None], (v - y0)[..., None]
    return (img[y0, x0] * (1 - a) + img[y0, x0 + 1] * a) * (1 - b) + (img[y0 + 1, x0] * (1 - a) + img[y0 + 1, x0 + 1] * a) * b


def main():
    d, prefix = sys.argv[1], sys.argv[2]
    modes = sys.argv[3].split(',') if len(sys.argv) > 3 else [m for m in ('single', 'nv', 'nv_s0', 'ours', 'fix', 'fix_seed') if os.path.exists(os.path.join(d, m + '.npy'))]
    seq = np.load(os.path.join(d, 'seq.npy')); R = {m: np.load(os.path.join(d, m + '.npy')) for m in modes}
    n = min(len(r) for r in R.values()); s = np.s_[M:-M, M:-M]
    stats = {}
    for m, r in R.items():
        rows = []
        for k in range(n):
            e = r[k] - seq[k]; es = R['single'][k] - seq[k] if 'single' in R else e
            ehf = e - box(e)
            inst = np.abs(r[k] - warp(r[k - 1], *motion(k)))[s].mean() - np.abs(seq[k] - warp(seq[k - 1], *motion(k)))[s].mean() if k else np.nan
            ps = 10 * np.log10(1 / max(((r[k] - R['nv'][k])[s] ** 2).mean(), 1e-12)) if 'nv' in R else np.nan
            rows.append((np.abs(e[s]).mean(), np.sqrt((ehf[s] ** 2).mean()), lap(r[k][s]) / lap(seq[k][s]), np.corrcoef(e[s].ravel(), es[s].ravel())[0, 1], ps, inst))
        stats[m] = np.array(rows)
    hdr = 'edit|.|   editHF   detail   corr1f  PSNRnv   inst+'
    for m, a in stats.items():
        print(f'== {m}  (frames 0, 1, 4, 16, last; mean over frames 8..)\n   {hdr}')
        for k in (0, 1, 4, 16, n - 1):
            print(f'k={k:2d} ' + '  '.join(f'{v:7.4f}' if i != 4 else f'{v:6.2f}' for i, v in enumerate(a[k])))
        print('8..  ' + '  '.join(f'{v:7.4f}' if i != 4 else f'{v:6.2f}' for i, v in enumerate(np.nanmean(a[8:], 0))))
    np.save(prefix + '_stats.npy', stats, allow_pickle=True)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for m, a in stats.items():
            for i, (j, t) in enumerate(((0, 'edit strength mean|out-in|'), (1, 'edit high-frequency rms'), (5, 'instability added (inst out - in)'))):
                ax[i].plot(a[:, j], label=m); ax[i].set_title(t); ax[i].set_xlabel('frame')
        ax[0].legend(); fig.tight_layout(); fig.savefig(prefix + '_curves.png', dpi=90)
    except ImportError:
        print('(no matplotlib: curves skipped)')
    from PIL import Image
    g = lambda x: (np.clip(x, 0, 1) * 255 + .5).astype(np.uint8)
    k = n - 1; y0, x0, ch, cw = 40, 120, 160, 256   # crop of the last frame
    top = [seq[k]] + [R[m][k] for m in modes]; bot = [np.zeros_like(seq[k])] + [0.5 + 8 * (R[m][k] - seq[k]) for m in modes]
    img = np.concatenate([np.concatenate([t[y0:y0 + ch, x0:x0 + cw] for t in top], 1), np.concatenate([b[y0:y0 + ch, x0:x0 + cw] for b in bot], 1)], 0)
    Image.fromarray(g(img)).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(prefix + '_crop.png')
    print('columns:', ['input'] + modes, '; bottom row = edit x8 + 0.5')


if __name__ == '__main__':
    main()
