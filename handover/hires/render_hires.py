"""Offline model-resolution comparison: one capture through the network at several model resolutions, composited to
the frame size exactly as dlssnr_nr.cpp does it (LDR path, gain 1), PNGs + zoomed crops + edit / high-frequency stats.

    py -I render_hires.py prep <frame.bin RGBA16F FWxFH> <FW> <FH> <work dir> <WxH>...    model inputs <work>/in_WxH.bin
    (GPU, per size)  NR_BENCH_IN=in_WxH.bin NR_BENCH_OUT=out_WxH.bin nr_bench <dll dir> <data dir> W H 8 1 1
    py -I render_hires.py comp <frame.bin> <FW> <FH> <work dir> <png dir>[,<png dir>...] <WxH>...

Model input (shader imp): frame == model size -> the frame saturate()d (LDR); else colAt = 4 bilinear taps at +-FW/W/4 around the
model pixel centre, each tap saturate()d (LDR). Output (shader exp), every size:
max(frame + bil(out) - bil(in), 0), bil = bilinear at model coords (id + 0.5) * W / FW, NaN -> 0.
PNGs: clamp [0,1], gamma 1/2.2 (as handover/accuracy). Crops: 2x nearest, left to right native then each size.
"""
import sys, os
import numpy as np
from PIL import Image


def load(path, W, H):
    return np.fromfile(path, np.uint16).reshape(H, W, 4).view(np.float16).astype(np.float32)


def bil(img, px, py):   # img HxWxC, sample coords (pixel centres at .5), clamp to edge, as the shader's bil / colAt taps
    H, W = img.shape[:2]
    x = np.clip(px - 0.5, 0, W - 1); y = np.clip(py - 0.5, 0, H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    tx = (x - x0)[None, :, None]; ty = (y - y0)[:, None, None]
    r0 = img[y0][:, x0] * (1 - tx) + img[y0][:, x1] * tx; r1 = img[y1][:, x0] * (1 - tx) + img[y1][:, x1] * tx
    return r0 * (1 - ty) + r1 * ty


def model_input(F, W, H):
    FH, FW = F.shape[:2]
    L = F.copy(); L[..., :3] = np.clip(L[..., :3], 0, 1)   # ld(): saturate(rgb), alpha as is
    if (FW, FH) == (W, H): return L
    cx = (np.arange(W) + 0.5) * FW / W; cy = (np.arange(H) + 0.5) * FH / H; sx, sy = FW / W * 0.25, FH / H * 0.25
    return sum(bil(L, cx + dx * sx, cy + dy * sy) for dx in (-1, 1) for dy in (-1, 1)) * 0.25


def composite(F, inp, out):
    FH, FW = F.shape[:2]; H, W = inp.shape[:2]
    out = np.nan_to_num(out, nan=0.0, posinf=65504, neginf=0.0)
    px = (np.arange(FW) + 0.5) * W / FW; py = (np.arange(FH) + 0.5) * H / FH
    d = np.nan_to_num(bil(out[..., :3], px, py) - bil(inp[..., :3], px, py))
    return np.maximum(F[..., :3] + d, 0)


def disp(img): return (np.clip(img, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
def luma(img): return (np.clip(img, 0, 1) ** (1 / 2.2)) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
def lap(y): return 4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]


def crops(y, n=256):   # (name, y0, x0): detail = most Laplacian energy, edge = most gradient energy, shadow = darkest textured
    H, W = y.shape; best = {}
    g = np.abs(np.diff(y, axis=1))[:-1] + np.abs(np.diff(y, axis=0))[:, :-1]; l = np.abs(lap(y))
    for y0 in range(0, H - n, n // 2):
        for x0 in range(0, W - n, n // 2):
            m = y[y0:y0 + n, x0:x0 + n].mean(); gs = g[y0:y0 + n, x0:x0 + n]
            ls = l[y0:y0 + n, x0:x0 + n].mean(); edge = np.percentile(gs, 99.5) - gs.mean()
            for k, v in (('detail', ls), ('edge', edge), ('shadow', ls / (m + 0.02) if m < 0.25 else -1)):
                if v > best.get(k, (-2,))[0]: best[k] = (v, y0, x0)
    return [(k, best[k][1], best[k][2]) for k in ('detail', 'edge', 'shadow')]


def main():
    cmd, frame, FW, FH, work = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    F = load(frame, FW, FH); os.makedirs(work, exist_ok=True)
    if cmd == 'prep':
        for s in sys.argv[6:]:
            W, H = map(int, s.split('x')); model_input(F, W, H).astype(np.float16).view(np.uint16).tofile(os.path.join(work, 'in_%s.bin' % s))
        return
    outs, sizes = sys.argv[6].split(','), sys.argv[7:]
    imgs = {'native': np.maximum(np.nan_to_num(F[..., :3]), 0)}
    for s in sizes:
        W, H = map(int, s.split('_')[-1].split('x'))   # <tag>_WxH: files in_<tag>_WxH.bin / out_<tag>_WxH.bin (a variant input)
        imgs[s] = composite(F, load(os.path.join(work, 'in_%s.bin' % s), W, H), load(os.path.join(work, 'out_%s.bin' % s), W, H))
    yn = luma(imgs['native']); hn = (lap(yn) ** 2).mean()
    print('| model res | mean abs edit (linear) | mean abs edit (display) | HF energy (Laplacian^2 of display luma) vs native | HF energy of the edit |')
    print('| --- | --- | --- | --- | --- |')
    for k, im in imgs.items():
        y = luma(im); e = y - yn
        print('| %s | %.4f | %.4f | %.3f x | %.2e |' % (k, np.abs(im - imgs['native']).mean(), np.abs(e).mean(), (lap(y) ** 2).mean() / hn, (lap(e) ** 2).mean()))
    cr = crops(yn)
    for o in outs:
        os.makedirs(o, exist_ok=True)
        for k, im in imgs.items(): Image.fromarray(disp(im)).save(os.path.join(o, '%s_%dx%d.png' % ('native' if k == 'native' else 'model_' + k, FW, FH)), optimize=True)
        for name, y0, x0 in cr:
            row = [np.repeat(np.repeat(disp(im[y0:y0 + 256, x0:x0 + 256]), 2, 0), 2, 1) for im in imgs.values()]
            sep = np.full((512, 8, 3), 255, np.uint8)
            Image.fromarray(np.concatenate(sum([[r, sep] for r in row], [])[:-1], 1)).save(os.path.join(o, 'crop_%s_x%d_y%d.png' % (name, x0, y0)), optimize=True)
    print('crops (256x256 at x,y; 2x; order %s):' % ' | '.join(imgs), ', '.join('%s %d,%d' % c for c in cr))


if __name__ == '__main__':
    main()
