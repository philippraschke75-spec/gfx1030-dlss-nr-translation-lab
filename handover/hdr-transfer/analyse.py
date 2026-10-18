"""Numbers and PNG crops for handover/hdr-transfer (dumps of batch.sh: <tag>_in.bin / <tag>_out.bin, f32 RGB 2560x1440).

    python -I analyse.py <dump dir> <png dir> [<png dir 2> ...]

Per crop: edit = mean |dY| (out - in), HF = mean |Laplacian of Y| out / in, LC = local contrast (mean 7x7 std of Y) out / in.
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

W, H = 2560, 1440
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
S = W / 1920
CROPS = {'club': {'face_woman': (300, 330, 220, 260), 'face_man': (1050, 380, 170, 170)},
         'hall': {'hand': (940, 780, 360, 300), 'texture_hall': (1150, 300, 560, 300)}}


def load(d, tag, k):
    return np.fromfile(f'{d}/{tag}_{k}.bin', np.float32).reshape(H, W, 3)


def box(y, r):
    c = np.cumsum(np.cumsum(np.pad(y, ((r + 1, r), (r + 1, r)), mode='edge'), 0), 1)
    n = 2 * r + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)


def lap(y):
    return np.abs(4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]).mean()


def lc(y):
    return np.sqrt(np.maximum(box(y * y, 3) - box(y, 3) ** 2, 0)).mean()


def main():
    d, pngs = sys.argv[1], sys.argv[2:]
    hall_in = np.clip(load(d, 'hall_960x540_def_t0', 'in'), 0, None) @ LUMA
    b = box(hall_in, 20); i = np.unravel_index(np.argmax(b[200:-200, 300:-300]), b[200:-200, 300:-300].shape)
    cy, cx = i[0] + 200, i[1] + 300
    CROPS['hall']['highlight'] = (int((cx - 200) / S), int((cy - 133) / S), 300, 200)
    for src, crops in CROPS.items():
        for m in ['960x540', '1920x1080']:
            for cfg in ['def', 'show']:
                tags = [f'{src}_{m}_{cfg}_t{t}' for t in ('0', '0.5', '1')]
                if not all(os.path.exists(f'{d}/{t}_out.bin') for t in tags): continue
                inp = load(d, tags[0], 'in'); outs = [load(d, t, 'out') for t in tags]
                for cn, (x, y, w, h) in list(crops.items()) + [('frame', (0, 0, 1920, 1080))]:
                    X, Y, Wc, Hc = int(x * S), int(y * S), int(w * S), int(h * S)
                    yi = np.clip(inp[Y:Y + Hc, X:X + Wc], 0, 1) @ LUMA
                    row = []
                    for t, o in zip(('0', '0.5', '1'), outs):
                        yo = np.clip(o[Y:Y + Hc, X:X + Wc], 0, 1) @ LUMA
                        row.append(f't{t}: edit {np.abs(yo - yi).mean():.4f} HF x{lap(yo) / lap(yi):.3f} LC x{lc(yo) / lc(yi):.3f}')
                    print(f'{src:4s} {m:9s} {cfg:4s} {cn:12s} | ' + ' | '.join(row))
                    if cn == 'frame' or cfg != 'show': continue
                    tiles = []
                    for lab, im in [('input', inp)] + [(f'transfer {t}', o) for t, o in zip(('0', '0.5', '1'), outs)]:
                        c = np.clip(im[Y:Y + Hc, X:X + Wc], 0, 1)
                        tl = Image.fromarray((c * 255 + 0.5).astype(np.uint8))
                        if tl.width < 400: tl = tl.resize((tl.width * 2, tl.height * 2), Image.NEAREST)
                        ImageDraw.Draw(tl).text((6, 4), f'{lab} ({m} model)', fill=(255, 255, 0))
                        tiles.append(tl)
                    sheet = Image.new('RGB', (sum(t.width for t in tiles), tiles[0].height))
                    xo = 0
                    for tl in tiles: sheet.paste(tl, (xo, 0)); xo += tl.width
                    for p in pngs:
                        os.makedirs(p, exist_ok=True); sheet.save(f'{p}/{cn}_{m}_showcase.png')


if __name__ == '__main__':
    main()
