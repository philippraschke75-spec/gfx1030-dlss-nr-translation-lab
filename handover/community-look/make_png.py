"""PNG panels from run.sh dumps (float32 RGB, <tag>_in.bin / <tag>_out.bin).

    python make_png.py <dump dir> <W>x<H> <out.png> <crop x,y,w,h | full> <label=tag> [label=tag ...]

The first panel is the input of the first tag. Values are clipped to [0,1] and written as they are (the captures are
display-referred). Prints mean |luma edit| and mean |Laplacian| per panel.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def lap(y):
    return np.abs(4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]).mean()


def main():
    d, size, out, crop = sys.argv[1:5]
    w, h = map(int, size.split("x"))
    pairs = [a.rsplit("=", 1) for a in sys.argv[5:]]
    load = lambda t, k: np.clip(np.fromfile(f"{d}/{t}_{k}.bin", np.float32).reshape(h, w, 3), 0, 1)
    x, y, cw, ch = (0, 0, w, h) if crop == "full" else map(int, crop.split(","))
    inp = load(pairs[0][1], "in")
    panels = [("input", inp)] + [(lab, load(t, "out")) for lab, t in pairs]
    yi = inp @ LUMA
    for lab, im in panels:
        yo = im @ LUMA
        print(f"{lab:28s} edit |dY| {np.abs(yo - yi).mean():.5f}  mean dY {np.mean(yo - yi):+.5f}  HF {lap(yo):.5f} (x{lap(yo) / lap(yi):.3f})")
    tiles = []
    for lab, im in panels:
        t = Image.fromarray((im[y:y + ch, x:x + cw] * 255 + 0.5).astype(np.uint8))
        ImageDraw.Draw(t).text((6, 4), lab, fill=(255, 255, 0))
        tiles.append(t)
    cols = 2 if len(tiles) == 4 else len(tiles) if len(tiles) <= 3 else 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * ch))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * cw, (i // cols) * ch))
    sheet.save(out)


if __name__ == "__main__":
    main()
