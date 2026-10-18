"""Side-by-side sheet + stats for encoding_probe.py output against NVIDIA's frame (RenoDX video, NR on).

    py tools/encoding_sheet.py <probe_dir> <nvidia_on.png> <out.png> [x0 y0 x1 y1]
Region default: the face in the FO4 RenoDX screenshots (right of the ReShade overlay).
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw

d, nv_png, out = sys.argv[1:4]
x0, y0, x1, y1 = map(int, sys.argv[4:8]) if len(sys.argv) > 7 else (980, 120, 1700, 1000)
x = np.load(os.path.join(d, 'in.npy'))
nv = np.asarray(Image.open(nv_png).convert('RGB'), np.float32) / 255
lum = lambda a: a @ np.float32([0.2126, 0.7152, 0.0722])
def stats(name, o):
    r = (slice(y0, y1), slice(x0, x1)); e = o[r] - x[r]
    hp = lambda a: a - np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((a.shape[1] // 8, a.shape[0] // 8), Image.BOX).resize((a.shape[1], a.shape[0]), Image.BILINEAR), np.float32) / 255
    print('%-6s region: mean |edit| %.4f  luma mean %.3f (in %.3f)  luma std %.3f (in %.3f)  fine-detail energy %.4f (in %.4f)' % (
        name, np.abs(e).mean(), lum(o[r]).mean(), lum(x[r]).mean(), lum(o[r]).std(), lum(x[r]).std(),
        np.abs(hp(lum(o[r]))).mean(), np.abs(hp(lum(x[r]))).mean()))
panels = [('input (NR off)', x)]
for n in os.environ.get('PANELS', 'raw,tm,odn,port720').split(','):
    p = os.path.join(d, n + '.npy')
    if os.path.exists(p): o = np.load(p); stats(n, o); panels.append((n, o))
stats('NVIDIA', nv); panels.append(('NVIDIA (RenoDX video)', nv))
crop = lambda a: Image.fromarray((np.clip(a[y0:y1, x0:x1], 0, 1) * 255 + 0.5).astype(np.uint8))
w, h = x1 - x0, y1 - y0; sheet = Image.new('RGB', (w * len(panels), h + 28), (0, 0, 0))
for i, (n, a) in enumerate(panels):
    sheet.paste(crop(a), (i * w, 28)); ImageDraw.Draw(sheet).text((i * w + 8, 6), n, fill=(255, 255, 255))
sheet.save(out); print('wrote', out)
