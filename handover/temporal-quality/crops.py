# crops.py <out.png> <W> <H> <x0,y0,w,h> <scale> <label=dumpprefix> ...
# Dumps: test_sync NR_TEST_DUMP=<prefix> (f32 RGB W x H: <prefix>_in.bin, <prefix>_out.bin).
# Top row: input, then each variant's output (gamma 2.2 for viewing). Bottom row: |out - in| edit x8 (0.5 = no edit).
import sys
import numpy as np
from PIL import Image, ImageDraw

out, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
x0, y0, cw, ch = map(int, sys.argv[4].split(','))
sc = int(sys.argv[5])
ld = lambda p: np.fromfile(p, np.float32).reshape(H, W, 3)[y0:y0 + ch, x0:x0 + cw]
view = lambda a: (np.clip(a, 0, 1) ** (1 / 2.2) * 255 + .5).astype(np.uint8)
edit = lambda a: (np.clip(a * 8 + .5, 0, 1) * 255 + .5).astype(np.uint8)
vs = [a.split('=', 1) for a in sys.argv[6:]]
inp = ld(vs[0][1] + '_in.bin')
top = [('input', view(inp))] + [(l, view(ld(p + '_out.bin'))) for l, p in vs]
bot = [('', np.full_like(top[0][1], 128))] + [(l + ' edit x8', edit(ld(p + '_out.bin') - ld(p + '_in.bin'))) for l, p in vs]
img = Image.new('RGB', (len(top) * cw * sc, 2 * ch * sc + 2 * 14), (0, 0, 0))
d = ImageDraw.Draw(img)
for r, row in enumerate((top, bot)):
    for i, (l, a) in enumerate(row):
        img.paste(Image.fromarray(a).resize((cw * sc, ch * sc), Image.NEAREST), (i * cw * sc, r * (ch * sc + 14) + 14))
        d.text((i * cw * sc + 3, r * (ch * sc + 14) + 1), l, fill=(255, 255, 255))
img.save(out)
print(out, img.size)
