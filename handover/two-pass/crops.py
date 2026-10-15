# crops.py: side-by-side 2x crops (input | 1 pass | 2 passes old | 2 passes fixed) from run.sh last-frame dumps
import os, numpy as np
from PIL import Image
D = 'E:/dlss-runtime/twopass_runs/'; W, H = 1920, 1080; here = os.path.dirname(os.path.abspath(__file__))
def load(t): return np.fromfile(D + t + '.bin', np.float32).reshape(H, W, 3)
def srgb(x): x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
tiles = [load('fix_s1_in'), load('fix_s1_out'), load('base_s2_out'), load('fix_s2_out')]
for name, (y, x) in {'crop_a': (400, 800), 'crop_b': (650, 300)}.items():
    row = [np.kron(srgb(im[y:y + 192, x:x + 192]), np.ones((2, 2, 1))) for im in tiles]
    Image.fromarray((np.concatenate(row, 1) * 255 + .5).astype(np.uint8)).save(os.path.join(here, name + '.png'))
