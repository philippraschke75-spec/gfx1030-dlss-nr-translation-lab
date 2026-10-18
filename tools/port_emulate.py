"""Our port's compose (compose=add) on a reference run at model resolution: out = frame + bilinear_up(NR_low - frame_low).

    py tools/port_emulate.py <full_probe_dir> <low_probe_dir> <name>     -> <full_probe_dir>/<name>.npy
"""
import sys, os
import numpy as np
from PIL import Image

full, low, name = sys.argv[1:4]
x = np.load(os.path.join(full, 'in.npy')); H, W = x.shape[:2]
e = np.load(os.path.join(low, 'raw.npy')) - np.load(os.path.join(low, 'in.npy'))
up = np.stack([np.asarray(Image.fromarray(e[..., c]).resize((W, H), Image.BILINEAR)) for c in range(3)], -1)
np.save(os.path.join(full, name + '.npy'), np.clip(x + up, 0, 1).astype(np.float32))
