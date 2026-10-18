"""Input-encoding probe: the PyTorch reference (dlssnr-pytorch-flex, run with that folder on PYTHONPATH) on one sRGB
frame, with candidate encodings of the frame before the network and the matching decode after it.

    py tools/encoding_probe.py <nvngx_dlssnr.dll> <in.png> <out_dir> [W H]      CODECS=raw,tm,odn (default all)

raw : the frame's code values as is (our import mode 0; NVIDIA's path per OpenDLSS-NR parity and dlssnr-pytorch)
tm  : sRGB(x / (1 + x)) of the code values (k_import mode 1: what a re-encode of an RGBA16F frame would look like)
odn : OpenDLSS-NR's demo proxy with the code values taken as scene-linear: sRGB(shoulder(x)), paper white 1
Controls = RenoDX showcase: Cinematic (2), intensity 2, tone 2, structure 1.88, skin 2, auto mask off (CTL=default: 1/0/1/1/-1/on).
"""
import sys, os, contextlib, io
import numpy as np, torch
from PIL import Image
from dlssnr.extract import extract
from dlssnr.io import Controls, _mirror
from dlssnr.model import NRNet, _main

enc = lambda x: np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.clip(x, 0, 1), 1 / 2.4) - 0.055)
dec = lambda y: np.where(y <= 0.04045, y / 12.92, np.power((np.clip(y, 0, 1) + 0.055) / 1.055, 2.4))
sh = lambda v: np.where(v > 0.75, 0.75 + 0.25 * (1 - np.exp(-5.770780 * (v - 0.75))), v)
shi = lambda p: np.where(p > 0.75, 0.75 - np.log(np.clip(1 - (p - 0.75) / 0.25, 1e-6, 1)) / 5.770780, p)
CODECS = {'raw': (lambda x: x, lambda y: y),
          'tm': (lambda x: enc(x / (1 + x)), lambda y: (lambda z: z / (1 - z))(np.minimum(dec(y), 0.999))),
          'odn': (lambda x: enc(sh(x)), lambda y: shi(dec(y)))}
assert np.allclose(CODECS['tm'][1](CODECS['tm'][0](np.float32([0.1, 0.5, 0.9]))), [0.1, 0.5, 0.9], atol=1e-4)
assert np.allclose(CODECS['odn'][1](CODECS['odn'][0](np.float32([0.1, 0.5, 0.9]))), [0.1, 0.5, 0.9], atol=1e-4)


def main(dll, png, out, W=None, H=None):
    img = Image.open(png).convert('RGB')
    if W: img = img.resize((int(W), int(H)), Image.BICUBIC)
    x = np.asarray(img, np.float32) / 255; H, W = x.shape[:2]
    with contextlib.redirect_stdout(io.StringIO()): w = extract(dll, None)
    net = NRNet(w, W, H)
    rows, cols = _mirror(torch.arange(net.padding.height), H), _mirror(torch.arange(net.padding.width), W)
    pad = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[:, rows][:, :, cols].unsqueeze(0).contiguous()
    ctl = {} if os.environ.get('CTL') == 'default' else dict(style=2, intensity=2.0, local_tone=2.0, local_structure=1.88, skin_structure=2.0, auto_mask=False)
    os.makedirs(out, exist_ok=True)
    lum = lambda a: a @ np.float32([0.2126, 0.7152, 0.0722])
    np.save(os.path.join(out, 'in.npy'), x)
    for name in os.environ.get('CODECS', 'raw,tm,odn').split(','):
        e, d = CODECS[name]; p = e(x).astype(np.float32)
        with torch.no_grad():
            y = _main(net.run(pad(p), Controls(seed=0, **ctl))[len(net.plan) - 1])[0, :, :H, :W].clamp(0, 1).permute(1, 2, 0).numpy()
        o = np.clip(d(y), 0, 1).astype(np.float32)
        Image.fromarray((o * 255 + 0.5).astype(np.uint8)).save(os.path.join(out, name + '.png'))
        np.save(os.path.join(out, name + '.npy'), o)
        print('%s: net-in mean %.3f std %.3f | mean |out-in| %.4f (net space %.4f) | luma std in %.3f out %.3f | luma mean in %.3f out %.3f' % (
            name, p.mean(), p.std(), np.abs(o - x).mean(), np.abs(y - p).mean(), lum(x).std(), lum(o).std(), lum(x).mean(), lum(o).mean()), flush=True)


if __name__ == '__main__': main(*sys.argv[1:])
