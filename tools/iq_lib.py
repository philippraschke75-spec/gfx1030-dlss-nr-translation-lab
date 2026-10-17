"""Image-quality helpers (handover/iq-guides): frame readers, resampling, metrics, PNG crops.

Frames are float32 [H, W, 3] in the network's input space (display-referred code values, clamped to [0, 1]).
"""
import json, os
import numpy as np


def r11g11b10(u):
    """uint32 R11G11B10_FLOAT -> float32 [..., 3]."""
    def f(bits, mb):   # 5-bit exponent, mb-bit mantissa, no sign
        e = (bits >> mb) & 31; m = bits & ((1 << mb) - 1)
        v = np.where(e == 0, m / (1 << mb) * 2.0 ** -14, (1 + m / (1 << mb)) * 2.0 ** (e.astype(np.int32) - 15))
        return np.where(e == 31, np.inf, v).astype(np.float32)
    return np.stack([f(u & 0x7FF, 6), f((u >> 11) & 0x7FF, 6), f((u >> 22) & 0x3FF, 5)], -1)


def load_capture(frame_dir):
    """A dlss-capture frame dir (meta.json + color.bin/depth.bin/motion.bin) -> (rgb [H,W,3], depth [H,W] or None, meta)."""
    m = json.load(open(os.path.join(frame_dir, 'meta.json')))
    c = m['files']['color']; W, H = c['width'], c['height']
    raw = np.fromfile(os.path.join(frame_dir, c['file']), np.uint8).reshape(H, c['row_bytes'])
    if c['dxgi_format'] == 26: rgb = r11g11b10(raw[:, :W * 4].copy().view(np.uint32))
    elif c['dxgi_format'] == 10: rgb = raw[:, :W * 8].copy().view(np.float16).reshape(H, W, 4)[..., :3].astype(np.float32)
    else: raise ValueError('colour format %d' % c['dxgi_format'])
    dep = None
    if 'depth' in m['files']:
        d = m['files']['depth']
        if d['dxgi_format'] in (19, 20, 40, 41):   # R32 typeless/float variants (D32 / R32_FLOAT), first 4 bytes
            dep = np.fromfile(os.path.join(frame_dir, d['file']), np.uint8).reshape(d['height'], d['row_bytes'])[:, :d['width'] * 4].copy().view(np.float32)
    return np.clip(np.nan_to_num(rgb, posinf=1.0), 0, 1), dep, m


def proxy(lin, exposure):
    """Scene-linear -> the network's display proxy (OpenDLSS-NR docs/frame.md, janblade SoftKnee): x * exposure, luminance
    shoulder above 0.75, peak channel brought to 1 by one scalar (hue kept), sRGB OETF."""
    d = np.maximum(np.nan_to_num(lin, posinf=65504.0), 0) * exposure
    L = luma(d)[..., None]
    rolled = 0.75 + 0.25 * (1 - np.exp(-(L - 0.75) / 0.25))
    d = np.where(L > 0.75, d * rolled / np.maximum(L, 1e-6), d)
    pk = d.max(-1, keepdims=True); d = np.where(pk > 1, d / np.maximum(pk, 1e-6), d)
    return np.where(d <= 0.0031308, d * 12.92, 1.055 * np.power(np.clip(d, 0, 1), 1 / 2.4) - 0.055).astype(np.float32)


def load_capture_lin(frame_dir):
    """Unclamped scene-linear capture colour (for proxy())."""
    m = __import__('json').load(open(os.path.join(frame_dir, 'meta.json'))); c = m['files']['color']
    raw = np.fromfile(os.path.join(frame_dir, c['file']), np.uint8).reshape(c['height'], c['row_bytes'])
    return r11g11b10(raw[:, :c['width'] * 4].copy().view(np.uint32)), m


def load_half4(path, W, H):
    x = np.fromfile(path, np.uint16).reshape(H, W, 4).view(np.float16)[..., :3].astype(np.float32)
    return np.clip(np.nan_to_num(x), 0, 1)


def to_half4(rgb):
    H, W, _ = rgb.shape
    return np.ascontiguousarray(np.concatenate([rgb, np.ones((H, W, 1), np.float32)], -1).astype(np.float16).view(np.uint16))


def resize(img, W, H):
    """Area-ish resample through PIL (float32 per channel, bicubic up / box down)."""
    from PIL import Image
    h0, w0 = img.shape[:2]
    if (w0, h0) == (W, H): return img.copy()
    flt = Image.BOX if W < w0 else Image.BICUBIC
    return np.stack([np.asarray(Image.fromarray(img[..., k].astype(np.float32), 'F').resize((W, H), flt)) for k in range(img.shape[2])], -1)


def psnr(a, b):
    return 10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-12))


def luma(x):
    return x @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def stats(inp, out, ref=None, mask=None):
    """Edit metrics. inp/out/ref [H,W,3] same size. Shadow = input luma below its 25th percentile (AO / indirect proxy)."""
    e = out - inp; L = luma(inp)
    sh = L <= np.percentile(L, 25)
    sel = (lambda a: a[mask]) if mask is not None else (lambda a: a.reshape(-1, *a.shape[2:]))
    r = {'edit': float(np.abs(sel(e)).mean()), 'edit_shadow': float(np.abs(e[sh]).mean()),
         'dL_shadow': float((luma(out) - L)[sh].mean()), 'white': float((out.max(-1) >= 0.999).mean() - (inp.max(-1) >= 0.999).mean())}
    if ref is not None:
        er = ref - inp
        r['psnr'] = psnr(sel(out), sel(ref)); r['edit_ref'] = float(np.abs(sel(er)).mean())
        r['corr'] = float(np.corrcoef(sel(e).ravel(), sel(er).ravel())[0, 1])
    return r


def save_png(path, img, scale=1):
    from PIL import Image
    a = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a, 'RGB')
    if scale != 1: im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)


def strip(path, tiles, labels=None, scale=1):
    """Side-by-side PNG of equally sized tiles (with a 4 px gap)."""
    h = max(t.shape[0] for t in tiles); g = np.ones((h, 4, 3), np.float32)
    row = []
    for t in tiles: row += [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))), g]
    save_png(path, np.concatenate(row[:-1], 1), scale)
    if labels:   # a black caption bar on top, one label per tile
        from PIL import Image, ImageDraw
        im = Image.open(path); bar = Image.new('RGB', (im.width, 16)); d = ImageDraw.Draw(bar); x = 0
        for t, l in zip(tiles, labels): d.text((x + 3, 2), l, fill=(255, 255, 255)); x += (t.shape[1] + 4) * scale
        out = Image.new('RGB', (im.width, im.height + 16)); out.paste(bar, (0, 0)); out.paste(im, (0, 16)); out.save(path)
