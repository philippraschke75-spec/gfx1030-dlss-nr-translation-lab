"""Temporal audit: PyTorch reference (dlssnr-pytorch-flex, new 310.9 small model, CPU) over a moving sequence with
different temporal policies. Writes <out>/<mode>.npy = float32 [N, H, W, 3] outputs (network space, intensity 1) and
<out>/seq.npy (inputs) + <out>/w_<mode>.npy (mean blend weight per frame).

    PYTHONPATH=<flex> DLSSNR_TRITON=0 <venv python> temporal_ref.py <out dir> <mode> [frames]

Sequence (env): SRC (half4 1920x1080 capture, default E:/dlss-runtime/color_1920x1080.bin) box-downsampled 2x to the
960x540 model scale, crop CW x CH (default 512x288) at CX,CY, panned by VX,VY model px per frame (default 1.3, 0.45)
plus Halton(2,3) jitter +-JIT px (0.25) and 1 % multiplicative noise (NOISE). Motion is exact: warp(prev)(x) = prev(x + d_k).

Modes (rules from OpenDLSS-NR docs/frame.md + demo/shaders, MIT, which reproduces NVIDIA's frame pipeline):
  single : no history, seed 0 (NVIDIA without history; our history=0)
  nv     : NVIDIA temporal: seed = frame index; frame 0 no history; history = previous *blended* output truncated to f16,
           reprojected with the 5-tap Catmull-Rom; lanes 7-9 = centred(history); post lerp weight
           clamp(sigmoid(head.a) * blend_scale); where there is no history (off screen) lanes 7-9 = current, weight 0.
  nv_s0  : nv with the seed held at 0
  ours   : our DLL v0.4.3 (hwp): seed 0, bilinear reprojection, colour gate 1 - smoothstep(0.08, 0.25, max|h - cur|)
           towards the *current input* (no edit), padding rows/cols = 0, off screen = current input, post lerp everywhere.
  fix    : ours with the fix: Catmull-Rom, no colour gate, reflected padding, off screen -> weight 0 (seed 0)
  fix_seed : fix + seed = frame index (= nv up to f16 truncation)
"""
import contextlib, io, os, sys
import numpy as np
import torch

import dlssnr.extract as ex
ex.RESOURCE_NAME = 'WEIGHTS_CC_SMALL_FP8'
from dlssnr.extract import extract
from dlssnr.io import Controls, _mirror
from dlssnr.model import NRNet, _main
from dlssnr.layers import pre_post

E = lambda k, d: type(d)(os.environ.get(k, d))
CW, CH, CX, CY = E('CW', 512), E('CH', 288), E('CX', 200), E('CY', 120)
VX, VY, JIT, NOISE = E('VX', 1.3), E('VY', 0.45), E('JIT', 0.25), E('NOISE', 0.01)
DLL = os.environ.get('DLL', 'E:/dlss-runtime/v041/nvngx_dlssnr.dll')


def halton(i, b):
    f, r, i = 1.0, 0.0, i + 1
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r


def shift(k):
    return k * VX + JIT * 2 * (halton(k % 8, 2) - .5), k * VY + JIT * 2 * (halton(k % 8, 3) - .5)


def motion(k):  # warp(prev)(x) = prev(x + d)
    a, b = shift(k), shift(k - 1)
    return a[0] - b[0], a[1] - b[1]


def bilin(img, u, v):  # img [h, w, 3]; u, v pixel-centre coords (0 = first pixel centre), clamped to the image
    h, w, _ = img.shape
    u = np.clip(u, 0, w - 1.0001); v = np.clip(v, 0, h - 1.0001)
    x0, y0 = u.astype(int), v.astype(int); a, b = (u - x0)[..., None], (v - y0)[..., None]
    return ((img[y0, x0] * (1 - a) + img[y0, x0 + 1] * a) * (1 - b) + (img[y0 + 1, x0] * (1 - a) + img[y0 + 1, x0 + 1] * a) * b).astype(np.float32)


def catmull5(img, u, v):  # OpenDLSS-NR historySample: 5-tap Catmull-Rom with the bilinear-weight trick (pixel-centre coords)
    h, w, _ = img.shape
    px, py = u + 0.5, v + 0.5                       # GLSL position in pixels (texel centres at +0.5)
    bx, by = np.floor(px - 0.5) + 0.5, np.floor(py - 0.5) + 0.5
    fx, fy = np.clip(px - bx, 0, 1), np.clip(py - by, 0, 1)
    def wts(f):
        sq, cu = f * f, f * f * f
        w0 = -0.5 * (f + cu) + sq; w1 = cu * 1.5 - sq * 2.5 + 1; w3 = (cu - sq) * 0.5; w2 = 1 - w0 - w1 - w3
        return w0, w1, w2, w3
    w0x, w1x, w2x, w3x = wts(fx); w0y, w1y, w2y, w3y = wts(fy)
    mx, my = w1x + w2x, w1y + w2y
    cl = lambda p, n: np.clip(p, 0.5, n - 0.5) - 0.5   # back to pixel-centre coords for bilin
    lowx, lowy = cl(bx - 1, w), cl(by - 1, h); cx, cy = cl(bx + w2x / mx, w), cl(by + w2y / my, h); hix, hiy = cl(bx + 2, w), cl(by + 2, h)
    left, top, mid, bottom, right = bilin(img, lowx, cy), bilin(img, cx, lowy), bilin(img, cx, cy), bilin(img, cx, hiy), bilin(img, hix, cy)
    a, b, c, d, e = w0x * my, w0y * mx, mx * my, w3y * mx, w3x * my
    val = left * a[..., None] + top * b[..., None] + mid * c[..., None] + bottom * d[..., None] + right * e[..., None]
    return (val / (a + b + c + d + e)[..., None]).astype(np.float32)


def trunc_f16(x):  # stored history: truncated toward zero to the half grid
    h = x.astype(np.float16); over = np.abs(h.astype(np.float32)) > np.abs(x)
    h[over] = np.nextafter(h[over], np.float16(0)); return h.astype(np.float32)


def sequence(n):
    src = np.fromfile(os.environ.get('SRC', 'E:/dlss-runtime/color_1920x1080.bin'), np.uint16).reshape(1080, 1920, 4).view(np.float16)[..., :3].astype(np.float32)
    src = np.clip(np.nan_to_num(src), 0, 1); half = src.reshape(540, 2, 960, 2, 3).mean((1, 3))
    ys, xs = np.mgrid[0:CH, 0:CW].astype(np.float32); rng = np.random.default_rng(1); out = []
    for k in range(n):
        sx, sy = shift(k)
        f = bilin(half, xs + CX + sx, ys + CY + sy)
        out.append(np.clip(f * (1 + NOISE * rng.standard_normal(f.shape).astype(np.float32)), 0, 1))
    return np.stack(out)


def main():
    out, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32
    os.makedirs(out, exist_ok=True)
    seq = sequence(n); np.save(os.path.join(out, 'seq.npy'), seq)
    with contextlib.redirect_stdout(io.StringIO()): w = extract(DLL, None)
    net = NRNet(w, CW, CH)
    rows, cols = _mirror(torch.arange(net.padding.height), CH), _mirror(torch.arange(net.padding.width), CW)
    PH, PW = net.padding.height, net.padding.width
    pad = lambda img: torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[:, rows][:, :, cols].unsqueeze(0).contiguous()
    def zpad(img):  # our hwp: padding = 0
        z = np.zeros((PH, PW, img.shape[2]), np.float32); z[:CH, :CW] = img; return torch.from_numpy(np.ascontiguousarray(z.transpose(2, 0, 1))).unsqueeze(0)
    post_layer = next(l for l in net.layers if isinstance(l, pre_post.PostBlock)); s = post_layer.spec
    blend = float(w.f16(s.block, s.layer, 'blend_scale', 1)[0])
    head_a = w.f16(s.block, s.layer, 'out_project', 16, pre_post.CHANNELS)[3]
    H = {}   # 'p': padded history [1,3,PH,PW]; 'm': padded lerp mask [1,1,PH,PW] (1 = history there)
    feats, post = pre_post.PreBlock.features, pre_post.PostBlock.forward

    def features(self, image):
        f = feats(self, image)
        if 'p' in H:
            f = f.clone(); hc = pre_post._centred_colour(H['p']); m = H['m']
            f[:, 7:10] = m * hc + (1 - m) * f[:, 7:10]
        return f

    wm = []
    def post_forward(self, x, skip, frame):
        if 'p' not in H:
            return post(self, x, skip, frame)
        wide = pre_post.swin_core(self.params, self.blend(x, skip), self.spec.shift_x, self.spec.shift_y)
        res = pre_post.compose(frame, torch.einsum('rc,bchw->brhw', self.project, wide), self.controls.intensity)
        wt = (blend * torch.sigmoid(torch.einsum('c,bchw->bhw', head_a, wide)[:, None])).clamp(0, 1) * H['lm']
        wm.append(float(wt[..., :CH, :CW].mean()))
        return res + wt * (H['p'] - res)

    pre_post.PreBlock.features, pre_post.PostBlock.forward = features, post_forward
    ys, xs = np.mgrid[0:CH, 0:CW].astype(np.float32)
    res, prev, wlog = [], None, []
    for k in range(n):
        cur = seq[k]; H.clear(); wm.clear()
        if mode != 'single' and prev is not None:
            dx, dy = motion(k); u, v = xs + dx, ys + dy
            on = ((u >= -0.5) & (u <= CW - 0.5) & (v >= -0.5) & (v <= CH - 0.5)).astype(np.float32)[..., None]
            if mode == 'ours':
                h = bilin(prev, u, v); a = np.abs(h - cur).max(-1, keepdims=True)
                t = np.clip((a - 0.08) / (0.25 - 0.08), 0, 1); wg = (1 - t * t * (3 - 2 * t)) * on
                H['p'] = zpad(cur + wg * (h - cur)); H['m'] = torch.ones(1, 1, PH, PW); H['lm'] = torch.ones(1, 1, PH, PW)
            else:   # nv / nv_s0 / fix / fix_seed
                h = catmull5(prev, u, v)
                H['p'] = pad(np.where(on > 0, h, cur)); H['m'] = H['lm'] = pad(on)
        seed = k if mode in ('nv', 'fix_seed') else 0
        with torch.no_grad():
            o = _main(net.run(pad(cur), Controls(seed=seed))[len(net.plan) - 1])[0, :, :CH, :CW].permute(1, 2, 0).numpy().copy()
        res.append(o); wlog.append(wm[0] if wm else 0.0)
        prev = trunc_f16(o) if mode in ('nv', 'nv_s0') else o
        e = o - cur
        print(f'{mode} frame {k}: edit mean|.| {np.abs(e).mean():.5f} rms {np.sqrt((e * e).mean()):.5f} blend w {wlog[-1]:.3f}', flush=True)
    np.save(os.path.join(out, f'{mode}.npy'), np.stack(res)); np.save(os.path.join(out, f'w_{mode}.npy'), np.array(wlog))


if __name__ == '__main__':
    main()
