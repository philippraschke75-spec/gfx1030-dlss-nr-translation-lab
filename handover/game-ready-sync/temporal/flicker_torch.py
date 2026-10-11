"""PyTorch reference for the temporal-flicker harness (see handover/temporal-2026-10-11/HARNESS.md).

Reads the model-size sequence test_sync dumps with NR_TEST_REF=1 NR_TEST_DUMPDIR=<dir> (f%03d_in.bin / f%03d_out.bin,
float32 linear RGB, input clamped to [0,1], output = nr_run of our HIP port), runs dlssnr-pytorch on the same inputs and
prints the frame-0 sanity check (our port vs PyTorch, single frame) and test_sync's flicker metric per mode:
  ours     : our port's dumped outputs (no network run; the like-for-like baseline)
  single   : PyTorch, every frame alone (history-valid = 0), what our port does today
  hist     : PyTorch with history as in SPEC.md / GEN.md: Phist = previous PyTorch output (P space) warped by the known
             motion; pre-block lanes 7-9 = centred(Phist), and after the network out += clamp(0.7397*sigmoid(head.a)) * (Phist - out),
             head.a = row 3 of out_project. Frame 0 (no history): Phist = current P (the vendor fallback).
  histslot : hist without the post lerp (lanes 7-9 only)
Input mapping: the combo data runs import/export mode 0 (values passed as is), so PyTorch gets the dumped values directly;
PSPACE=1 maps to P = sRGB(x/(1+x)) and back (mode-1 data, SPEC.md section 1).
Run from the dlssnr-pytorch folder: uv run python <this file> <dump dir> [frames] [modes, comma separated]
Sequence parameters must match the test_sync run (env MW, MH, FW, VX, VY, JIT; defaults 960, 540, 2560, 0.37, 0.13, 0.25).
"""
import os
import sys

import numpy as np
import torch

from dlssnr.io import Controls, _mirror
from dlssnr.layers import pre_post
from dlssnr.model import NRNet, _main
from dlssnr.weights import Weights

MW, MH, FW = int(os.environ.get("MW", 960)), int(os.environ.get("MH", 540)), int(os.environ.get("FW", 2560))
VX, VY, JIT = (float(os.environ.get(k, d)) for k, d in (("VX", 0.37), ("VY", 0.13), ("JIT", 0.25)))
BLEND = 0.7397  # post-block +0x48 gate (blend_scale) in all our data dirs


def halton(i, b):
    f, r, i = 1.0, 0.0, i + 1
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def shift(k):  # frame px, as test_sync
    return (k * VX + JIT * 2 * (halton(k % 8, 2) - .5), k * VY + JIT * 2 * (halton(k % 8, 3) - .5))


def motion(k):  # model px: warp(prev)(x) = prev(x + d)
    a, b, sc = shift(k), shift(k - 1), MW / FW
    return (a[0] - b[0]) * sc, (a[1] - b[1]) * sc


def warp(img, dx, dy):  # img [H, W, 3]; bilinear fetch at x + d (edges clamped; the metric skips them)
    h, w, _ = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = np.clip(xs + dx, 0, w - 1.001), np.clip(ys + dy, 0, h - 1.001)
    x0, y0 = u.astype(int), v.astype(int)
    a, b = (u - x0)[..., None], (v - y0)[..., None]
    return ((img[y0, x0] * (1 - a) + img[y0, x0 + 1] * a) * (1 - b) + (img[y0 + 1, x0] * (1 - a) + img[y0 + 1, x0 + 1] * a) * b).astype(np.float32)


def to_p(x):  # linear -> proxy P = sRGB(x / (1 + x))
    y = np.maximum(x, 0) / (1 + np.maximum(x, 0))
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055).astype(np.float32)


def from_p(p):
    y = np.where(p <= 0.04045, p / 12.92, np.power((np.clip(p, 0, 1) + 0.055) / 1.055, 2.4))
    return (y / np.maximum(1 - y, 1e-6)).astype(np.float32)


if os.environ.get("PSPACE") != "1":  # combo data = import/export mode 0 (identity): measured 31.5 dB vs 24.2 dB with the P mapping
    to_p = from_p = lambda x: np.asarray(x, np.float32)

L = np.array([.2126, .7152, .0722], np.float32)


def measure(ci, co, pi, po, k):  # test_sync's metric: (inst out, inst in, white-jump frac, visible frac)
    dx, dy = motion(k)
    ci, co, pi, po = (np.clip(x, 0, 1) for x in (ci, co, pi, po))
    wo, wi = warp(po, dx, dy), warp(pi, dx, dy)
    m = 8 + int(np.ceil(max(abs(dx), abs(dy))))
    s = np.s_[m:-m, m:-m]
    d = ((co - wo) @ L - (ci - wi) @ L)[s]
    return np.abs(co - wo)[s].mean(), np.abs(ci - wi)[s].mean(), (d > 3 / 255).mean(), (np.abs(d) > 2 / 255).mean()


def to_tensor(img, net):  # float [H, W, 3] -> padded [1, 3, rows, cols], as frame_to_tensor
    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    rows, cols = _mirror(torch.arange(net.padding.height), MH), _mirror(torch.arange(net.padding.width), MW)
    return t[:, rows][:, :, cols].unsqueeze(0).contiguous()


def main():
    d, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["ours", "single", "hist", "histslot"]
    load = lambda k, q: np.fromfile(os.path.join(d, f"f{k:03d}_{q}.bin"), np.float32).reshape(MH, MW, 3)
    weights = Weights.load("weights.safetensors")
    net = NRNet(weights, MW, MH)
    controls = Controls(seed=int(os.environ.get("SEED", 0)))
    hist = {}  # "p": padded Phist tensor while history is on; "lerp": apply the post lerp

    feats = pre_post.PreBlock.features

    def features(self, image):  # lanes 7..9 = the "previous colour" slot (PreBlock docstring, SPEC.md)
        f = feats(self, image)
        if "p" in hist:
            f = f.clone()
            f[:, 7:10] = pre_post._centred_colour(hist["p"])
        return f

    post = pre_post.PostBlock.forward
    post_layer = next(l for l in net.layers if isinstance(l, pre_post.PostBlock))
    s = post_layer.spec
    head_a = weights.f16(s.block, s.layer, "out_project", 16, pre_post.CHANNELS)[3]

    def post_forward(self, x, skip, frame):  # GEN.md: res += clamp(gate * sigmoid(head.a), 0, 1) * (aux - res)
        if not hist.get("lerp"):
            return post(self, x, skip, frame)
        wide = pre_post.swin_core(self.params, self.blend(x, skip), self.spec.shift_x, self.spec.shift_y)
        res = pre_post.compose(frame, torch.einsum("rc,bchw->brhw", self.project, wide), self.controls.intensity)
        a = torch.einsum("c,bchw->bhw", head_a, wide)[:, None]
        return res + (BLEND * torch.sigmoid(a)).clamp(0, 1) * (hist["p"] - res)

    pre_post.PreBlock.features = features
    pre_post.PostBlock.forward = post_forward
    for mode in modes:
        hist.clear()
        acc, pi, po, pp = [], None, None, None
        for k in range(n):
            ci = load(k, "in")
            cp = to_p(ci)
            if mode == "ours":
                out = load(k, "out")
            else:
                if mode in ("hist", "histslot"):
                    hist["p"] = to_tensor(cp if pp is None else warp(pp, *motion(k)), net)  # frame 0: current P (fallback)
                    hist["lerp"] = mode == "hist"
                with torch.no_grad():
                    outp = _main(net.run(to_tensor(cp, net), controls)[len(net.plan) - 1])[0, :, :MH, :MW].clamp(0, 1).permute(1, 2, 0).numpy()
                out, pp = from_p(outp), outp
            if k == 0 and mode == "single":
                ours = np.clip(load(0, "out"), 0, 1)
                oc = np.clip(out, 0, 1)
                psnr = lambda a, b: 10 * np.log10(1 / max(float(((a - b) ** 2).mean()), 1e-12))
                print(f"frame 0 sanity (linear, clamped): our nr_run vs PyTorch {psnr(ours, oc):.2f} dB (input vs PyTorch {psnr(ci, oc):.2f} dB); "
                      f"in P space {psnr(to_p(ours), outp):.2f} dB; edit correlation {np.corrcoef((ours - ci).ravel(), (oc - ci).ravel())[0, 1]:.3f}, "
                      f"edit rms ours {np.sqrt(((ours - ci) ** 2).mean()):.4f} torch {np.sqrt(((oc - ci) ** 2).mean()):.4f}", flush=True)
            if k > 0:
                acc.append(measure(ci, out, pi, po, k))
                print(f"{mode} frame {k}: inst out {acc[-1][0]:.5f} in {acc[-1][1]:.5f} mean(out-in) {float((np.clip(out, 0, 1) - ci).mean()):+.5f}", flush=True)
            pi, po = ci, out
        a = np.array(acc).mean(0)
        print(f"TORCH {mode} {MW}x{MH} over {len(acc)} frames: inst out {a[0]:.5f} in {a[1]:.5f} excess {a[0] - a[1]:+.5f} (x{a[0] / a[1]:.2f}) "
              f"| white jumps {100 * a[2]:.3f}% of px, visible flicker {100 * a[3]:.3f}% of px", flush=True)


if __name__ == "__main__":
    main()
