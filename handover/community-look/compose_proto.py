"""Offline prototype of the DLL compose modes (numpy, CPU).

The network ran at a model size below the frame. Each mode carries its answer back onto the full-size frame.
Scored against the network run AT the frame size (what NVIDIA does when the host hands it that size):

    python compose_proto.py <out dir> <tag: def|com> <full WxH> <low WxH> [intensity]

Inputs (from run.sh, NR_TEST_DUMP): net_<tag>_<WxH>_in.bin / _out.bin, float32 RGB, last frame.
Modes (the DLL's compose= values):
  add     today: frame + I * up(N - L)
  nv      NVIDIA's intensity at model size (luminance ratio ^ I, guard 2, clamp), then added as `add`
  ratio   OkLab matched residual (janblade Transfer 2, NvidiaResidualModel): log-luminance ratio and chroma
          difference carried up and applied to the frame's own OkLab -- multiplicative, so the frame's detail
          scales with the network's local brightening
  guided  local-linear transfer (He et al. guided filter on luma, dlssnr-pytorch scripts/m4_upsample.py):
          y = up(a) * Y_frame + up(b), chroma by the upsampled chroma edit
"""
import sys

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def load(path, w, h):
    return np.fromfile(path, np.float32).reshape(h, w, 3)


def up(x, w, h):
    """Bilinear, pixel centres at .5, edges clamped (the DLL's bil())."""
    hs, ws = x.shape[:2]
    px = np.clip((np.arange(w) + 0.5) * ws / w - 0.5, 0, ws - 1)
    py = np.clip((np.arange(h) + 0.5) * hs / h - 0.5, 0, hs - 1)
    x0 = np.floor(px).astype(int); y0 = np.floor(py).astype(int)
    x1 = np.minimum(x0 + 1, ws - 1); y1 = np.minimum(y0 + 1, hs - 1)
    tx = (px - x0)[None, :, None]; ty = (py - y0)[:, None, None]
    r0 = x[y0][:, x0] * (1 - tx) + x[y0][:, x1] * tx
    r1 = x[y1][:, x0] * (1 - tx) + x[y1][:, x1] * tx
    return r0 * (1 - ty) + r1 * ty


def luma(x):
    return x @ LUMA


def box(x, r):
    """(2r+1)^2 mean, edges clamped."""
    k = 2 * r + 1
    p = np.pad(x, ((r, r), (r, r)) + ((0, 0),) * (x.ndim - 2), mode="edge")
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)) + ((0, 0),) * (x.ndim - 2))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def oklab(rgb):
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
    return np.cbrt(np.maximum(np.clip(rgb, 0, 1) @ m1.T, 1e-12)) @ m2.T


def from_oklab(lab):
    m1 = np.array([[1.0, 0.3963377774, 0.2158037573], [1.0, -0.1055613458, -0.0638541728],
                   [1.0, -0.0894841775, -1.2914855480]], np.float32)
    m2 = np.array([[4.0767416621, -3.3077115913, 0.2309699292], [-1.2684380046, 2.6097574011, -0.3413193965],
                   [-0.0041960863, -0.7034186147, 1.7076147010]], np.float32)
    return (lab @ m1.T) ** 3 @ m2.T


def nv_intensity(L, N, i):
    """dlssnr-pytorch compose(): blend up to 1, luminance ratio ^ I past it (guard 2), clamped."""
    out = np.clip(L + min(i, 1.0) * (N - L), 0, 1)
    if i > 1:
        ratio = (luma(out) + 1 / 512) / (luma(L) + 1 / 512)
        amp = np.clip(np.maximum(ratio, 1e-6) ** i, 0.5, 2.0)
        out = np.clip(out * (amp / np.maximum(ratio, 1e-6))[..., None], 0, 1)
    return out


def modes(F, L, N, i, r=2, eps=1e-3):
    h, w = F.shape[:2]
    out = {}
    out["add"] = F + i * up(N - L, w, h)
    out["nv"] = F + up(nv_intensity(L, N, i) - L, w, h)
    # ratio: L' = L_F * exp(I * up(log ratio)), ab' = ab_F + min(I,1) * up(d ab), ratio guard 2 (ln 2)
    la, lb = oklab(L), oklab(N)
    res = np.stack([np.log(np.maximum(lb[..., 0], 0.1)) - np.log(np.maximum(la[..., 0], 0.1)),
                    lb[..., 1] - la[..., 1], lb[..., 2] - la[..., 2]], -1)
    res[..., 0] *= np.clip(10 * la[..., 0], 0, 1)
    R = up(res, w, h); lf = oklab(F)
    lf[..., 0] *= np.exp(np.clip(i * R[..., 0], -np.log(2), np.log(2)))
    lf[..., 1:] += min(i, 1.0) * R[..., 1:]
    out["ratio"] = from_oklab(lf)
    # guided: per window y_N ~ a * y_L + b on the model raster, applied to the frame's luma
    yl, yn, yf = luma(L), luma(N), luma(F)
    ml, mn = box(yl, r), box(yn, r)
    a = (box(yl * yn, r) - ml * mn) / (box(yl * yl, r) - ml * ml + eps)
    b = mn - a * ml
    A, B = up(box(a, r)[..., None], w, h)[..., 0], up(box(b, r)[..., None], w, h)[..., 0]
    dy = A * yf + B - yf                      # the luma edit, at full detail
    ce = up((N - yn[..., None]) - (L - yl[..., None]), w, h)
    out["guided"] = F + i * (dy[..., None] + ce)
    return out


def lap(y):
    return np.abs(4 * y[1:-1, 1:-1] - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]).mean()


def psnr(a, b):
    return 10 * np.log10(1 / np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))


def main():
    d, tag, full, low = sys.argv[1:5]
    i = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    fw, fh = map(int, full.split("x")); lw, lh = map(int, low.split("x"))
    F = load(f"{d}/net_{tag}_{full}_in.bin", fw, fh)
    T = load(f"{d}/net_{tag}_{full}_out.bin", fw, fh)       # the network at full size: the target
    L = load(f"{d}/net_{tag}_{low}_in.bin", lw, lh)
    N = load(f"{d}/net_{tag}_{low}_out.bin", lw, lh)
    T = nv_intensity(F, T, i)                              # NVIDIA's own intensity on the full-size answer
    yF, yT = luma(F), luma(T)
    print(f"{tag} {low} -> {full}, intensity {i}: target edit |dY| {np.abs(yT - yF).mean():.5f}, "
          f"HF target {lap(yT):.5f} frame {lap(yF):.5f}; no-NR PSNR {psnr(F, T):.2f}")
    for name, o in modes(F, L, N, i).items():
        yo = luma(np.clip(o, 0, 1))
        print(f"  {name:7s} PSNR {psnr(o, T):6.2f}  edit |dY| {np.abs(yo - yF).mean():.5f}  "
              f"HF {lap(yo):.5f} (x{lap(yo) / lap(yT):.3f} of target)  corr(edit) "
              f"{np.corrcoef((yo - yF).ravel(), (yT - yF).ravel())[0, 1]:.3f}")


if __name__ == "__main__":
    main()
