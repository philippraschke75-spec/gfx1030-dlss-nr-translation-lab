"""cmp.py ref.npy test.npy: per-frame RGB error of test vs ref (f16 outputs; inf/nan treated as 0 like test_sync)."""
import sys, numpy as np
a = np.load(sys.argv[1]).view(np.float16).astype(np.float64)[..., :3]
b = np.load(sys.argv[2]).view(np.float16).astype(np.float64)[..., :3]
for k in range(a.shape[0]):
    x, y = np.nan_to_num(a[k], posinf=0, neginf=0), np.nan_to_num(b[k], posinf=0, neginf=0)
    x, y = np.clip(x, 0, None), np.clip(y, 0, None)
    d = np.abs(x - y); mse = (np.minimum(d, 1) ** 2).mean()
    print(f"frame {k}: identical {np.mean(a[k] == b[k]) * 100:.3f}%  max abs {d.max():.4g}  mean abs {d.mean():.3g}  "
          f"PSNR(peak 1) {10 * np.log10(1 / mse) if mse else float('inf'):.1f} dB  >0.01: {(d > 0.01).mean() * 100:.4f}%  >0.05: {(d > 0.05).mean() * 100:.4f}%")
