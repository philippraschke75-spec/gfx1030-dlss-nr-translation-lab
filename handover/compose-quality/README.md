# compose-quality (branch from v0.3-rc)

New ini keys (dlssnr_nr.cpp, kSyncHlsl `exp`):

- `upsample=bilinear|guided` (default bilinear), `upsample_sigma=0.1`: guided = joint-bilateral 2x2 upsample of the
  960x540 edit, weight = bilinear x similarity of the live pixel to each tap's network input. Used in the resampled,
  mv_reuse and nr_split paths.
- `reuse_depth_reject=1` (default, same as v0.3-rc), `reuse_depth_thresh=0.05`: relative depth gate on warped
  edits / history (fade from thresh to 4x thresh). 0 turns it off.

Measured (test_sync, 2560x1440 frame, 960x540 model, history=1 nr_split=1 vit_every=2 reuse=bottleneck, 30 frames):

| variant | HF out/in | flicker excess | visible flicker | PSNR vs 1280x720-model output | edit gradient at edges |
|---|---|---|---|---|---|
| v0.3-rc / new build, options off | x1.012 | +0.00088 | 3.14 % | 39.32 dB | 0.00617 |
| guided sigma 0.1 | x1.015 | +0.00093 | 3.52 % | 39.25 dB | 0.00787 |
| guided sigma 0.05 | x1.024 | +0.00097 | 3.81 % | 39.20 dB | 0.00848 |
| guided sigma 0.2 | x1.012 | +0.00089 | 3.29 % | 39.29 dB | 0.00703 |
| 1280x720 model (ref) | x1.006 | +0.00085 | 3.30 % | - | 0.00726 |

Options off: OUTHASH identical to v0.3-rc (074f0daf94012383). Ghost test (960x540, moving object, depth 0.5/0.1):
depth reject on 0.02489 / object 0.05290, off 0.02398 / 0.05263 (the metric compares with NR of the same frame,
so rejecting to the live frame always scores slightly worse there). Cost of guided: within noise at 2880x1620
(15.55-15.59 vs 15.55-15.64 ms per frame).

Crops: `crop_*.png`, columns input, v0.3-rc, guided, guided 0.05, guided 0.2, 720p model; bottom row = edit x8.

Credits: ideas from xenmods/DLSSNR-Cost-Scaler ("matched residual resolve", MIT, (c) 2026 xen) and
matiasLombo/neural-upstream (delta reprojection with depth rejection, MIT, (c) 2026 Matias Lombo). No code copied verbatim.
