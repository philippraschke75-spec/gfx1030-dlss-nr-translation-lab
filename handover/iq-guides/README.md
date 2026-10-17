# Image quality: guide inputs, model resolution, Strong preset (branch `iq-guides`, 2026-10-17)

User report: compared with real DLSS 5, indirect light and ambient occlusion are missing and faces look worse. Base:
`origin/v0.4.1-rc`, new model (310.9.16 `WEIGHTS_CC_SMALL_FP8`, id `ca6f64f1dd88b965`), RX 6900 XT.

## 1. Guide inputs: nothing is missing

The network's pre-block takes 16 lanes per pixel and nothing else (OpenDLSS-NR `docs/network.md` and `docs/frame.md`,
dlssnr-pytorch `layers/pre_post.py`, our `pre_native.hip`):

| lane | content | ours (v0.4.1 data, kernarg of k_pre_native) |
| --- | --- | --- |
| 0-2 | Gaussian noise, hash of the padded pixel + seed | seed 0 (constant; NVIDIA counts frames, the reference uses 0 for one frame) |
| 3 | constant 1 | 1 |
| 4-6 | display proxy (sRGB code value), `(x - 0.5) * 0.125`, f16 | cscale 0.0625 (`* 2`), correct |
| 7-9 | reprojected previous output, same encoding (copy of 4-6 without history) | `+0x40` history buffer: the DLL warps the last NR output with the game's motion vectors, gates by colour and depth (history=1) |
| 10 | style / 128 | from the host (OptiScaler Style), `nr_set_conditioning` |
| 11 | local tone | from the host |
| 12-14 | structure / skin / other (auto mask: 1, skin, structure) | from the host, same mapping as `Controls.conditioning` |
| 15 | 0 | 0 |

There are no depth, normal, albedo, material, character-mask or exposure lanes. NVIDIA's API takes depth and motion
(`DLSSNR.Depth`, `DLSSNR.MotionVectors`, `DepthInverted`, MV scale; janblade `dlssnr_forwarder.cpp`) only to reproject the
history for lanes 7-9 and for the 4th output channel (temporal blend logit, `blend_scale` 0.74/0.77). Our DLL receives
the same depth and motion from the host (`dlssnr_call_evaluate_v2`) and uses them for exactly that. Exposure and the
display proxy (paper white, soft knee above 0.75, sRGB) are prepared by the host before our DLL sees the frame
(janblade `dlssnr.hlsl` SoftKnee / encode pass; our `hdr_mode` only acts when a host passes scene-linear HDR). The
skin "auto mask" is learned inside the network; lanes 12-14 only carry the strengths. All pre-block values were read
back from the v0.4.1 data (`cscale 0.0625, seed 0, c28/c2c/c30 = 1, 1, 0, c48/c4c = 1, 1`).

So the "3D-guided" part is the temporal path, which we already feed. The gap is elsewhere: model resolution (2) and
strength (3). Accuracy at the same input is not the problem: ours vs PyTorch, same model size, 4 real frames, edit
correlation 0.93 (540p) / 0.95 (720p, 1080p) / 0.96-0.99 (1440p), for every slider setting in the sweep (`tools/iq_analyse.py`).

## 2. Model resolution (faces)

Same frames (1080p capture `color_1920x1080.bin`: hand + dark hall; 3 Cyberpunk club captures, 720p, proxy as the host
builds it), network at 540p / 720p / 1080p with the edit carried to 1440p as the DLL does (bilinear), our network run
natively at 1440p, and PyTorch at 1440p. Note: the 1440p content is upscaled from 1080p/720p, so "native" means the network
at output resolution, not extra scene detail. Numbers: PSNR / edit correlation vs PyTorch 1440p, default conditioning.

| crop | 540p (today) | 720p | 1080p | 1440p native |
| --- | --- | --- | --- | --- |
| face (man, small) | 29.3 / 0.59 | 32.2 / 0.77 | 38.5 / 0.93 | 42.1 / 0.97 |
| face (woman) | 30.0 / 0.81 | 30.8 / 0.86 | 33.0 / 0.92 | 39.5 / 0.98 |
| hand (skin) | 28.8 / 0.89 | 30.5 / 0.93 | 31.5 / 0.94 | 36.8 / 0.99 |
| dark corner (AO / indirect) | 38.3 / 0.78 | 39.8 / 0.85 | 39.0 / 0.83 | 44.3 / 0.95 |
| whole frame (hall / club) | 33.4 / 0.79, 34.0 / 0.74 | 34.4 / 0.84, 34.2 / 0.79 | 33.7 / 0.81, 36.6 / 0.88 | 40.5 / 0.96, 41.9 / 0.96 |

On a small face the 540p network does less than half of the edit it does at 1080p (|edit| 0.013 vs 0.030), and what it
does correlates 0.59 with the full-resolution result. That is the "faces look worse" report. The dark-corner edit is
similar in size at every resolution (|edit| ~0.009) but its shape follows the full-res result better at 1440p.

Cost (nr_bench, exclusive, whole network per job; release = bottleneck reuse every 3rd job, mean over jobs; with
nr_split=1 a job is spread over two frames, so a displayed frame pays about half):

| model | release (reuse 3) median / mean | every job full | est. per displayed frame vs 540p | est. fps from 43 at 540p |
| --- | --- | --- | --- | --- |
| 960x540 | 7.0 / 8.3 ms | 11.0 ms | - | 43 |
| 1280x720 | 12.3 / 14.0 ms | 17.5 ms | +2.9 ms | ~38 |
| 1920x1080 | 28-30 / 30-35 ms | 36.7 ms | +11-13 ms | ~28 |
| 2560x1440 | 41.3 / 47.4 ms | 59.5 ms | +19.5 ms | ~23 |

(fps column is an estimate from the GPU time, not a game measurement.) 720p is the practical "High quality" step (data
already ships). 1080p / 1440p are not real-time on a 6900 XT with this network; data dirs can be built with
`tools/build_plan.py <dll> 1920x1080 ...` if someone wants to try (arena 0.9 / 1.4 GB).

Flicker (harness: test_sync, release ini, 60 frames, 2560x1440 output): 540p 1.83 % visible, 720p 2.69 %.

PNGs (columns labelled): `res1440_*.png` input | 540p (today) | 720p | 1080p | 1440p native | PyTorch 1440p.

## 3. Strength: sliders, Strong preset

Sweep at 540p (4 frames, carried to 1080p), |edit| = mean |out - in|, shadow = darkest 25 % of the input
(AO / indirect-light proxy), flicker = visible flicker share in the harness (default 1.83 %).

| setting | edit | shadow edit | shadow brightening | faces (woman, man) | flicker |
| --- | --- | --- | --- | --- | --- |
| default (tone 1, structure 1, Standard, I 1) | 0.0178 | 0.0121 | +0.0096 | 0.038, 0.013 | 1.83 % |
| local_tone 1.5 | 0.0189 | 0.0124 | +0.0092 | 0.036, 0.018 | 1.82 % |
| local_tone 2 | 0.0206 | 0.0130 | +0.0088 | 0.042, 0.020 | 1.87 % |
| local_structure 1.5 / 2 | 0.0160 / 0.0142 | 0.0129 / 0.0108 | +0.011 / +0.009 | - | 3.10 / 2.89 % |
| skin_structure 2 | 0.0169 | 0.0113 | +0.0094 | 0.034, 0.014 | 1.76 % |
| intensity 1.3 | 0.0222 | 0.0163 | +0.0127 | 0.046, 0.017 | 2.76 % (old blend 2.81) |
| intensity 1.6 | 0.0266 | 0.0207 | +0.0159 | - | 3.81 % (old 3.92) |
| intensity 2.0 | 0.0322 | 0.0265 | +0.0203 | - | 5.23 % (old 5.47) |
| **tone 2 + intensity 1.3 (Strong)** | **0.0258** | **0.0171** | **+0.0116** | **0.051, 0.025** | **2.60 %** |
| tone 1.5 + intensity 1.3 | 0.0236 | 0.0166 | +0.0123 | 0.045, 0.022 | 2.54 % |
| tone 2 + intensity 1.6 | - | - | - | - | 3.43 % |
| style Natural | 0.0352 | 0.0213 | -0.0053 | - | 3.49 % |
| style Cinematic | 0.0385 | 0.0255 | +0.0203 | - | 3.79 % |
| Cinematic + tone 2 | 0.0446 | 0.0290 | +0.0213 | - | 4.99 % |
| 720p + Strong | - | - | - | - | 3.76 % |

Findings:
- local_tone extrapolates sanely above 1 (PyTorch agrees: +34 % edit at 2) and costs no flicker. local_structure above
  1 makes the edit smaller and flickers more (PyTorch agrees on the size): leave it at 1. skin_structure 2 does not
  strengthen faces in this model.
- Intensity is the shadow-brightening (indirect light) lever but costs flicker, roughly linearly.
- Style is NVIDIA's biggest in-distribution lever: Cinematic doubles the edit and the shadow brightening (PyTorch
  corr 0.97 to ours), at the flicker cost of intensity 1.6; it also changes the colour grade (warmer skin). Natural
  darkens shadows. `style_*.png`.
- None of the settings above produced blocky artefacts or a white-out increase in the crops (white-out share falls in
  every case: the network pulls clipped pixels down).

Intensity above 1 now uses the reference compose (dlssnr-pytorch `PostBlock.compose`, NVIDIA's library): the luminance
ratio out/in is raised to the intensity (bounded 0.5..2x) and the colour rescaled by it, which keeps the hue; the old
DLL extrapolated `in + I * (nr - in)` per channel (chroma shift 0.0226 vs 0.0112 at I 2, values > 1 on 1.4 % of pixels).
Over-range is brought back by one scalar instead of a per-channel clamp. Intensity <= 1 is bit-identical to v0.4.1
(strict test_sync 60/60 matched). Flicker is 2-4 % lower than the old blend at the same intensity.

Recommended **Strong** preset (ini, overrides the OptiScaler sliders; or set the same in the OptiScaler NR menu, Pass 1):

```
local_tone=2
intensity=1.3
```

+45 % edit, +41 % shadow edit, faces +35 % / +84 %, flicker 1.83 -> 2.60 %. For a much stronger, graded look:
`style=2` (Cinematic) on top (2.5x default edit, flicker ~5 %). Presets: `presets/Strong.ini`, `presets/HighQuality.ini`.

OptiScaler menu (janblade `DlssNr_Menu.cpp`, Pass 1): Style Standard / Natural / Cinematic, Intensity 0-2, Local
structure 0-2, Local tone 0-2, Skin structure -1..2 (-1 = follow structure), Auto skin mask on/off; passes 2/3 have the
same sliders. Our DLL takes all of them from the host each frame; an ini key overrides its slider.

PNGs: `strong_*.png` input | default | tone 1.5 + I 1.3 | tone 2 + I 1.3 | tone 1.5 + I 1.6 (540p, reference compose).
`best_*.png`: input | today (540p default) | 720p + Strong | 1440p native + Strong | PyTorch 1440p + Strong.

## Not done / for other branches

- The 4th output channel (blend logit) and seed per frame belong to temporal stabilisation (temporal-quality).
- Host options (ReplaceDetail, Transfer, multipass) and edit-add vs full replace: covered by another agent.
- The async fallback path (no D3D12 interop) still blends linearly above intensity 1.

## Reproduce

Scripts in `tools/` (`iq_lib.py`, `iq_run.py`, `iq_analyse.py`, `iq_res.py`, `iq_strong.py`); work dir
`E:\dlss-runtime\iqg` (frames `in_<WxH>.npy`, outputs, `ours.sh`, `ref.ps1`, `flk.sh`, `bench.sh`).
