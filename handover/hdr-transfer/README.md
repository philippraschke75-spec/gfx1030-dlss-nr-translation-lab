# RenoDX "HDR Transfer Strength" (branch `hdr-transfer`, base `origin/v0.4.3-rc` 28d1475)

Task: copy the look of RenoDX's dlss5 addon (`renodx-dlss5.addon64` 0.2026.827.2036, YouTube "DLSS 5 Fallout 4 with Mods + ENB"),
"Control-compatible color transfer": Scene Paper-White Scale 1.1, HDR Transfer Strength 1 vs 0, Color Strength 1.

## Source

- clshortfuse/renodx has no dlss5 addon on any branch or in its history (only `src/addons/dlssfix`); the addon binary has no published source.
- **Yukikaze20170315/RenoDX-DLSS5-Generic** (MIT, "Copyright (c) 2025 Carlos Lopez Jr.", the RenoDX licence): the addon source
  (v7.5.0-rc5 / v7.6 / v8.5dev, newer than the video's build). The three sliders are in `dlssnr.hpp` (`paper_white_scale`, `transfer_strength`,
  `color_strength`) and the maths is in `shaders/legacy_encode|legacy_common` (`ResolvePixel`, `UpgradeToneMap`) and `v6_common.hlsli`.
  lunks/stray-dlssnr confirms "Control-compatible color transfer" is RenoDX's heading for these three controls, and that Transfer 0 is an exact bypass.
- OpenDLSS-NR `demo/shaders/nr_composite.comp` uses the same `upgradeToneMap` (at strength 1) after its own encode (`nr_preprocess.comp`: paper white, sRGB, `(x - 0.5) * 0.125`).

## The maths (RenoDX, legacy codec = the video's build: "FP16 working surface; HDR soft-clip/sRGB only when NGX marks HDR")

Encode (only for HDR): `p = x / paperWhite` (PQ: linearise, `/ (203/10000 * paperWhite)`), hue-preserving max-channel shoulder above 0.75
(`0.75 + 0.25 (1 - exp(-5.77 (m - 0.75)))`), sRGB encode. SDR is passed through unchanged.

Decode, per pixel, O = original, P = proxy (network input), N = network output (HDR: P, N sRGB-decoded, O = x / paperWhite):
```
UpgradeToneMap: if O_y < P_y: U = N * O_y / P_y          (the edit as a luminance ratio)
                else:         U = N * (N_y + O_y - P_y) / N_y   (the excess above the proxy added back)
                U = HueOkLab(U, N)                       (== U for a positive scalar; OkLab chroma scales like L)
upgraded = lerp(O, U, TransferStrength)        HDR path;   SDR path: upgraded = lerp(O, N, TransferStrength)
result   = lerp(O * clamp(upgraded_y / O_y, 0, 4), upgraded, ColorStrength)
```
So **Transfer Strength is the blend between the untouched frame and the network result. At 0 the output is the game frame,
bit for bit (NR off).** Paper white does nothing on an SDR frame. Color Strength 0 keeps the frame's colour and takes only the luminance change.

Screenshots 1469 / 1470 (transfer 1 / 0) are therefore NR on / NR off, not a finer transfer. The micro-contrast the user sees is the
whole NR edit at Cinematic, intensity 2, tone 2, structure 1.88, skin 2, with the network running at the output size (2560x1440) on NVIDIA.

## What our DLL now does (ini keys, `[nr]`)

| key | default | meaning |
| --- | --- | --- |
| `transfer_strength` | 0 | 0 = today's compose (bit-exact v0.4.3). t > 0: shown = lerp(today, RenoDX transfer, t) |
| `color_strength` | 1 | RenoDX Color Strength (used when t > 0) |
| `paper_white` | 1 | HDR (`hdr_mode`) only: divides the exposure E (`Ex()`), so encode, decode, gates, history and edit_temporal all move together |

Our transfer `rtx()` (kSyncHlsl): O = live colour in linear light (LDR: sRGB-decoded frame, HDR: x E), P / N = sRGB-decoded network input and
answer at this pixel, then RenoDX's UpgradeToneMap and Color Strength, back to the frame's encoding. Deviations, both deliberate:
- **t = 0 means today, not "NR off"**: our compose already equals RenoDX's transfer 1 where O == P (full-size path), so the key only blends the
  formula difference in.
- Above the proxy the excess is added per channel (`N + max(O - P, 0)`, same luminance as RenoDX) because our LDR input clips per channel and N
  has lost the hue of a > 1 pixel (with RenoDX's version strict test_sync showed max error 20 on HDR specks; now it passes).
- The encode stays ours (LDR: clamp; HDR: `T(x) = sRGB(xE / (1 + xE))`). RenoDX's soft-clip only exists for HDR frames, and the host
  (OptiScaler) already builds the display proxy before our DLL in the real game.

Applied on every compose path of `exp`: full size, resampled, resampled + edit_temporal, nr_split (+ edit_temporal), mv_reuse, fg reuse.
P is the network input belonging to the edit (resampled: `bil(p, 0)`; split / mv_reuse: the stored input `o`; edit_temporal / fg: the live
frame's model-footprint proxy `colAt`, as `imp` feeds the network). The network input itself is unchanged by `transfer_strength`, so history and
the edit_temporal blend are untouched; `paper_white` changes the HDR input and goes through the same `T/Ti/L` everywhere.

## Numbers (`stills.txt`, `flicker.txt`, `exact.txt`)

Stills: test_sync, static camera, frame 2560x1440, new model (310.9.16 small), release Standard preset (history, edit_temporal 0.7, nr_split 3).
Crops: edit = mean |dY|, HF = mean |Laplacian| out / in, LC = 7x7 local contrast out / in.

| crop, model | default t0 / t1 | Showcase t0 | Showcase t0.5 | Showcase t1 |
| --- | --- | --- | --- | --- |
| face woman, 540p | 0.026, HF 1.01 / 1.02 | 0.052, HF 1.11, LC 1.24 | 0.051, HF 1.13, LC 1.23 | 0.050, HF 1.18, LC 1.23 |
| face man, 540p | 0.012, LC 1.13 / 1.12 | 0.036, HF 1.36, LC 1.68 | 0.036, 1.34, 1.66 | 0.036, 1.35, 1.64 |
| face woman, 1080p | 0.030, HF 1.01 / 1.01 | 0.053, HF 1.27, LC 1.23 | 0.053, 1.27, 1.23 | 0.052, 1.28, 1.23 |
| hand, 540p | 0.082, HF 1.07 / 1.12 | 0.107, HF 1.01 | 0.107, 1.04 | 0.106, 1.07 |
| textured wall, 540p | 0.009, HF 1.04 / 1.04 | 0.057, HF 1.46, LC 1.42 | 0.056, 1.45, 1.41 | 0.056, 1.43, 1.40 |
| highlight, 540p | 0.010, HF 1.03 / 1.01 | 0.073, HF 1.34, LC 1.28 | 0.072, 1.31, 1.26 | 0.072, 1.28, 1.25 |
| whole club frame, 540p | 0.016 / 0.015 | 0.029, HF 1.17, LC 1.22 | 0.028, 1.18, 1.21 | 0.028, 1.19, 1.21 |

- transfer 0 -> 1 changes the edit by -2 % and HF / LC by a few % either way (+ on skin and hands, - on highlights and textures). Visually the
  crops are indistinguishable (`*_showcase.png`). It is not the lever.
- The Showcase settings are: edit x2 to x6, HF up to x1.5, local contrast up to x1.7 (`levers_*.png`: input | default | Showcase 540p | Showcase 1080p | + transfer 1).
- Flicker harness (moving, 60 frames, edit_temporal 0.7, 540p): default 0.81 / 0.79 / 0.94 % visible (t 0 / 0.5 / 1), Showcase 3.35 / 3.27 / 3.65 %.
- Cost: evaluate GPU time (whole list, network excluded by the split) median 3.1 / 2.9 / 3.8 ms default, 3.5 / 3.4 / 3.4 ms Showcase: within
  noise (shared GPU). Analytically the transfer adds 3 pow per channel per output pixel (+ 16 taps on edit_temporal paths), well under 0.3 ms at 1440p.

Bit-exactness, transfer off (`exact.txt`): OUTHASH identical to the v0.4.3-rc DLL built from git (28d1475) on full size (9a0f2da610ec1383),
1080p resampled (7eb1c29ffd704383), Cinematic preset with nr_split 3 + edit_temporal + intensity 1.3 (9435a6a9813ae383), mv_reuse
(5c70d911ed6e8383), and resampled + edit_temporal (be438d1cb9334383 in all 4 runs of `exact2.sh`, base and ours alternating; one
earlier run of ours, the first after the build on a busy GPU, gave 97d388c2bc5b0383: run-to-run noise of that config, not the patch). Strict test_sync 960x540: PASS 60/60 at t 0 and at t 1.
(The old `/e/dlss-runtime/v043/build` predates edit_temporal; do not use it as the v0.4.3 reference.)

## Recommendation

To get the video's look: the Showcase settings (in `E:\dlss-runtime\build_transfer\nr_port.ini`) and a larger model resolution
(1080p: HF x1.27-1.80 on faces vs x1.11-1.36 at 540p; the iq-guides table: a 540p edit on a small face is under half the 1080p one).
`transfer_strength` can stay at 0; 1 is available for A/B.

## Reproduce

`b.bat` (ROCm 6.4) and `b_base.bat` in `E:\dlss-runtime\hdr_transfer`; `exact.sh`, `exact2.sh`, `batch.sh` (each run via `run.sh` ->
gpu2.sh, 150 s cap), then `python -I analyse.py E:/dlss-runtime/hdr_transfer/out <png dirs>`. Sources: `club1_1920x1080.bin` (Cyberpunk club
frame 1 of `E:\dlss-runtime\iqg\in_1920x1080.npy`, faces) and `color_1920x1080.bin` (hand, hall, highlights).

Credit: transfer maths from RenoDX (Carlos Lopez Jr., MIT) via RenoDX-DLSS5-Generic.
