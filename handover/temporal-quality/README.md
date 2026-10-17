# temporal-quality (branch from v0.4.1-rc + nr-splitn)

New ini keys in `dlssnr_nr.cpp` (kSyncHlsl: new passes `tea` and `teb`, plus a branch in `exp`):

- `edit_temporal=0..0.95` (default 0 = off, bit-exact; recommended 0.7). This stabilises the edit on the output side.
  - `tea` computes, at each model texel, the edit that exp would show:
    - nr_split: the last job's edit, warped by the vectors, the acc chain and the anchor, then colour- and depth-gated;
    - full NR: out - in.
  - `teb` blends that edit with the previous frame's stabilised edit S. S is warped by this frame's vectors, depth-rejected on disocclusion, and clamped to the 3x3 min/max of the current edit (TAA-style).
  - `exp` then shows S, upsampled bilinearly from model size. It no longer applies its own per-frame-pixel colour gate.
  - Works in nr_split mode and in the resampled full-NR path. nr_every, fg and mv_reuse paths are untouched.
- `edit_limit=k` (default 0 = off) caps |edit| at k x (3x3 luma range + 0.02). It is measured below and **not recommended**.
- `edit_warp=bilinear|bicubic` (default bilinear): Catmull-Rom fetches of the job edit, the history and S. Sharper, but flickers more.
- Cost: with stabilisation on, `exp`, `tea` and `teb` are compiled without `globallycoherent` (macro NC). Every read in them sits behind a UAV barrier; only `spin` polls. With coherent views the three passes cost +0.33 ms, with plain views about 0.
- test_sync additions:
  - `EDIT` line: effect strength, mean |out - in|;
  - `FLICKER steady(k>=8)`: leaves out the first edit's fade-in, which is about half of the old all-frames number at N=3;
  - `NR_TEST_PERFRAME`: per-frame visible %;
  - `NR_TEST_TS`: GPU timestamps around evaluate;
  - `NR_TEST_DUMP` in ghost mode.

## Results
Setup: test_sync, new model (`v041/nr_data`, ca6f64f1dd88b965), release ini (history=1, vit_every=3, reuse=bottleneck), NVIDIA default conditioning.
- Flicker: 2560x1440 frame, 960x540 model, 60 frames.
- Ghost: 960x540, 13 px/frame, pace 8/12 ms.

| N=3 | steady visible flicker | all-frames visible | white jumps (steady) | EDIT mean abs | HF out/in |
|---|---|---|---|---|---|
| base | 0.827 % | 1.706 % | 0.199 % | 0.01471 | x1.013 |
| edit_temporal=0.7 (final) | **0.075 %** | 0.85 % | 0.043 % | 0.01448 (-1.6 %) | x1.004 |
| edit_temporal=0.5 | 0.156 % | 0.97 % | 0.062 % | 0.01496 | x1.005 |
| edit_temporal=0.85 | 0.073 % | 0.78 % | 0.033 % | 0.01469 | x1.003 |
| compose without temporal (gate per model texel only) | 0.530 % | 1.45 % | 0.134 % | 0.01502 | x1.006 |
| old "delta" compose 0.5 / 0.75 (frame-res gate kept, dropped) | 0.496 / 0.416 % | | | 0.0146 | x1.012 |
| 0.5 + edit_warp=bicubic | 0.257 % | | | 0.01507 | x1.008 |
| 0.5 + edit_limit=1 / 2 / 4 | 0.166 / 0.219 / 0.157 % | | 0.024 / 0.048 / 0.061 % | 0.01361 / 0.01476 / 0.01496 | x1.005 |
| static camera (jitter + noise), base -> 0.7 | 0.752 -> 0.184 % | | | | |

| other N / presets (steady visible, EDIT) | base | edit_temporal=0.7 |
|---|---|---|
| N=0 (no split) | 0.951 %, 0.01569 | 0.057 %, 0.01559 |
| N=2 | (all-frames 1.83 %) | 0.050 %, 0.01537 |
| N=4 | (all-frames 1.76 %) | 0.110 %, 0.01501 |
| Strong (local_tone=2, intensity=1.3), N=3 | 1.565 %, 0.02053, HF x1.028 | 0.200 %, 0.02029, HF x1.009 |
| Cinematic (style=2), N=3 | 2.186 %, 0.04031, HF x1.035 | 0.424 %, 0.04001, HF x0.999 |

Ghost test: mean |out - NR(same frame)|. Lower is better; "live" = no edit = 0.00608.

| | N=2 | N=3 | N=4 | moving object N=3: all / object region | moving object N=4: all / object region |
|---|---|---|---|---|---|
| base | 0.00208 | 0.00236 | 0.00265 | 0.00209 / 0.00654 | 0.00231 / 0.00689 |
| edit_temporal=0.7 | 0.00208 | 0.00236 | 0.00268 | 0.00196 / 0.00526 | 0.00215 / 0.00565 |
| edit_temporal=0.85 | | 0.00246 | | 0.00202 / 0.00500 | |

Cost:
- GPU time of the evaluate call (timestamps, flicker sequence, N=3, `gpu2.sh --exclusive`, mean of the faster half of frames >= 8, 3 alternating rounds): off 3.21 / 3.19 / 3.12 ms, on 3.18 / 3.14 / 3.13 ms. That is about -0.02 ms: the direct `exp` reads less than the old one.
- The first version with coherent views cost +0.33 ms (tea 0.15, teb 0.10, exp +0.08).

Bit-exactness: with everything off, OUTHASH is identical to the base merge build: N=3 866c2f5b6bd24383, N=0 40f89a7223c0e383.

Reading:
- In steady state, most of the base flicker had two sources.
  - The job switch: every N frames the edit jumps (1.5-3 % visible on those frames, 0.1-0.3 % between them).
  - The per-frame-pixel colour gate in `exp`, which turns the 1 % input noise into gate noise. Doing the gate per model texel alone halves the flicker.
- The temporal blend removes most of the rest.
- The effect strength stays within -2 %.
- High-frequency energy of the output falls by about 0.9 % of the total (HF out/in x1.013 -> x1.004). Most of the lost HF was the gate speckle at edges; see the edit rows of the crops. For a sharper edit, `edit_warp=bicubic` trades some of the stability back.
- `edit_limit` cuts white jumps but either weakens the effect (k=1: -9 %) or adds visible flicker (k=2), so it stays off.

## Item 2: noise seed
- The pre-block's seed (`PreParams +0x24`, `pre_native.hip`) is baked into kernargs.bin as 0 by build_plan. nr_runtime never writes it.
- So our port already holds the seed constant every frame. That matches NVIDIA's behaviour without history (danielblnc's log). It is also the temporally coherent choice: a per-frame seed only adds fresh noise detail each frame.
- No change made.
- Not done: noise that moves with the content. The hash uses the padded screen coordinate, so generated detail is screen-fixed under a pan. That would need a runtime change: offsetting the hash coordinate by the camera motion.

## Item 4: ghosting
- With 0.7, ghosting matches the base at N=2, 3 and 4 (N=4 +1 %). The moving-object region improves by 18-20 %: the 1-frame-warped history plus the clamp replaces part of the 2-6 frames old chained edit.
- Bicubic warp makes no difference on the integer-motion ghost test and adds flicker on sub-pixel pans.
- MV dilation (closest depth) was not added. The harness passes exact per-pixel vectors, so its benefit cannot be measured here. Add it when NR runs after an upscaler with render-res vectors.
- Disocclusion: S is rejected by the relative depth gate (0.05-0.2, as the existing gates).

## Crops
- Columns: input, base, edit_temporal=0.7. Bottom row: edit x8. Gamma 2.2.
- `crop_pan_mural.png` / `crop_pan_light.png`: pan sequence, last frame, detailed mural / light blotch.
- `crop_static_mural.png`: static camera.
- `crop_object.png`: 960x540 moving object (ghost test, N=3).
- Script: `crops.py`.
- The crops come from the build before the last gate change (live colour colAt instead of the frame pixel; steady flicker 0.098 % instead of 0.075 %).

Credits: TAA-style neighbourhood clamp, plus ideas from AMDNR ("Residual temporal", "Still-surface steadiness"; GPL-3) and matiasLombo/neural-upstream (delta reprojection with depth rejection, MIT). No code copied.
