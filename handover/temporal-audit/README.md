# Temporal audit (branch temporal-audit, from v0.4.3-rc)

Question: does our history path make the NR edit weaker than NVIDIA's temporal DLSS-NR?
Answer: yes, by about 10-16 %. The causes were our history colour gate and the bilinear history warp. Both are fixed; the defaults now follow NVIDIA.

## NVIDIA's temporal inputs per frame
Source: OpenDLSS-NR `docs/frame.md`, `demo/nr_pass.cpp` and the demo shaders. It is MIT and reproduces NVIDIA's frame pipeline.
- **History source:** the previous frame's *blended* network output, `lerp(clamp(proxy + rgb/4), history, w)`. It is taken in proxy code space, before style and intensity, and stored truncated to f16.
- **Reprojection:** by motion vectors (current to previous), with a **5-tap Catmull-Rom**. NVIDIA notes that bilinear softens the image frame over frame, because the history feeds the network's input.
- **Rejection:** only an "is there a history" flag, for pixels that come from off screen. There is **no colour gate and no depth gate**; disocclusion is left to the network's per-pixel blend logit. Where there is no history, lanes 7-9 get the current proxy and the blend weight is 0.
- **Lanes 7-9:** centred(history), with the same encoding as lanes 4-6. Padding is a reflected copy.
- **Post blend:** `w = clamp(sigmoid(head.a) * blend_scale, 0, 1)`, with blend_scale 0.767 for the new model.
- **Seed:** the frame counter (`seed = frames`). It is 0 with no history or after a reset (flex: "the library counts frames").
- **Reset:** the host's `DLSSNR.Reset`, the first frame, or a resize. History is off for that frame.

## Ours (v0.4.3) vs NVIDIA

| | NVIDIA | v0.4.3 | now (defaults) |
|---|---|---|---|
| colour gate | none | `1 - smoothstep(0.08, 0.25, max\|hist - cur\|)` towards the *unedited input* | off (`history_gate=1` restores it) |
| warp | Catmull-Rom 5-tap | bilinear | Catmull-Rom (`history_filter=bilinear` restores it) |
| history padding | reflected | 0 | reflected |
| seed | frame counter, 0 without history | always 0 | frame counter (`history_seed=0` restores it); runtime `nr_set_seed_mode` |
| off-screen / depth reject | weight 0 | current input, lerp still applied | unchanged: edge strip only; depth gate kept |
| f16 truncation of history | yes | f32 | f32 (no measurable effect) |
| nr_split | - | history is N frames old (job to job, acc-warped) | unchanged (inherent to split) |
| edit_temporal | - | extra EMA plus 3x3 clamp of the shown edit | unchanged, costs about 3 % of the edit |

The v0.4.3 colour gate compared the previous *output* (edit included) with the input without the edit. Wherever NR changed a pixel by more than 0.08, it threw history away, and the post lerp then pulled the output back toward the unedited input. On top of that, the network rejected our blurred and gated history: its mean blend weight was 0.2-0.3, against 0.55 with NVIDIA's rules.

## PyTorch reference (flex, new model, CPU)
- Sequence: 512x288 crop at 540p scale, 32 frames, pan 1.3 / 0.45 px per frame, jitter, 1 % noise.
- Files: `temporal_ref.py`, `analyse.py`, crop `pan_crop.png`.
- `pan_crop.png` columns: input, single, nv, nv_s0, ours, fix, fix_seed. Bottom row: edit x8.

Means over frames 8-31:

| mode | edit mean\|.\| | edit HF rms | corr with single-frame edit | PSNR vs NVIDIA rules | added instability |
|---|---|---|---|---|---|
| single (no history) | 0.0112 | 0.0056 | 1 | 40.9 dB | +0.0014 |
| **nv (NVIDIA rules)** | **0.0117** | **0.0064** | 0.90 | - | -0.0003 |
| nv, seed 0 | 0.0114 | 0.0063 | 0.90 | 49.9 dB | -0.0003 |
| **ours v0.4.3** | **0.0098 (-16 %)** | **0.0051 (-20 %)** | 0.94 | 44.1 dB | +0.0004 |
| fix (seed 0) | 0.0116 | 0.0064 | 0.90 | 50.3 dB | -0.0003 |
| fix + seed | 0.0117 | 0.0065 | 0.90 | 52.0 dB | -0.0003 |

What this shows about NVIDIA's temporal path:
- It makes the edit **slightly stronger** (+4 %) and its fine detail stronger (+14 % edit HF) than single-frame.
- It adds no instability: the output is about as stable as the input.
- It does **not** keep building the effect up over time. The edit is flat from frame 2 on.
- So temporal is not where the large gap to real DLSS 5 lies. Our old path was 16 % below single-frame, which made it about 20 % below NVIDIA.

## Our port (GPU)
`gpu_all.sh` / `gpu_rel.sh`: test_sync, frame = model 960x540, same pan, 32 frames. Base = the v0.4.3 build.

Network alone (nr_split=0, edit_temporal=0, vit_every=1):

| | EDIT mean\|out-in\| | HF out/in | steady instability vs input |
|---|---|---|---|
| history=0 | 0.01574 | x1.188 | x1.28 |
| v0.4.3 history=1 | 0.01462 (-7 % vs no history) | x1.032 | x1.01 |
| **new history=1** | **0.01609 (+10 % vs v0.4.3)** | x0.996 | **x0.90** |
| new + history_gate=1 | 0.01597 | x0.969 | x0.87 |
| new + history_filter=bilinear | 0.01521 | x1.041 | x1.01 |
| new + history_seed=0 | 0.01634 | x0.990 | x0.89 |

Release Standard preset (nr_split=3, edit_temporal=0.7, reuse=bottleneck):
- On this pan: EDIT 0.01444 -> **0.01559 (+8 %)**, visible flicker 3.33 % -> 0.58 %.
- With edit_temporal=0: 0.01610.

Re-checks:
- **Flicker harness** (2560x1440, Standard), steady visible flicker: 0.827 % -> **0.107 %**. White jumps: 0.199 % -> 0.013 %.
- **Ghost test, N=3** (|out - NR of the same frame|):

  | case | v0.4.3 | new |
  |---|---|---|
  | moving image | 0.00241 | 0.00237 |
  | moving object, whole frame | 0.00214 | 0.00229 |
  | moving object, object region | 0.00660 | 0.00644 |

  No new ghosting.
- On the GPU, the seed counter changes almost nothing: +-1.5 %, within noise. In PyTorch it moves PSNR vs NVIDIA from 50.3 to 52.0 dB. It stays on because it is NVIDIA's behaviour.

## Not done
- No per-pixel weight 0 where there is no history. That needs a post-kernel change, and it only affects the entering edge strip.
- No f16 truncation.
- The depth gate is kept: NVIDIA has none, but ours fires only on depth discontinuities.
- `edit_temporal` still trims about 3 % of the edit. Turning it off is a flicker trade-off.
