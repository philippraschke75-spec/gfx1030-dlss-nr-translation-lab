# Community look: NVIDIA path vs ours, and new compose modes (branch `community-look`, 2026-10-17)

Question: community users on NVIDIA run strong host settings (e.g. an RE Requiem log: Intensity 2, Style 1,
LocalStructure 2, LocalTone 2, ReplaceDetailStrength 1.5, Transfer 2, WorkingScale 0.8, ReversibleMode 2, RunBeforeSR,
2-3 passes with per-pass styles) and get a near photoreal look. Why does our AMD port look weaker, and what can the
compose in our DLL do about it?

Files here:
* `host-options.md`: every community option, where it runs in janblade's host, and whether it reaches our DLL.
* `compose_proto.py`: offline (numpy) prototype and scoring of the compose modes against the network run at full size.
* `run.sh`: one guarded test_sync flicker-mode run (`gpu2.sh`, 150 s cap) with extra `nr_port.ini` lines.
* `renders.sh`, `make_png.py`: the still renders and PNG panels below.

## 1. What differs between the NVIDIA path and ours

Our DLL replaces janblade's forwarder (`nvngx.dll_dlssnr.dll`), so the host's encode, WorkingScale downsample,
ClampProxy/PassFeedback between passes, enlarge, Transfer, Replace/ReplaceDetail, guard, colour strength, paper white
and exposure all run on our answer exactly as on NVIDIA's (details in `host-options.md`). The host's "replace detail"
and "transfer" do NOT assume a full NVIDIA output: they take whatever the feature returns at the size it was asked for,
and both are no-ops unless WorkingScale < 1 (at RunBeforeSR with WorkingScale 1, as in the RE log after its first
minute, the host passes the answer through unchanged in Replace mode on a tone-mapped frame).

What never reaches us: `Preset` (NVIDIA's render preset hint), the UI-correction inputs (`set_extras`), and the NvAPI
kernel options (`VitEvery`, `KernelProfile`, `Precision`).

What reaches us but acts differently inside our DLL:

1. **Intensity above 1.** NVIDIA's post block blends towards the answer up to 1 and above 1 raises the luminance ratio
   to the intensity (bounded 2x either way, hue kept, clamped to [0,1]) (dlssnr-pytorch `layers/pre_post.py:compose`).
   We extrapolated the RGB edit linearly, so at 2 the chroma edit doubled too and nothing was bounded.
2. **Extra passes.** Our `pass2_gain` (default 0.5) halves every pass after the first; NVIDIA applies each in full.
   A 2-3 pass community preset is therefore much weaker with us.
3. **Model size.** NVIDIA's network runs at exactly the size the host hands it; we resample to the nearest data step
   (max 1280x720) and carry the edit back bilinearly. This is the part compose cannot fix (section 3), and it is the
   iq-guides agent's model-resolution topic.

## 2. New compose modes (`nr_port.ini`, sync path)

```
compose=add     ; default, today's code, bit-exact (OUTHASH bd1112e0d83a8383 resampled, ece991ef072bd383 full size, old == new)
compose=nv      ; NVIDIA's intensity rule (above): blend up to 1, luminance ratio ^ intensity past it, bounded 2x, hue kept
compose=ratio   ; OkLab matched residual (port of janblade's Transfer 2 / NvidiaResidualModel, GPL-3, attributed in the
                ; source): log-luminance ratio x intensity (bounded 2x) and chroma difference x min(intensity, 1),
                ; applied to the live frame's own OkLab -- multiplicative, so the frame's own detail scales with the edit
detail=1        ; adds (detail - 1) x the live frame's detail above what the network saw (live - model input).
                ; "full replace + detail transfer" = up(answer) + detail x (live - up(input)), which at detail 1 IS add;
                ; so compose=add + detail=s is task (b). janblade's ReplaceDetail idea, measured against the model raster.
```

Implementation: `ced()` in the sync shader of `handover/game-ready-sync/dll/dlssnr_nr.cpp`, used by the resampled path,
the full-size path, `nr_split` and `mv_reuse` (the `fg_mode` reuse path keeps add). Two more root constants (40).
The modes act on `gain x intensity x pass2_gain` as the intensity. HDR (`hdr_mode`) goes through the same `cmp()` bound.

Also measured and dropped: a local-linear (guided filter) luma transfer from dlssnr-pytorch `scripts/m4_upsample.py`:
-2 dB and half the target's fine detail (it replaces flat-window frame detail with the window mean).

## 3. Measurements

### Against the network at full size (static, offline, `compose_proto.py`)

Network run at 640x360 and composed onto a 1280x720 frame, scored against the same network run at 1280x720 with
NVIDIA's intensity rule (what NVIDIA computes when the host hands it 1280x720). Capture A (`color_1920x1080.bin`).

| conditioning, intensity | add | nv | ratio | guided | HF of ours / HF of full-size target |
| --- | --- | --- | --- | --- | --- |
| default, 1 | 35.44 dB | 35.44 | 35.43 | 33.58 | 0.85 |
| community (style 1, structure 2, tone 2), 1 | 35.59 | 35.59 | 35.61 | 33.65 | 0.83 |
| community, 2 | 28.70 | **29.74** | **29.81** | 27.00 | 0.71 |

* At intensity 2 the NVIDIA rule is +1.0-1.1 dB closer to what NVIDIA computes than our old linear extrapolation.
* The full-size network adds fine detail (HF 1.21x the frame at intensity 1, 1.52x at 2); composed from half size we
  reach only 0.71-0.85 of it. The missing part is uncorrelated with the frame's own detail (correlation 0.01-0.08), so
  it is synthesis, not a contrast boost: `detail` > 1 matches the HF energy (detail ~1.25-1.5) but costs 0.5-1.5 dB.
  The only real fix is a larger model size.

### Game-like sequence (test_sync flicker mode, 1920x1080 frame, 960x540 model, release nr_port.ini, 16 frames)

Release ini: `history=1, nr_split=3, vit_every=0, reuse=bottleneck, int8_swin=1, max_inflight=3`. Community =
`intensity=2, style=1, local_structure=2, local_tone=2`. Pan 0.37/0.13 px, jitter, 1 % noise (HARNESS.md).

| config | mean luma edit | HF (out/in) | flicker (inst out/in) | visible flicker |
| --- | --- | --- | --- | --- |
| today, default settings | +0.0024 | x1.019 | x1.41 | 5.3 % |
| today (add) + community | +0.0054 | x1.208 | x2.37 | 15.1 % |
| nv + community | +0.0080 | x1.207 | x2.24 | 15.5 % |
| ratio + community | +0.0075 | x1.247 | x2.27 | 15.8 % |
| nv + community + detail 1.5 | +0.0080 | x1.559 | x2.60 | 17.2 % |
| ratio + community + detail 1.5 | +0.0075 | x1.602 | x2.64 | 17.9 % |
| 2 passes, add, pass2_gain 0.5 (today) | +0.0102 | x1.304 | x2.99 | 20.0 % |
| 2 passes, nv, pass2_gain 1 (NVIDIA-like) | +0.0187 | x1.447 | x3.60 | 26.1 % |

* The community preset itself (intensity 2, tone/structure 2) is what makes the look strong -- and it makes the edit
  flicker 2-3x more on our port. On NVIDIA the full-size network and NVIDIA's own temporal path hide much of that;
  that is the temporal-quality agent's topic.
* nv/ratio are a little steadier than add at intensity 2 (x2.24/2.27 vs x2.37) and slightly brighter on average.
* Cost: the compose is per-pixel ALU in the existing output pass (no extra pass, no extra memory). test_sync timing mode
  (960x540 model, 1920x1080 frame, 60 frames, shared GPU): add p50 14.12 ms, ratio p50 13.64 ms -- below the noise.
* `pass2_gain=1` almost doubles the 2-pass edit -- this is the single biggest strength difference to NVIDIA for
  multi-pass presets.

### Stills (static camera, release ini, `renders.sh` + `make_png.py`)

* `A_full.png` (half size), `A_crop.png`: capture A, 1920x1080 frame, 960x540 model. Panels: input | today, default
  settings | today + community preset (old linear intensity) | compose=nv + community | compose=ratio + community.
  Old linear intensity 2 pushes the floor and walls to over-saturated cyan and crushes the darks; nv/ratio keep the
  game's hues with the same local-contrast gain.

  | panel | mean abs luma edit | mean luma edit | HF (Laplacian) vs input |
  | --- | --- | --- | --- |
  | today, default | 0.0179 | +0.0075 | x1.040 |
  | today + community | 0.0361 | +0.0110 | x1.190 |
  | nv + community | 0.0373 | +0.0137 | x1.193 |
  | ratio + community | 0.0372 | +0.0132 | x1.249 |

* `B_full.png` (half size), `B_crop.png`: capture B (Cyberpunk club, 1280x720 render-size colour, i.e. RunBeforeSR),
  clipped to [0,1]. Panels: input | today + community (add) | nv + community (both 960x540 model) | the same network
  run at the frame's own size (1280x720 step) with NVIDIA's rule = what NVIDIA computes when handed 1280x720.

  | panel | mean abs luma edit | HF vs input | PSNR vs full-size network |
  | --- | --- | --- | --- |
  | input | - | x1.000 | 27.12 dB |
  | today + community (add) | 0.0218 | x1.134 | 28.85 dB |
  | nv + community | 0.0218 | x1.134 | 29.43 dB |
  | full-size network, NVIDIA rule | 0.0211 | **x1.590** | - |

  The edit is as strong as NVIDIA's (0.022 vs 0.021): our port is not "weaker". What the full-size network has and we
  lack is fine detail (HF x1.59 vs x1.13): skin, fabric, the panel grids. That comes from the model size, not the
  compose.

Side finding: when the frame size equals the model step (no resampling, e.g. RunBeforeSR at 1280x720 with
`model_res=1280x720`), `imp` passes the colour to the network unclipped. An LDR frame with values above 1 (2.6 % of the
club capture) then gets green/magenta speckle along the bright strips (seen in an unclipped run of capture B). The
resampled path saturates (`ld()`); the full-size path should too. Not changed here.

## 4. Recommendation

For users copying NVIDIA community presets: `compose=nv` (NVIDIA's own intensity rule; `ratio` is equivalent within
0.1 dB and carries slightly more frame detail) and `pass2_gain=1` when they run 2-3 passes. Leave `detail=1` unless the
host is not already restoring detail (janblade's ReplaceDetailStrength does the same thing host-side in Replace mode
below WorkingScale 1). Defaults are unchanged (`compose=add`, `pass2_gain=0.5`) until a game test.
