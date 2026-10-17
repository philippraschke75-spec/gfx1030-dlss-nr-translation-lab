# Host options: where each community setting runs (janblade OptiScaler-F5-DLSSNR v0.1.25)

Source read: `E:\dlss 5\janblade-src` (GPL-3, read-only). Main files: `OptiScaler/shaders/dlssnr/DlssNr_Dx12.cpp`
(D3D12 dispatch), `OptiScaler/shaders/dlssnr/precompile/dlssnr.hlsl` (encode / resolve / interpass shader),
`OptiScaler/dlssnr/forwarder/dlssnr_forwarder.cpp` (the NVIDIA path), `OptiScaler/dlssnr/PassProfiles.h`,
`OptiScaler/Config.h`.

## How our DLL is plugged in

The host loads `nvngx.dll_dlssnr.dll` as its "forwarder" and calls `dlssnr_call_create` / `dlssnr_call_evaluate_v2`
through function pointers (`DlssNr_Dx12.cpp:682`). On NVIDIA this file is janblade's forwarder, which hands the call to
NVIDIA's `nvngx_dlssnr.dll` snippet. With our port the file IS our DLL (`handover/game-ready-sync/dll/dlssnr_nr.cpp`).
The host's only branch on the port is `isPort` (it exports `dlssnr_backend_id`): it then skips creating the NGX
capability block (`DlssNr_Dx12.cpp:2563`). Everything else in the pipeline is identical:

```
game frame --encode (mode 0: /white point, proxy curve, sRGB; or passthrough)--> proxy
proxy --mode 2 area downsample to WorkingScale--> model input (workWidth x workHeight)
for pass in passes: evaluate(feature[pass], input, style/intensity/... of that pass) -> answer
                    if more passes: ClampProxy (mode 12, PassFeedback) -> next input
answer --SGSR1 or bilinear enlarge (WorkingScale < 1)--> resolve (Composed / Replace, Transfer, guard,
        colour strength, ReplaceDetail, skin protection) --> game frame
```

So every host-side step runs on our output exactly as on NVIDIA's. The difference is only what happens INSIDE
`evaluate`: NVIDIA's feature returns the network's answer computed at the size it is given; ours resamples the input to
the nearest model step of the data dir (640x360, 832x468, 960x540, 1280x720), runs there, and returns
`live input + gain * intensity * bilinear-up(answer - what the network saw)` (default `compose=add`).

## Option by option

| option (OptiScaler.ini `[DlssNr]`) | what it does | where it runs | reaches our DLL? |
| --- | --- | --- | --- |
| `Intensity` (0-2) | passed to the feature (create and every evaluate) | inside the feature. NVIDIA: post block, linear blend up to 1, above 1 the luminance ratio raised to the intensity, bounded 2x, hue kept (dlssnr-pytorch `compose()`). Ours: `gain * intensity * edit`, linear extrapolation of all three channels | yes, but **different maths above 1** (fixed by `compose=nv` / `compose=ratio`) |
| `Style` (0-2), `Pass2Style`, `Pass3Style` | model conditioning, per pass (`PassStyle`) | inside the feature; each pass has its own feature | yes: `setHost` -> `applyCond` -> `nr_set_conditioning` per feature |
| `LocalStructure`, `LocalTone`, `SkinStructure`, `AutoMask` (+ `Pass2*`, `Pass3*`) | model conditioning per pass (`PassTuning`) | inside the feature | yes (range 0..2; values outside are ignored, the previous value stays) |
| `Preset` / `Pass2Preset` / `Pass3Preset` | `DLSSNR.Hint.Render.Preset` at create | NVIDIA feature only | **no** (our create ignores the argument) |
| UI correction (`dlssnr_call_set_extras`: UI, UIAlpha, Backbuffer) | NVIDIA's interface correction inputs | NVIDIA feature only | **no** (our export is a stub) |
| `VitEvery`, `VitEveryPlain`, `KernelProfile`, `Precision` (NVFP4) | NvAPI kernel-launch hooks around NVIDIA's model (`DlssNrNative::BeginEvaluate`) | NVIDIA kernels only | **no** (ours: `vit_every` / `reuse` in `nr_port.ini`) |
| `Passes` (1-3, 30 unlocked) | one feature per pass, sequential, each with its own history | host | yes. **But our DLL scales every extra pass by `pass2_gain` (default 0.5)**, NVIDIA applies each pass in full |
| `PassFeedback` (0-1) | ClampProxy (mode 12) between passes: `next = proxy + f * (CubeScale(answer) - proxy)` | host, between our calls | yes |
| ClampProxy | the same mode 12: the answer cube-scaled back into [0,1] before it becomes the next input (hue-preserving, one scalar) | host | yes |
| `WorkingScale` | proxy area-downsampled (mode 2, in linear light) before the feature; the answer enlarged after (`ReducedUpscaleMethod`: SGSR1 default, bilinear) | host | yes. Ours then resamples AGAIN to its own step (e.g. 0.8 x 1080p = 1536x864 -> 960x540) |
| `Transfer` (0 classic, 1 matched residual = default, 2 OkLab residual) | how a below-size answer comes back: 1 = frame's own full-res proxy + cube-scaled edit, 2 = `NvidiaResidualModel` (log-luminance ratio + chroma difference in OkLab, bilinear from the answer grid, applied to the full-res proxy) | host resolve, **only when `WorkingScale` < 1** (`gModelWorkScale < 0.999`) | yes, applies to our answer the same way. At WorkingScale 1 (e.g. RunBeforeSR at render size) it is a no-op on both paths. Note: the v0.1.16 log prints "transfer classic" for Transfer 2 (it only names 1) |
| `TransferStrength` ("Detail strength") | Composed: blend towards the model, above 1 the luminance ratio to a power | host resolve | yes |
| `ColourStrength` | Composed: model colour vs game colour, above 1 OkLab chroma boost | host resolve | yes |
| `ReversibleMode` (0 knee, 1/2 Neutwo, 3/4 hybrid, 5 HLG, 6 PQ, 7 linear; 2 and 4 = Replace) | proxy curve in the encode; Composed (ratio composition, guard) or Replace (the answer IS the picture, decoded) | host encode + resolve | yes. On a tone-mapped frame (passthrough, colour transform off) Replace = the answer as is |
| `ReplaceDetailStrength` ("Restore Sharpness") | Replace modes only: multiplies the result by `(Y + s * (Y - blur5(Y)) + floor) / (Y + floor)` from the native frame, blur radius `round(1/WorkingScale)` | host resolve, **only Replace and WorkingScale < 1** | yes, same as NVIDIA. A no-op at WorkingScale 1 |
| `MaxRatio` ("guard", default 2) | Composed: caps brightening (one scalar from luminance); Replace: two-sided bound | host resolve | yes |
| paper white (`WhitePointSource` 0, `WhitePointScale`) / exposure (sources 1 game, 2 scan, 3 automatic; trims and anchors) | the white point the encode divides by before the curve, and the resolve multiplies back | host encode + resolve | yes |
| colour transform / `ColourEncoding` | passthrough on a tone-mapped frame (no white point, no curve), else the HDR encode | host | yes |
| `RunBeforeSR` / `FinishedPicture` / `DeferredDlss` | where in the frame NR runs (render size before SR, after SR, finished picture) | host | yes (decides the size we are handed) |
| `DetailReuse` (+ Steady, Fill, Depth...) | host runs the model every other frame, moves the saved answer-minus-input by the motion vectors on the others | host, around our calls | yes (on top of our own `nr_split` / `mv_reuse`; using both reuses twice) |
| `SkinProtection`, skin/environment detail and colour | attenuate the final edit by a skin-colour mask | host resolve | yes |
| `ApplyModel`, `HoldFrame`, `DebugView`, `Compare*` | A/B and debug views | host | yes |

## Community options that never reach our DLL

* `Preset` (and per-pass presets): NVIDIA's `DLSSNR.Hint.Render.Preset`. Our create ignores it.
* UI correction inputs (`dlssnr_call_set_extras`): stub.
* `VitEvery` / `VitEveryPlain` / `KernelProfile` / `Precision`: NvAPI hooks on NVIDIA's kernels. Our equivalents are
  `nr_port.ini` keys.

## Community options that reach us but act differently

* `Intensity` above 1: NVIDIA raises the luminance ratio to the intensity (bounded 2x, hue kept, clamped to [0,1]);
  we extrapolated the RGB edit linearly (chroma doubles too, no bound in LDR). Measured against the network run at
  full size with NVIDIA's rule, linear extrapolation is ~1 dB worse at intensity 2 (README.md). `compose=nv` ports
  NVIDIA's rule.
* `Passes` 2-3: every extra pass is scaled by our `pass2_gain` (0.5). NVIDIA runs each pass at full intensity.
  `pass2_gain=1` matches NVIDIA (the HDR bound for extra passes uses the same value).
* `WorkingScale` / any host size that is not one of our steps: we resample to the nearest step and carry the edit
  back bilinearly; NVIDIA computes at exactly that size. At high output sizes (WorkingScale 0.8 of 4K = 3072x1728)
  NVIDIA's network runs at 2.4x our 1280x720 in each axis. That is model resolution, not compose: the fine detail the
  full-size network synthesises is uncorrelated with the frame's own detail (corr 0.01-0.08, README.md), so no
  transfer of frame detail recovers it.
