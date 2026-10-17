# Model resolution 540p / 720p / 1080p / 1440p (v0.4.2-hires, 2026-10-17)

One capture (`color_1920x1080.bin` bilinear-upscaled to 2560x1440), new model (310.9.16 `WEIGHTS_CC_SMALL_FP8`), vit_every=1,
history on (8 static frames), composited to 2560x1440 as the DLL does (`render_hires.py`). PNGs are gamma 2.2. Crops: 256x256
at 2x, left to right: native | 960x540 | 1280x720 | 1920x1080 | 2560x1440 | 2560x1440 with the input clamped to [0,1].

| model res | network ms (RX 6900 XT) | runtime VRAM | mean abs edit (linear / display) | HF energy vs native | HF energy of the edit |
| --- | --- | --- | --- | --- | --- |
| 960x540 | 9.68 | 335 MB | 0.0204 / 0.0285 | 0.95x | 2.5e-05 |
| 1280x720 | 14.93 | 494 MB | 0.0225 / 0.0288 | 1.11x | 4.1e-05 |
| 1920x1080 | 29.52 | 856 MB | 0.0185 / 0.0247 | 1.11x | 4.9e-05 |
| 2560x1440 | 51.21 | 1424 MB | 0.0258 / 0.0224 | 1.52x | 6.7e-04 |
| 2560x1440, input clamped | | | 0.0284 / 0.0255 | 1.19x | 1.6e-04 |

Old model 1920x1080: 54.67 ms, 892 MB. HF energy = mean squared Laplacian of the display luma.

**1440p artifact.** When the frame size equals the model size, the DLL feeds the frame to the network unclamped (LDR, HDR
not latched) and shows the raw network output. Pixels above 1 (1.5 % of this capture, up to 38) then cause blocky cyan/green
and magenta patches around highlights (`crop_*`, 5th tile); with the input clamped they are gone (6th tile). The resampled
path (every model size below the frame size) clamps already. Fix (not done here): in the 1:1 path, clamp the input as `ld()`
does and output `c + gain * (e - saturate(c))` where the frame exceeds 1.

1080p geometry: build_plan now pads 1080 rows to 1088 (NVIDIA's `Geometry::fromValid` = dlssnr-pytorch-flex `plan.padding`,
checked on 17936 sizes); it was 1152. vs PyTorch at 1920x1080: 41.01 -> 41.42 dB. 540p / 720p / 832x468 / 640x360 data
are byte-identical.
