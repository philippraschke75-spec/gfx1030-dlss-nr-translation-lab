# Model resolution 540p / 720p / 1080p / 1440p (v0.4.2-hires, 2026-10-17)

One capture (`color_1920x1080.bin` bilinear-upscaled to 2560x1440), new model (310.9.16 `WEIGHTS_CC_SMALL_FP8`), vit_every=1,
history on (8 static frames), composited to 2560x1440 as the DLL does (`render_hires.py`). PNGs are gamma 2.2. Crops: 256x256
at 2x, left to right: native | 960x540 | 1280x720 | 1920x1080 | 2560x1440.

| model res | network ms (RX 6900 XT) | runtime VRAM | mean abs edit (linear / display) | HF energy vs native | HF energy of the edit |
| --- | --- | --- | --- | --- | --- |
| 960x540 | 9.68 | 335 MB | 0.0204 / 0.0285 | 0.95x | 2.5e-05 |
| 1280x720 | 14.93 | 494 MB | 0.0225 / 0.0288 | 1.11x | 4.1e-05 |
| 1920x1080 | 29.52 | 856 MB | 0.0185 / 0.0247 | 1.11x | 4.9e-05 |
| 2560x1440 | 51.21 | 1424 MB | 0.0204 / 0.0253 | 1.21x | 1.6e-04 |

Old model 1920x1080: 54.67 ms, 892 MB. HF energy = mean squared Laplacian of the display luma.

**Fixed: frame = model size.** The DLL used to feed the frame unclamped to the network when the frame size equals the
model size (LDR) and showed the raw network output: pixels above 1 (1.5 % of this capture, up to 38) made blocky cyan/green
and magenta patches (also seen at 1280x720 frame = model). Now that path clamps the input to [0,1] as the resampled path
does and shows frame + (NR - clamped input), so highlights keep their excess (async fallback: same clamp). Before the fix
the 1440p row read 0.0258 / 0.0224, 1.52x, 6.7e-04.

1080p geometry: build_plan now pads 1080 rows to 1088 (NVIDIA's `Geometry::fromValid` = dlssnr-pytorch-flex `plan.padding`,
checked on 17936 sizes); it was 1152. vs PyTorch at 1920x1080: 41.01 -> 41.42 dB. The other sizes keep their padding (960x540 / 640x360 rebuilt
byte-identical).
