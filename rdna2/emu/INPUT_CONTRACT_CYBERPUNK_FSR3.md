# Real upscaler inputs from Cyberpunk 2077 on RX 6900 XT (measured 2026-09-21, 2560x1440 output, FSR3 native path)

Source: `rdna2/diagnostics/2026-09-21/fsr3-pixels/` (raw planes + `upscaler_meta.txt` + log). Captured by the F9 one-shot in the hook (records copies into the
game's own command list before `ffxFsr3UpscalerContextDispatch`; verified by a byte-exact isolated GPU test first). One frame, camera nearly still.

## Proven by measurement (observed values, not assumptions)
* The game calls the native `ffxFsr3UpscalerContextDispatch` (~6,000 calls in the session); the FFX-API `ffxDispatch` is never called.
* Render 1707x960, output 2560x1440. Description resource entries every 0xB0 bytes from +0x08 (ptr, 32-byte description, state u32 at +0x28 = 12 for all):
  colour +0x08, depth +0xb8, motion vectors +0x168, (exposure +0x218 null), reactive-mask-like +0x2c8 (R8), further entries +0x378..+0x588, output +0x638 (2560x1440 RGBA16F).
* Colour: DXGI R16G16B16A16_FLOAT, linear HDR: all finite, R/G/B 0.0005..65, mean ~0.12-0.27, luma p1/p50/p99 = 0.012/0.150/0.974, alpha exactly 1.
* Depth: DXGI R32G8X24_TYPELESS, plane 0 read as R32_FLOAT: all finite, 0.0006..0.914, median 0.0017 (1,023,607 distinct values).
* Motion vectors: DXGI R16G16B16A16_FLOAT; 93.6% of pixels exactly zero (camera almost static), xy magnitude p99 = 0.0004; z/w channels also populated (z -0.017..0.001, w 0..1) meaning unknown.
* Scalars at the end of the description: jitter (+0x6e8) = (0.375, -0.0556) this frame (changes every frame: earlier (-0.25,-0.167), (0.25,0.389), (-0.375,0.056));
  motion-vector scale (+0x6f0) = (1707, 960); render size (+0x6f8) = (1707, 960).

## Read from the raw description with the SDK field order (plausible, NOT independently verified)
+0x700 = 1 (sharpening enabled?), +0x704 = 0.2 (sharpness), +0x708 = 25.9 (frame time delta, ms), +0x70c = 1.0 (pre-exposure), +0x710 = 0 (reset),
+0x714 = 16000.0, +0x718 = 0.02 (camera near/far in some order), +0x71c = 1.190977 (vertical FOV rad, ~68.2 deg), +0x720 = 1.0 (view-space-to-meters), +0x724 = 418 (flags).
Depth values are large near the camera and tiny far away, which suggests reversed-Z; this is an inference from the value distribution only.

## Implications
* The model input is linear HDR RGBA16F at RENDER resolution (1707x960), not the 10-bit backbuffer capture used before. Exposure is not supplied (auto-exposure path).
* Depth, motion vectors, jitter and pre-exposure exist and are now available as real data for an offline replay.
* Still unknown: which `k_import` format code matches RGBA16F, what host-side conversions (tonemap, exposure, padding to a multiple of 128) precede the network,
  how depth/motion vectors are packed for the network, and the meaning of the mv z/w channels.

## Capture cost measured for the F9 path (one shot, 1707x960)
recording into the game's list 13.8 ms CPU on the render thread (includes readback allocation), fence wait 0.02 ms, map+write of 32.8 MB 17.4 ms, next Present 19.8 ms after the F8 capture in the same session.
