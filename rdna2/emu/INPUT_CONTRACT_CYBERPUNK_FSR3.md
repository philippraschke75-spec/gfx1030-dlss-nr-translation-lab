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

## k_import format code resolved (2026-09-22): format=0 is RGBA16F

Read from the gfx1100 disassembly (`analysis/gfx1100-disassembly.txt`, function `_Z8k_import12ImportParams` at 0xA9D00): the branch
`s_cmp_lt_i32 s5,3 / cbranch 0xAA11C` (format group 0/1/2) eventually reaches 0xAA440-0xAA488 for format==0, which computes
`addr = pixel_index*8` (8 B/pixel), issues `global_load_b32` (packed R,G as two f16 halves of one dword) + `global_load_u16 offset:4`
(B channel, alpha dropped), then three `v_cvt_f32_f16` conversions. This is an RGBA16F decode (RGB kept, alpha discarded).

Verified in the emulator (`verify_import_fmt0.py`): six representative half-float values (0.125, 2.5, 65.0, 0.0009, 1.0, -3.0) round-trip
through format=0/mode=0 exactly (`np.allclose`, both pixels matched).

Verified on the RX 6900 XT with REAL captured Cyberpunk pixels (`difftest_import_real.py`, sandbox `rdna2/build/import-real-20260922`):
5 tiles from different regions of the captured frame (corners, centre, far background), format=0, mode=0 (raw decode) - **all 5 byte-exact,
0 mismatches**, output floats finite and in a plausible tonemapped-HDR range (0.03..0.70). Modes 1 and 3 on the same real tile reproduce the
same ULP-class noise seen earlier on synthetic data (max 3 ULP, 101/9216 float elements differing) - not a new bug, the known
transcendental-approximation difference between the emulator and GPU hardware.

This resolves one of the two "still unknown" items from the previous entry: the source format code for the real FSR3 colour resource is 0.
Depth (`R32G8X24_TYPELESS`) and motion-vector (`RGBA16F`) format codes are not yet determined the same way.
