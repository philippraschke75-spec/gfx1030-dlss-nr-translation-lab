# DLL side of temporal history (branch temporal-dll, 2026-10-11)
Build: E:\dlss-runtime\build_temporal_dll (dirs h0/h1/h2 = ini variants; run.sh = locked test runner). Data: combo nr_data 960x540 (release, import mode 0). No new data needed.

## ini
- `history=0|1|2`, default 0 (bit-exact, unchanged behaviour).
  - 1 = runtime hist_valid=1: pre-block +0x40 plus the post-block lerp.
  - 2 = hist_valid=2: post-block lerp only. Until the runtime implements 2, test it with env NR_HIST_MASK=2.
- `history_gate_lo/hi`: colour gate in network space, default 0.08 / 0.25.
- Needs nr_hist_layout, nr_ext_history(in,out) and nr_ext_enqueue2 (temporal-runtime). If any is missing, history stays off and the log says so.

## Design (sync path)
- **io layout:** two PAD_W x PAD_H RGB32F slots after the flag. hist_in is written by the DLL; hist_out (P_t) is written by the runtime.
- **hwp** (after imp, padded model size), per in-frame pixel:
  - Writes the current colour, or hist_out warped by the game's MVs, sampled bilinear. MV convention: previous = current + mv*mvScale, in frame px.
  - Padding pixels get 0.
  - Rejects: off screen, NaN, |mv| >= 0.25 UV.
  - Gates: colour (max|cur-hist|) and relative depth (0.05..0.2) against the depth kept from the previous frame.
- **hst** (after the spin): keeps the depth and st[12] = frame (0 when the spin timed out).
- **Validity:**
  - CPU side: no reset, and the previous frame was an NR frame. Then hist_valid = mode, else 0.
  - GPU side: st[12] == frame-1, else hwp writes the current colour.
- k_export +0x38 is untouched.
- **ponytail:** the colour space is assumed to be import mode 0 (raw, release data). Mode-1 data needs cb tmap=1 from a runtime query.

## Results (960x540 RGBA16F, 2% relative input noise)
**Correctness**
- history=0: strict test_sync PASS, max err 0. r11 PASS.
- e5 and UP=2 FAIL identically on the unmodified base build (pre-existing).

**Flicker**, F = mean|O_t - warp O_{t-1}| minus the same for the inputs (input term 0.0020):

| mode | static F | static brightness drift | moving (4,2 px/frame) abs dO |
|---|---|---|---|
| h0 | 0.01290 | 0.0100 | 0.0314 |
| h1 | 0.01370 | 0.0141 | ~0.0315 |
| h2 | 0.01266 | 0.0099 | 0.0299 |

**Ghosting** (object moving 6 px/frame), mean |P(O) - NR without history|:

| mode | background | object | uncovered |
|---|---|---|---|
| h2 | 0.0017 | 0.0060 | 0.0003 (no ghost) |
| h1 | 0.021 | 0.025 | 0.031 |

**Cost:** below the noise, < 0.5 ms (list p50 ~30.7 ms vs 31.0 ms).

**Conclusion**
- h1 (pre-block loop) is unstable and slightly worse than h0.
- h2 is stable with no ghosting, but it reduces flicker by only 2-5%.
- The network amplifies small input noise about 7x, and history does not fix that.

## Update 2: mode-1 data (temporal-runtime a0c1e77)
- **Colour space.** The history fill now uses the data's k_import space. The DLL reads the k_import kernarg (+0x28 mode, +0x2c scale) from manifest.txt and kernargs.bin.
  - Mode 0: raw colour.
  - Otherwise: max(0, scale*x), then x/(1+x) clamped, then the sRGB OETF.
  - The log says `history: k_import scale ...`.
- **Test regression.** history=0 strict test_sync still PASSes with max err 0.

### Flicker and ghosting
Same test as above (NR_TEST_HIST: 2% noise, exact MVs), m1 = E:\dlss-runtime\temporal\m1.

**Static frames, mean |O_t - warp O_{t-1}|** (input term 0.0020):
- h0 0.00404
- h1 0.00125, a 3.2x cut, below the input noise (drift 0.0046, settling)
- h2 0.00401

**Moving frames** (4,2 px per frame):
- h0 0.00657
- h1 0.00227
- h2 0.00652

**Moving object, mean |P(O) - NR without history|** (background / object / uncovered):
- h1: 0.0069 / 0.0119 / 0.0099. The uncovered region is only about 1.4x the background, so ghosting is mild.
- h2: about 0 everywhere. hist_valid=2 has no measurable effect.

### HARNESS metric
Frame path at 2560x1440, model 960x540, 60 frames, run with fl.sh:

| data | history | x | excess | white jumps | visible |
|---|---|---|---|---|---|
| m0 | 0 | 1.50 | +0.00130 | 0.72% | 3.9% |
| m0 | 1 | 1.11 | +0.00029 | 0.22% | 1.1% |
| m0 | 2 | 1.49 | | 0.70% | |
| m1 | 0 | 1.80 | +0.00209 | 2.71% | 10.7% |
| m1 | 1 | 1.31 | +0.00080 | 1.26% | 5.1% |
| m1 | 2 | 1.79 | | 2.67% | |

- Network alone, m1 history off: x1.64, white jumps 4.08%. The network-alone path uses nr_run, so it has no history.
- On this sequence, m0 with history=1 is the best combination. m1 is worse than m0 both with and without history.

### Single frame, m0 vs m1 (history off)
- Files: m0.png, m1.png, diff.png (|m1-m0| x8); 2560x1440 versions are in build_temporal_dll\png.
- No visible dark blocks in m1.
- Crude counter (out luma < in luma - 0.1; a dark block is a 16 px block with more than 50% such pixels):

| data | dark px at 960x540 | dark blocks at 960x540 | dark blocks at 2560x1440 |
|---|---|---|---|
| m0 | 1.35% | 14 | 144 |
| m1 | 3.09% | 51 | 444 |

- mean |m1-m0| is 0.021, max 11.5 (a few hot pixels from the mode-1 inverse at the top of the range).
- Script: game-ready-sync/temporal/render_modes.py.

### Game package
- Location: E:\dlss-runtime\build_temporal_dll\game_pkg, containing nr_runtime.dll, nvngx.dll_dlssnr.dll, nr_data\960x540 (m1) and nr_port.ini.
- The ini sets history=1 and ldr_input=1, with model_res=960x540.
- The vendor nvngx_dlssnr.dll must sit next to nr_data, as before.
- Alternative per the harness: point data_dir at the combo (m0) data with history=1.
