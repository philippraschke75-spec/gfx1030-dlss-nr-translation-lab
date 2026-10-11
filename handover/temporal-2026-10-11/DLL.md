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
