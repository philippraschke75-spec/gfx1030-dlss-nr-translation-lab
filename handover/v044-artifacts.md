# v0.4.4: high model resolution artefacts (FO4, 1080p / 1440p, style 2 / tone 2 / structure 2 / intensity 2)

Raw results are in `release-v0.4.4/results/`, and the scripts are in `release-v0.4.4/scripts/`.

## Verified (offline: test_sync, torch_ref on the RX 6900 XT, new model)

### Network vs the reference
`tools/torch_ref.py` now takes `COND=style,tone,structure,skin,auto` and `INT`. With INT > 1, our output also goes through a numpy copy of the DLL's `boost()`. The reference is dlssnr-pytorch-flex, run with the `dlssnr-pytorch/.venv` python.

| model | sliders | ours vs PyTorch | edit corr | mean \|edit\| ours / ref |
|---|---|---|---|---|
| 540 | default | 39.95 dB | 0.936 | 0.0161 / 0.0151 |
| 540 | 2,2,2 | 40.18 dB | 0.980 | 0.0328 / 0.0347 |
| 540 | 2,2,2 + int 2 | 37.23 dB | 0.988 | 0.0565 / 0.0601 |
| 1080 | default | 41.42 dB | 0.963 | 0.0171 / 0.0169 |
| 1080 | 2,2,2 | 40.37 dB | 0.984 | 0.0341 / 0.0367 |
| 1080 | 2,2,2 + int 2 | 37.51 dB | 0.989 | 0.0573 / 0.0613 |
| 1440 | default | 42.04 dB | 0.972 | 0.0176 / 0.0177 |
| 1440 | 2,2,2 | 42.65 dB | 0.974 | 0.0208 / 0.0222 |
| 1440 | 2,2,2 + int 2 | 39.15 dB | 0.984 | 0.0351 / 0.0374 |

With history (HIST=1, 540, 2,2,2) we get 39.37 dB, so our history recurrence also matches the reference.

- Tone 2 and structure 2 are inside the reference's valid range, and the reference produces the same output.
- The intensity 2 rule matches the reference's `compose()`.
- At these sliders, the strong look (about 3.5x the edit) and the 4-5x flicker come from the network itself.
- The edit strength is about the same at every model size. This explains the "little visible gain" at hi-res.

### Pipeline vs the per-frame reference
Metric: PIPE in test_sync (`NR_TEST_REF` + `NR_TEST_COND` / `NR_TEST_INT`). Frame 2560x1440.

- **model = frame (1440):** PIPE 0.00000 at default and 2,2,2, and 0.00007 with int 2. This path is bit-identical to the network. The geometry and the clamp-fix composition are OK.
- **Resampled 540 / 1080:** PIPE is about 10 % of the edit. This is the input-resampling difference, and it does not depend on model size.
- **history=1:** the output departs from the no-history reference by 30-45 % of the edit. With exact motion vectors, the moving-object test (`hist.sh`) shows ghost / uncovered error at about the same relative size from 540 to 1440. That is no resolution-specific failure.
- **history=1 with 2,2,2 / int 2:** the history feeds the strong edit back and builds it up (brightness drift 0.016-0.03). The reference recurrence does the same (see above).
- **Flicker at 2,2,2 / int 2:**

  | model | clean | history | Standard (et 0.7 + split 3) |
  |---|---|---|---|
  | 540 | 13.5 % | 8.3 % | 3.4 % |
  | 1080 | 9.3 % | 7.7 % | 1.7 % |
  | 1440 | 12.0 % | 11.5 % | 2.6 % |

  The user's FO4 ini had edit_temporal=0, nr_split=0 and history=1, which is the "history" column.

### HDR latch
- hdr auto applies only to RGBA16F / R9G9B9E5 colour.
- The FO4 log has no `hdr:` lines, so the latch was off.
- The latch needs pixels > 1, so a dark frame cannot set it. Not a cause here.

### Self-check, our bug (fixed)
- **The bug:** `self_check` read io's input after the job's flag was out. By then the D3D12 side may already have imported the next frame, so the reported `mean |out-in|` was inflated by motion.
- **The fix:** the input is now snapshotted at the job's part 1.
- **Not verified:** that this caused the FO4 0.13 at 1440p. The harness cannot show it, because its motion is small.

## Not verified / open
- **REGRESSION, open:** the recreate stress test (`recreate.sh <build> 0 6`) hangs after cycle 3 with the v0.4.4 DLL. It also hangs with the v0.4.3 runtime and the v0.4.3 test_sync (build3), so the cause is in dlssnr_nr.cpp. v0.4.3 passes the same test. Symptoms:
  - About 15 s extra before the eviction ("idle 26 s" vs 11 s).
  - Cycle 4 never finishes, killed at the 150 s bound.
  - Not bisected yet. Suspects: review-fix changes in evaluate (lock / state snapshot), the nr-preset create path, ptConvert.
  - The timestamped run `ts_recreate.sh` did not get the GPU (another session held the locks).
  - The eviction-race test itself passes (evaluate returns 1, the device is OK).
  - **Do not ship before this is bisected.**
- `NR_TEST_DROP_PART=20` with Performance (split 4) never triggers within 60 frames. Only drop=10 is exercised (it passes).
- The nr_every=2 runs time out at 150 s on the v0.4.4 build. Base finished at about the bound, so this is probably harness slowness, not a hang. Recheck with NF=40.
- No in-game test (as instructed).
