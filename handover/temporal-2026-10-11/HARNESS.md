# HARNESS.md - temporal flicker harness and baseline (2026-10-11)

Branch `temporal-harness` (from `origin/combo-mv-flicker`). Files:
- `handover/game-ready-sync/dll/test_sync.cpp`: flicker mode (`NR_TEST_FW/FH`) rewritten as a game-like sequence + temporal metric. It now passes motion vectors and depth to evaluate.
- `handover/game-ready-sync/temporal/run_flicker.sh`: one guarded run (Fallout check, `/e/dlss-runtime/gpu.lock`, 150 s timeout).
- `handover/game-ready-sync/temporal/flicker_torch.py`: PyTorch reference (single-frame and history modes) on the dumped sequence. CPU only.

## Sequence (test_sync flicker mode)
- Source: `E:\dlss-runtime\color_1920x1080.bin` (a real captured frame), bilinear-resampled to the 2560x1440 frame.
- Camera pan: 0.37 / 0.13 frame px per frame (sub-pixel, `NR_TEST_VX/VY`).
- TAA-like jitter: Halton(2,3), amplitude +-0.25 px (`NR_TEST_JIT`).
- Per-pixel, per-frame multiplicative noise: sigma 1 % (`NR_TEST_NOISE`).
- Optional bright specks (`NR_TEST_SPECK`, default 0).
- Motion vectors: RG16F uv, previous = current + mv, mvScale = W,H, the exact per-frame motion. Depth: R32F 0.1, inverted. Both are passed to evaluate_v2. The current build ignores them unless mv_reuse is on.

## Metric
Computed for frames k >= 1 over the interior (8 px plus the motion margin), with values clamped to [0,1]:
- `inst` = mean over RGB of |x_k - warp(x_{k-1})|, where warp is a bilinear fetch at the exact known motion. It is computed for the output and for the input.
- `excess` = inst_out - inst_in: the instability NR adds. This is SPEC's F. `x` = inst_out / inst_in.
- `white jumps`: % of pixels whose luma rises against the warped previous output by more than 3/255 beyond the input's own rise.
- `visible flicker`: % of pixels where that excess luma change is larger than 2/255, either sign.
- `network alone` (`NR_TEST_REF=1`): the same metric for nr_run on the same sequence generated at model size (input clamped to [0,1]). This takes the DLL's resample/composite path out of the measurement.
- flicker_torch.py also prints `mean(out-in)` per frame. This is the brightness bias the SPEC asks for.

## Baseline: current build (history-valid = 0), combo data, frame 2560x1440, 60 frames

| case | model | inst out | inst in | excess | x | white jumps | visible flicker |
|---|---|---|---|---|---|---|---|
| game sequence (defaults) | 960x540 | 0.00392 | 0.00262 | +0.00130 | 1.50 | 0.718 % | 3.93 % |
| game sequence (defaults) | 1280x720 | 0.00371 | 0.00262 | +0.00110 | 1.42 | 0.611 % | 3.23 % |
| network alone, defaults | 960x540 | 0.00621 | 0.00439 | +0.00182 | 1.41 | 1.48 % | 6.69 % |
| network alone, defaults | 1280x720 | 0.00536 | 0.00389 | +0.00147 | 1.38 | 1.17 % | 5.22 % |
| static camera (VX=VY=0), jitter + noise, 30 fr | 960x540 | 0.00380 | 0.00256 | +0.00124 | 1.49 | 0.647 % | 3.49 % |
| static camera, 30 fr | 1280x720 | 0.00364 | 0.00256 | +0.00108 | 1.42 | 0.560 % | 3.05 % |
| clean pan (no noise, no jitter), 30 fr | 960x540 | 0.00303 | 0.00121 | +0.00182 | 2.49 | - | - |
| fully static (no motion, no noise) | 960x540 | 0 | 0 | 0 | - | 0 | 0 |
| known-bad control: `ldr_input=0` (old HDR input) | 960x540 | 0.01261 | 0.00262 | +0.01000 | 4.82 | 16.4 % | 49.6 % |

Reading:
- NR always adds instability: 1.4-1.5x the input's own frame-to-frame change, and about 2.5x on a clean sub-pixel pan.
- 960x540 is worse than 1280x720 on every metric. This matches the in-game report, where the flicker is seen at 540.
- The network is deterministic: identical input gives identical output, so a static, noise-free scene gives exactly 0. The flicker therefore comes from the network not being shift-equivariant and from it amplifying sub-pixel jitter and noise. It is not run-to-run nondeterminism.
- The metric separates the known white-flicker bug (`ldr_input=0`, excess 8x higher) from the current state. The current residual is the low-level "slightly white" flicker: 0.6-0.7 % of pixels per frame jump brighter by more than 3/255.
- Not checked: whether the numbers match what the eye sees in game. No game runs were allowed.

## PyTorch reference (dlssnr-pytorch, CPU, 960x540, same 12 model-size frames)
- dlssnr-pytorch has no temporal mode: depth, motion, history and reset are ignored. flicker_torch.py adds history by monkeypatching, following SPEC.md and GEN.md, without editing the package:
  - `hist`: Phist = previous PyTorch output warped by the known motion. Pre-block lanes 7-9 = centred(Phist). The post-block then applies `out += clamp(0.7397*sigmoid(head.a)) * (Phist - out)`, where head.a = row 3 of `out_project`. On frame 0, Phist = current input.
  - `histslot`: lanes 7-9 only, without the post lerp.
- Sanity check on frame 0 (single frame, our nr_run vs PyTorch): 31.5 dB, while the input vs PyTorch is 27.0 dB. Edit correlation is 0.81, edit rms is 0.031 for ours and 0.045 for PyTorch.
  - Our port's edit is about 30 % weaker than PyTorch's, but points the same way.
  - Feeding PyTorch P = sRGB(x/(1+x)) instead is worse (24.2 dB). So the combo data looks like import mode 0 (values passed as is), and `PSPACE=1` is off by default.
  - Seed 0 and seed 1 give the same 31.4-31.5 dB.
  - The gap to the README's 45-49 dB (vs NVIDIA) is UNRESOLVED and is worth checking separately (intensity or conditioning baked into the capture?).

| mode | inst out | inst in | excess (F) | x | white jumps | visible | mean(out-in) |
|---|---|---|---|---|---|---|---|
| ours (nr_run dump) | 0.00611 | 0.00429 | +0.00181 | 1.42 | 1.49 % | 6.64 % | - |
| PyTorch single (history off) | 0.00628 | 0.00429 | +0.00198 | 1.46 | 1.89 % | 8.37 % | +0.0027..+0.0037 |
| **PyTorch hist (slot + lerp) = target** | **0.00339** | 0.00429 | **-0.00090** | **0.79** | **1.17 %** | **6.08 %** | +0.0012 -> 0.0000 |
| PyTorch histslot (no lerp) | 0.00884 | 0.00429 | +0.00455 | 2.06 | 5.66 % | 32.5 % | oscillates +-0.005, growing |

Reading:
- Our port without history matches PyTorch without history on flicker (x1.42 vs x1.46). The baseline is the network's behaviour, not a port bug.
- With history in both places (lanes 7-9 and the post lerp), the output becomes more stable than its input: F goes from +0.0020 to -0.0009, which passes SPEC's F_hist < 0.5 F_nohist, and inst_out is 0.54x.
- Do not ship lanes 7-9 alone: that oscillates and diverges (x2.06 and growing). The post lerp is required.
- Caveat: with history on, the brightness edit mean(out-in) decays from +0.0034 to about 0 over 10 frames. Because Phist on frame 0 is the input, the lerp keeps pulling the output back toward it.
  - This may be correct (the vendor damps), or it may mean the frame-0 fallback or the head.a scale is wrong.
  - Check it against the real library or a longer run before using "hist" as the bit-level target. The flicker reduction itself is clear.

## How to run
Build (PowerShell): `$env:OUT='E:\dlss-runtime\build_temporal'; cmd /c "<repo>\handover\game-ready-sync\build.bat"`

Baseline (bash, from anywhere). Each command is one guarded GPU job of about 1 min:
```
R=<repo>/handover/game-ready-sync/temporal/run_flicker.sh
bash $R 960x540 60 NR_TEST_REF=1                 # frame + network-alone metric
bash $R 1280x720 60 NR_TEST_REF=1
bash $R 960x540 30 NR_TEST_VX=0 NR_TEST_VY=0     # static camera
INI=$'ldr_input=0\n' bash $R 960x540 60          # extra ini lines via INI (here: the known-bad control)
B=<build dir> D=<data dir> FW=.. FH=.. bash $R ...   # other build / data / frame size
```
Output: the `FLICKER frame ...` line, plus a `FLICKER network-alone ...` line when REF is on.

PyTorch reference (CPU, about 30 s per frame):
```
mkdir -p /e/dlss-runtime/temporal_dump
bash $R 960x540 12 NR_TEST_REF=1 NR_TEST_DUMPDIR='E:\dlss-runtime\temporal_dump'
cd "E:/dlss 5/dlssnr-pytorch" && uv run python <repo>/handover/game-ready-sync/temporal/flicker_torch.py 'E:\dlss-runtime\temporal_dump' 12 ours,single,hist,histslot
```
If the test_sync run changes the sequence (VX/VY/JIT) or the size, pass the same values as env MW/MH/FW/VX/VY/JIT.

For the temporal implementation: once history is wired into the DLL, the same `run_flicker.sh` lines are the acceptance test. Targets:
- network alone at 960x540: x <= 0.8 (PyTorch hist 0.79) and white jumps <= 1.2 %, with no brightness drift beyond the hist reference;
- frame path: excess <= 0.

For a bit-level check, dump the history run's outputs, compare them per frame with `hist` mode, and check that the error does not grow.

The old flicker mode (frame-res fine texture, 20.0 specks) is replaced. To get the specks back, use `NR_TEST_SPECK=20`.
The normal test_sync mode is unchanged: 30/30 frames matched, PASS.
