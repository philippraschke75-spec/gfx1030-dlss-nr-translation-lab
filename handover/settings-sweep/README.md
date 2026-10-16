# Settings sweep (2026-10-16, branch `settings-sweep`)

Base: v0.3.1-rc + accuracy + int8-swin + perf-tile. In perf-tile, nr_runtime.cpp, prepost/build.sh and
post_native.hip were kept from rc: rc already has the generic `<sym>_prep` path and its own pre/post.
Kernels: the c32fix translated set, natives rebuilt from rdna2/native (int8 swinvar + tiled vit), and pre/post .co from perf_tile.
build_plan.py works at 960x540, 832x468, 768x432 and 640x360.
Data and logs: `E:\dlss-runtime\sweep` (cfg/<name>/, png/, nr_data/).

Test setup: test_sync pace, cpu 8 ms, game GPU load 12 ms, 2560x1440 frame, 250 frames, `--exclusive`. fps is the median of 3 runs.
- PSNR: last frame (1440p, display range) of the flicker run vs `ref` (540p, NR every frame, vit_every 1, int8 off).
- Flicker: harness at 1440p, 60 frames. Ghost: NR_TEST_GHOST at frame = model size.
- Release ini otherwise (history=1, nr_split=1, busy_mode=wait, int8_swin=1).

See table.md. Short version:

| config | fps | vs base | PSNR | visible flicker | pacing |
|---|---:|---:|---:|---:|---|
| 960x540 vit2 (Quality, current) | 37.3 | 0 | 46.2 | 1.79 % | ok |
| 960x540 vit3 / vit4 | 40.1 / 40.0 | +7 % | 46.2 / 46.1 | 1.84 % | ok |
| 832x468 vit2 (Balanced) | 42.7 | +14.5 % | 40.5 | 2.02 % | best, 0 spikes |
| 768x432 vit2-4 | 41-43 | +10-16 % | 39.1-39.3 | 2.0 % | worse than 468, no reason to use |
| 640x360 vit4 (Performance) | 48.8 | +30.8 % | 36.6 | 2.00 % | ok |
| nr_every=3 + mv_reuse (no split) | 44-47 | +19-27 % | 45.3 / 39.3 | 1.66 / 2.08 % | REJECT: 13/38 ms alternating frames (sd 7-10) |

nr_bench mean network ms (int8 on): 540p 22.4 / vit2 19.4 / vit3 18.6 / vit4 18.0. 468p 18.4 / 15.4 / 14.7 / 14.0.
432p 17.0 / 14.4 / 14.3 / 13.1. 360p 13.9 / 11.1 / 10.3 / 9.9.
torch_ref (STR=1,1,1,1, one frame, ours vs PyTorch at the same res): 41.3 / 41.2 / 39.5 / 43.0 dB (540/468/432/360). So every size has the reference geometry; 432 is the weakest.
int8_swin: +1.6 % fps, -0.3 dB vs ref.

## Frame spikes
- About 1 frame per 250 is 40-60 ms (about 2 frame times). Occasionally a whole run degrades (sd 7-17, many spikes): this happened only in one late time window, for three configs in a row, so it is external GPU contention.
- They do not come from HDR (hdr_mode=off still spikes), logging (log= still spikes), the self-check (it runs once, at NR frame 10, and spikes land at random frames 50-240), or eviction (none logged).
- The v0.3.1-rc base build shows them too (max 66.8 ms in this test), so this is not a v0.3.1 regression.
- Pattern: one isolated late HIP job, so D3D12 spins for one extra job.
- `ref` (no split) had a 224 ms frame close to the 250 ms spin bound: a job that was dropped.
- Likely cause: the GPU scheduler arbitrating between the D3D12 queue and the HIP queue under full load. It needs HIP-side timestamps per job to prove (not done).
- test_sync now prints `pace spikes` (frames over 1.5x median).

Presets: preset_quality.ini, preset_balanced.ini, preset_performance.ini. PNGs: input, base, r468, r360v4 (all 1440p; the rest are in sweep/png).
