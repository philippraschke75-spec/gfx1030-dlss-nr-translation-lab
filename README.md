# gfx1030 DLSS-NR translation lab

Experimental AMD RDNA2 translation and runtime research: NVIDIA DLSS Neural
Rendering's 71-block network, ported from its gfx1100 (RDNA3) kernels to
gfx1030 (RDNA2, tested on an RX 6900 XT), by disassembly-derived kernel
translation and instruction-level GPU/emulator diff-testing.

## Current status — September 25, 2026

**A full captured Cyberpunk 2077 frame now renders correctly end to end on
the RX 6900 XT.** Structural correlation against a "network ignores its
input" baseline: **S_mid +0.9247, S_fine +0.9258** (input-blind baseline is
+0.651). No dark blocks, no colour cast. This is a genuine correctness
breakthrough, not incremental — as recently as this week the same frame
scored +0.71 with a visible colour cast and tile pattern; three host-read
fixes (a spurious extra `k_repack` dispatch that was never part of the real
schedule, and the `k_import`/`k_export` tonemap-mode field, both traced to
exact host addresses) closed the remaining gap.

Still open: performance (a full frame currently takes several hundred ms of
GPU time; two kernels dominate at ~40% combined and are being profiled
statically to determine whether there is a safe lever without touching
kernel correctness — see `PERF_ANALYSIS.md`), and a native-resolution,
no-upscaler capture path (the existing capture hooks a specific upscaler's
internal dispatch, which does not fire when no upscaler is active).

### Reverse engineering: complete

The whole 71-block schedule is recovered from the vendor host binary, and every
weight block has an identified owner:

| blocks | stage |
|---|---|
| 0 | pre-block (`k_pre_block_1h_32_fp8`) |
| 1-22 | encoder, 4 stages of 4/4/6/8 at C=32/64/128/256 |
| 23-30, 40-47 | C=512 windowed attention (`k_ffwd2` -> residual -> `k_qkv_attn2` -> residual) |
| 31-38 | ViT, C=1024 |
| 39 | decoder-upsample transition (`k_dec_upsample`) |
| 48-69 | decoder, 4 stages of 8/6/4/4 — the encoder mirrored, its own flag convention |
| 70 | post-block (`k_post_block_1h_32_fp8`) — the sole writer of the network's result buffer |

Also resolved: the U-net skip wiring, the `ctx` buffer map, per-block window
modes and origins (all 4 shift-window origins now difftested bit-exact), the
frame-level pipeline, and the true tail of the network (the post-block, not
`k_final_head` as earlier notes assumed). Block-to-block sync needs no flag
kernels — they are not in the forward pass at all.

### Verified bit-exact against the reference emulator, on real hardware

* The full 22-block encoder as one dispatch chain, all four stages — 0 mismatches.
* The C=512 attention stage (`k_conv_res2`, `k_qkv_attn2`) at real geometry,
  **all four shift-window origins**, 3-D grid dispatch (16 attention heads via
  `workgroup_id_z`) — 0 mismatches each.
* `k_export` across every format/mode and history-flag combination — 0 mismatches.
* The post-block and pre-block kernels on finite input — 0 mismatches (the
  only known divergence is how an e4m3 NaN code propagates, unresolvable
  without real gfx1100 hardware, and immaterial unless the frame produces
  NaN activations).
* `k_import` on the real captured Cyberpunk frame at full 1707x960.

A coverage-guard utility (`rdna2/emu/coverage_guard.py`) now makes every
difftest's "bit-exact" claim a counted, blocking property: a test declares
the full parameter space a claim needs and refuses PASS until every required
combination has a *current* (code-fingerprinted) passing result — prompted
by an external review that caught us reporting "3 of 4 origins" as complete.

### What closed the remaining correctness gap

Two host-read fixes, each traced to an exact address in the vendor binary:

1. The runner was dispatching an extra `k_repack` between the encoder and
   the C=512 stage that the host never calls at that point — it scrambled
   every activation one dispatch after a clean encoder output. Dropping it:
   S_mid +0.71 -> +0.89.
2. `k_import`/`k_export`'s tonemap-mode field was read from the wrong stack
   argument. With the correct host value, 26 previously-saturated 16px
   blocks (>x9 activation blowup at specific encoder blocks) became finite:
   S_mid +0.89 -> +0.9247, no colour cast.

## Developer entry points

* [Frame runner state, running log](rdna2/emu/FRAME_STATE_2026-09-24.md) — the primary log of every fix this session, in order, with host citations.
* [Kernel/host contract](rdna2/emu/VARPARAMS_HOST_CONTRACT.md)
* [Continuation notes](rdna2/emu/ASTRA_CONTINUATION.md)
* [Capture implementation](rdna2/LAUNCH_CAPTURE.md)
* [Cyberpunk test history](rdna2/CYBERPUNK_CAPTURE_PREFLIGHT.md)
* [Selected diagnostic evidence](rdna2/diagnostics/2026-09-21/README.md)
* [Performance analysis](PERF_ANALYSIS.md) — static + dynamic profiling of the two most expensive kernels; in progress.

The `gfx1032-dlss-nr-research-main` source/notes were supplied by another
contributor. Their historical claims are not automatically validated by this
project. The test machine is an RX 6900 XT (`gfx1030`); a `gfx1032` label does
not establish compatible translated kernels.

Vendor DLLs, extracted payloads/disassemblies, compiled executables and local
build directories are not distributed in this update. Local launch scripts
require generated configuration and privately supplied runtime files; this
repository is not a ready-to-install game mod.
