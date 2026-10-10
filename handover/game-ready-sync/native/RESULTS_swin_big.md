# swin c64/c128/c256 speed work (perf-swin-big, 2026-10-10)

Base: `swin_var.hip` = astra-work/validated-attention-candidate (shared attention normalization, bit-exact, -0.42 ms).
Tools: `cmpvar.py ref.co test.co` (byte compare and timing on all 36 g_A steps), `benchvar.py`, `bsv.sh` (build plus VGPR/LDS/scratch),
`gpu_time_swin.sh` (bounded GPU timing, 3 repeats; not run because GPU use is currently blocked).

## Measured on the GPU before the no-GPU rule (one plain step per width, ms, against the original native-mid source)
Knockout profile, 256 threads (C64 / C128 / C256): full 1.08 / 0.93 / 0.79.
- All GEMM inner loops removed: 0.32 / 0.33 / 0.28. The dot2 loops are about 65-70% of the time.
- Removing one phase saves: expand 0.20/0.17/0.19, qkv 0.10/0.12/0.13, context 0.12/0.05/0.06, mid 0.10/0.06/0.04,
  logits 0.08/0.05/0.03, contract and out_proj about 0.03/0.04.
- Removing weight staging (put16) saves 0.07 ms at C256 only.

Rejected variants (bit-exact on all 36 steps, but slower):
- 512 threads per workgroup (fewer outputs per thread; the 128-VGPR cap causes scratch spills): +48% / +31% / +34%.
- C-wide GEMMs on 128 threads with 8x8 tiles (less LDS traffic per dot2): +18% / +13% / +23% (256 VGPRs, spills at C256).

Conclusion: the inner loops already issue one ds_read_b128 per 4-8 dot2 and are limited by dot2 issue. LDS and VGPRs cap occupancy.
Untested ideas, which need the GPU:
- C256: process a window pair per workgroup to fix the 2-round tail (126 workgroups on 80 CUs).
- C256: split the MLP head loop across 2 workgroups and sum the partials in head order.

No new candidate is CPU-verified. Nothing should be installed beyond the astra attention candidate.
