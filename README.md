# gfx1030 DLSS-NR translation lab

Experimental AMD RDNA2 translation and runtime research. **Not ready for
in-game rendering. No successful DLSS-NR game frame has been demonstrated here.**

## Current status — September 22, 2026

**Still not a mod, and still no rendered DLSS-NR frame.** What changed is that the
network's structure is now fully reverse-engineered and most of it has been executed
on the physical GPU.

### Reverse engineering: complete

The whole 71-block schedule is recovered from the vendor host binary, and every
weight block has an identified owner:

| blocks | stage |
|---|---|
| 0 | pre-block |
| 1-22 | encoder, 4 stages of 4/4/6/8 at C=32/64/128/256 |
| 23-30, 40-47 | C=512 windowed attention (`ffwd` -> `ffwd2` -> residual -> `qkv_attn` -> residual) |
| 31-38 | ViT, C=1024 |
| 39 | the decoder-upsample transition weight (not a block) |
| 48-69 | decoder, 4 stages of 8/6/4/4 — the encoder mirrored |
| 70 | final head |

Also resolved: the U-net skip wiring (encoder and decoder index the same buffer array
with deliberately opposite formulas), the `ctx` buffer map, per-block window modes, and
the frame-level pipeline. Block-to-block sync needs no flag kernels — they are not in
the forward pass at all.

### Executed on the RX 6900 XT, bit-exact against the interpreter

* **The full 22-block encoder as one dispatch chain**, all four stages including the
  fused stage transitions — 0 mismatches.
* **A complete C=512 attention block** (5 chained dispatches, real weights) — 0 mismatches.
* **A decoder stage** (blocks 66-69) — 0 mismatches.
* **`k_import` on the real captured Cyberpunk frame** at full 1707x960, producing a
  viewable image whose statistics match the capture's own.
* 16 of 18 registry kernels pass individually; see [the contract notes](rdna2/emu/VARPARAMS_HOST_CONTRACT.md).

### Two findings that reframed earlier results

Both remaining kernel "failures" turned out to be defects in *the reference interpreter*,
not in the gfx1030 translation: a dropped `stride64` scale on LDS 2addr addressing, and
`v_cvt_f16_f32` destroying the upper half of its destination. Fixing the second also
fixed `k_mean`, which had been separately misdiagnosed. What remains is a 1-ULP rounding
difference in WMMA — a reference-model imprecision, not a port bug.

These difftests judge the *pair* (translation, reference model). For two sessions a
reference-model defect read as a translation failure.

### What is still missing for a frame

Assembling the verified pieces into one 71-block run with real buffer allocation, the
ViT chain (decoded but not yet executed), and `k_export`, which cannot be validated
standalone because it consumes the network's own output format.

## Developer entry points

* [Continuation notes](rdna2/emu/ASTRA_CONTINUATION.md)
* [Capture implementation](rdna2/LAUNCH_CAPTURE.md)
* [Cyberpunk test history](rdna2/CYBERPUNK_CAPTURE_PREFLIGHT.md)
* [Selected diagnostic evidence](rdna2/diagnostics/2026-09-21/README.md)

The `gfx1032-dlss-nr-research-main` source/notes were supplied by another
contributor. Their historical claims are not automatically validated by this
project. The test machine is an RX 6900 XT (`gfx1030`); a `gfx1032` label does
not establish compatible translated kernels.

Vendor DLLs, extracted payloads/disassemblies, compiled executables and local
build directories are not distributed in this update. Local launch scripts
require generated configuration and privately supplied runtime files; this
repository is not a ready-to-install game mod.
