# Frame runner state, 2026-09-24

The chain renders a full 1707x960 frame: 170 dispatches, ~620 ms, guards intact,
`k_export` finite over all 960 rows and 1707 columns. The scene is sharp and
legible. What remains is one colour defect and a set of parameters the capture
never recorded.

## Fixed this session, with the evidence

| fix | env | evidence |
|---|---|---|
| The tail is the post-block, not `k_final_head` | always on | R6/R6b/R6c: epilogue dispatches handle `0x180066350`; kernarg layout read from the stores at `0x180030e13`-`0x180030ea4`; grid `(PAD_W//8, PAD_H//8)` from the same idiom as the prologue's |
| `ctx+0x100` is the export input | always on | allocated `ctx[0x18]*ctx[0x1c]*16` at `0x18002ded7`-`0x18002df37`; `k_export` +0x00 built at `0x18002d7ca` |
| `k_export` +0x08 = `PAD_W` | always on | no shear at the padded pitch |
| Encoder skip leak removed | always on | `enc_stage_out.get(3 - di, ...)` sat in the encoder loop with `di` leaking from the allocation loop, so stages 2-4 read stage 1's output |
| U-net skip wired in the decoder | `DEC_SKIP=1` | `head input` went from 11.66% to 100% of bytes |
| Pre-block at the padded full frame | `PRE_FULL=1` | R9: grid and H/W come from `ctx+0x18` (`0x18002ee72`, `0x18002edbd`), same idiom as the post-block. Removed the hard edge at row 256 |
| `k_pre_block_1h_32_fp8` as block 0 | `PRE_1H=1` | R10: the prologue branches are exclusive and C:67 had them inverted. `byte [0x18009b1f0] == 1` pairs pre_block_1h with post_block_1h |
| **Decoder last block takes flag 2** | `DEC_LAST2=1` | **R5, `0x180030a0e`-`0x180030a22`. This removed the column stripe: dec s1 phase-var 0.992 -> 0.008, dec s4 0.986 -> 0.001** |
| Previous stage at +0x30, not +0x38 | `DEC_PTRA=1` | E3: `trace_reads 256_0 8` shows flag 8 reads +0x30 (32,768 B) and never touches +0x38 |
| Skip parity | `SKIP_PARITY=1` | encoder stage s ping-pongs `ctx+0x1d8[s]` / `ctx+0x2a8[3-s]`; the first block writes `ctx+0x1d8[s]`, so with even block counts the decoder's skip is the second-to-last output |

## Corrections to the contract that this session established

* The activation layout is **NCHW16c**: `(c//16)*(H*W*16) + (y*W + x)*16 + (c%16)`.
  Proven by a 16-impulse probe, 16 of 16 landing at the predicted positions. The
  "4x4 tile" reading of C:22 was inferred and is refuted.
* The U-net skip array is **`ctx+0x1d8`**, indexed by `s` in the encoder and
  `3-d` in the decoder. `ctx+0x2a8[d]` is the decoder's first-block output and
  half of its ping-pong pair. The skip is passed by **buffer aliasing**, never as
  a pointer argument.
* C:67 had the prologue branches inverted.
* `difftest`s ran at `H = W = 16`, two windows per axis, and were blind to
  column-periodic structure. `net_encoder_stage1.py` now takes `DIFF_H`/`DIFF_W`.
  Block 1 passes at W=64 with 0 mismatches, and the **reference emulator shows the
  same period-4 structure**, so that structure is intrinsic to the kernel and not
  a translation defect.

## The one remaining defect

A's output anti-correlates with the input: block-mean luminance correlation
-0.2574 / -0.3875 / -0.5463 at n = 1 / 8 / 32, and mid-band structural
correlation S_mid ~ 0 in every configuration tried. Green is specifically
suppressed: A's channel means are R 0.3133, G 0.0827, B 0.2631 against an input
of R 0.1218, G 0.2307, B 0.2721.

No scalar moved it. `POST_F30`, `POST_F48`, `EXP_S0`, `EXP_S1`, every channel
permutation and every input substitution left it in place.

**The open reading:** if A is a residual-form post-block, i.e. it produces a
correction rather than an image, then anti-correlation with the input is expected
and something should add the input back. That would make `k_export`'s +0x38 term
an add with the wrong sign rather than a spurious subtract, and would change what
the right value for it is.

## Parameters that cannot be read from the binary

Three remaining unknowns are **application-supplied per frame**, not constants:

* `ctx+0x30` (post-block +0x30). Never written anywhere in the frame, setup or
  driver code; only read twice, both in the epilogue.
* `k_export` +0x38 = `xmm6` = job param `+0x40`, loaded unconditionally at
  `0x18002d47b`.
* `k_export` +0x3c = `xmm7` = job param `+0x3c` when positive, else the constant
  at `0x18006d2c8`.

`EXP_S0=0` removes a subtraction that otherwise puts -65.125 into an RGBA16F
display surface, and 0 is an ordinary value for a strength control. That is a
parameter choice, not a fitted constant.

`POST_F30=0.125` is different: it was fitted against a frame-wide R:G:B ratio
that has since been retired as a correctness proxy. It should not be treated as
derived.

**Consequence for what "finished" means:** the kernels and the schedule can be
fully correct while the output still looks wrong, because the per-frame job
parameters were never captured. Closing that gap needs a capture that records
the job struct at the API boundary (the block at `rax` in the frame function,
fields +0x08 through +0x40), not more disassembly.

## Update: the export parameters are derived, not application-supplied

Host read of the embedded PE (`dlssnr_on_amd_setup.exe` @0x47c00). The job pointer is the
frame function's 10th argument (`[rsp+0x320]` at `0x18002d40c`). Its two call sites:

* `0x180015b24` passes NULL: +0x38 = 0 (`xorps xmm6`), +0x3c = 1.0 (`0x18006d2c8`).
* `0x18001a84a`/`0x18001a89e` pass the local job at `rbp+0x40`, initialised at `0x18001a078`-`0x18001a0c5`:
  * job +0x3c = f32 table `[0x180096cd0 + 4*slot]`, which is **1.0 for all four slots**.
  * job +0x40 = **int** history-valid flag. It is 0 at init and set to 1 at `0x18001a822` only when
    `byte [0x18009ac5c] == 1`, which `setns` sets over the history resource creation calls at `0x1800152e4`.

So `k_export` +0x38 is an int flag, not a float strength. Packing it as int 1 reproduces the
-65.12 subtraction exactly. That term is the history blend, which is correctly off on a first frame. The runner
now packs `EXP_HIST` (default 0) as i32 and `EXP_S1` defaults to 1.0. Both are host-derived values.
S_mid is unchanged (+0.0017), so **the anti-correlation is not an export-parameter problem**. It is
upstream, in the post-block or before it.

**Post-block scalars, also host-derived** (`ctx` is the global `0x18009a100`):

* +0x30 = `ctx+0x30` is the ini `[DlssNrOnAmd] Scale`, **default 0.03125**. It is read at `0x1800082b0`-`0x1800082d3` into
  `0x18009ad14` and copied to `ctx+0x30` every frame at `0x18001a735`. It was not unwritten: the store is
  RIP-relative, which a `0x30(%reg)` search misses. The fitted 0.125 is retired.
* +0x48 = `xmm6`. It is zero only when env `DLSSNR_NOBLEND` is set (`0x1800314d6`-`0x1800314e5`). Otherwise it is
  `ctx+0xd8` = f16 `block70.layer0.blend_scale` read back at `0x180030b7c`-`0x180030c4f`, which is **0.7397**.
  The runner now loads it from `block70_layer0_blend_scale.bin`.

With every host value derived: S_mid +0.0097, S_fine +0.0026, lag8 0.88, and about 30% of export pixels
clamp at 0. **No host parameter is left to blame; the defect is in the network's computation.**

The addresses in this file refer to the embedded PE, not NVIDIA's `nvngx_dlssnr.dll`.

## Update 2: the "anti-correlation" was a sheared reference

`k_import` writes at the **padded** pitch: it receives PAD_H/PAD_W and a padded grid, and the buffer is
PAD_H*PAD_W*12. The runner read it packed, as SRC_H x SRC_W, which shears each row by 85 px (adjacent-row corr
0.21 vs 0.75). `imported.png`, the reference that `smid.py` and every correlation above used, was that
sheared image. With the read fixed:

| | before | after |
|---|---|---|
| S_mid / S_fine | +0.0097 / +0.0026 | **+0.651 / +0.751** |
| block-mean lum corr n=1/8/32 | -0.26/-0.39/-0.55 | **+0.72/+0.71/+0.71** |
| channel means R/G/B (input 0.33/0.46/0.49) | "green suppressed" | 0.19/0.31/0.33 |

Every S_mid figure earlier in this file, and in earlier sessions, was measured against the sheared
reference and is void. That includes the "no scalar moved it" conclusion.

**What is still wrong, and where:** `POST_CONST=0x00,0x08` (network inputs to the post-block replaced by
constants) gives S_mid **+0.999**, so the passthrough/blend path is right. With the real network, the post-block adds
a correction that has no fine structure. A held-out linear probe (luminance band-passed and regressed on all channels,
NCHW16c e4m3) finds R^2 ~ 0 in the pre-block output (act A as f16 or e4m3, act B) and in encoder stage 1. It rises only at
coarse pools (enc s4 pool ~0.36), where coarse structure survives misregistration. **Loss starts at the pre-block.**

Pre-block scalars are now host-derived and made no difference: +0x20 = ctx+0x34 = 0.0 (never written; ctx is BSS),
+0x28 = ctx+0x20 = (0,0) (ToneChannels default 0), +0x48 = ctx+0x28/0x2c = (1.0, 1.0) f32 (UseAutoMask default 1,
SkinStructure -1 -> LocalStructure 1.0).

## Update 3: the network never saw the image; fixed

The pre-block output was **byte-identical** for the real frame, a black frame and an impulse frame, with
deterministic repeat runs. Two kernarg errors, both required for the kernel to read its input:

* **+0x40 is a second RGB source, not scratch.** The kernel loads 12 B/pixel from +0x00 and from +0x40 at the same
  index, using +0x00 alone when +0x40 is null (`s_cmp_eq_u64 s[26:27],0` / `v_cndmask`, code 0x2EFC-0x2FE8).
  The host passes ctx+0x118 (`0x18002ed8d` -> `[rbp+0x618]` -> `0x18002ef4a`), which is 0 on the `0x18002d6da` path and ctx+0x108 on
  `0x18002dd04`. The runner had the scratch buffer there. Now `PRE_40=null` (default) | ctx118 | scratch.
* **+0x20 = ctx+0x34 = 0.0625.** The ctx constructor (`0x18001f211` -> `0x180020470`) stores the 8 bytes {0.03125, 0.0625}
  from `0x18006d2e0` into ctx+0x30/0x34. My earlier "never written" missed this 8-byte store. At 0 the kernel ignores its input.

After the fix (linear-probe R^2 against input luminance): encoder input (pre-block +0x38, C=64 at 512x896) **1.000**;
enc s2/s3/s4 pools **0.998 / 0.995 / 0.997**; **dec s1 pong 0.46**. Ping/pong and act A still probe ~0, but the pools
computed from them are ~1, so those buffers use another layout (probe limitation, not data loss).
S_mid is now +0.547. That is lower than the +0.651 of the input-blind network, because the network's correction
now depends on the image but is degraded. **The next loss point is the decoder, from stage 1 on.**

Kept from the earlier note: the pre-block difftests ran at H=W=16 and never exercised the real input path.

## Update 4: the middle section (C=512 -> ViT -> k_dec_upsample) does not follow the host

Impulse-vs-black footprints (layout-free: which output bytes change) at full frame:
enc s4 pool peaks exactly at the four impulse positions (encoder intact). From c512_1 on, every pixel changes,
the ViT buffers barely react, and block39's output buffer is all zero. Ridge probe: enc s4 pool R^2 0.98, c512_1 w0 0.43.

What the host does (embedded PE):
* `k_dec_upsample` (handle 0x1800663e8, launch 0x180030509-0x1800305b8): +0x00 = ctx+0x250 (dumped as 'vit1d'),
  **+0x08 = ctx+0x228 ('vit512a', the C=512 stage-1 output, set at 0x18002fa17)**, +0x10 = ctx+0x298, +0x18 = block 39
  weights, +0x20 = [rbp+0x544]; grid ([rbp+0x604], 1, 1). The runner passes the never-written off_b39 at +0x08.
* **`k_repack` (handle 0x1800663e0) also runs before the ViT loop**, not only at enc -> C=512 (launch 0x18002fcfe).
* ViT loop (blocks 31-38, 0x18002fd47-0x1800303a0): input/output **ping-pong between ctx+0x258 and ctx+0x290**;
  `k_expand2` grid = (ctx[0x314]/64, **32**, 1), block 256. The runner uses (7,4) for all five ViT kernels, in place on one
  buffer. The ViT's output is ~0: B260 3.6% nonzero, B288 0.11%, and a separate output buffer stays entirely zero.
  So the ViT contributes nothing, and the c512 features or zeros reach the decoder.

VIT_SEP=1 (separate ViT buffer, block 39 +0x08 = c512_1[0]) is in the runner but off by default. With the ViT
producing zeros it is no better. **Next: rebuild the mid section from the host.** That means k_repack into the ViT
token buffer, the per-kernel grids, the 0x258/0x290 ping-pong, k_repack back, and k_dec_upsample with the skip.

## Method note

Six rounds of hypotheses moved nothing. One literal read of the host's flag
computation moved everything. Where a host read is available, take it before
reasoning from symptoms.

Two metrics were used and both had to be discarded: frame-wide R:G:B against the
input (a denoiser need not match its input's channel balance) and "sharper by
eye" (amplified noise also looks sharp). The surviving measure is the band-passed
structural correlation in `smid.py`, with the column phase-variance as a guard.
