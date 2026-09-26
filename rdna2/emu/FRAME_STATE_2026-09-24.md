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

## Update 5: MID_HOST=1 rebuild runs clean but does not help

`MID_HOST=1` implements Update 4's plan: `k_final_head` into off_pooled/off_headb,
`k_repack` into a dedicated ViT token buffer (`off_tok`), true ping-pong across
the 8 ViT blocks (no more in-place aliasing), and `k_dec_upsample` reading
`off_headb`/`c512_1[0]` as its +0x00/+0x08 pair. Both paths run clean, guards
intact, k_export finite.

Scored with `smid.py` against the corrected (unsheared) input reference, same
code, same frame, only `MID_HOST` toggled:

| | S_mid | S_fine | lag8 |
|---|---|---|---|
| MID_HOST=0 (old, in-place ViT) | +0.5466 | +0.6475 | +0.93 |
| MID_HOST=1 (host rebuild) | +0.5282 | +0.6288 | +0.97 |

**Slightly worse on every measure, not better.** Not a regression in the sense
of anything breaking - the chain still completes - but the rebuild did not
close the gap Update 4 identified, and lag8 creeping toward 1.0 suggests the
column-periodic structure is leaking back in rather than being resolved by the
real ping-pong.

## Update 6: the MID_HOST=1 launches match the host; the loss is not in launch geometry

An earlier draft of Update 5 listed three open guesses: shared repack dims, one shared ViT grid, and unchecked
`k_final_head` wiring. None of them applies to the committed code (b866d1c). All three were read from the host PE
(`version.dll`, base 0x180000000) and compared field by field with `net_frame_full.py`'s `MID_HOST` branch.
In every launch below, `0x180065910` takes the grid in rcx and the block (256) in rdx.

* `k_final_head` (0x1800663d8, launch 0x18002fbbf): grid (ctx[0x30c],1,1). +0x00/+0x08 = ctx+0x248/+0x250 (one
  16-byte copy at 0x18002faea). +0x10 = `block30.layer4.layer`. The host dumps ctx+0x250 at ctx[0x30c]<<14 bytes.
* `k_repack` forward (0x1800663e0, launch 0x18002fd1a): grid (256,1,1). +0x00 = ctx+0x250, +0x08 = ctx+0x258,
  +0x10 = [rbp+0x4a4] = ctx+0x1d0/0x1d4 = (H6,W6) (set at 0x18002f957), +0x18 = ctx[0x314], +0x1c = 1.
* `k_repack` back (launch 0x180030486): the same, except +0x00 = the last ViT output (ctx+0x258 after 8 swaps),
  +0x08 = ctx+0x250, +0x1c = 0. The trailing field is a direction flag. The C=512 dims are not used.
* The five ViT launches all have gx = ctx[0x314]/64. The gy values are expand2 32, contract2 **8**, qkv2 32,
  attention2 32, contract2 **8** (r14/r12 = 0x100000020/0x100000008). contract2 +0x20 is (4096, 0x400000), then
  (1024, 0x100000), with +0x28 = 4. attention2 +0x20 is ctx+0x310 after `pshufd 0xe1` = (NTOK, H6*W6).
  ctx+0x310 = H6*W6 (0x18002e468).
* `k_dec_upsample`: +0x20 = ctx+0x1c4/0x1c8 = (H5,W5), grid (ctx[0x308],1,1).

The ping-pong (swap at 0x18002fd80), the weight layers 0/1/2/4 and the pooled pointer of block 30 also match.
**No host-launch parameter in this section is left to change**, so the drop from 0.547 to 0.528 comes from inside the
kernels or buffers. The remaining suspects, from cheapest to check to most expensive:
1. The ViT kernels at the real size. `net_vit.py`'s bit-exact difftest should be re-run at NTOK=448 with
   ctx+0x310 = 16*28. This is the same trap as the pre-block difftests at H=W=16, and it needs no GPU.
2. The layout of block 30's pooled write (+0x38 -> ctx+0x248) versus the layout `k_final_head` reads.
3. Whether ctx+0x228 really is `c512_1[0]` after blocks 23-30.

## Update 7: the ViT difftest's own blind spot - block_count_x was hardcoded to 1

`net_vit.py` (block 31, five dispatches per ViT block) had `block_count_x=1,
block_count_y=1` hardcoded for every kernel, in both the GPU dispatch manifest
and the emulator's workgroup loop - identical in kind to the earlier
`net_encoder_stage1.py` H=W=16 blind spot (FRAME_STATE, top of file), except
here it affected every ViT difftest ever run, at any H/W, because the grid was
never derived from H/W to begin with.

At the real frame size (1707x960), the C=512->ViT stage is H6,W6=16,28 (NTOK=448,
grid.x=7); the five kernels' grid.y is 32/8/32/32/8 per FRAME_STATE Update 6's
host read. None of that was ever exercised - every ViT translation "PASS" to
date covered exactly one 64-token workgroup.

Fixed: `net_vit.py` now derives grid.x/grid.y for both the kernarg hidden args
AND the GPU dispatch manifest (which had its own separate `|1|1|` hardcoded in
the launch line - two places, not one) from `VIT_REAL`/`VIT_GX`/`VIT_GY_CAP`,
and the emulator side loops over the real (wx, wy) grid instead of (0,0).

**Result: still 0 mismatches at grid.x=2** (all 5 kernels, including
k_attention2, the one with a plausible cross-workgroup reduction). This is the
first time any ViT kernel has been tested at more than one workgroup, and the
translation holds. Full grid.x=7 (real frame size) is running to confirm at
scale; grid.y is capped low (VIT_GY_CAP) to keep emulator runtime bounded,
since y is independent per-workgroup parallelism, not part of the token-count
tiling grid.x tests.

**Consequence:** a cross-workgroup translation bug is looking less likely as
the explanation for the mid-section's low S_mid. The remaining candidates from
Update 6 (block 30's pooled-write layout, whether ctx+0x228 really is
c512_1[0] after blocks 23-30) move back up.

**Update 7 confirmed at full scale.** `VIT_GX=7 VIT_GY_CAP=1` (the real frame's
grid.x, y capped for runtime): 0 mismatches across all 5 kernels, 214,176 bytes
checked, including k_attention2's 14,278-byte delta. The ViT block's
translation is bit-exact at real token count and real grid.x. **The
cross-workgroup / split-K concern is closed: it is not a translation defect.**
The mid-section's low S_mid is somewhere else - most likely the block 30
pooled-write layout or the ctx+0x228 identity, per Update 6's remaining list.

## Update 8: net_block512.py had the SAME blind spot as net_vit.py - and here it FAILS at scale

`net_block512.py` (the C=512 stage's 5-kernel chain, blocks 23-30/40-47) had the identical
`block_count_x/y` hardcoded to `(1,1)` bug as net_vit.py, in the kernarg hidden args, the
GPU dispatch manifest, and the emulator's workgroup loop - three places, same as before.
Its own docstring even said "Verifying the recipe at its REAL geometry is the point" while
never actually doing so. Fixed the same way: `GX, GY = ceil(W/8), ceil(H/8)` (these are
2-D k_swin_var-family kernels), threaded through all three places.

**Unlike the ViT fix, this one FAILS as soon as more than one workgroup runs:**

| grid | result |
|---|---|
| (1,1) - the old default, BH=BW=8 | 0 mismatches, PASS (unchanged from before the fix) |
| (2,1) - BH=8 BW=16, the minimum that adds a second workgroup | **23,466 mismatches, FAIL** |

This is very likely the actual defect behind the mid-section's low S_mid
(+0.53-0.55, Updates 3-6): the C=512 stage runs in the real frame at grid
(7,4) = 28 workgroups (56x32, stage_hw(4) of 1707x960), and its translation
has never been checked at more than one workgroup until now. The real chain
in net_frame_full.py has been dispatching a translation nobody verified
correct at that scale.

**Not yet done, and the natural next step:** localise the mismatch. Candidates,
cheapest first:
* Which of the 5 kernels first diverges (test steps [0], [0,1], [0,1,2], ... like
  net_vit.py's incremental checks already do for the ViT chain).
* Whether the mismatch is spatially structured (confined to one workgroup's tile,
  or a border/seam between tiles) or scattered throughout both.
* `k_qkv_attn`'s window-origin field (`+0x20/+0x24`, fixed at `(0,0)` for block 23
  regardless of which workgroup tile) - every other verified kernel in this
  codebase takes one shift/mode value per DISPATCH, not per workgroup, so this is
  probably not it, but it is the one kernarg field whose per-tile correctness has
  never been separately confirmed at grid>1.
* An LDS or barrier assumption inside one of these kernels that happens to be
  invisible at grid=(1,1) (e.g., a reduction that silently uses uninitialised
  neighbour-tile memory when there IS a neighbour tile, which a single-workgroup
  run can never expose).

`BH`/`BW` env vars (now wired to a real grid) make any smaller reproduction case
cheap to test before spending emulator time on the full 28-workgroup grid.

## Update 9: Update 8's FAIL was the harness, not the translation - the C=512 block passes at grid > 1

The 23,466 mismatches came from `net_block512.py`'s emulator loop, not from any kernel. `k_ffwd_inpview` and
`k_conv_res_views` are 1-D (`.amdhsa_system_sgpr_workgroup_id_y 0`). Their translated prologue is
`s_mov_b32 s15, s2`: the original reads its tile index from s15, and the hardware X id supplies it
(net_frame_full.py's `wants_2d` docstring, :454-462, already says so). The harness gave every kernel
`{14: wx, 15: wy}`. So at grid (2,1), every emulated workgroup of the 1-D kernels got s15 = 0 and recomputed
tile 0, while the GPU computed tiles 0 and 1. The 2-D kernels (`k_ffwd2`, `k_qkv_attn`: s14 = x, s15 = y) were
wired correctly.

Fixed in `net_block512.py`:
* 1-D kernels get `{14: 0, 15: i}` in the emulator.
* They also get net_frame_full.py's own buffer-sized grid, `ceil(512*ceil(H/4)*ceil(W/4)*16 / 16384)`, instead of
  the 2-D (GX, GY). The fix covers the kernarg hidden args, the GPU manifest and the emulator loop.
* `STEPS=` selects a prefix of the chain.

| run | grids (1-D / 2-D) | result |
|---|---|---|
| STEPS=0, BH=8 BW=16 | (4,1) / - | 0 mismatches |
| all 5, BH=8 BW=16 | (4,1) / (2,1) | **0 mismatches**, 163,185 bytes written |
| all 5, BH=16 BW=16 | (8,1) / (2,2) | **0 mismatches**, 326,390 bytes written |

`k_qkv_attn`'s window origin (+0x20 = (0,0)) is per dispatch. It is also correct at grid (2,2), so no per-tile
value is needed. **Update 8's conclusion is withdrawn: the C=512 translation is not shown to be defective.** This
was not run at the real 32x56 (28 two-D and 28 one-D workgroups). Nothing here suggests size-specific behaviour,
but that is an assumption.

**The host grid for the 1-D kernels is twice what the runner launches (read from the launcher; no frame run yet).**
The launcher at 0x180033660 receives rdi = &{&[rbp+0x604], ctx, &stage4, &stage5} (0x18002f972-0x18002f99c).
[rbp+0x604] = ctx[0x308] = ceil(H5/4)*ceil(W5/4) (0x18002e3ed-0x18002e430). All three 1-D launches use
grid ([rdi], 1, 1):
* `k_ffwd_inpview` (handle 0x1800663c8, grid at 0x18003375d-0x180033787)
* `k_conv_res_views` layer 1 (grid at 0x180033931-0x18003395b)
* `k_conv_res_views` layer 3 (grid at 0x180033fe1-0x18003400b)

The runner uses `grid_for(..., N512, per_wg=C512_WG=16384)` = 512*tiles*16/16384 = **tiles/2** (56 instead of
112 at 32x56). Each workgroup covers one 4x4 tile across 512 channels = 8192 bytes. `k_dec_upsample` already
uses the same 8192 B per workgroup and ctx[0x308]. With the host grid in `net_block512.py` (BH=8 BW=16, 1-D grid 8
instead of 4), the chain still matches the emulator exactly (0 mismatches), and it writes **261,119 bytes instead
of 163,185**. At the runner's grid, part of every 1-D output is never written. In the frame, that is about half of
the C=512 stage's tiles in 3 of 5 dispatches, across all 16 blocks (23-30, 40-47).

Runner change to test (no code change needed): `C512_WG=8192` gives `grid_for` exactly ctx[0x308]. If it scores
better, change the default at net_frame_full.py:518 from '16384' to '8192', citing the addresses above.

Still open behind that: Update 6's two remaining suspects (block 30's pooled-write layout, the identity of
ctx+0x228), and the unmapped 2-D paths of the launcher (0x1800336cc: grid ((tiles+3)/4, 8, 1);
0x180033b7b: ((tiles+3)/4, 4, 1)). Their handles have not been matched to kernels yet, so whether the runner's
(ceil(W/8), ceil(H/8)) for `k_ffwd2`/`k_qkv_attn` is right is a hypothesis to check next.

## Update 10: C512_WG=8192 changes nothing - grid.x is not what limits coverage here

Ran the real frame with `C512_WG=8192` (grid.x=112, matching the host per Update 9)
against the default `C512_WG=16384` (grid.x=56). **The rendered output is byte-for-byte
identical** (same SHA256, both runs) - not just close in `smid.py`'s score
(+0.5282 both times), the actual PNG bytes match exactly.

So doubling the 1-D kernels' workgroup count across all 16 C=512 blocks changed
nothing visible in the final frame. That rules out "half the tiles are simply
unwritten" as the mechanism, at least for the visible output - either:
* something downstream (a later block's dispatch, or the C=512->ViT repack)
  fully overwrites or recomputes whatever the extra workgroups would have
  filled in, so grid.x here is coverage-redundant rather than coverage-limited, or
* the kernel derives its own internal loop bound from H,W (baked into the
  kernarg) rather than purely from block_count_x, so launching more workgroups
  at the SAME H,W just means some of them redundantly recompute the same tiles
  rather than reaching previously-unwritten ones.

Either way, Update 9's C512_WG lead does not explain the mid-section's low
S_mid on its own. Runner reverted to C512_WG=16384 default (no observed
difference, and it matches the previously-passing net_block512.py baseline
grid more closely pending further investigation).

**Next, in order:** Update 6's two still-untouched suspects (block 30's pooled-write
layout into off_pooled/off_headb; whether ctx+0x228 is really c512_1[0] after
blocks 23-30) are now the most promising remaining leads, since the grid-count
and translation-correctness questions (Updates 7-10) have both come back clean
or inconclusive-but-inert.

## Update 11: the C=512 blocks run the VIT512_OLD kernels; the host's default path uses different ones

**Update 6's two leads are confirmed correct, with no change needed:**
* Block 30's pooled write matches. When the launcher's 6th argument r15 is nonzero (only for block 30:
  ctx+0x248, 0x18002f9c5-0x18002f9ca), step 5 takes the `k_conv_res_views` path (0x180033e95 -> 0x180033fce) with
  +0x38 = r15 and +0x40 = stage-5 (H6,W6) (0x180034089-0x1800340a2), grid (ctx[0x308],1,1). `k_final_head` +0x00
  reads the same ctx+0x248 (0x18002faea). The runner's `_pool30` dispatch matches field for field.
* ctx+0x228 is `c512_1[0]`. In each block the launcher writes it only through step 5's +0x18 (0x180033f2f, 0x180034042).
  The runner's step 5 writes `work[0]` there too.

**What is wrong is which kernels run.** Every branch in the launcher 0x180033660 tests the byte at 0x18009b208.
That byte is a static initialised to `atoi(getenv("VIT512_OLD"))`, or **0** when the variable is unset
(0x18003422d-0x180034275; the string is at 0x18007fd0d). With 0, every `test byte,N` is clear. Handles were
resolved from the registration table (`lea rdx,slot; lea r8,name`), not from earlier notes:

| step | runner / net_block512.py (flag bit set = VIT512_OLD path) | host default (flag 0) |
|---|---|---|
| 1 | `k_ffwd_inpview` (0x1800663c8) | **not dispatched** |
| 2 | `k_ffwd2`, grid (ceil(W/8), ceil(H/8)), +0x28 = 4 | `k_ffwd2` (0x180066370, launch 0x180033a4d), grid **((T+3)/4, 8, 1)**, +0x28 = **T** |
| 3 | `k_conv_res_views` | **`k_conv_res2`** (0x180066378, launch 0x180034216), grid ((T+3)/4, 4, 1) |
| 4 | `k_qkv_attn`, origin (0,0) | **`k_qkv_attn2`** (0x180066380, launch 0x180033e5e), 3-D grid, **shifted origins** |
| 5 | `k_conv_res_views` | **`k_conv_res2`** (launch 0x1800340fb via 0x180033fc2), grid ((T+3)/4, 4, 1); block 30 only: `k_conv_res_views` + pooled |

Here T = ctx[0x308] = ceil(H5/4)*ceil(W5/4). The grids are set at 0x1800336cc-0x180033721, 0x180033b7b-0x180033bd0
and 0x180033eab-0x180033f00. "First block" means r14 = the stage input, which is nonzero only for block 23
(0x18002f9e7-0x18002f9f0).

Host kernargs (default path):
* `k_ffwd2` (0x18003399c-0x1800339f4): +0x00 = first block ? **0** : ctx+0x228, +0x08 = first ? input : **0**,
  +0x10 = ctx+0x230, +0x18 = layer 0, +0x20 = (H,W), +0x28 = T.
* `k_conv_res2` step 3 (0x18003413e-0x1800341bd): +0x00 = ctx+0x230, +0x08 = first ? 0 : ctx+0x228,
  +0x10 = first ? input : 0, +0x18 = ctx+0x238, +0x20 = 0 (qword), +0x28 = layer 1, +0x30 = (H,W), +0x38 = T.
* `k_qkv_attn2` (0x180033c0f-0x180033cda): +0x00/+0x08 = ctx+0x238/+0x240, +0x10 = layer 2, +0x18 = (H,W),
  +0x20 = origin = table 0x180066410[(blk-23)%4] = (0,0), (-4,-4), (-4,0), (0,-4). Grid = ((W+7-ox)>>3, (H+7-oy)>>3,
  **16**) (constant (7,7,0,0) at 0x18006d570). The translated .s enables workgroup_id_z.
* `k_conv_res2` step 5 (0x180033f0d-0x180033f7b): +0x00 = ctx+0x240, +0x08 = ctx+0x238 (pshufd 0x4e swap),
  +0x10 = 0, +0x18 = ctx+0x228, +0x20 = first ? input : 0, +0x28 = layer 3, +0x30 = (H,W), +0x38 = T.

All three kernels have translations in `build/kernels-hw-scratch/`. Blocks 40-47 (the second C=512 stage) use the
same launcher, but their call site and first-block input were not read here. That is an assumption to check.
Applying this needs z-grid support: net_run.cpp:102 passes gz = 1, and `ka_for` writes block_count_z = 1.

## Update 12: the real C=512 kernels are wired (C512_HOST=1); k_qkv_attn2 produces 50% NaN

Wired the host's default C=512 path from Update 11's kernarg citations: `k_ffwd2` (same symbol,
new fields/grid), `k_conv_res2` (new symbol `_Z11k_conv_res211Conv2Params`, replaces
`k_conv_res_views` for steps 3/5 except block 30's pooled write, which is untouched), and
`k_qkv_attn2` (new symbol `_Z11k_qkv_attn210AttnParams`, replaces `k_qkv_attn`, needs a 3-D
grid). `ka_for` and both manifest builders now support an optional `gz` (3rd grid component);
`net_run.cpp` was rebuilt to pass it through (commit d35cff9).

**Verified against the kernel's own disassembly, not just Update 11's notes:** `_Z11k_qkv_attn210AttnParams.s`
loads `s[4:7] <- kernarg+0x00` (128-bit: the two activation pointers) and `s[4:7] <- kernarg+0x18`
(128-bit: H, W, ox, oy - matches the field layout exactly), and `s[0:1] <- kernarg+0x10` (the
weight pointer, single-ABI, no dispatch_ptr). The field offsets are right.

**But it produces exactly 50% `0x7f` (e4m3 NaN) at block 23, every time**, regardless of grid.y:

| QKV2_GY | result |
|---|---|
| 4 (my `(H5+7-oy)>>3` formula) | 50.00% nonzero, distinct=2 (0x00/0x7f 50/50) |
| 8 (matching k_ffwd2's own gy) | byte-identical: 50.00%, distinct=2, same split |

Its input (`work[2]`, written by the preceding `k_conv_res2` step) is healthy at that point (254
distinct values, sane distribution). So the input is not the problem, and grid.y is not the
limiting dimension either - two different grid.y values gave byte-identical broken output, which
rules out simple under-coverage in y. The grid.z=16 dimension (workgroup_id_z, the one genuinely
new mechanism here) is the remaining candidate; I did not have the budget to trace how the kernel
uses the z workgroup id/lane before this session's cutoff.

**C512_HOST now defaults to 0** (the old, known-working - if wrong-by-default per Update 11 -
path) so nothing renders the broken output unless explicitly requested. The scaffolding
(CONVV2, ATTN2_C512 symbols, `_c512_stage_old` fallback, gz support end to end) is real,
tested infrastructure and stays in the tree either way.

**Next:** read `_Z11k_qkv_attn210AttnParams.s` for how it uses the z workgroup id (likely
`s_load` of a hidden `workgroup_id_z`-related SGPR, or `v_mbcnt`/lane arithmetic combined with
it) and compare against the ORIGINAL gfx1100 disassembly at the same point, to see whether the
translation handles z correctly or whether the runner is supplying the wrong VALUE for something
z-dependent (not the grid size, per the gy test above - possibly gz itself is wrong, worth
sweeping GZ the same way GY was, or a kernarg field's semantics differ from what Update 11 assumed
for the z-tiled case specifically).

## Update 13: the k_qkv_attn2 NaN was k_conv_res2 reading its weights from address 1

**Cause (confirmed):** in C512_HOST's steps 3 and 5, +0x28 was packed as the integer layer index,
`(0x28, '<i', (1,))` and `(0x28, '<i', (3,))`. The host stores the return value of
`0x180031bc0(ctx, blk, layer)` there, which is the layer's weight pointer (0x180034193 -> 0x180034198 and
0x180033f51 -> 0x180033f56). The kernel uses it as an address: `_Z11k_conv_res211Conv2Params.s` loads +0x20..+0x3f
into s[48:55], then `s_add_u32 s74, s50, 0x40000` and `v_add_co_u32 v52, vcc_lo, s50, v0`. So step 3 ran on
garbage weights, and k_qkv_attn2 turned that input into 50% 0x7f. Fixed: +0x28 = `L(1)` / `L(3)`.

Block 23 trace after the fix: qkv_attn2 -> w3 is 49.99% nonzero with **215 distinct values**, and no 0x7f spike.
The ~50% zero fraction is shared by every step, ffwd2 through conv_res2_2, so it is not specific to qkv_attn2.

**grid.z = 16 is a genuine hardware dimension (read, not swept).** The translated prologue maps the z id into
s15 (`s_mov_b32 s15, s6`). The original uses s15 as a head index: `s_mul_i32 s37, s15, 0x60`,
`s_lshl_b32 s3, s15, 2` added to the kernarg base, `s15 << 13` (8 KiB per head) and `s15 << 5` (32 channels per
head). 16 x 32 = 512 = C. The kernel has no internal loop that stands in for z. The GZ sweep was not needed once
the cause was found, so it was not run.

**Score, same code, same frame, MID_HOST=1:**

| | S_mid | S_fine | lag8 |
|---|---|---|---|
| C512_HOST=0 (VIT512_OLD kernels) | +0.5282 | +0.6288 | +0.97 |
| C512_HOST=1 (host default kernels, fixed) | **+0.4770** | +0.5757 | +0.96 |

The host-default path now runs clean end to end: 157 dispatches, guards intact. It scores lower, not higher.
C512_HOST stays default 0. What remains unverified on this path:
* `k_conv_res2` and `k_qkv_attn2` have never been difftested against the emulator at any grid.
* The first-block (+0x08/+0x10/+0x20 input) wiring for stage 40-47 was not read from the host.
* Every C512_HOST=1 dispatch should be diffed against its Update 11 citation one field at a time. This update
  found one wrong field by reading; others of the same kind may remain.

## Update 14: k_qkv_attn2's grid.y default was wrong too, and it barely matters

Re-audited every C512_HOST=1 field against Update 11's citations by hand, since the +0x28
weight-pointer bug (Update 13) was mine and there was no reason to assume it was the only one.
Everything else checks out field-for-field (step 5's `+0x08 = ctx+0x238` note "pshufd 0x4e swap"
is how the two adjacent pointers get loaded together in the disassembly, not a value transform -
already correctly wired).

One more mismatch: Update 11 cites `k_qkv_attn2`'s grid as "constant (7,7,0,0) at 0x18006d570" -
a literal table read, not derived from geometry. The runner computed grid.y as
`(H5+7-oy)>>3 = 4` at H5=32, not the cited 7. Defaulted to the literal 7 instead (QKV2_GY stays
overridable).

**This barely moves the score:** S_mid +0.4819 (gy=4) -> +0.4858 (gy=7), S_fine +0.5757 -> +0.5779.
So it is a real citation-accuracy fix, worth keeping, but not what explains C512_HOST=1 scoring
below C512_HOST=0. The open item is unchanged from Update 13: k_conv_res2 and k_qkv_attn2 have
never been difftested against the emulator, and that is the only way left to tell whether their
own translation is correct - reading fields one at a time has now caught two real bugs but is
running out of new mismatches to find this way.

One thing worth flagging for whoever builds that difftest: gy=7 matches gx (also 7) at this
particular H5,W5=32,56. Whether that is because the constant genuinely does not depend on
frame geometry, or because it happens to equal ceil(W5/8) at every C=512 stage size this project
uses, was not read from the host and would fail silently at a different resolution.

## Update 14: stage 40-47 wiring read from the host - C512_HOST=1 reaches S_mid +0.711

The stage-2 call site (0x1800305c4-0x18003064d) and the launcher's step-5 argument reload show three wiring
errors in C512_HOST=1. Each was taken from a host read:

1. **Stage 2 has no first block.** Before the loop the host swaps ctx+0x228 <-> ctx+0x298, so ctx+0x228 becomes
   k_dec_upsample's output (0x1800305c4-0x1800305e0). It then calls the launcher with a 3rd arg (the stage input)
   of `xor r8d, r8d` = 0 for every block 40-47 (0x18003063a). The runner gave block 40 the first-block wiring.
   Now `first = bi == 0 and blocks[0] == 23`.
2. **Step 5's +0x20 is not the stage input.** Before step 5 the launcher reloads r14 from its 5th stack arg
   (`mov r14, [rsp+0x3b0]`, 0x180033e8d). That arg is 0 for all of stage 1 (0x18002f9fd) and ctx+0x2a0 at block 47
   (0x180030605-0x180030627). In k_conv_res2 it is an optional second output: `s_cmp_lg_u64 s[48:49], 0` plus a
   store address built from s48. The runner passed the stage input there at block 23, so block 23 wrote its
   output over the stage input.
3. **The decoder reads ctx+0x2a0** (0x180030849 -> [rbp+0x668]), block 47's second output, not ctx+0x228.
   New buffer `off_2a0`, written by block 47's step 5 and used as `stage_in` for the decoder when C512_HOST=1.

Score, same frame, MID_HOST=1 C512_HOST=1. Each row reverts one fix:

| | S_mid | S_fine | lag8 |
|---|---|---|---|
| all three fixes | **+0.7106** | +0.6844 | +0.98 |
| revert 1 (stage 2 first-block wiring) | +0.3888 | +0.4732 | +0.90 |
| revert 2 (block 23 +0x20 = input) | +0.6485 | +0.6580 | +0.97 |
| revert 3 (decoder from ctx+0x228, no 0x2a0 write) | +0.5080 | +0.5989 | +0.92 |
| before this update (Update 13) | +0.4770 | +0.5757 | +0.96 |
| C512_HOST=0 (old kernels) | +0.5282 | +0.6288 | +0.97 |

All runs had guards intact, and the full-fix result reproduces. **This is the first score above the input-blind
network's +0.651.** lag8 stays at 0.98, so the column-periodic structure is still there.

C512_HOST still defaults to 0. Flipping it is the obvious next change, but it waits on the net_block512_2.py
difftest below.

`net_block512_2.py` is new. It difftests k_ffwd2 -> k_conv_res2 -> k_qkv_attn2 -> k_conv_res2 on the GPU
against the gfx1100 original in the emulator. Kernargs are packed exactly as C512_HOST=1 packs them. The
emulator loop is 3-D, with ids in s14/s15 for ffwd2/conv_res2 and s13/s14/s15 for qkv_attn2, following each
translated prologue. It also has STEPS/FIRST/MODE/OUT2 switches.

**Difftest result (net_block512_2.py, BH=8 BW=16, block 23, first block, origin (0,0)):** every prefix has
0 mismatches.

| steps | wrote | mismatches |
|---|---|---|
| k_ffwd2 (grid 2,8,1) | 65,278 B | 0 |
| + k_conv_res2 (2,4,1) | 130,580 B | 0 |
| + k_qkv_attn2 (2,1,**16**) | 195,842 B | 0 |
| + k_conv_res2 | 326,365 B | 0 |

This run started before the +0x20 fix, so step 3's +0x20 pointed at the input slot. The optional second output
(slot 4 in "wrote") was therefore exercised as well. **The host-default C=512 translations are bit-exact at grid > 1,
including grid.z = 16.** Not run here: the real 32x56 size, MODE 1-3 (shifted origins) and FIRST=0.

## Update 15: (7,7,0,0) at 0x18006d570 is the rounding addend, not grid.y - Update 14's grid.y "fix" reverted

Read from the host (`version.dll`, capstone). The table is a `paddd` operand in `.rdata`, not a grid literal:

```
0x180033c26  movq  xmm0, [stage5+4]        ; (H, W)
0x180033c32  movq  xmm6, [0x180066410+i*8] ; (ox, oy)
0x180033c37  pshufd xmm0, xmm0, 0xe1       ; (W, H)
0x180033c3c  paddd xmm0, [0x18006d570]     ; + (7,7,0,0)
0x180033c44  psubd xmm0, xmm6              ; - (ox, oy)
0x180033c48  psrad/psrld/paddd/psrad 3     ; signed /8
0x180033c65  mov   dword [rsp+0x78], 0x10  ; gz = 16
```

So grid = ((W5+7-ox)>>3, (H5+7-oy)>>3, 16), which was the runner's original formula. At H5=32 that gives gy = 4 or 5,
never 7. Both C=512 stages go through the same launcher 0x180033660, so the formula holds at every stage size.
Update 14's "citation-accuracy fix" (the one about grid.y, lines above) misread the addend as the grid and is reverted.
`QKV2_GY` now defaults to `(H5+7-oy)>>3` and can still be overridden.

Score, MID_HOST=1 C512_HOST=1: S_mid **+0.7106**, S_fine +0.6844, lag8 +0.98. This is identical to gy=7: the extra
workgroup rows gy=7 launched lie past H5 and write nothing that counts. So the literal was harmless at this size, but
it was only right by accident.

## Update 16: k_export difftested for the first time - PASS at every mode

`net_export.py` (new) difftests `k_export` on the real GPU against the gfx1100 original in the emulator, sweeping
`+0x18` (mode 0-8), `+0x28` and `+0x38` (hist), 32x56 geometry with a non-matching input row stride (`+0x08 != W`)
and a real output pitch, so the packing net_frame_full.py actually uses is exercised, not a simplified stand-in.

**Result: PASS, 0 mismatches at every (mode, +0x28, hist) combination**, including `mode=0, +0x28=0, hist=0` -
the exact fields `net_frame_full.py`'s default (`EXPORT_MODE=0`, `EXP_HIST=0`) packs. The only non-PASS rows are
`+0x28=1` at modes 5/7/8 (f32 path, 16 B/pixel; only mode 0 is RGBA16F), which differ by up to 9 ULP in f32 words - `+0x28=1` takes the branch
that runs `v_exp_f32` (hardware-approximate transcendental, ~1 ULP per the ISA; the emulator computes it exactly),
consistent with the same tolerance already documented and accepted for other kernels in this project. Not a defect.

**`k_export`'s own translation is now verified**, closing the "never validated" item this file and
`net_frame_full.py`'s docstring both flagged. Whatever remaining defect produces the frame's colour cast and
tile pattern is not in `k_export`.

Also this update: `net_block512_2.py`'s three untested shift-window origins (MODE=1,2,3, i.e. `ENC_MODES[1..3]`)
now all PASS at 0 mismatches too, same as MODE=0 - all four origins of the C=512 stage are bit-exact.

## Update 17: post_block (writes ctx+0x100, the sole network-result buffer) still never difftested - emulator gap found

Ran `difftest_spec.py post_block` (existing Spec, never executed before). Crashes before producing a result:

```
File "gfx11emu.py", line 617, in swap
    w.next = s.addr2i[tgt]
KeyError: 48384   # = 0xbd00
```

The kernel's translated `.s` has a computed jump (`s_getpc_b64` + `s_add_u32 s4, s4, .Lpc_bd00-.Lgetpc_31538` +
apparently `s_swappc_b64`/`s_setpc_b64` reading s4:s5) targeting a label `.Lpc_bd00`, with two more near-identical
labels `.Lpc_1bd00`/`.Lpc_2bd00` elsewhere in the same file - looks like a 3x-duplicated code path. `s.addr2i`
(built from the loaded kernel's own decoded instruction stream) does not contain 0xbd00. No prior kernel in this
project's difftests exercised this computed-jump pattern, so this may be a genuine `gfx11emu.py` gap (addr2i not
covering everything the loader should have decoded) rather than a translation defect - not yet determined.
Handed to the terminal session for the disassembly-level dig.

## Update 18: post_block and k_pre_block_1h_32_fp8 difftested - Update 17's "emulator gap" was the fixture

Update 17's `KeyError: 48384` is not a gfx11emu.py gap and not a PC miscount. 0xbd00 is
`_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti`, the only non-kernel function in the image, which the pre and post
blocks call through `s_getpc_b64 / s_add_u32 / s_swappc_b64` (pre: 0x2fc94-0x2fcd4). `.Lpc_bd00` names that
original address. `kernelspec.emulate` loaded only the kernel's own symbol, so the target was not in `addr2i`.

Two fixture fixes in `kernelspec.py`:
1. Load `swin_layer` after the kernel (`prog[0]` is the entry, so the kernel must come first).
2. Kernels whose descriptor has `.amdhsa_user_sgpr_dispatch_ptr 1` (pre and post block) get a modeled AQL
   packet in s[0:1], with the kernarg in s[2:3]. Before, s[0:1] was the kernarg too, and these kernels read the
   workgroup size from the packet right before the call (post: 0x30eac, pre: 0x2fc2c `s_load_b64 s[0:1], s[0:1], 0x4`).
   The crash had hidden this. Kernels with dispatch_ptr 0 are unchanged (ffwd2 still PASS).

`difftest_pre.py` had the same entry-point bug (it loaded swin_layer and kernel together, and swin_layer comes
first in the file). It faulted at 0xbd08 on every run, including the committed version, so it had never
produced a result. Also fixed there: hidden args at +0x50/+0x5c/+0x90 per the metadata (they were packed at
+0x58/+0x64 in a 0x100 B kernarg), and the host scalars net_frame_full.py packs (+0x20=0.0625, +0x48=(1,1),
+0x40=null) as defaults.

| test (sandbox import-real-20260922) | result |
|---|---|
| difftest_pre 8x8 / 16x16 (seeds 1, 2) / 32x32 | PASS, 0 |
| difftest_pre 16x16 with +0x40 = random slot / +0x20=0 / old guesses (1.0; 0,0) | PASS, 0 (each output differs) |
| difftest_spec post_block_const | PASS, 0 |
| difftest_spec post_block (random e4m3 inputs) | FAIL, 3072 B: emulator writes f32 qNaN, GPU writes 0 |
| same, e4m3 NaN codes 0x7f/0xff removed from slots 0/1 | PASS, 0 |
| difftest_spec ffwd2 (dispatch_ptr 0 control) | PASS, 0 |

So the translations of both kernels are verified on finite inputs. The one divergence is how an e4m3 NaN input
propagates. Whether the emulator or the gfx1030 translation matches real gfx1100 hardware there cannot be
settled without that hardware. It only matters if the frame ever produces NaN activations.

## Update 19: env-var defaults now match the configuration every score in this file used

`C512_HOST` and `MID_HOST` both defaulted to 0 (the old, unverified paths) despite every S_mid/S_fine number in
this file - including the +0.7106 result in Update 14 - being measured with both explicitly set to 1.
`C512_HOST`'s own comment already claimed "=1 (default)" while the code said otherwise. A plain
`net_frame_full.py` run with no env vars was silently exercising kernels this file argues against, not the ones
it reports on. Both now default to 1; `C512_HOST=0`/`MID_HOST=0` still restore the old paths for comparison.

No other env-var default in this file was found inconsistent with FRAME_STATE's "Fixed this session" table
(`PRE_FULL`, `PRE_1H`, `DEC_LAST2`, `DEC_SKIP`, `DEC_PTRA`, `SKIP_PARITY`, `DEC_FLAGS` all already default to
their documented-correct value).

## Update 20: the encoder -> C=512 k_repack is not the host's - removing it lifts S_mid +0.711 -> +0.891

**Localisation.** `STRUCT_PROBE` with `IMPULSE_BASE` (flat 0.2 grey vs the same frame with five 16x16 squares at 32-px
cells (4,6) (4,40) (16,23) (26,10) (26,48)) maps |impulse - flat| per buffer under 7 candidate layouts and reports
`lift` = share of the change within the impulse neighbourhoods / their area (1 = scattered). The layout that gives a
compact footprint is the buffer's layout, and the first buffer where no layout does is where locality is lost.

* Layouts, now settled by footprint: encoder pools and decoder pongs are `nchw16c` (as documented); encoder ping/pong
  are 4x4-tile-major (`tile_c_pix`: [ty][tx][C][16 px]) - their old R^2 ~ 0 was the probe's layout, not lost signal;
  the C=512 buffers are `tile_c_pix` too (one 4x4 tile x 512 ch = 8192 B per workgroup).
* With the runner's default, lift is 5.7 -> 3.9 through the encoder (enc s4 pool 3.92) and **1.26 at c512_1 w0**, in
  every layout: the pixels are scrambled one dispatch after the clean pool. That dispatch was the encoder `k_repack`.

**Host read (`version.dll`).** Block 23's input is the launcher's 3rd argument, `r8 = [rbp+0x630]` only for
edi == 0x17 (`0x18002f9ea mov r8d,0` / `0x18002f9f0 cmove r8,[rbp+0x630]`), and `[rbp+0x630] = ctx+0x1f8[rbx]`
(`0x18002f846 mov rax,[r13+rbx*8+0x1f8]` / `0x18002f84e`) - the encoder's stage-4 pool. From 0x18002f800 to the
launcher call at 0x18002fa0b the k_repack handle (0x1800663e0) is never referenced. The only k_repack in the
"enc -> mid" phase is `0x18002fcfe`, after the C=512 stage and after k_final_head: the pre-ViT repack that MID_HOST
already runs. The phase table's "enc -> mid: k_repack, k_final_head" (VARPARAMS 609/2018) was read as a repack
before block 23; it is the one before the ViT.

**Result**, same frame, only the encoder repack dropped (`NO_REPACK=1`, now the default; `REPACK_ENC=1` restores it):

| | S_mid | S_fine | lag8 | c512_1 w0 lift |
|---|---|---|---|---|
| with encoder k_repack (before) | +0.7106 | +0.6844 | +0.98 | 1.26 |
| host: pool fed directly | **+0.8912** | **+0.8853** | +0.98 | **3.72** |

The rendered frame is now clearly the scene. Still wrong: dark square blocks roughly one C=512 cell (32 px) to two
cells in size, a blue/magenta cast in places, and the 8-px column periodicity (lag8 unchanged at 0.98). After the
ViT the footprint spreads (lift ~1.5 from c512_2 on), which global attention allows, so it does not by itself
point at a defect. `FIX_IN_LOAD` fixtures are k_repack outputs and now require `REPACK_ENC=1`.

## Update 21: k_import/k_export mode is the host's -1, not 0 - the dark blocks are gone, S_mid +0.891 -> +0.925

**Trace.** The 26 dark 16-px blocks (edges on a 16-px grid) were traced with `DEFECT_MASK` (per buffer, in its
Update 20 layout, statistics inside vs outside the blocks) and a per-encoder-block prefix trace
(`C512_TRACE=2 PROBE_ENC=1 DEFECT_MASK=...`): |x| inside/outside is x0.74-x1.28 through blocks 1-8, then **x3.72 at
block 9** (first block of stage 3, first saturated e4m3 codes) and **x9.43 at block 15** (first of stage 4). The input
at those spots is ordinary content (max luminance 2.6; all 27 blocks with luminance > 8 are elsewhere).

`ENC_EMU_CHECK=9` ran block 9's exact frame kernarg in the emulator on the GPU's pre-block-9 arena for the 6
workgroups covering the blocks plus 2 controls: **0 mismatches in 195,641 bytes**. The translation computes exactly
what the original does. Flags (1|4*last, 0x18002f717) and the weight records (last block of a stage carries the
pooling weights) match the host. So the cause was the input.

**Host read.** `k_import` +0x28 = `ebp` = the frame function's 9th argument (`[rsp+0x318]`, 0x18002d533 -> 0x18002d5ff);
+0x2c = `xmm7` = 1.0 (0x18006d2c8), overridden by a positive job value. At both first-frame call sites the 9th
argument is the global dword 0x18009a488 (0x180015add, 0x18001a880); the history path passes a literal 0
(0x18001a870). The constructor sets 0x18009a488 to **-1** (0x18001f26b) and no other direct write exists. In
k_import any mode != 0 tonemaps: `max(0, scale*x)`, `x/(1+x)` clamped, then the sRGB OETF (0xaa498-0xaa518). Mode 0
passes raw HDR (65.1 max here). The same `ebp` is k_export's +0x28, the v_exp (inverse) branch.

| IMPORT_MODE / EXPORT_MODE | S_mid | S_fine | lag8 | dark 16-px blocks | output range |
|---|---|---|---|---|---|
| 0 / 0 (before) | +0.8912 | +0.8853 | +0.98 | 26 | 0..1 |
| **-1 / -1 (host, new default)** | **+0.9247** | **+0.9258** | +0.98 | **0** | 0..47.9 (linear HDR) |
| -1 / 0 | +0.9718 | +0.9753 | +0.98 | 0 | 0..1 |

-1/0 scores higher only because its output stays in the tonemapped domain the PNG reference is compared in; -1/-1
is what the host does. The rendered frame is now clean: no dark blocks, no blue/magenta cast. **Open:** lag8 stays
at 0.98 in every variant, and whether an application ever overwrites 0x18009a488 before the first frame is not
settled (no direct write found; a block copy cannot be excluded statically).

## Update 22: the k_import mode is derived from the colour format (host value 1), and lag8 is not a stripe measure

**Correction to Update 21's mechanism.** Update 21 counted the frame function's stack arguments one short. With 8
pushes and `sub rsp,0x288`, `[rsp+0x318]` is the **10th** argument and `[rsp+0x320]` (the job) the 11th. So:

* `k_import` +0x0c = the 4th argument (`ebx = r9d`, 0x18002d400) = global 0x18009a488, the **source-format code**,
  written per frame at 0x180018ee4 from the colour texture's DXGI format (jump table at 0x180013ec7: RGBA16F 0,
  UNORM8 1, sRGB8 2, 10:10:10:2 3, R11G11B10F 4, RGBA32F 5, RGB9E5 6). Its constructor value -1 never reaches a
  frame. The runner's +0x0c = 0 was right.
* `k_import`/`k_export` +0x28 = the 10th argument = global **0x18009a850**, written per frame at 0x180015028:
  `mode = [rbp+0x240]` (0 for RGBA16F), `= 1 if (format & 3) == 0` (0x180014ffe: RGBA16F and R11G11B10F, the HDR
  float formats), `= 1 if [rbp+0x244]`, then overridden by INI **`[DlssNrOnAmd] Tonemap`** (`GetPrivateProfileIntA`,
  default -1, 0x180007ff6 -> 0x18009acf8) when >= 0.

The capture's colour texture is DXGI 10 (RGBA16F, `INPUT_CONTRACT_CYBERPUNK_FSR3.md`), so the host passes
**mode 1**. Both kernels only test the mode against 0 (k_import 0xaa434/0xaa48c, k_export 0xab778), so 1 and -1 are
byte-identical and Update 21's scores stand. Defaults are now `IMPORT_MODE=1`, `EXPORT_MODE=1`. The "does an app
overwrite -1" question is answered: the value is computed every frame from the texture format, plus the INI key.

**lag8.** `smid.py`'s lag8 is the lag-8 autocorrelation of the column-mean profile. Any smooth profile scores
near 1: the **input** itself is +0.98, the same as every output. After removing the smooth trend, the share of
the column profile periodic in x mod 8 is 0.02-0.04 % in input and output alike (random control 0.01 %). Row- and
column-wise spectra show no peak at periods 16, 8 or 4: output/input energy there equals the neighbouring
frequencies within 5 %, before and after the fixes. **There is no 8-px stripe pattern**; lag8 cannot detect one and
should not be read as a defect. Side observation: the output carries ~2.6x the input's fine-scale energy (4-6x
before the fixes), a contrast/detail question, not periodicity.

## Update 23: translator lowering was costing half the frame - GPU time 1.94x lower, output bit-identical

Profiling (`net_run.cpp` phase timers, `NET_RUN_VERBOSE=1`) showed three lowering choices in `translate_kernels.py`
dominating the ~560 ms of GPU time. Each is now a switch; the new values are the defaults.

1. **`TX_DELAY`** (default `none`, was `safe`): every RDNA3 `s_delay_alu` / `s_waitcnt_depctr` was lowered to
   `s_waitcnt_depctr 0` + `s_nop 7` - a full dependency drain plus 8 idle cycles, 6,225 times in the pre-block alone
   (32 % of its emitted instructions). RDNA2 interlocks ordinary VALU dependencies in hardware; they are dropped.
2. **`TX_WAITCNT`** (default `faithful`, was `full`): every `s_waitcnt` became a wait for all outstanding memory. The
   original counters are now kept. This is sound only while each translated instruction issues exactly as many VMEM
   and LGKM operations as its original: checked statically over all 19 frame kernels, **0 deviations** (the WMMA
   expansion ends in `lgkmcnt(0)`; the translator-inserted prologue has no memory ops).
3. **`TX_WMMA`** (default `batched`, was `serial`, in `translate_final_head.wmma`): each emulated
   `v_wmma_f32_16x16x16_f16` issued 64 `ds_bpermute_b32`, each followed by its own `s_waitcnt lgkmcnt(0)`. Now the 8
   permutes of one output register go to gather registers `scratch+16..+23`, one wait, then the 8 `v_dot2c_f32_f16` in
   the unchanged k order - same accumulation order, so bit-identical. VGPRs 210 -> 220 for WMMA kernels (still 4
   waves/SIMD; LDS already limits the pre/post block to one workgroup per CU).

**Correctness.** The full-frame post-run arena (all 1.4 GB, every intermediate buffer) hashes to `d54273b81de7cf88`
with the old kernels and with every variant, in 9 runs; S_mid/S_fine unchanged (+0.9247/+0.9258). Emulator
difftests with the new kernels: pre-block 16x16 and 32x32 PASS, post_block_const PASS, ffwd2 PASS, the C=512 chain
steps 0-3 at 8x16 MODE 1 PASS, k_export (8 combinations) unchanged. All 34 kernels rebuild; the 19 frame kernels
are byte-identical to the verified variant build. Old kernels: `build/kernels-hw-scratch-safe`.

**Timing.** Absolute numbers moved with the machine's state (the GPU was underclocked and driving Wallpaper Engine on
two monitors), so the comparison was run interleaved, old/new alternating, three rounds:

| kernels | round 1 | round 2 | round 3 | median |
|---|---|---|---|---|
| old (`safe`/`full`/`serial`) | 667 ms | 682 ms | 677 ms | 677 ms |
| new (defaults) | 378 ms | 349 ms | 329 ms | **349 ms (1.94x faster)** |

Per kernel (one earlier run, unloaded): pre-block 115.7 -> 70.8 ms, post-block 116.2 -> 57.8 ms, k_swin_var C=256
78.2 -> 37.3 ms, C=128/64/32 143.2 -> 64.3 ms, k_contract2 + k_qkv_attn2 50.8 -> 24.5 ms. The pre-block gained
nothing from WMMA batching and is now the largest single kernel.

**Clean absolute figure** (normal GPU clocks, Wallpaper Engine paused), interleaved, three rounds, hash identical in all
six: old **461.2 / 461.2 / 461.7 ms**, new **244.3 / 244.1 / 244.6 ms** - **1.89x**, spread +-0.3 ms. Breakdown of
the new 244.5 ms: pre-block 60.9 (25 %), post-block 49.6 (20 %), k_swin_var C=256 32.0, C=128 23.2, C=64 20.5, C=32
11.7, k_qkv_attn2 12.3, k_contract2 8.9.

## Update 24: WMMA emulation without LDS (v_permlanex16 + DPP row_share) - 244 -> 209 ms, bit-identical

The batched WMMA lowering (Update 23) still gathered every A operand through `ds_bpermute_b32`. Its address
`((lane>>4)<<2) + 8*j` reads lane `(lane>>4) + 2j`: lanes 0-15 take lane 2j, lanes 16-31 lane 2j+1. `TX_WMMA=dpp`
(now the default) builds, per A register, a vector whose row 0 is A and whose row 1 lane i holds A's row-0 lane i+1
(`v_permlanex16_b32` with selects `0x87654321`/`0xffedcba9`, merged by `v_cndmask` on mask `0xffff0000`), then each
`v_dot2c_f32_f16_dpp ... row_share:2j` reads lane 2j of its own row. Same operands and accumulation order, no LDS,
no waits: per WMMA 16 + 64 VALU instead of 64 LDS round-trips + 64 VALU. Needs 3 spare SGPRs; `k_conv_res2` has
none (106 used) and keeps the batched form automatically.

Verification: pre-block 8x8 and 16x16, post_block_const, ffwd2, k_swin_var C=32/64/128 (flags 0, 1, 4) and C=256
(flags 0, 1) PASS with 0 mismatches against the emulator; the full-frame 1.4 GB arena hash stays `d54273b81de7cf88`
in every run. The k_qkv_attn2 8x16 difftest and C=256 flag 4 were stopped for time; both kernels are covered at full
size by the frame hash.

Timing, interleaved, normal clocks: old 244.8 / 244.2 / 244.5 ms, new **208.9 / 209.2 / 210.4 ms** (-14 %).
Pre-block alone ~61 -> ~56 ms.

Also: the emulator is 1.25-1.7x faster (`0d9cf4b`: single-region memory fast path, table-lookup `v_perm_b32`,
cached exec mask), output hash-identical on 6 kernels (`emu_selfcheck.py`).

## Update 25: e4m3 encoder's power-of-two division becomes v_ldexp - 209.7 -> 206.6 ms, bit-identical

The float -> e4m3 encoder (inlined 212 times in pre/post block's swin_layer path) computes `e = floor(log2 x)`, builds
`2^e` with `v_ldexp_f32 vP, 1.0, e`, then divides `x / 2^e` with the full 10-instruction IEEE sequence
(`v_div_scale` x2 ... `v_div_fmas`, `v_div_fixup`). 2,980 of the pre-block's 2,982 executed divisions are this. The
encoder's guard makes the division exact: `v_cndmask vX, 0x43e00000, ...` clamps x <= 448, `v_cmpx_o` / `v_cmpx_neq 0`
drop NaN and zero, `v_cmpx_ngt_f32 0x3c800000, vX` keeps only x >= 2^-6. So 2^e and x/2^e are normal floats and the
quotient is exactly `ldexp(x, -e)`, in any denorm mode.

`TX_DIVPOW2` (default on) replaces a site with `v_sub_nc_u32 t, 0, e` + `v_ldexp_f32 vD, vX, t` only when all hold:
the divisor is `v_ldexp_f32 vP, 1.0, vE`; the division has exactly the IEEE shape; no branch target inside; the
448-clamp and 2^-6 guard precede it with x, e, 2^e unchanged since; and a CFG liveness analysis in the compiler's own
model (write kills, read uses, helper call/return modelled) shows every temporary the division wrote, vcc included,
dead afterwards. 94 of 212 sites in the pre-block and 91 in the post-block qualify; the rest keep the division
because some temporary is read later.

Verification: pre-block 8x8 and 16x16 and post_block_const PASS against the emulator; full-frame arena hash unchanged
(`d54273b81de7cf88`). Timing, interleaved, 3 rounds: 209.4 / 209.5 / 210.3 -> **206.5 / 206.6 / 206.7 ms** (-1.4 %).
Pre-block 54.6 -> 53.3 ms.

**Where this leaves the encoder:** its remaining cost is its own arithmetic (`v_log_f32`, floor, rounding, carry,
clamp) and the exec-mask branching around it. Translation-level cleanup has now removed the waits, the LDS gathers
and the redundant divisions; what is left is the original code's work, so further large gains need kernels rewritten
for RDNA2 rather than translated.

## Update 26: RDNA2-native e4m3 encoder spliced into pre_block/post_block - 206.6 -> 183.9 ms, bit-identical

The cloud track (rdna-2-cloud-work) designed a branch-free 11-VALU f16->e4m3 encoder (11 instructions vs ~40
VALU + 15 scalar/branch in the translated original) and verified it in isolation on the GPU: 0/65536 mismatches
over every f16 bit pattern (`kernels/e4m3/encode_fast.s`, `ENCODER.md`).

Splicing it into the real kernels turned out safer than first estimated. The old encoder is inlined 214 times
each in `k_pre_block`/`k_post_block` (not the ~212/~418 first guessed), and the concern that instances might be
interleaved by the scheduler was checked directly: only 1 of 213 gaps between consecutive instances is small
enough to suggest interleaving (median gap 300 lines); the other 213 are cleanly isolated, each bounded by
`v_cvt_f32_f16_e64 vTMP, |vIN|` through the matching `s_or_b32 exec_lo, exec_lo, sSAVE` that restores the
EXEC mask the old branchy code saved right after entry, with the result converging into one destination
register via two `v_or_b32` sites (normal-path and subnormal-path).

`perf/splice_encoder.py` (new) finds every clean site, extracts vIN/vOUT/the save register, picks a temp
register never referenced anywhere inside that site's span, and replaces the whole bounded region with the new
11-instruction sequence writing into the same vOUT the old code used - nothing outside the spliced span changes.
210 of 214 sites spliced in `k_pre_block`, 209 of 214 in `k_post_block` (the 1 interleaved pair, and a couple
that didn't match the exact save/restore pattern, left on the old path - negligible weight).

**Verification, in order:**
* `k_pre_block` difftest at 8x8, 16x16 (seeds 1, 2): PASS, 0 mismatches each.
* `k_post_block_const` difftest: PASS, 0 mismatches.
* `k_post_block` difftest (random inputs, includes the pre-existing e4m3-NaN divergence): identical failure,
  byte-for-byte, `out_sha256` byte-for-byte identical between the spliced and unspliced kernel - confirms the
  splice changes nothing about that already-documented, unrelated issue.
* Full frame (both spliced kernels installed): **arena hash `d54273b81de7cf88`** - byte-identical to the known-good
  reference from Update 25. GPU time **206.6 -> 183.9 ms** (-11.0%).

Not yet done: wiring the splice into the regular build pipeline (currently a manual post-process step on the
translated `.s`/`.co` - a from-scratch `translate_kernels.py` run would need it reapplied). The `.s`/`.co` files
themselves are gitignored build artifacts as always; `perf/splice_encoder.py` is the artifact that's committed.
