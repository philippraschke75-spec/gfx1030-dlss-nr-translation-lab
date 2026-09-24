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
