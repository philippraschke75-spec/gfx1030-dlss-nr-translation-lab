# k_swin_var launch contract read from the AMD port's host code (static; 2026-09-21)

Source: PE image embedded in `G:\dlss\dlssnr_on_amd_setup.exe` at file offset 0x47c00 (x86-64, .text 413,696 B, sha256 prefix b89fa08c…),
analysed with `llvm-objdump -d`. This is the third-party host that already launches the gfx1100 kernels. VAs are its image VAs.
Nothing here was run; it is code reading. Items marked (inferred) are deductions from the code.

## Per-block launcher `0x180032df0` (args: ctx, C, flags, in_ptr, [stack] out_ptr, weights_ptr, H, W, mode, ptrA, ptrB)
VarParams (424 B) as filled (offsets are struct offsets):
| off | value | evidence |
|---|---|---|
| +0x00 | input activation ptr (`r9`; first block of a stage uses the stage's initial buffer, later blocks the previous output) | caller 0x18002f727 |
| +0x08 | output activation ptr (buffers alternate: ping-pong) | 0x18002f693..6af |
| +0x10 | weight record ptr = return value of `0x180031bc0(ctx, block_index, 0)` (one WEIGHTS_HT record per block) | 0x18002f712/75b |
| +0x18 / +0x1c | stage H, W (i32) from the stage tuple (C,H,W) | 0x18002f6fd..706 |
| +0x20 / +0x24 | X, Y origin offsets from table `0x180066410[mode]`: mode0 (0,0), mode1 (-4,-4), mode2 (-4,0), mode3 (0,-4) | 0x180032e23..2e |
| +0x28 | flags = (block==first_in_stage) \| 4*(block==last_in_stage) (encoder path); callee also tests bit 3 (0x8) and mask 0x38 | 0x18002f717..736, 0x180032e6a, 0x180032f31 |
| +0x30 / +0x38 | two further pointers (stack args 10/11) | 0x180032fe8/ff0 |
| +0x40..+0x9f | zeroed (96 B) | xorps/movups run |
| +0xa0 | scratch buffer ptr (`ctx+0x180`, sized per workgroup: C=32 -> 4 or 8 KiB, 64 -> 8 KiB, 128 -> 16 KiB, 256 -> 32 KiB, times workgroup count) | 0x180032e53..ecf, 0x180032f1a |
Grid: x = (W - Xoff + 7)/8, y = (H - Yoff + 7)/8 (C-style truncating division), z = 1; block = 256 threads (0x100000100 dims constant).
Kernel choice: `(C-0x20) rol 27` jump table over C in {32,64,128,256} -> k_swin_var<32|64|128|256>; C=32 has a second entry when flags&8 (inferred: the <32,true>/<32,false> split).
Activation buffer size (bytes): C * ceil(H/4) * ceil(W/4) * 16 (0x18002f8xx) => 4x4-tile-major layout, 16 B per tile-channel? (inferred).

## Network schedule (loop at 0x18002f624, 4 encoder stages)
| stage | C | blocks | per-block modes |
|---|---|---|---|
| 1 | 32 | 1..4 | 0,1,2,3 |
| 2 | 64 | 5..8 | 0,1,2,3 |
| 3 | 128 | 9..14 | 0,1,2,3,0,1 |
| 4 | 256 | 15..22 | 0,1,2,3,0,1,2,3 |
Each block launches with its own record. This agrees with, and is now confirmed by, the size/extent topology in WEIGHTS_HT_FINDINGS.md.
The first/last-in-stage flag values (1 and 4) match the extents traced for flags 1 and 4 (records 20672 / 22720).

## Consequences for the fixtures used so far
* The (-4,-4) origin offsets used in earlier synthetic runs are the real shifted-window mode 1, not an arbitrary choice.
* Tests with flags 16/32/63 or non-zero +0x40..+0x9f pointers exercise code paths this encoder loop never sets. They remain useful
  for translation correctness but are not the deployed configuration. The decoder path (`swinup%d_C%d`, 0x180030968) is unread.
* Legit dims: H,W are stage sizes (network input divided per stage); H=7,W=9 is not a plausible real size.

## Still unread
Stage tuple derivation (ctx+0x190: C,H,W per stage), pre-block/post-block/import/export/reproject launchers, decoder stages,
flag_wait/flag_set protocol, meaning of +0x30/+0x38 pointers, ViT/1D stages, and exact tone/exposure/mvec/depth packing in `k_import`.

## Addendum: input-side launchers (same host image; static reading, 2026-09-21)
Network input size: the host pads the frame up to a multiple of 128 in both dimensions (`((x+127)/128)*128`, function 0x18002d3b0),
then `0x18002dd20(ctx, H, W)` builds the stage table: stage k has C = 32*2^k (k=0..5: 32..1024) and (H, W) halved (k+1) times with
truncating division, and allocates `H*W*12` B (float RGB network input) and `H*W*16` B buffers.
### ImportParams (0x30 B explicit; kernel `k_import`; launcher 0x18002d3b0; kernel prologue confirms field use)
| off | field |
|---|---|
| +0x00 | source image ptr |
| +0x08 | source row pitch (bytes, i32) |
| +0x0c | source format code (i32; kernel branches include R11G11B10F (case 4) and RGB9E5 (case 6); other codes 0..7 exercised) |
| +0x10 / +0x14 | source H, W (i32); out-of-range rows/columns are mirror-reflected (`2*H-r-2`) and clamped |
| +0x18 / +0x1c | padded H, W (i32) |
| +0x20 | destination ptr (float RGB, 12 B/pixel) |
| +0x28 | mode (i32; mode 0 = raw decode, modes >=1 apply a nonlinear transfer, output in [0,1]) |
| +0x2c | f32 scale (job exposure/intensity; default from a host constant, overridden by a positive job value) |
Grid = (ceil(padW/256), padH, 1), 256 threads. Hidden args start at +0x30 (block counts) with group size at +0x3c.
### PreParams (0x58 B; kernel `k_pre_block_1h_32_fp8`; launcher 0x18002ee.. -> handle 0x180066348) (partial)
+0x00 input float-RGB ptr (`ctx+0xf8`), +0x08 output ptr (`ctx+0x218`), +0x10 block-0 weight record ptr, +0x18 H,W (i32 pair),
+0x20 f32 (`ctx+0x34`), +0x24 i32 (`ctx+0x38`), +0x28 two f32 (`ctx+0x20`), +0x30 i32 0, +0x38 ptr (`ctx+0x220`), +0x40 ptr, +0x48 two 32-bit values (`ctx+0x28`).
Grid = (ceil(W/8), ceil(H/8)), 256 threads; scratch `ctx+0x180` sized 8 KiB * gridx * gridy. Meaning of the ctx floats/ints (tone, exposure,
jitter) is not yet identified.

## Correction/addendum: the real first block (pre-block) is `k_swin_var<32,true>` (launcher branch at 0x18002efb8; handle 0x18006e388)
The `PreParams` interpretation above (handle 0x180066348) is the *other* branch (`byte [0x18009b1f0] != 1`). The active-looking branch builds VarParams:
+0x00 = 0 (null), +0x08 out buffer, +0x10 block-0 weights (record `block0.layer0.layer`, 21,696 B), +0x18/+0x1c H,W, +0x20/+0x24 = 0,
+0x28 flags = 0x14 (16|4), +0x2c..+0x37 = 0, +0x38 ptr (`ctx+0x220`), +0x40 float-RGB input (`ctx+0xf8`, the k_import output), +0x48 ptr (optional),
+0x50 f32 (`ctx+0x34`), +0x54 f32x2 (`ctx+0x20`), +0x5c = 0, +0x60 f32x2 (`ctx+0x28`), +0x68 i32 (`ctx+0x38`), +0x6c unwritten padding,
+0x70..+0x9f zero, +0xa0 scratch. Verified consistent with the traced 12-byte loads at +0x40 and the 21,680 B extent. Which of the two branches is live
depends on a host boolean (unresolved); both kernels are translated.

### ExportParams (0x40 B; kernel `k_export`; launcher tail at 0x18002d7b4..d876, handle 0x180066400) (partial, static)
+0x00 ptr (`ctx+0x100`, network result buffer), +0x08 i32, +0x0c i32 (=host arg `[rsp+0x2f0]`, an image dimension), +0x10 i32 (`[rsp+0x2f8]`, the other dimension),
+0x14/+0x18 i32 (`[rsp+0x308]`, `[rsp+0x310]`), +0x20 ptr (`[rsp+0x300]`, destination surface), +0x28 i32 (format/mode), +0x30 ptr (`ctx+0xf8`, the
original float-RGB copy from k_import, i.e. the blend source), +0x38 f32, +0x3c f32 (job strengths). Grid = (ceil(dimA/256), dimB, 1), 256 threads.
Not yet run: needs a fixture with real semantic values; `k_export` decodability/translation status unchecked.

## Open item: the stage-1 -> stage-2 transition (C=32 -> C=64, spatial size halved) is NOT yet located (2026-09-22)

Ruled out by reading the AMD port host code:
* The per-block launcher's kernel-selection jump table (0x18006e6b0, keyed by `(C-0x20) rol 27`) only ever loads one of
  the five `k_swin_var` handles - no other kernel is reachable from inside a block dispatch.
* The encoder stage loop (0x18002f624..0x18002f92f) calls only the per-block launcher plus buffer-clear/logging calls
  (`0x1800321e0` with a size matching the activation-buffer formula) between stages; no extra kernel launch is inserted
  between a stage's last block and the next stage's first block in this loop.
* Block4 (last of stage 1) is dispatched as `k_swin_var<32,false>` with flags=4 only - the SAME channel width as blocks
  1-3 (confirmed by its WEIGHTS_HT record size matching the `<32,false>` extent in RESULTS_REAL_CONFIG.md), so it does
  not itself change the channel count.
* `k_repack` (RepackParams; complex integer div/mod index arithmetic, no arithmetic reduction - consistent with a pure
  relayout/space-to-depth reshape, a plausible mechanism for channel-doubling without an average-pooling loop) is
  called from a *different* host function (starting ~0x18002fba3, which begins by calling `k_final_head` and then loops
  over block indices 0x1f..0x27 = 31..38, the previously-identified 5-layer/ViT block range) - not obviously the
  stage-1/2 transition; this may be a different code path (e.g. a reduced-quality preset) rather than the main encoder.
* `k_mean` structurally contains an 8-element summation loop, but the DLL's own log string ("auto-exposure: encoded
  mean %.3f -> exposure %.4f") indicates it computes the frame's auto-exposure scalar, not a spatial downsample.

Not yet checked: whether the transition is handled inside `k_swin_var` itself via an unexplored flag combination (only
1, 4, 0x14, 16, 32, 63 have been exercised - see RESULTS_VAR.md/RESULTS_REAL_CONFIG.md), or by a host function this
disassembly excerpt does not cover. This blocks extending the real-pixel chain (RESULTS_REAL_CONFIG.md) past stage 1.

## Resolved: the stage-1 -> stage-2 transition is fused into the last block's own kernel call (2026-09-22)

A partner's independent reimplementation, [OpenDLSS-NR](https://github.com/maanHimself/OpenDLSS-NR) (Vulkan/NVIDIA,
not run or executed by us - only its documentation and source were read), describes the transition as a 2x2 box
pool of the last block's raw output (`((a+b)+(c+d))*0.25`, all in half) followed by a channel-doubling GEMM, and
its source shows this can be fused directly into the last block's own kernel dispatch as a second output pointer
(`pooledOutput` in `nr_graph.cpp`). Independent corroboration for that project's credibility: its documented window
phase origins `(0,0),(-4,-4),(-4,0),(0,-4)` and the tensor name `block70.layer0.blend_scale` match what we found
ourselves from the AMD port disassembly and the WEIGHTS_HT container, before either side could have seen the other's
work on those specific details.

This was then independently checked against OUR OWN translated kernel (`k_swin_var<32,false>`, host-only, emulator
only): `trace_block4_pool.py` shows that with `flags=4` (last-in-stage) and *only* then, the kernel writes to
kernarg **+0x38** - previously one of the "unclear semantics" pointer fields - in addition to its normal output at
+0x08. At H=W=32 the write is exactly 16,384 bytes; at H=W=16 the earlier documented activation-buffer formula gives
`C(64) * ceil(H(16)/4) * ceil(W(16)/4) * 16 = 16,384` - an exact match, confirming the buffer is C=64 (doubled),
H=W halved. `probe_real_transition.py` fed this real +0x38 output (real captured pixels, real block0-4 weights) into
`k_swin_var<64,false>` (block5) as its +0x00 input at H=W=32: it terminates cleanly and produces non-degenerate
output. The output's magnitude distribution (E4M3-decoded median ~96, some values at the 448 saturation ceiling) is
plausible but not yet confirmed correct - block5's own real flags/origin/mode and whatever else may be needed have
not been independently verified, and this was checked in the emulator only, not on the GPU.

**+0x30 remains unexplained** (not written in this trace for any tested flags value; may be a third output, an input,
or unused for the encoder path).

Next: verify this on the GPU (byte-exact against the emulator, the way every other stage in RESULTS_REAL_CONFIG.md
was), then extend the real-pixel chain (RESULTS_REAL_CONFIG.md) through blocks 5-8 (stage 2) using this mechanism.

## k_ffwd (FfwdParams): first real progress on the 512-wide/FFN kernel family (2026-09-22)

Read directly from the kernel's own gfx1100 prologue (`analysis/gfx1100-disassembly.txt`, entry 0x32200) and cross-checked
against the AMD port host launcher (function at 0x180033660, which builds this struct at [rsp+0x310] before dispatch):

* `s_load_b128 s[4:7], s[0:1], null` at kernel entry -> **+0x00 and +0x08 are two 64-bit pointers** (input, output).
  Confirmed by the host: `movups xmm0,[ctx+0x228]` (16 raw bytes = two pointers) copied straight into this slot.
* `s_load_b64 s[2:3], s[0:1], 0x10` -> **+0x10 is a weight pointer**, confirmed by the host calling the *same*
  per-block weight-lookup helper (`0x180031bc0`) that `k_swin_var`'s launcher uses, indexed by the block number.
* `s_load_b32 s2, s[0:1], 0x2c`, masked to 16 bits, then used as the denominator of a fixed-point reciprocal division
  -> **+0x2c is a channel-count-like scalar**. Leaving it zero (an untested guess) causes the kernel to spin forever
  on a broken division - this is why an earlier structural probe hung; setting it to a plausible value (32) fixed it.
* Kernel signature (host launcher args): `(ctx, block_index, optional_secondary_output_ptr, mode_flag)`; a null test on
  the third argument selects between `k_ffwd_inpview` and `k_ffwd` earlier in the same host function.
* `kernarg_size` = 288 for `k_ffwd`/`k_ffwd_inpview`, 304 for `k_ffwd2`; grid is 1D (`workgroup_id_x` only, unlike
  `k_swin_var`'s 2D grid) - expected for a per-token FFN with no spatial windowing.

Empirically probed (`trace_ffwd.py`, host-only, with +0x2c=32): the kernel now **terminates cleanly** (it previously
hung with +0x2c=0) and shows a coherent access pattern: input +0x00 fully read (8,192 B), output +0x08 fully written
(8,192 B, same size), weight +0x10 read sparsely up to ~512 KB. 8,192 B matches 256 tokens x 32 channels in FP8
(1 B/element) - consistent with a per-workgroup token tile, and with the OpenDLSS-NR reference's description of the
dense `32 -> 128 -> 32` FFN structure for narrow blocks. This is a structural/plausibility result (synthetic weights,
guessed channel count), not yet a GPU-verified or real-data-chained result.

**FfwdParams is now fully resolved for the fields that matter**: a kernarg-read trace (`trace_ffwd_kernarg.py`, hooking
every scalar load against the kernarg base directly, not just the arena pointer slots) shows the kernel reads *only*
+0x00 (16 B, the input+output pointer pair), +0x10 (8 B, weight pointer) and +0x2c (4 B, channel count) - nothing else
in the 288-byte kernarg is ever touched. There is no separate row/token-count field: the amount of work is derived
entirely from the dispatch grid size (1D, `workgroup_id_x` only), the same way `k_import`'s bounds come from its grid.

Still unknown: the real host values for block 23-30's launch (they use the "expert" grouped-FFN variant per the
reference, C=512 not 32, and the launcher's `test byte[0x18009b208],1` branch selects between at least two structurally
different FFN paths we have not both traced), and the launch contracts for `k_qkv_attn`, `k_conv_res`/`k_conv_res2`,
and the projection kernels this stage also needs.
