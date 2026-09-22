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

## k_qkv_attn (AttnParams): RESOLVED (2026-09-22)

Kernel descriptor enables **both** `dispatch_ptr` (s[0:1]) and `kernarg_segment_ptr` (s[2:3]) - the same convention
as the pre-block kernel, not the single-pointer convention `k_ffwd`/`k_swin_var` use. A modeled AQL dispatch packet
is required to test it at all (reused the construction from `difftest_pre.py`).

**Handle and host launcher, recovered from the actual host binary** (embedded PE in
`G:\dlss\dlssnr_on_amd_setup.exe` at file offset 0x47c00 - "MZ" header confirmed there; carved out and disassembled
directly with `llvm-objdump -d --x86-asm-syntax=intel`, since the outer setup.exe's own PE headers hide this
embedded payload from a direct disassembly pass). The registration table in `.rdata` pairs each kernel's name
string with its handle slot (`lea rdx,[handle_slot]; lea r8,[name_string]; call 0x180065930`): `_Z10k_qkv_attn10AttnParams`'s
name string lives at `0x18006ce25` and its handle slot is **`0x180066368`**. That handle is dispatched from the
same launcher function that already handles `k_ffwd`/`k_ffwd2` (continuing past `0x180033660`), specifically the
branch taken when `test byte[0x18009b208],0x4` is *not* set (the alternate branch dispatches handle `0x180066380`,
presumably `k_qkv_attn2`, the mangled `_Z11k_qkv_attn210AttnParams`).

**Kernarg construction at `0x180033c22`-`0x180033cda`** (this resolves and *corrects* the earlier partial finding -
the previous read-trace's "+0x18 (16 B, another pointer pair)" was a wrong inference; it's one 128-bit load
covering four 32-bit scalars, not two pointers):

| off | value | evidence |
|---|---|---|
| +0x00 | input ptr | `movups xmm0,[rsi+0x238]` -> `[rsp+0x170]` (kernarg base) |
| +0x08 | output ptr | same 16 B move (input/output pair, like `k_swin_var`) |
| +0x10 | weight ptr = `0x180031bc0(ctx, block_index, layer=2)` - this block's **layer2** tensor (`qkv_weight+attn_scale+attn_bias`, one combined blob per `DLL_HOST_EVIDENCE.md`'s layer-tensor naming) | `0x180033c9c..cac` |
| +0x18 | H (i32), from `ctx_sub+4` where `ctx_sub=[rdi+0x10]` | `0x180033cb4..cbb` |
| +0x1c | W (i32), from `ctx_sub+8` | `0x180033cc2..cc5` |
| +0x20 | window origin X (i32), from the same mode table `0x180066410[mode]` `k_swin_var` uses | `0x180033c2b..c44`, `0x180033ccc` |
| +0x24 | window origin Y (i32), table's second lane | `0x180033cd5..cda` |
AttnParams is **exactly 40 bytes** (`+0x00`-`+0x27`) - confirmed from the assembled `.s`'s own kernel metadata:
`.args: [.offset 0, .size 40, .value_kind by_value]`, followed by the compiler-inserted HSA hidden-args block
(`hidden_block_count_x/y/z` at `+0x28/0x2c/0x30`, `hidden_group_size_x/y/z` at `+0x34/0x36/0x38`, etc.) - see
below, this resolves what an earlier pass of this section wrongly treated as a fifth AttnParams field.

Grid: the launcher computes `(size - origin + 7) >> 3` per axis (0x180033c37-c5f) - the same 8-wide window-tile
formula as `k_swin_var`, confirming the reference's "window is 8x8 tokens."

**Why the kernel hung before, precisely diagnosed by reading its own body, not guessed:**
1. Feeding `+0x18`/`+0x20` a huge 64-bit arena pointer (as if it were a second pointer pair) instead of small H/W/
   origin integers made the kernel's internal window/grid arithmetic operate on astronomically large "sizes,"
   producing a loop that cycled with an exact ~13,000,000-step period and never converged even at an
   80,000,000-step budget (`trace_qkv_attn_zeroed.py`, `trace_qkv_attn_64thread.py` - both hypotheses tested and
   ruled out along the way: it wasn't NaN from random attention-shaped data, and it wasn't a workgroup-size
   mismatch, since zeroed data and a 64-thread retry reproduced the identical non-terminating pattern).
2. After fixing H/W/origin, a *second*, smaller bug appeared: `analysis/gfx1100-disassembly.txt` lines
   33761-33767 (`0x3392f0`-`0x39930c` region... i.e. `0x3930c`) show a tight loop `v1 += s12; branch back` that
   exits only once `v1 > 0x7fff`, where `s12` is `+0x34` masked to 16 bits (`s_load_b32 s8,s[2:3],0x34` /
   `s_and_b32 s12,s8,0xffff` right at kernel entry). Leaving `+0x34 = 0` made `v1` never advance - a genuine
   increment-by-zero infinite loop, the same failure class as `k_ffwd`'s original zero-divisor bug.

**Verified**: `trace_qkv_attn_fixed.py` with `+0x18=8, +0x1c=8` (one 8x8 window), `+0x20=+0x24=0` (phase-0 origin)
terminates cleanly in ~14,000,000 steps (330 s), all 8 waves finishing normally, zeroed weight/bias data. (This
script's very first working version filled `+0x34` with an arbitrary placeholder before the hidden-args nature
of that offset was understood - see below; the placeholder happened to be enough to avoid the increment-by-zero
hang described next, which is why the kernel terminated even though the value's *meaning* was still wrong.)

**Which real block dispatches this kernel, identified from weight sizes**: scanning `weights_ht_index.json` for
every `blockN.layer2.layer` size shows two distinct families - blocks 23-30 and 40-47 at 917,568 B
(`3*512*512 + 16 heads * 64*64*2` bytes = the C=512 QKV weight *plus* the reference's documented 64x64-per-head
`prior` bias term), and blocks 31-38 at 3,145,856 B (`3*1024*1024` bytes almost exactly, *no* extra prior-bias
bytes). This matches the reference's own stated distinction ("prior: learned 64x64 per head [window blocks] /
none [ViT blocks]") exactly: blocks 23-30/40-47 are `k_qkv_attn` (window attention, handle `0x180066368`,
resolved here), blocks 31-38 are the separate `k_qkv_attn2`/ViT variant (`_Z11k_qkv_attn210AttnParams`, handle
`0x180066380`, not yet investigated). Block23 is the first block of the C=512 stage and continues directly from
the already-GPU-verified encoder chain (block22's pooled output was C=512, H=W=4 - `RESULTS_REAL_CONFIG.md`).

**GPU hardware-verified with structural (random) data** (`difftest_qkv_attn.py`, real block23 `layer2.layer`
weight bytes extracted from `nvngx_dlssnr.dll`, H=W=4 matching block22's output, origin (0,0)): dispatched on the
real RX 6900 XT via the rebuilt `_Z10k_qkv_attn10AttnParams.co` module - **PASS, 0 mismatches**, 8,169 bytes
written to the output slot, 5.923 ms GPU time, arena guards intact. This is the same emulator-vs-hardware
differential method used for every other kernel in this file.

**`+0x34` is not a kernel-defined field - it's the compiler's HSA hidden-args block, resolved 2026-09-22:**
chaining real (not random) input activations initially produced a real, reproducible **7,840/8,192-byte GPU-vs-
emulator mismatch** (see `RESULTS_REAL_CONFIG.md`). The host launcher itself was searched for any explicit write
to the stack slot corresponding to kernarg `+0x34` on either dispatch path and found none, which was the right
signal that this offset isn't launcher-controlled at all. `statebisect_qkv_attn.py` (adapting the existing
`statebisect.py` bisection tool to this kernel's dual dispatch_ptr/kernarg_ptr ABI - it needed two fixes: the
dump prologue's "read the kernarg pointer" must use `s[2:3]` not `s[0:1]` for this ABI, and register comparison
must skip both pointer pairs, `s0-s3`, not just `s0/s1`) bisected the exact first-diverging instruction:
`s_load_b32 s8, s[2:3], 0x34` at kernel entry, where the emulator read whatever placeholder value the test had
written (e.g. 512) while real hardware read `0x10100` - decodes as two packed u16 values `(256, 1)`, exactly the
test's launch `workgroup_size_x, workgroup_size_y`. Cross-checked against `_Z10k_qkv_attn10AttnParams.s`'s own
`.args` metadata: `AttnParams` is `.offset 0, .size 40, .value_kind by_value` - **exactly 40 bytes**, and offset
`0x34` (52) falls inside the compiler-appended implicit-args block (`hidden_block_count_x/y/z` at `0x28/0x2c/0x30`,
`hidden_group_size_x/y/z` at `0x34/0x36/0x38`, `hidden_remainder_x/y/z` at `0x3a/0x3c/0x3e`, `hidden_global_offset_x/y/z`
at `0x50/0x58/0x60`, `hidden_grid_dims` at `0x68`) - fields HIP's runtime fills in from the real launch dimensions,
overwriting whatever bytes a manually-constructed kernarg buffer places there. There was never a "channel-count
scalar"; the earlier "+0x34" framing throughout this document's history was a misreading of a hidden-args offset
as a kernel parameter. **Fixed** by populating the emulator's kernarg with the correct hidden-args values matching
the real launch config instead of a guess - `gpu_verify_block23_real.py` re-run with this fix: **PASS, 0
mismatches**, real chained encoder-pipeline input, real block23 weight data, real hardware dispatch.

**`layer2` blob's internal sub-layout, derived by arithmetic (not yet disassembly-confirmed):** `DLL_HOST_EVIDENCE.md`
names `layer2` as `qkv_weight+attn_scale+attn_bias` combined into one blob. For `C=512`, `heads=C/32=16`: `qkv_weight`
(`3*C*C` bytes, e4m3 1 B/elem) = 786,432 B; `attn_bias`/the reference's "prior" (`heads*64*64*2` bytes, f16 64x64
per head) = 131,072 B; `attn_scale` (`heads*4` bytes, f32 per-head) = 64 B. Sum = 917,568 B - an **exact** match to
the real `block23.layer2.layer` size pulled from `nvngx_dlssnr.dll`. So the likely sub-offsets within the blob are
`qkv_weight` at `+0`, `attn_bias` at `+786432`, `attn_scale` at `+917504` (or `attn_scale` first at `+0` and
`qkv_weight` shifted by 64 - the order isn't pinned down, only the three sizes). Not yet used: the current GPU-
verified test copies the whole blob to the weight pointer undifferentiated, which is fine for a termination/
correctness-of-plumbing check but means the kernel's *output values* haven't been checked against a real reference
render - only that it computes *something* deterministic and matches the emulator bit-for-bit on real hardware.

**`k_qkv_attn`'s full contract is now closed**: 40-byte `AttnParams` + correct HSA hidden-args, real chained
input from the encoder, real block23 weight data, GPU hardware-verified bit-exact against the emulator (see
`RESULTS_REAL_CONFIG.md`). **Still open**: confirming the `layer2` sub-offset order (qkv_weight/attn_bias/
attn_scale - sizes are confirmed, ordering is not).

## k_qkv_attn2 (the ViT variant, `_Z11k_qkv_attn210AttnParams`, handle `0x180066380`): RESOLVED (2026-09-22)

Blocks 31-38 per the weight-size analysis above (no `attn_bias`/prior term, matching the reference's documented
ViT-vs-window-block distinction). Applied every lesson from `k_qkv_attn` up front rather than re-discovering
them:

- **Single kernarg-pointer convention**, not the dual dispatch_ptr/kernarg_ptr one - confirmed directly from its
  own prologue (`s_load_b128 s[4:7], s[0:1], 0x18`, `s_load_b32 s30, s[0:1], 0x34`: both read straight off `s[0:1]`,
  no `s[2:3]` involved, unlike `k_qkv_attn`).
- **AttnParams is 40 bytes here too** (`_Z11k_qkv_attn210AttnParams.s`'s own `.args` metadata: `.offset 0, .size 40`),
  with the identical HSA hidden-args block after it - populated correctly from the very first test this time,
  not guessed.
- **Exhaustive kernarg-read trace** (`trace_qkv_attn2_kernarg.py`): reads exactly `+0x00` (16 B, input+output
  pointer pair), `+0x10` (8 B, weight pointer), `+0x18` (16 B, H/W/originX/originY) - **only 3 real pointers**,
  confirming no `attn_bias` buffer is needed, exactly matching the reference's documented ViT/window-block
  difference. Terminated cleanly in 1.5 s with random data - no infinite-loop issue at all, since the hidden-args
  mistake that caused `k_qkv_attn`'s bugs was avoided from the start.

**GPU hardware-verified** (`difftest_qkv_attn2.py`, real `block31.layer2.layer` weight bytes extracted from
`nvngx_dlssnr.dll` - 3,145,856 B, large enough to span past one 1 MiB arena slot into the next two, which is
fine since nothing else uses those slots for this kernel): **PASS, 0 mismatches**, 2,037 bytes written, GPU
dispatch in 0.468 ms, guards intact. (First attempt used `H=W=2`, a guess at the post-block30 pooled spatial
size, and produced 0 bytes written - almost certainly a fully-masked/degenerate case from the token-padding
logic, not a bug; retried with `H=W=8`, a known-non-degenerate size, for this structural pass.)

**Still open for both attention kernels**: chaining real (not random) input activations past block22/30 (the
ViT's real input would need the C=512-to-1024 pooling/channel-doubling transition after block30 decoded first,
analogous to the encoder's fused-pool mechanism); the real spatial size the ViT actually operates at; and
confirming the `layer2` blob's internal sub-offset ordering for both kernels.

## k_conv_res is dead code; the real projection kernel is k_conv_res_views (RESOLVED, 2026-09-22)

The earlier structural pass on `k_conv_res` (handle `0x180066360`) PASSED (0 mismatches) but with an unresolved
pointer-order guess (`written_slots: [2]` contradicted the assumed input/output/weight ordering). Searching the
**entire host `.text` section** for that handle found only two references: the `.rdata` name-string
registration pair, and a single generic init-time "warmup" trampoline near the very start of `.text`
(`0x180001180`-`0x1800011dc` - a trivial, argument-free dispatch pattern shared by every registered kernel,
almost certainly a JIT/cache-priming pass at startup, not a real per-block call site). **`k_conv_res` is never
dispatched with real per-block arguments anywhere in this binary** - the earlier PASS was real (proves the
*translation* works) but tested a kernel this host build doesn't actually use for rendering.

The real per-block dispatch for a C=512 block's projection step, found continuing in the same orchestration
function that builds `k_ffwd`'s and `k_qkv_attn`'s kernargs (`0x180033f76`-`0x1800340e9`), uses handle
**`0x1800663d0`** = `_Z16k_conv_res_views12ConvPlParams` (`k_conv_res_views` - the `_views` partial-input-view
variant, consistent with the naming pattern `k_ffwd_inpview`/`k_conv_res_views` already known from
`DLL_HOST_EVIDENCE.md`). Full kernarg decoded directly from this launcher:

| off | value | evidence |
|---|---|---|
| +0x00 | pointer = `ctx+0x240` | `0x180034018..1f` |
| +0x08 | always 0 (null) | `0x180034027` |
| +0x10 | pointer = `ctx+0x238` | `0x180034033..3a` |
| +0x18 | pointer = `ctx+0x228` - **the real output**, per the GPU test below | `0x180034042..49` |
| +0x20 | `r14` (role not yet identified) | `0x180034051` |
| +0x28 | weight ptr = `0x180031bc0(ctx, block, layer=3)` - this block's **layer3** (`projection_weight+attn_cos_skip`) | `0x180034059..69` |
| +0x30 / +0x34 | H, W (i32) from `ctx_sub1+4/+8`, `ctx_sub1=[rdi+0x10]` | `0x180034071..82` |
| +0x38 | `r15` (role not yet identified) | `0x180034089` |
| +0x40 / +0x44 | a second H/W pair from `ctx_sub2+4/+8`, `ctx_sub2=[rdi+0x18]` - plausibly the *other* view's dimensions, matching the "views" name | `0x180034091..a2` |

`ConvPlParams` is exactly 72 bytes per its own `.s` metadata (`.offset 0, .size 72`) - an exact match to this
field count. `trace_conv_res_views_kernarg.py`'s exhaustive read trace confirms it: reads exactly `+0x00`
(32 B), `+0x20` (16 B), `+0x30` (16 B), `+0x40` (8 B), and `+0x54` (4 B = `hidden_group_size_x`, correctly
populated this time) - nothing else. Terminated cleanly in 1.9 s with random data.

**GPU hardware-verified** (`difftest_conv_res_views.py`, real `block23.layer3.layer` weight bytes at `+0x28`):
**PASS, 0 mismatches**, 8,163 bytes written, guards intact. Output confirmed landing at the `+0x18` pointer
(`ctx+0x228`), resolving the ordering ambiguity the earlier `k_conv_res` test left open. **All three kernel
types needed for one full C=512 block's forward pass are now individually resolved and GPU-verified**:
`k_ffwd`, `k_qkv_attn`, `k_conv_res_views`.

**Still open**: the roles of `+0x20`/`+0x38` (r14/r15 - possibly view/origin indices), which of `ctx+0x238`/
`ctx+0x240` is which view's input, and the real per-block orchestration order and skip-scale combination that
assembles a full block's output from these three kernels' results.

## k_expand (`_Z8k_expand12ExpandParams`): RESOLVED (2026-09-22)

Likely the ViT's FFN channel-expand step (`Cin=1024 -> 4096`, per `DLL_HOST_EVIDENCE.md`'s "Vit Cin=1024 (FFN
contract 4096)"). `ExpandParams` is only 24 bytes per its own `.s` metadata (`.offset 0, .size 24`) - confirmed
by its own prologue (`s_load_b128 s[4:7],s[0:1],null` for the input/output pair, `s_load_b64 s[8:9],s[0:1],0x10`
for the weight pointer) and by `trace_expand_kernarg.py`'s exhaustive read trace, which shows **only 3 pointers**
(no H/W, no other scalar) plus the hidden-args' `+0x24` (`hidden_group_size_x`, correctly populated - same
divisor-use pattern as `k_conv_res`/`k_qkv_attn2`).

Weight: `block31.layer0.layer` is 4,194,320 bytes, within 16 bytes of an exact `1024*4096` e4m3 match - the
strongest available candidate for this role (not yet confirmed via host-launcher tracing, only by size).

**GPU hardware-verified** (`difftest_expand.py`): **PASS, 0 mismatches**, 65,284 bytes written to the output
slot, 4.2 ms GPU dispatch, guards intact. The weight spans about 4 arena slots (1 MiB each), so this test uses
a wider 8-slot arena to keep it clear of the input/output slots.

## k_ffwd2 (`_Z7k_ffwd211Ffwd2Params`): RESOLVED (2026-09-22)

The second FFN dispatch in a C=512 block - the last unresolved kernel in that block's forward path.
`Ffwd2Params` is 48 bytes per its own `.s` metadata (`.offset 0, .size 48`), so the hidden-args block starts
at `+0x30` and `hidden_group_size_x` lands at `+0x3c`. Three independent sources agree field-for-field:

| off | value | evidence |
|---|---|---|
| +0x00 | pointer = `ctx+0x228` (the same buffer `k_ffwd` reads as its input) | `0x18003399c..a3` |
| +0x08 | pointer = `r14`; the host null-checks it (`test r14,r14`), so it is optional | `0x1800339ab` |
| +0x10 | pointer = `ctx+0x230` - **the output**, confirmed by the GPU test below | `0x1800339b3..ba` |
| +0x18 | weight ptr = `0x180031bc0(ctx, block, layer=0)` - this block's **layer0** | `0x1800339c2..cf` |
| +0x20 | H (i32) from `ctx_sub1+4`, `ctx_sub1=[rdi+0x10]` | `0x1800339d7..de` |
| +0x24 | W (i32) from `ctx_sub1+8` | `0x1800339e5..ec` |
| +0x28 | i32 from `[[rdi]]` - a **count of 16-token groups** (see below) | `0x1800339ef..f4` |
| +0x2c | padding (loaded as part of the 128-bit load, never used) | - |

The kernel's own prologue reads exactly this and nothing else - `s_load_b256 s[16:23], s[0:1], null` (the four
pointers), `s_load_b128 s[8:11], s[0:1], 0x20` (the four scalars), `s_load_b32 s2, s[0:1], 0x3c`
(`hidden_group_size_x`) - and `trace_ffwd2_kernarg.py`'s exhaustive read trace confirms it at runtime
(`+0x00`/32 B, `+0x20`/16 B, `+0x3c`/4 B, nothing else; terminated cleanly in 2.1 s).

**`+0x28` is a live count, not padding, and `0` is a silent no-op.** The kernel computes
`s29 = (field_0x28 << 4) - (workgroup_id_x << 6)` as its remaining-work counter, so the field counts
**16-token groups**. An emulator sweep at H=W=8 makes this exact: `n=1` writes 1,024 B, `n=2` writes 2,048 B,
`n=4` writes 4,096 B, and `n>4` writes the same 4,096 B - i.e. the effective count is `min(n*16, H*W)`, and
`4*16 == 8*8` is the saturation point. A first run with `n=0` dispatched happily on both the emulator and the
real GPU, agreed bit-for-bit, and wrote **nothing** - a vacuous pass. Any future fixture for this kernel must
set `+0x28` to at least `ceil(H*W/16)`, or it silently tests nothing.

Weight: `block23.layer0.layer` is 524,288 bytes = exactly `512*1024`, a clean C=512 -> 1024 FFN expand weight,
consistent with the host passing `layer=0` here.

**GPU hardware-verified** (`difftest_ffwd2.py`, real `block23.layer0.layer` weight bytes): **PASS, 0
mismatches**, 4,096 bytes written (4,084 differing from the random fill, the rest coinciding), guards intact.
The output landed in the `+0x10` slot, confirming `ctx+0x230` as the output - the same buffer `k_ffwd` writes,
so the two FFN dispatches share one destination.

**Emulator gap found and fixed on the way:** `s_abs_i32` was unimplemented and faulted the trace. Implemented
from the RDNA3 ISA's own definition (`D0.i = S0.i < 0 ? -S0.i : S0.i; SCC = D0.i != 0`), including the
`S_ABS_I32(0x80000000) => 0x80000000` wrap case, and checked against all six worked examples the ISA prints.

**Still open**: the role of the optional `+0x08` pointer, and how `k_ffwd`'s and `k_ffwd2`'s shared writes to
`ctx+0x230` compose (accumulate? disjoint channel ranges? two halves of one expand?).

## Seven more kernels resolved and GPU-verified via a shared spec harness (2026-09-22)

Hand-writing a ~75-line difftest per kernel was the bottleneck, so the boilerplate moved into
`kernelspec.py` (build kernarg, place pointers in arena slots, fill hidden-args, run the grid,
dispatch, diff) and each kernel is now one entry in `difftest_spec.py`'s registry recording only its
field layout. Two things the harness centralizes because both were repeatedly gotten wrong by hand:

* **Hidden-args offsets are derived, not guessed** - they begin immediately after the kernel's own
  explicit struct, whose size comes from the `.s` `.args:` metadata. Assuming a fixed offset is what
  cost a whole session on `k_qkv_attn`'s `+0x34`.
* **A kernel that writes nothing is a FAIL**, not a pass. `k_ffwd2` once "passed" while writing zero
  bytes. The harness now reports that explicitly as a vacuous test.

Ported first to `k_ffwd2`, `k_expand` and `k_conv_res_views`, which it reproduces (same written
slots, same emulator step counts). Newly resolved and **GPU-verified, 0 mismatches**, all with real
weight bytes:

| kernel | struct | layout | weight |
|---|---|---|---|
| `k_final_head` | 24 B | 3 pointers | `block70.layer0` - block70 is the last block and the only one with a `layer0.blend_scale` |
| `k_expand2` | 24 B | 3 pointers | `block31.layer1` |
| `k_dec_upsample` | 40 B | 5 pointers | `block48.layer0` - blocks 48-69 are the decoder path |
| `k_qkv` | 40 B | 5 pointers | `block31.layer2` = `1024*1024*3` + 128, exactly a ViT QKV weight |
| `k_contract2` | 48 B | 4 pointers + H/W + count | `block31.layer4` = `1024*1024` + 2048 |
| `k_conv_splitk` | 48 B | 4 pointers + H/W + count | `block31.layer4` |
| `k_repack` | 32 B | 2 pointers + four i32 | none |
| `k_ffwd_inpview` | 32 B | 3 pointers + H/W | `block23.layer1` |
| `k_export` | 64 B | 3 pointers + dims/mode + two f32 strengths | none |
| `k_mean` | 32 B | ptr, three i32, ptr at `+0x18` | none |

**The 40-byte structs are five pointers - but this does not generalize by load pattern.** For the
40-byte ones (`QkvParams`, `AttnParams1d`, `DecUpParams`), `s_load_b256` at `+0x00` then
`s_load_b64` at `+0x20` is exactly `5*8 == 40`. Reading `+0x20` as an H/W scalar pair made `k_qkv`
fault dereferencing `0x800000088` (the packed `8,8`) and made `k_attention` write into the weight
slot; as a fifth pointer `k_qkv` passes and writes **three** separate slots, which is what a QKV
projection should produce.

The 48-byte `ConvParams1d` pair (`k_contract2`, `k_conv_splitk`) has the *same* load pattern plus an
i32, but is **four** pointers with an H/W pair at `+0x20` and a count at `+0x28`. `k_contract2`
passes in that form and faults when `+0x20` is promoted to a pointer - the exact opposite of
`k_qkv`. I briefly "generalized" the five-pointer reading across both and turned a passing
`k_contract2` into a faulting one; the sweep caught it. **Field shapes are per-struct and have to be
confirmed against hardware individually, not inferred from a matching `s_load` sequence.**

**Weight-block map** (71 blocks): 0 pre-block, 1-22 encoder (C=32/64/128/256), 23-30 C=512 attention
(4 layers each), 31-38 ViT (5 layers, C=1024), 39 transition, 40-47 decoder-side attention,
48-69 the decoder/upsampling path (sizes shrinking as resolution grows), 70 the final head.

**Emulator gap found and fixed:** `global_atomic_cmpswap_b32` was unimplemented. Added from the
RDNA3 ISA definition (`tmp = MEM; src = DATA[31:0]; cmp = DATA[63:32]; MEM = tmp==cmp ? src : tmp`),
serializing lanes in ascending order so two lanes contending on one address cannot both observe the
original value.

**Fixture lesson - random bytes are not valid float input.** `k_mean` is a reduction, and random
bytes read as f32 span ~60 orders of magnitude and include NaNs, making summation order-dependent;
the emulator and hardware then disagree for reasons that are not translation defects. With
well-conditioned floats they agree exactly. The harness grew a `fill={slot: 'f32'|'zero'}` option
for this. `k_mean` still needs a scalar sweep to find a non-degenerate config.

**Still open**: `k_attention` (`AttnParams1d`) strides past a 6-slot arena, so it needs a larger
working buffer than its pointer count implies; and `k_mean`'s scalars need a sweep like `k_ffwd2`'s.

## k_mean: fixture solved, but a real emulator/hardware divergence remains open (2026-09-22)

`MeanParams` is 32 bytes: a pointer at `+0x00`, four i32 at `+0x08`/`+0x0c`/`+0x10`/`+0x14`, and a
second pointer at `+0x18`. A scalar sweep shows `+0x0c` and `+0x10` scale the emulator step count
identically (0 -> 782, 1/2/4 -> 868, 16 -> 1126, 64 -> 1838, 256 -> 4046), i.e. a symmetric H/W
pair, while `+0x08` and `+0x14` do not affect control flow at all.

Two fixture requirements, both of which silently produce a *vacuous* test if missed:

* **The input must be well-conditioned f32.** Random bytes read as f32 span ~60 orders of magnitude
  and include NaNs, which makes the summation order-dependent; the emulator and hardware then
  disagree for reasons that have nothing to do with translation.
* **The output must start zeroed**, because this kernel CAS-accumulates into it. Over random bytes
  it writes nothing whatsoever. With a zeroed accumulator the result scales exactly linearly with
  the workgroup count (0.6227 at 1x1, 1.2455 at 2x1, 2.4910 at 4x1), as a cross-workgroup
  accumulating reduction should.

**Open, reproducible divergence.** With correct fixtures the emulator and the real GPU still
disagree on the value: emulator 0.6227, hardware 0.1514, for a single workgroup with no concurrency
at all. Evidence gathered so far, which should save the next investigator the same steps:

* It is **not** a concurrency artifact - it reproduces at grid 1x1, and the hardware result is
  byte-identical at 1x1 and 4x1 while the emulator's scales with the grid.
* It is **not** the newly added `global_atomic_cmpswap_b32`. Instrumenting the arena shows exactly
  **one** write into the output slot, carrying the already-divergent value, so the wrong number is
  computed upstream of the atomic.
* It is **not** a mistranslation of the atomic: `translate_kernels.py` maps
  `global_atomic_cmpswap_b32` to RDNA2's `global_atomic_cmpswap` with identical operands, which is
  what the generated `.s` contains.
* It is **not** a cross-lane lowering problem: the kernel has no DPP, permlane, swizzle or bpermute
  at all. Its only lane-id op is a single `v_mbcnt_lo_u32_b32`, whose emulator implementation is
  correct for wave32.

So the defect is in the ordinary per-lane reduction arithmetic somewhere in this kernel's 257
instructions. `statebisect.py`'s methodology (binary-search the first divergent instruction against
saved hardware state) is the right next tool; it was built for exactly this and resolved the
`k_qkv_attn` case.

**Emulator additions this round:** `global_atomic_cmpswap_b32`, implemented from the RDNA3 ISA
(`tmp = MEM; src = DATA[31:0]; cmp = DATA[63:32]; MEM = tmp==cmp ? src : tmp; RETURN_DATA = tmp`),
serializing lanes in ascending order so two lanes contending on one address cannot both observe the
original value, and copying both source registers up front because the destination commonly aliases
the data pair (`global_atomic_cmpswap_b32 v0, v2, v[0:1]`).

## k_export: GPU-verified, completing the output path (2026-09-22)

`ExportParams` (64 B) needed no new decoding - the static host reading recorded earlier in this file
and the kernel's own load pattern agree field-for-field: pointer at `+0x00` (network result), i32 at
`+0x08`, four i32 from the `b128` at `+0x0c` (two of them the image dimensions), pointer at `+0x20`
(destination surface), i32 at `+0x28` (format/mode), pointer at `+0x30` (the original float-RGB copy
from `k_import`, i.e. the blend source), and two f32 job strengths at `+0x38`/`+0x3c`.
`hidden_group_size_x` lands at `+0x4c`, consistent with the 64-byte explicit size.

Both the network result and the blend source are float-RGB buffers, so they get well-conditioned
f32 fills rather than random bytes.

**GPU hardware-verified**: **PASS, 0 mismatches**, 128 bytes written into the destination-surface
slot, guards intact. This also exercises the restored `v_pack_b32_f16` fix on hardware - `k_export`
is one of the four kernels that contains a `v_pack_b32_f16` with a floating inline constant, the
case the old lowering got wrong.

With `k_final_head` and `k_export` both verified, the output path now has hardware coverage at both
ends, where it previously had none.

**Probed but not yet resolved:**
* `k_conv_res2` (`Conv2Params`, 64 B = eight 8-byte fields). Treating all eight as pointers faults,
  so some are scalars; a `k_conv_res_views`-shaped layout runs cleanly for 154k steps but writes
  nothing, so it needs a count sweep like `k_ffwd2`'s `+0x28`.
* `k_flag_set` needs `s_sendmsg(MSG_RTN_GET_REALTIME)` in the emulator - the realtime clock a
  spin-wait uses for backoff. `k_flag_set`/`k_flag_wait` are the inter-block sync primitives.
* `k_attention`, `k_attention2` and `k_conv_splitk` scan to the very end of whatever arena they are
  given (they faulted at exactly 6 MiB with a 6-slot arena and exactly 24 MiB with a 24-slot one),
  which looks like a scan for a sentinel that neither random nor zeroed data ever produces. They
  need a structured fixture rather than a larger buffer.
