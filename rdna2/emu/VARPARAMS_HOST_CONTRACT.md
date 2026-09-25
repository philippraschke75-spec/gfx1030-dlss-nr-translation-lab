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
branch taken when `test byte[0x18009b208],0x4` IS set (jne at 0x180033c09 -> 0x180033d2f). **Corrected
2026-09-24 (FRAME_STATE Update 11):** the bit-clear default branch dispatches `0x180066380` = `_Z11k_qkv_attn210AttnParams`
(registration table), and the flag byte is `atoi(getenv("VIT512_OLD"))`, 0 when unset (0x180034246-0x180034263).

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
* `k_conv_res2` (`Conv2Params`, 64 B) - **now resolved and GPU-verified, 0 mismatches.** Shaped
  like `ConvPlParams`: pointers at `+0x00`/`+0x10`/`+0x18`/`+0x28`, a null at `+0x08`, H/W at
  `+0x30`/`+0x34`. The sweep puts its count at `+0x38`: only that field enables any write at all,
  and it saturates at 4 for H=W=8 - the same `min(n*16, H*W)` rule as `k_ffwd2`'s `+0x28`. Output
  lands in the `+0x18` slot, matching `k_conv_res_views`. Treating all eight 8-byte fields as
  pointers faults, which is how the scalar positions were found.
* `k_flag_set` needs `s_sendmsg(MSG_RTN_GET_REALTIME)` in the emulator - the realtime clock a
  spin-wait uses for backoff. `k_flag_set`/`k_flag_wait` are the inter-block sync primitives.
* `k_attention` / `k_attention2` (`AttnParams1d`). The end-of-arena scanning was **my own layout
  error**, not kernel behaviour: as five pointers they read off the end of any arena, but
  `AttnParams1d` is **4 pointers + H/W** like `ConvParams1d`, with `+0x18` the output. In that form
  both run cleanly and write to the right slot - but they still **mismatch** (250 of 2023 bytes for
  `k_attention`, 248 of 2022 for `k_attention2`), so they are *not* verified.

  Characterising the mismatch: 88% of written bytes agree exactly, and the differing ones are far
  too large to be rounding (mean |delta| 62, max 199, with a pronounced cluster near 128 - i.e. sign
  bit flips in e4m3 bytes). WMMA is **not** the suspect: `k_qkv` (4 WMMA) and `k_contract2` (4 WMMA)
  both pass with the same software lowering, as does the fully verified `k_swin_var`. What
  distinguishes these kernels is transcendentals - `k_attention` has **328** exp/log/rcp/sqrt ops
  (softmax) against 37 in `k_contract2`. Two candidates remain, and they need separating rather than
  guessing: a genuine divergence in a transcendental's translated approximation, or fixture realism,
  since random bytes make softmax inputs extreme and e4m3's 4-bit exponent then quantizes
  near-boundary values to opposite sides. Realistic activations plus `statebisect` is the way in.

## Registry sweep baseline (2026-09-22, seed 1, real RX 6900 XT)

`for k in $(py difftest_spec.py --list); do ... difftest_spec.py $k 1; done` - a regression baseline
for the whole registry. **15 PASS / 3 FAIL**, and all three failures are investigated above rather
than merely observed.

| kernel | status | bytes written | out slots |
|---|---|---|---|
| align_probe | PASS | 2 | 0 |
| attention | **FAIL** (250 mism) | 2023 | 3 |
| attention2 | **FAIL** (248 mism) | 2022 | 3 |
| contract2 | PASS | 8159 | 2 |
| conv_res2 | PASS | 8167 | 2 |
| conv_res_views | PASS | 8167 | 2 |
| conv_splitk | PASS | 2038 | 2 |
| dec_upsample | PASS | 8168 | 2 |
| expand | PASS | 65306 | 1 |
| expand2 | PASS | 8157 | 1 |
| export | PASS | 128 | 1 |
| ffwd2 | PASS | 4083 | 2 |
| ffwd_inpview | PASS | 8167 | 1 |
| final_head | PASS | 16330 | 1 |
| mean | **FAIL** (4 mism) | 4 | 1 |
| qkv | PASS | 1529 | 1,2,3 |
| qkv2 | PASS | 6122 | 1,2,3 |
| repack | PASS | 8167 | 1 |

Run this before and after touching `gfx11emu.py`, `translate_kernels.py` or `translate_final_head.py`.
It already earned its keep once: it caught a `k_contract2` regression I introduced by generalizing
`k_qkv`'s five-pointer layout onto `ConvParams1d`, which does not share it.

**What this does and does not establish.** Each row is translation equality - the gfx1030 module
computes, bit for bit, what the gfx1100 interpreter computes for that fixture. It is *not* evidence
that the network's numerics are right: the reference is `gfx11emu.py`, not a real gfx1100 GPU, and
most fixtures use synthetic activations with real weights. Nothing here has rendered a frame.

## The whole-network driver: schedule recovered (2026-09-22)

The long-standing "still unread" items - decoder stages, ViT/1D stages, the post-block and
final-head path - are now mapped. The entire network is driven by **one function,
`0x18002ea60`-`0x180031465`** (~2,050 instructions, 139 calls to 29 targets). Its backward jumps
give the stage skeleton directly:

| region | range | what it dispatches |
|---|---|---|
| prologue | `0x18002ea60` | `k_pre_block_1h_32_fp8`, `k_swin_var<32,true>` |
| **encoder loop** | `0x18002f630`-`0x18002f929` | launcher A (`k_swin_var<C>`), 4 stages / 22 blocks |
| enc -> mid | `0x18002f929`-`0x18002fd80` | `k_repack`, `k_final_head` |
| **ViT loop** | `0x18002fd80`-`0x1800302d2` | `k_expand2`, `k_contract2` x2, `k_qkv2`, `k_attention2` |
| mid -> dec | `0x1800302d2`-`0x180030870` | `k_repack`, `k_dec_upsample`, launcher B x4 |
| **decoder loop** | `0x180030870`-`0x180030b18` | launcher A x2, launcher B x3 |
| epilogue | `0x180030b18`-`0x180031465` | `k_post_block_1h_32_fp8`, `k_swin_var<32,true>` |

Handle -> kernel names were recovered properly rather than guessed: each registration site loads the
handle into `rdx` and its mangled-name string into `r8`, so walking those pairs and reading the
strings out of `.rdata` (image base `0x180000000`) yields all **29** mappings.

### The two per-block launchers

**Launcher A = `0x180032df0`** - the `k_swin_var` dispatcher already documented at the top of this
file. It references the five `k_swin_var` handles (`0x18006e388`-`0x18006e3a8`) through the
`(C-0x20) rol 27` jump table plus the window-mode origin table `0x180066410`. Used by the encoder
**and** the decoder loop.

**Launcher B = `0x180033600`** - the attention/FFN block dispatcher, and the answer to the
"how does a C=512 block compose?" question this file has been circling. Its handle references in
address order give the per-block recipe:

| addr | kernel |
|---|---|
| `0x18003381e` | `k_ffwd_inpview` |
| `0x1800338ea` | `k_ffwd` |
| `0x180033a3b` | `k_ffwd2` |
| `0x180033b4a` | `k_conv_res_views` |
| `0x180033d23` | `k_qkv_attn2`  (ViT variant) |
| `0x180033e4c` | `k_qkv_attn`   (window variant) |
| `0x180033fc2` | `k_conv_res2` |
| `0x1800340e9` | `k_conv_res_views` |
| `0x180034204` | `k_conv_res2` |

So one block is `[k_ffwd_inpview | k_ffwd] -> k_ffwd2 -> k_conv_res_views ->
[k_qkv_attn2 | k_qkv_attn] -> [k_conv_res2 | k_conv_res_views]`, with the variant chosen by the
branches at `0x1800336a9`/`0x1800336ca`/`0x180033728`/`0x180033a73`. **Every kernel in that recipe
is GPU-verified** except `k_qkv_attn2`'s ViT sibling path.

**Why this matters:** it also explains the `ctx+0x230` vs `ctx+0x238` puzzle from earlier - those are
not two halves of one FFN, they are distinct stage buffers consumed by different kernels inside this
launcher, which is why no explicit copy between them was ever found.

**Still needed before an offline frame can run:** the per-stage loop bounds (which block indices each
loop covers), the buffer allocation/ping-pong layout across stages, and the `k_flag_set`/`k_flag_wait`
sync protocol. The kernels themselves are largely done; what is missing is the bookkeeping around them.

### Stage loop bounds (partial, 2026-09-22 - investigation was mid-flight)

**ViT loop - CONFIRMED, blocks 31..38.** The loop counter is initialised
`mov r15d, 0x1f` (=31) at `0x18002fd55`, compared `cmp r15d, 0x27` (=39) at `0x18002fd93`, and the
body indexes relative to it with `lea eax, [r15 - 0x1f]` at `0x1800302cb` before the back edge at
`0x1800302d2`. That is blocks **31-38 inclusive**, independently confirming the ViT block range that
was previously only inferred from the 5-layer weight-record shape.

**Decoder loop - 4 stages, like the encoder.** `cmp r9d, 0x4` guards both entry (`0x18003085a`) and
the back edge (`0x180030b14`), so `r9d` runs 0..3. Each iteration reads a **3-dword table entry**:
`lea rdx, [rax + 2*rax]` (i.e. `rax*3`) then `mov r10d, dword ptr [r8 + 4*rdx]` with the table base
at `[rbp+0x498]` (`0x180030878`-`0x180030883`). A 4-entry table of (C,H,W)-shaped triples mirrors
the encoder's own stage tuple table at `ctx+0x190`, so the decoder is almost certainly 4 stages with
per-stage width/size, walking the encoder's stages in reverse.

**Not yet read:** the encoder loop's own per-stage block counts are already documented at the top of
this file (4/4/6/8); the decoder's per-stage block counts come from that 3-dword table and have not
been dumped yet. That table dump is the obvious next step - it should turn blocks 48-69 into a
concrete per-stage schedule the same way the encoder's is.

## The complete 71-block schedule (2026-09-22, second session)

All 71 weight blocks are now accounted for, with the decoder resolved from two independent
directions that agree exactly.

| blocks | stage | C | dispatched by |
|---|---|---|---|
| 0 | pre-block | 32 | `k_swin_var<32,true>` (prologue) - **FAILS difftest, 28825; see end of file** |
| 1-4 / 5-8 / 9-14 / 15-22 | **encoder**, 4 stages (4/4/6/8) | 32 / 64 / 128 / 256 | launcher A |
| 23-30 | C=512 attention | 512 | launcher B (`k_qkv_attn` path) |
| 31-38 | **ViT** | 1024 | ViT loop: `k_qkv2`, `k_attention2`, `k_expand2`, `k_contract2` |
| 39 | transition | - | single 525,312 B record |
| 40-47 | C=512 attention | 512 | launcher B (`k_qkv_attn` path) |
| 48-55 / 56-61 / 62-65 / 66-69 | **decoder**, 4 stages (8/6/4/4) | 256 / 128 / 64 / 32 | launcher A + launcher B |
| 70 | final head | - | `k_final_head` (+ `layer0.blend_scale`) |

### How the decoder schedule was established

**From the code.** The decoder loop (`0x180030870`-`0x180030b18`) runs 4 iterations (`cmp r9d, 0x4`
at entry and back edge) and indexes the stage-tuple table with **`3 - r9d`**
(`mov eax,0x3` / `sub eax,r9d` / `lea rdx,[rax+2*rax]` at `0x180030870`-`0x180030878`), i.e. it walks
the encoder's stages **in reverse**. The table base `[rbp+0x498]` is loaded from
`lea rcx,[r13+0x190]` at `0x18002f5c7` - so it is **`ctx+0x190`, the very stage tuple table the
encoder uses**, which this file previously listed as unread. Each iteration reads a 3-dword
`(C,H,W)` triple into `[rbp+0x690]`/`[rbp+0x688]`/`[rbp+0x628]`.

Per-stage descriptors live in a stack array of 40-byte entries at `rbp+0x560`
(`lea r13,[8*rdi+0x560]`, `rdi = stage*5`), constructed at `0x18003067c`-`0x18003083b`, with
`[r13]`=first block, `[r13+4]`=end, `[r13+0x10]`=per-block mode array - the same shape the encoder
loop uses. The inner per-block loop (`0x1800309d0`-`0x180030a6a`) then calls **launcher A**
(`k_swin_var`) with `(C,H,W)` from the stage tuple, plus a **launcher B** path at `0x180030a8e`.

**From the weights, independently.** Grouping blocks 48-69 by `layer0` size gives
`1+7 | 1+5 | 1+3 | 1+3` = **8/6/4/4 = 22 blocks**, the exact reverse of the encoder's documented
`4/4/6/8`. The per-stage sizes are *identical* to the encoder's (689,232 / 197,184 / 61,760 /
20,672), and the odd-sized block sits at the **end** of each encoder stage (the fused downsample,
already documented) and at the **start** of each decoder stage (the upsample + skip concat):

| C | encoder odd block | decoder odd block |
|---|---|---|
| 256 | block 22, 820,288 | block 48, 820,784 |
| 128 | block 14, 229,936 | block 56, 230,176 |
| 64 | block 8, 69,936 | block 62, 70,048 |
| 32 | block 4, 22,720 | block 66, 22,784 |

Two independent lines of evidence - control flow and weight-record topology - agreeing on 8/6/4/4
in reverse order is strong. This is a U-net, and the decoder mirrors the encoder exactly.

**Remaining for an offline frame:** the buffer ping-pong layout (the decoder reads three per-stage
pointer arrays at `ctx+0x2a8` / `ctx+0x2c8` / `ctx+0x2e8`, indexed by stage - likely skip input,
working buffer and output), and the `k_flag_set`/`k_flag_wait` sync protocol. The schedule itself is
no longer the blocker.

## U-net skip connections: the wiring is determined (2026-09-22)

`ctx+0x2a8` is a 4-entry array of per-stage activation buffers, and the encoder and decoder index it
with *deliberately opposite* formulas, which is what makes the skip connections line up.

**Encoder** (loop head `0x18002f630`, stage counter `ecx` = 0..3):
* stage tuple indexed **forward**: `lea rcx,[rdx+2*rdx]` then `lea rbx,[rdx+4*rcx]` = `table + 12*stage`
  (`0x18002f678`-`0x18002f683`)
* buffer array indexed **reversed**: `mov eax,0x3` / `sub eax,ecx` / `mov rax,[r13+8*rax+0x2a8]`
  (`0x18002f630`-`0x18002f637`) = `ctx+0x2a8[3-stage]`

**Decoder** (loop head `0x180030870`, counter `r9d` = 0..3) - the mirror image:
* stage tuple indexed **reversed**: `mov eax,0x3` / `sub eax,r9d` / `lea rdx,[rax+2*rax]` = `tuple[3-r9d]`
* buffer array indexed **forward**: `mov edx,r9d` / `mov r15,[rcx+8*rdx+0x2a8]` (`0x1800308ac`-`0x1800308af`)
  = `ctx+0x2a8[r9d]`

Those two cross exactly, so each decoder stage reads the encoder buffer of **matching channel width**:

| C | encoder | buffer | decoder |
|---|---|---|---|
| 32 | stage 0 (blocks 1-4) | `ctx+0x2a8[3]` | iter 3 (blocks 66-69) |
| 64 | stage 1 (blocks 5-8) | `ctx+0x2a8[2]` | iter 2 (blocks 62-65) |
| 128 | stage 2 (blocks 9-14) | `ctx+0x2a8[1]` | iter 1 (blocks 56-61) |
| 256 | stage 3 (blocks 15-22) | `ctx+0x2a8[0]` | iter 0 (blocks 48-55) |

The decoder pulls two further per-stage pointers alongside it - `ctx+0x2c8[r9d]` and `ctx+0x2e8[r9d]`
(`0x1800308b7`, `0x1800308bf`) - and `ctx+0x2a8[r9d]` is passed to launcher A as its `in_ptr`
argument (`mov r9, r15` at `0x180030a56`), which is what confirms these are activation buffers
rather than queues or events.

Buffer **sizes** follow the already-documented activation formula `C * ceil(H/4) * ceil(W/4) * 16`,
with `(C,H,W)` coming from the `ctx+0x190` stage tuple, so a runner can allocate these itself without
reproducing the host's allocator. No store to `ctx+0x2a8` with a literal offset exists anywhere in
the image, so the array is filled through a pointer by a helper - worth knowing if the exact
allocation ever matters, but it does not for replaying the schedule.

**This removes the buffer-layout unknown.** What is left before an offline frame is the
`k_flag_set`/`k_flag_wait` sync protocol (and `k_flag_set` needs `s_sendmsg(MSG_RTN_GET_REALTIME)`
in the emulator), plus writing the runner itself.

## The flag sync protocol is not in the inference path (2026-09-22)

`k_flag_set` (handle `0x180066440`) and `k_flag_wait` (`0x180066430`) have been carried as an open
item since this file was started ("flag_wait/flag_set protocol" under Still unread). They are **not
needed for a forward pass.**

Their handles are referenced in exactly three places - `0x180005038`, `0x180019f1f`, `0x18001fec2`
(plus the registration sites) - and **none** is inside the network driver
(`0x18002ea60`-`0x180031465`), launcher A (`0x180032df0`) or launcher B (`0x180033600`), nor inside
any function the driver calls. Those addresses sit in unrelated early subsystems, so the flags
belong to some other path (interop/frame-pacing) rather than to block-to-block ordering.

Consequence for a runner: **dispatch the 71 blocks sequentially with ordinary host-side
synchronisation and ignore the flag kernels entirely.** This also retires the need for
`s_sendmsg(MSG_RTN_GET_REALTIME)` in the emulator, which was only wanted in order to run
`k_flag_set`.

### Status: the structural reverse-engineering is done

Every structural unknown this file has tracked is now closed:

* the 71-block schedule - **done** (encoder 4/4/6/8, C=512 attention, ViT 31-38, decoder 8/6/4/4)
* per-block kernel composition - **done** (launcher A and launcher B recipes)
* the stage tuple table - **done** (`ctx+0x190`, indexed forward by the encoder, reversed by the decoder)
* U-net skip wiring - **done** (`ctx+0x2a8`, opposite indexing on each side)
* buffer sizes - **done** (`C * ceil(H/4) * ceil(W/4) * 16` from the stage tuple)
* flag sync - **done** (not in the inference path)

What remains for an offline frame is **engineering, not reverse-engineering**: a runner that
replays the schedule on the GPU using the verified kernels, fed by the captured Cyberpunk frame.
The honest caveats still stand - `k_attention`/`k_attention2` mismatch and sit in the ViT path, so a
first frame would be expected to be wrong in that region until that is resolved.

## k_export: contract corrected from the launcher, and why it cannot be tested standalone (2026-09-22)

The earlier `ExportParams` entry was a partial static reading, explicitly flagged unverified. Tracing
the launcher tail properly (`0x18002d7b4`-`0x18002d888`, dispatch of handle `0x180066400` at
`0x18002d876`) confirms the layout and **corrects the two dimension fields, which were the wrong way
round**:

| off | value | evidence |
|---|---|---|
| +0x00 | ptr = `ctx+0x100` - the **network result** buffer | `0x18002d7ca`-`0x18002d7d1` |
| +0x08 | i32 `edi`, materialised from a SIMD register (`movd edi, xmm10`, `0x18002d514`) | `0x18002d7d9` |
| +0x0c | i32 = `[rsp+0x2f0]` - **height** (this is grid.y verbatim) | `0x18002d7e0`-`0x18002d7e8` |
| +0x10 | i32 = `r15d` = `[rsp+0x2f8]` - **width** (this is the value chunked by 256 for grid.x) | `0x18002d6fa`, `0x18002d7f0` |
| +0x14 / +0x18 | i32 from `[rsp+0x308]` / `[rsp+0x310]` | `0x18002d7f8`, `0x18002d7ff` |
| +0x20 | ptr = `[rsp+0x300]` - destination surface | `0x18002d806` |
| +0x28 | i32 `ebp` = `[rsp+0x318]` - format/mode | `0x18002d80e`, `0x18002d533` |
| +0x30 | ptr = `ctx+0xf8` - the float-RGB copy from `k_import`, i.e. the blend source | `0x18002d815`-`0x18002d81c` |
| +0x38 / +0x3c | f32 `xmm6` / `xmm7` - job strengths | `0x18002d824`, `0x18002d82d` |

The grid is built at `0x18002d761`-`0x18002d789` as `(ceil(r15d/256), [rsp+0x2f0], 1)`, and those two
values are exactly the fields at `+0x10` and `+0x0c`. That is what fixes the ordering: **`+0x0c` is
the height and `+0x10` the width**, the reverse of the earlier note.

**Why a passthrough test is not possible.** Feeding the imported float-RGB image into `+0x00` and
running `k_export` on its own was tried across both resolutions, several modes and several `+0x08`
values: coverage never exceeded ~10% and the surface was always non-finite. That is not a parameter
problem. `+0x00` is the *network's* output, which arrives in the network's own tiled/quantised
format from `k_final_head` - float RGB there is simply the wrong data, so the kernel reads nonsense
and writes inf/NaN (and because "coverage" counts non-zero bytes, a surface full of f16 NaN patterns
reads as low coverage even though it was fully written).

So `k_export` cannot be validated semantically until the network actually produces output. Its
**translation** is already verified independently - it passes the registry difftest with 0
mismatches - which is the part that was ever in question for the gfx1030 port. What remains is
supplying it real input, which means finishing the middle of the network, not further probing here.

## Correction + decode: the real attention/FFN block launcher is 0x180033660 (2026-09-22)

**Correction to the entry above.** `0x180033600` is *not* the block launcher. It is a five-line
wrapper ending at `0x18003365e` that just calls a formatter (`0x18003f194`) with a printf-style tag.
The driver's "launcher B" call sites are really label formatting; the tags recovered from them are a
useful structural map in their own right:

| tag | where |
|---|---|
| `enc%d`, `ds%d`, `swin%d_C%d` | encoder + downsample |
| `dec%d`, `swinup%d_C%d`, `swin%d_C%d` | decoder + upsample |
| `b%d`, `b%dh`, `b%dq`, `b%do`, `b%dx1` | the C=512 attention blocks and their variants |

The **real** attention/FFN block launcher is **`0x180033660`** (0x348-byte frame, and the function
that actually contains the `k_ffwd` / `k_ffwd2` / `k_qkv_attn` / `k_qkv_attn2` / `k_conv_res_views` /
`k_conv_res2` / `k_ffwd_inpview` handles). It has exactly **two** call sites, and they are the two
C=512 attention stages:

### Blocks 23-30 (`0x18002f9c0`-`0x18002fa15`), fully decoded

```
mov  eax, 0
cmp  edi, 0x1e            ; block == 30 (the last one)?
jne  +7
mov  rax, [r13 + 0x248]   ;   then pass ctx+0x248 as the 6th argument
lea  ecx, [rdi - 0x17]    ; ecx = block - 23
mov  edx, ecx
sar  dl, 7 / shr dl, 6 / add dl, cl / and dl, -4 / sub cl, dl
movsx r9d, cl             ; r9d = (block - 23) % 4   <- signed-modulo idiom
cmp  edi, 0x17            ; block == 23 (the first one)?
mov  r8d, 0
cmove r8, [rbp + 0x630]   ;   then pass that buffer as the 3rd argument
mov  [rsp+0x28], rax      ; 6th arg: ctx+0x248 on the last block, else 0
mov  [rsp+0x20], 0        ; 5th arg: always 0 here
mov  rcx, rbx             ; 1st arg: ctx
mov  edx, edi             ; 2nd arg: block index
call 0x180033660
inc  edi
cmp  edi, 0x1f            ; loop while block < 31
jb   ...
```

So the signature is
`launcher(ctx, block_index, first_block_input_or_null, mode, 0, last_block_extra_or_null)` and
**`mode = (block - 23) % 4`** - the same four-mode shifted-window cycle the encoder uses, confirming
these blocks are windowed attention rather than a flat sequence.

### Blocks 40-47 (`0x180030640`)

Structurally identical, with `lea ecx, [rdi - 0x28]` - i.e. `mode = (block - 40) % 4`. Same
launcher, same argument shape. One decode covers both stages.

**What this gives a runner:** the per-block index, window mode, and the first/last-block special
buffers for both C=512 stages. Combined with the per-block kernel recipe and the already-documented
kernarg field-to-ctx-buffer mappings for each of those kernels, the remaining gap is narrow - the
per-block H/W and the count fields (the `+0x28`/`+0x38`-style token counts that `k_ffwd2` and
`k_conv_res2` need, whose semantics are already known: `min(n*16, H*W)`).

## The frame-level pipeline, and why a first frame needs no depth or motion vectors (2026-09-22)

The 71-block network driver is not the top of the pipeline. It is called from an **outer
frame-level function** which also owns the input and output kernels. Dispatches in address order:

```
0x18002d64f   k_import
0x18002d6f2   -> call the network driver (71 blocks)
0x18002d876   k_export
0x18002d983   k_mean
0x18002dced   k_reproject
0x18002dd15   -> call the network driver (71 blocks)
```

There are **two paths**, and the distinction matters a great deal for what a first render requires:

* **Colour-only path**: `k_import` -> network -> `k_export`. No depth, no motion vectors.
* **Temporal path**: `k_mean` -> `k_reproject` -> network. This is where motion vectors and depth
  enter, via `k_reproject` (`ReprojParams`, 128 B, still unverified).

`INPUT_CONTRACT_CYBERPUNK_FSR3.md` lists "how depth/motion vectors are packed for the network" as an
open unknown, and it has been carried as a blocker for an offline frame. **It is not one for a first
frame**: reprojection sits on the temporal branch, so a reset/first frame runs the colour-only path,
and `k_import` is already GPU-verified on the real capture at full resolution.

This removes the last item that could have been an unbounded surprise. What remains between here and
a full-network render is enumerable:

1. the internals of the attention/FFN launcher `0x180033660` for blocks 23-30 and 40-47 (call sites
   and per-block mode already decoded; the per-kernel kernarg field layouts are all documented)
2. the ViT loop's per-block wiring, blocks 31-38
3. the decoder's per-block wiring, blocks 48-69 (launcher A, the same one the encoder uses)
4. the block 39 transition
5. extending the encoder run from stage 1 to all four stages (the pattern is proven and bit-exact)
6. whole-network buffer allocation and the `ctx` pointer roles
7. `k_attention`'s mismatch - a correctness problem for the ViT region, not a blocker on getting a
   render to happen

None of these is an unknown unknown; each is bounded reverse-engineering or integration against
kernels that are already verified.

## The C=512 block recipe, with weight layers and buffers (2026-09-22)

Decoding inside `0x180033660` gives the full per-block sequence, and it maps exactly onto the four
weight layers each of blocks 23-30 and 40-47 carries:

| step | kernel | weight layer | role |
|---|---|---|---|
| 1 | `k_ffwd_inpview` (first block of the stage) **or** `k_ffwd` | 0 | FFN expand |
| 2 | `k_ffwd2` | 0 | FFN expand, second half |
| 3 | `k_conv_res_views` (`0x180033b4a`) | **1** | FFN contract + `ffn_cos_skip` |
| 4 | `k_qkv_attn` **or** `k_qkv_attn2` | 2 | QKV / attention |
| 5 | `k_conv_res_views` (`0x1800340e9`) / `k_conv_res2` | **3** | projection + `attn_cos_skip` |

This is why the launcher contains `k_conv_res_views` **twice**: once after the FFN with layer 1 and
once after attention with layer 3. The earlier entry in this file documented only the layer-3 site
and therefore read as if there were a single projection step.

The layer numbering now lines up with `DLL_HOST_EVIDENCE.md`'s tensor naming end to end:
layer0 = FFN expand, layer1 = contract + `ffn_cos_skip`, layer2 = `qkv_weight+attn_scale+attn_bias`,
layer3 = `projection_weight+attn_cos_skip`.

### Variant selection

* **`k_ffwd_inpview` vs `k_ffwd`**: `test r14, r14 / je` at `0x180033741`-`0x180033744`. `r14` is the
  launcher's 3rd argument, which the caller sets with `cmove` only when `block == first_of_stage`.
  So the input-view variant runs exactly once per stage, consuming the stage's incoming buffer;
  every other block uses plain `k_ffwd`.
* A further branch on `dword [rsi+0x98]` against 3 (`0x180033a6c`, `0x180033a92`) selects between
  the remaining paths.

### `k_ffwd_inpview` kernarg, built at `0x180033794`-`0x1800337d1`

| off | value |
|---|---|
| +0x00 | `r14` - the stage's incoming activation buffer |
| +0x08 | `ctx+0x230` |
| +0x10 | weight ptr, `0x180031bc0(ctx, block, layer=0)` |
| +0x18 / +0x1c | H / W from `[rdi+0x10]+4` / `+8` |

**This independently confirms today's empirical decode of `FfwdPlParams`**, which was found by
probing (three pointers plus H/W, because treating it as four pointers faulted). The host builds
precisely that layout - a static confirmation of a result that had only been established by
experiment.

### First `k_conv_res_views` kernarg, built at `0x180033aa4`-`0x180033b02`

| off | value |
|---|---|
| +0x10 | `ctx+0x228` |
| +0x18 | `ctx+0x238` |
| +0x20 | 0 |
| +0x28 | weight ptr, layer **1** |
| +0x30 / +0x34 | H / W |
| +0x38..+0x47 | zeroed (`xorps`/`movups`) |

Note the zeroed tail: the count field at `+0x38` that `k_conv_res2` needs is **0** here, consistent
with this dispatch not using the token-count path.

## The ViT block recipe, blocks 31-38 (2026-09-22, corrected 2026-09-23)

The ViT loop (`0x18002fd80`-`0x1800302d2`, counter `r15d` from `0x1f` to `0x27`) issues **five**
dispatches per block, matching the five weight records blocks 31-38 carry. **The ViT uses six
contiguous ctx buffers** (ctx+0x260, ctx+0x268, ctx+0x270, ctx+0x278, ctx+0x280, ctx+0x288),
each allocated in its own arena slot.

| step | dispatch at | kernel | weight layer | kernarg |
|---|---|---|---|---|
| 1 | `0x18002fe68` | `k_expand2` | 0 | `+0x00` input, `+0x08` ctx+0x260, `+0x10` weight |
| 2 | `0x18002ff83` | `k_contract2` | 1 | `+0x00` ctx+0x260, `+0x08` input, `+0x10` ctx+0x268, `+0x18` weight, `+0x28` = **4** |
| 3 | `0x180030070` | `k_qkv2` | 2 | base rbp-0x20 (see below) |
| 4 | `0x18003015d` | `k_attention2` | - (no lookup) | base rbp+0x10 (see below) |
| 5 | `0x180030278` | `k_contract2` | 4 | `+0x00` ctx+0x288, `+0x08` ctx+0x268, `+0x10` output, `+0x18` weight, `+0x28` = **4** |

### Step 3: k_qkv2 kernarg (host disassembly 0x180030008-0x18003002c, base rbp-0x20)

Host loads two 128-bit register pairs:
- `movups [rcx+0x268] -> [rbp-0x20]` at 0x180030008 (fields +0x00, +0x08)
- `movups [rcx+0x278] -> [rbp-0x10]` at 0x180030013 (fields +0x10, +0x18)
- `call 0x180031bc0` weight lookup with layer=2 at 0x180030027
- `mov [rbp], weight result` at 0x18003002c (field +0x20)

| offset | field |
|---|---|
| +0x00 | ctx+0x268 |
| +0x08 | ctx+0x270 |
| +0x10 | ctx+0x278 |
| +0x18 | ctx+0x280 |
| +0x20 | weight layer 2 |

### Step 4: k_attention2 kernarg (host disassembly 0x1800300ee-0x180030118, base rbp+0x10)

Host loads two 128-bit register pairs and one dword pair with element swap:
- `movups [rax+0x270] -> [rbp+0x10]` at 0x1800300f5 (fields +0x00, +0x08)
- `movups [rax+0x280] -> [rbp+0x20]` at 0x180030100 (fields +0x10, +0x18)
- `movq [rax+0x310] -> [rbp+0x30]` at 0x18003010b, then `pshufd xmm0, xmm0, 0xe1` at 0x180030113 
  (swaps the two dwords: element [0,1] becomes [1,0], so H and W are swapped)
- `movq [rbp+0x30] -> [rbp+0x20]` at 0x180030118 (fields +0x20, +0x24)

| offset | field |
|---|---|
| +0x00 | ctx+0x270 |
| +0x08 | ctx+0x278 |
| +0x10 | ctx+0x280 |
| +0x18 | ctx+0x288 |
| +0x20 | H/W pair from ctx+0x310, **swapped by pshufd** (original order W, H) |
| +0x24 | |

**Why attention takes no weight lookup**: ViT `layer3` is only **2 bytes** (`block31.layer3.layer`),
a scalar rather than a matrix - so `k_attention2` reads the scale, not a weight record. The other
four layers map cleanly onto expand / contract / qkv / projection.

**The `+0x28 = 4` the host writes is the token-count field**, and 4 is exactly the value derived
empirically for `k_contract2` from the `min(n*16, H*W)` sweep. Static and experimental agree.

**Ping-pong**: the loop head swaps buffers every iteration (`mov rax, rdi` / `mov rdi, r13` /
`mov r13, rax` at `0x18002fd83`-`0x18002fd90`), so step 1 reads the input and step 5 writes back
to the other.

**Verified on hardware** (2026-09-23): ViT block 31 executes as a complete five-dispatch chain on
gfx1030 with **0 mismatches** against the emulator for steps 1-3 and for the full 5-step chain.
The known 1-ULP WMMA rounding in k_attention2 does not manifest as divergence in the full-chain
result (likely absorbed by step 5's subsequent k_contract2).

## k_attention's mismatch: root cause is the emulator, not the translation (2026-09-22)

The mismatch had been characterized but not explained. Two experiments settled it.

**It is not fixture realism.** Running `k_attention` with random bytes, zeroed inputs, and
well-conditioned f32 gives **250 / 247 / 248** mismatches. Numerical sensitivity would vary with the
data; a near-constant count across wildly different inputs means a specific operation is wrong.
This retires the "softmax on extreme inputs quantised to opposite sides of an e4m3 boundary"
hypothesis recorded earlier.

**Narrowing by instruction set.** Comparing the instruction mix of the two failing attention kernels
against twelve passing ones leaves only **five** instructions used by the failures and by no passing
kernel:

```
ds_store_2addr_stride64_b32  x7      ds_load_2addr_stride64_b32  x6
ds_store_2addr_b64           x1      v_lshrrev_b64               x2
v_cmp_eq_u32_e64             x1
```

**The `stride64` LDS forms were being decoded and then ignored.** The emulator's `_ds` matcher
captured the variant as a *non-capturing* group, `(?:stride64_)?`, so the flag was parsed and
discarded, and the address used `offset * element_size`. Per the RDNA3 ISA the plain form is
`ADDR_BASE + OFFSET * 4`, but the stride64 form is `ADDR_BASE + OFFSET * 4 * 64` ("with a larger
stride"). Every stride64 access therefore aliased onto the wrong LDS row.

Fixed by capturing the group and scaling by `element_size * 64`. **Mismatches drop 250 -> 214**
(`k_attention2` likewise 248 -> 214), with all five emulator test files and every previously passing
kernel unaffected.

**The important consequence: this was an emulator defect, so the gfx1030 translation was right all
along.** These difftests use the interpreter as the reference, so a wrong reference reads as a
failing translation. Any future mismatch should be checked against the ISA before the translator is
suspected.

**Still open**: 214 mismatches remain, so there is a second defect. The residual suspects from the
list above are `ds_store_2addr_b64`, `v_lshrrev_b64` (whose implementation reads correctly against
the ISA) and `v_cmp_eq_u32_e64`; the 2addr **b64** store path is the most likely, since the b32
paths are now exercised correctly by several passing kernels. `statebisect`'s first-divergent-
instruction search is the tool for the rest.

## k_attention fully root-caused: two emulator bugs, then 1-ULP WMMA rounding (2026-09-22)

`statebisect` was adapted to this kernel (`statebisect_attn.py`) and binary-searched the
7,957-instruction first-visit trace. Two real defects came out, **both in the emulator**, plus a
final residue that is not a defect at all.

### Fixing the bisect harness first

Two things had to be corrected before the search would run, and both were latent bugs in the shared
tool rather than anything to do with this kernel:

* **`statebisect` hard-coded the workgroup-id SGPRs as `s4`/`s5`.** That index is
  `user_sgpr_count`, which is 4 for `k_swin_var` (what the tool was written against) but **2** for
  `k_attention`. The guard therefore compared garbage, every wave branched to `dumpskip`, and the
  GPU dump came back empty. `variant_source` now takes the base as a parameter.
* **`coverage.json` only ever holds the last-built kernel**, so `source_vgprs` was unavailable. It
  is now derived from the gfx1100 disassembly as the highest v-register referenced.

### Defect 1 - `v_cvt_f16_f32` destroyed the upper half of its destination

First divergence landed on `v_cvt_f16_f32_e32 v3, v3`, with hardware holding `0x3d902c80` where the
emulator had `0x00002c80` - **identical low halves, upper half zeroed**. The f16 result occupies
`D[15:0]` and `D[31:16]` must be left untouched; the emulator wrote the whole dword. Fixed by
merging into the existing destination. This moved the first divergence roughly **900 instructions
later**, which is how the fix was confirmed.

### The residue - 1-ULP differences out of WMMA

The next divergence is `v_wmma_f32_16x16x16_f16`, and the differences are now **one ULP**
(`4425e8d2` vs `4425e8d1`, `c38272f6` vs `c38272f5`). That is a rounding-model difference in the
reference, not a translation defect. Hardware's internal accumulation order is undocumented and is
**neither** of the obvious candidates: f64-then-round (the original) and naive sequential f32 were
both measured and left the count at 214 unchanged. The f64 form is kept as the more accurate
reference.

**Why other WMMA kernels pass**: `k_qkv`, `k_qkv2`, `k_contract2` and `k_conv_splitk` all use the
same lowering and report 0 mismatches, because their results are quantised to e4m3/f16 before being
stored, which absorbs a 1-ULP f32 difference. `k_attention` has values sitting on quantisation
boundaries, where 1 ULP flips the stored byte - which is exactly the large-magnitude, sign-flip-
shaped deltas recorded earlier when the mismatch was first characterised.

### Conclusion

**`k_attention`'s gfx1030 translation is correct.** The mismatch was two emulator defects (now
fixed) plus reference-model rounding imprecision in WMMA. It should be reclassified from "blocking
defect" to "known reference-model imprecision", and it does **not** block building or validating the
ViT path. Treating these difftests as a verdict on the translation, rather than on the pair
(translation, reference model), is what made it look like a port bug for two sessions.

## Decoder per-block dispatch verified; k_dec_upsample decoded (2026-09-22)

**The decoder's per-block dispatch is the encoder's.** Its inner loop (`0x1800309d0`-`0x180030a6a`)
calls **launcher A** (`0x180032df0`) with `(C,H,W)` from the reversed stage tuple, the per-block mode
from an array at `[r13+0x10]`, and `in_ptr = ctx+0x2a8[r9d]` - the encoder skip buffer. Same
launcher, same `k_swin_var` kernels, only the weights and tuple differ.

Verified by execution rather than inspection: blocks **66-69** (C=32, the last decoder stage) run as
a four-dispatch `net_run` chain with their real weight records and match the emulator with
**0 mismatches**, 56,304 bytes written. The stage-chain script is now parameterised by block list,
since encoder and decoder stages are the same construction.

### `k_dec_upsample` (`DecUpParams`), built at `0x180030510`-`0x180030563`

| off | value |
|---|---|
| +0x00 | `ctx+0x250` |
| +0x08 | `[rbp+0x690]`, a local (corrected - an intervening load made this look like `ctx+0x250`) |
| +0x10 | `ctx+0x298` |
| +0x18 | weight ptr, `0x180031bc0(ctx, block=**39**, layer=0)` |
| +0x20 | weight-derived |

Five pointers, matching the `DecUpParams` shape verified in the registry. This is the step that opens
each decoder stage, alongside `k_repack`.

## Registry sweep after the emulator fixes: 16 PASS / 2 FAIL

Re-running the whole registry after the `stride64` and `v_cvt_f16_f32` fixes:

* **`k_mean` now passes.** It had been failing with 4 mismatches and had been separately investigated
  and localised to "ordinary per-lane arithmetic". It was the same `v_cvt_f16_f32` defect. That is a
  second kernel whose "translation failure" was really the reference model.
* The only remaining failures are `k_attention` / `k_attention2` at 214 each, at the time attributed
  to 1-ULP WMMA rounding in the reference rather than translation defects.
  **SUPERSEDED** - see "k_attention2 is bit-exact in the real chain" at the end of this file. The
  WMMA-rounding explanation was wrong; the fixture was.

## Block 39 resolved: it is not a block, it is the decoder-transition weight (2026-09-22)

Block 39 was carried as an unexplained "transition" with a single 525,312-byte record and no known
dispatch. It is not dispatched as a block because it is not one: at `0x18003053a` the host does
`mov edx, 0x27` (= **39**) immediately before the weight lookup that feeds **`k_dec_upsample`**
(`0x180030542`, dispatch at `0x1800305a3`). Block 39 is simply the weight record for the first
decoder upsample, fetched once on the way from the ViT into the decoder.

The size corroborates it exactly: `525,312 = 1024*512 + 1024`, a C=1024 -> 512 projection plus a
1024-entry bias/scale - precisely the channel-halving needed to bring the ViT's C=1024 output down
to the decoder's C=512 path.

Method note: every weight lookup in the driver was listed with how its block index is set. Only
three use a literal - `edx=0` twice in the prologue (the pre-block, block 0), `edx=0x46` (= 70)
twice in the epilogue (the final head), and `edx=0x27` (= 39) here. Everything else is a loop
counter. That accounting is what showed block 39 had no loop to belong to.

### All 71 weight blocks now have an owner

| blocks | owner |
|---|---|
| 0 | pre-block (`k_swin_var<32,true>`, prologue, `edx=0`) |
| 1-22 | encoder, 4 stages via launcher A |
| 23-30 | C=512 attention via the block launcher `0x180033660` |
| 31-38 | ViT loop |
| **39** | **`k_dec_upsample` transition weight** |
| 40-47 | C=512 attention via the same launcher |
| 48-69 | decoder, 4 stages via launcher A |
| 70 | final head (`edx=0x46`) |

## The ctx buffer map (2026-09-22)

Every ctx-relative qword slot referenced as a pointer by the frame-level function, the network
driver, launcher A or the block launcher. This is the allocation a runner has to reproduce.

| ctx offset | used by | role |
|---|---|---|
| `+0x0f8` | frame-fn, `k_export` | float-RGB copy from `k_import` - the export blend source |
| `+0x100` | frame-fn, driver, `k_export` | **network result** |
| `+0x108` / `+0x110` / `+0x118` | frame-fn | input-side buffers |
| `+0x140` / `+0x148` | frame-fn (x57 / x27) | a list/vector pair, not a single buffer |
| `+0x180` / `+0x188` | driver, launcher A | per-workgroup **scratch** (launcher A's `+0xa0`) |
| `+0x218` / `+0x220` | driver, frame-fn | pre-block buffers |
| **`+0x228`** | block launcher | **C=512 persistent activation** - block output and next block's input |
| **`+0x230` / `+0x238` / `+0x240`** | block launcher | C=512 intermediates (FFN out, attn in, attn out) |
| `+0x248` | driver | the extra passed on the **last** block of a C=512 stage |
| `+0x250` / `+0x258` / `+0x298` | driver | decoder upsample |
| **`+0x260` / `+0x268` / `+0x288`** | driver | **ViT** intermediates |
| `+0x2a0`, `+0x2a8[4]` | driver | stage tuple / **per-stage skip buffers** (U-net) |

The three distinct working sets - `0x228`-`0x240` for C=512 blocks, `0x260`-`0x288` for the ViT,
and `0x2a8[]` for the encoder/decoder skips - are why no single ping-pong pair explains the whole
network: each stage type has its own.

Buffer **sizes** come from the activation formula `C * ceil(H/4) * ceil(W/4) * 16` with `(C,H,W)`
from the `ctx+0x190` stage tuple, so a runner can allocate these itself rather than reproducing the
host allocator.

## The full encoder runs as one chain on hardware (2026-09-22)

All **22 blocks across four stages** (C=32/64/128/256, 4/4/6/8) execute as a single `net_run`
dispatch sequence and match the emulator running the identical chain with **0 mismatches**,
175,633 bytes written.

This adds the piece the single-stage runs could not: the **fused stage transitions**. The slots
written are exactly what the schedule predicts and nothing else -

| slot | meaning |
|---|---|
| 18, 19 | the within-stage ping-pong pair |
| 21, 22, 23 | `STAGE_IN[1..3]` - all three transitions fired, each stage's last block emitting the next stage's input through `VarParams +0x38` |
| 17 | per-workgroup scratch (`+0xA0`) |
| 4 | block 22's pooled output, which correctly keeps the default slot because stage 4 has no successor |
| 24-45 | **untouched** - the weight records are not written |

### A fixture bug worth recording

The first version of this run also reported PASS with 0 mismatches, and was **wrong**. Weights were
placed from slot 6, which overlaps the 18 `VarParams` pointer slots that `make_kernarg` fills; the
scratch pointer at `+0xA0` is slot 17, so scratch overwrote block 12's weight record partway through
the chain. Emulator and GPU did the same wrong thing, so translation equality genuinely held and the
test passed while blocks 12-22 ran on corrupted weights.

This is the same failure mode as `k_ffwd2` "passing" while writing zero bytes: a bit-exact result
says the two implementations agree, not that the fixture asked a meaningful question. Any chain test
must place its buffers **beyond `len(V.PTR_FIELDS)`**, and checking that no weight slot appears in
the written set is a cheap way to catch it - which is why that slot list is recorded above.

## k_attention2 is bit-exact in the real chain — the 214 count is a fixture artefact

The ViT block-31 chain (`net_vit.py`) was run in two cuts against `net_run.exe`, seed 1, to isolate
step 4 (`k_attention2`) rather than trust a whole-chain PASS:

| cut | dispatches | emulator bytes written | slots touched | mismatches |
|---|---|---|---|---|
| steps 1-3 | 3 | 22439 | 1,2,3,4,5 | **0** |
| steps 1-4 | 4 | 24484 | 1,2,3,4,5,**6** | **0** |
| steps 1-5 | 5 | 32651 | 1..7 | **0** |

The 1-3 → 1-4 delta is **2045 bytes**, all in slot 6 (`ctx+0x288`, `k_attention2`'s `+0x18`
destination). So the kernel did substantial work *and* matched the emulator exactly. No weight slot
(8 and above) appears in any written-slot list, so the fixture is not the overlapping-weights trap.

**Control, run in the same session:** the per-kernel registry difftest for `attention2` still reports
**214** mismatches (2019 bytes written). The emulator did not change; the input data did.

This retires the standing explanation. The record previously attributed the 214 to *1 ULP out of
`v_wmma_f32_16x16x16_f16`*, with hardware's accumulation order blamed as undocumented. **That is not
supported.** Two independent observations contradict it:

* With the real wiring the kernel is bit-exact, which a genuine hardware-rounding difference would
  not be.
* The differing bytes are not adjacent codes. `emu=04 gpu=02`, `emu=36 gpu=03`, `emu=1e gpu=01` are
  far apart — that is not a last-place rounding difference in any of the stored formats.

The fixture is wrong in at least two ways relative to the real ABI:

1. Slots 0/1/3 were filled with **random bytes**, not valid floats — the trap already recorded for
   reduction and softmax kernels.
2. `+0x10` was given **`block31_layer2.bin`**, a weight file. In the real launch `+0x10` is
   `ctx+0x280`, one of `k_qkv2`'s three activation outputs. `k_attention2` takes **no weight**;
   the ViT layer-3 entry is a 2-byte scalar.

Adding `fill={0:'f32', 1:'f32', 2:'f32'}` (and dropping the bogus weight) moves the count
**214 -> 156**, confirming the random-byte defect is real but is not the whole story. The remaining
156 are **not yet explained**: the three inputs are still mutually inconsistent random blobs rather
than a single consistent QKV projection, which is the obvious next suspect, but it has not been
demonstrated.

**Status to carry forward:** `k_attention` / `k_attention2` are **not known translation defects**.
They are bit-exact where it has been possible to test them against real data. The registry's 16/18
should be read as 16 verified plus 2 whose fixtures ask an invalid question — not as two broken
kernels. Do not spend further sessions on the WMMA rounding model on their account.

## Block 0 (`k_swin_var<32,true>`) is NOT verified - correcting the record

Earlier handovers list block 0 as "pre-block | kernel verified". That is wrong. Run today,
standalone, with no chain code involved:

```
_Z10k_swin_varILi32ELb1EEv9VarParams  flags=20  hw=[16,16]  grid=[2,2]
emu_bytes_written 42812   mismatches 28825   status FAIL
```

The fixture is not at fault. `difftest_preblock.py` already fills the `+0x40` input with float RGB
in [0,1) by default (`PRE_RAW` is opt-in), so this is not the random-bytes-as-floats trap that
explains `attention2`. The kernel mismatches on valid input.

**`<32,true>` is exercised by exactly one dispatch in the whole network.** The 22 encoder blocks use
the `<...,false>` instantiations (`32_0`, `64_0`, `128_0`, `256_0`), which pass. So the passing
encoder says nothing about the pre-block, and the pre-block is the network's entry point - every
downstream block consumes its output.

Whether this is a translation defect or a `gfx11emu.py` defect is **not determined**. Do not assume
either. Three prior "translation failures" in this file turned out to be the reference model or the
fixture; do not let that make the opposite mistake and assume this one is too.

### Chain wiring fixed along the way (all real, none of them the cause)

While tracking this down, `net_full.py`'s block 0 was found to be wrong in three independent ways.
All are now corrected, and none of them account for the mismatch:

* It dispatched `D.SYMS['32_0']` - `k_swin_var<32,false>`, the wrong template instantiation. The
  contract has recorded `<32,true>` since the launcher was decoded. The wrong variant faults on a
  `ds_load` past the end of its LDS.
* It used the **encoder** calling convention - `flags=1|4`, input at `+0x00` - where the pre-block
  takes `flags=0x14`, `+0x00` null, and its float-RGB input at `+0x40`.
* `+0x30` was left holding `make_kernarg`'s default pointer; the pre-block is launched with it null.

The authoritative layout is `chain_import_preblock_real.py`, which feeds real `k_import` output into
the pre-block. Anything that dispatches the pre-block should copy it rather than build a kernarg from
the encoder pattern.

## Activation buffers at real resolution: two corrections

Running the frame path at 1707x960 (`net_frame_full.py`) pinned down two things the contract stated
ambiguously. Both were found by the arena guards, not by reasoning.

**The formula counts elements, and the element is f16.** `C * ceil(H/4) * ceil(W/4) * 16` is written
here without a unit. Taken as bytes, the pre-block writes past the end of its output and `net_run`
reports `guards CORRUPT`. The buffer is that count times **2 bytes**:

```
C=32 at 1707x960  ->  32 * 240 * 427 * 16 * 2  =  104,939,520 B  (104.9 MB)
```

The element width is confirmed by reading the result back, not by the allocation succeeding. As f32
the buffer holds values up to 1.7e38 and 9% of it exceeds 1e4; as f16 it is 100% finite and entirely
within +/-1e4. An over-allocation at 4 B/element also makes the guards pass, so allocation alone
proves nothing - check the values.

**`+0x48` is not optional.** It is described above as "ptr (optional 3rd buffer)". Nulling it while
giving every other pointer a real buffer trips the guards at the correct f16 size. It needs a buffer
the same size as the output. `difftest_preblock.py` never noticed because it leaves `make_kernarg`'s
default pointer there rather than nulling it.

### What this does and does not show about block 0

The full-resolution path now runs clean: `k_import` + pre-block, 2 dispatches, ~23 ms, guards intact,
558.9 MB arena. `k_import` output is correct (finite, 0.0005..65.12, 71.7% nonzero - it is the frame).

The pre-block output is **not**. It saturates f16 (+/-65504, `finite=False`), and a channel view of it
is structureless noise from an input that demonstrably contains the image. That is the same verdict
its standalone difftest gives, reached independently: **block 0 does not work**, and it is the
network's entry point.

Sizing and wiring are therefore no longer suspects for block 0. The remaining question is the one
`statebisect.py` answers: the first instruction where the translated kernel and the emulator diverge,
and hence whether the defect is in `k_swin_var<32,true>` or in the reference model.

## `v_lshlrev_b16` / `v_ashrrev_i16` destroyed `D[31:16]` (emulator defect, fixed)

Bisecting the pre-block (`statebisect.py 32_1 0x14 1`) localised the first divergence exactly:

```
LAST MATCH        trace[6812] = b9ca4
FIRST DIVERGENCE  trace[6813] = b9cac
last executed, result differs:  0xb9ca4  v_lshlrev_b16  v5, 8, v5
    emu = 0x00008f00      gpu = 0xffff8f00
```

The low half agrees; the emulator zeroed the upper half. A 16-bit VALU op writes `D[15:0]` and
**preserves** `D[31:16]`. `gfx11emu.py` masked the result to `0xffff` and stored the whole dword, so
every `BIN16` op and `v_ashrrev_i16` wiped the high half. This is the third instance of this exact
defect, after `v_cvt_f16_f32` and `v_pack_b32_f16`.

Fixed, and the fix is confirmed by the bisection moving to trace[6872] with `v_lshlrev_b16` gone.
Regression-checked: `mean`, `contract2`, `expand`, `qkv`, `conv_res_views`, `ffwd2` all still PASS at
0 mismatches.

**It does not fix block 0.** `difftest_preblock` still reports 28,825, byte-identical to before, with
a cleared bytecode cache. So the pre-block has at least one further defect.

### What the bisection can and cannot say here

* With strict comparison it now stops at `s_getpc_b64 s[18:19]` (`emu=0x000bbd24 gpu=0x00032c4c`).
  That is **benign**: `s_getpc` returns the current PC and the two binaries are laid out differently.
  The tool's pointer heuristic covers VGPRs but not SGPR pairs.
* With `BISECT_HEURISTIC_FILTERS=1` it reports *no divergence* at the last traced instruction while
  memory still differs by 28,825 bytes. **Do not read that as "the registers agree."** Filter line
  176 ignores differences where `((e ^ g) & 0xffff) == 0` - high-half-only differences, precisely the
  class of bug just fixed. The filters can hide the next one.
* The bisection only inspects the **first visit** to each PC. `k_swin_var` loops, and later
  iterations are never compared, so a defect that only manifests after iteration 1 is invisible.

Localising the rest of block 0 needs a method that follows memory writes, or a bisection extended to
repeat visits. Two further latent bugs in `statebisect.py` were fixed to get this far: `source_vgprs`
read from `coverage.json`, which only ever holds the last-built kernel (ported from
`statebisect_attn.py`), and SCC compared without honouring its poison sentinel of 2, which made every
run abort at entry as a "harness problem".

## The shape of block 0's divergence: not layout, not lanes, not rounding

`difftest_preblock`'s 28,825 stayed **byte-identical** across four separate emulator ALU fixes
(`v_lshlrev_b16`, `v_ashrrev_i16`, and then `v_mul_f16`, `v_add_f16`, `v_fmac_f16`). A count that
never moves under arithmetic changes is telling you the question is wrong, so
`preblock_diffshape.py` asks about shape instead of magnitude. At 16x16, seed 1, output slot `+0x8`:

```
emulator wrote 8162 bytes | gpu wrote 8163 | both untouched 8194 | differing 5348
differ but only one side wrote : 55
byte-value histogram L1 distance : 10.3%
best byte shift 64 -> 1.8% match     aligned -> 67.4% match
|delta| as u16 words:  <=1 8.0%   <=2 10.9%   <=4 14.0%   <=16 16.9%   <=256 40.8%
per-lane share of differing blocks: 25-50% across all 32 lanes, no structure
```

**Eliminated by this:**

* *Layout or addressing permutation* - a shift fits far worse than alignment (1.8% vs 67.4%), and
  both sides write the same byte count to the same region.
* *A lane or wave subset* - every one of the 32 two-byte lanes differs in 25-50% of dirty blocks,
  evenly. No wave is clean and no wave is wholly wrong.
* *Rounding or 1-ULP noise* - only 8% of differing words are within 1 ULP and 59% are further apart
  than 256. This is not the `k_attention` story.

**Trap worth recording**: the per-block view first appeared to show "128 blocks fully equal, 128
partly differing", which reads like an alternating pattern with half the work correct. It is not.
The clean half is the second half of the slot, which **neither side wrote**. It is equal because it
is untouched. Any "share of blocks equal" metric must be taken together with the untouched count, or
it flatters the result exactly where the result is emptiest.

**Where that leaves it.** Differences start at the very first written byte, yet `statebisect` in
strict mode finds matching registers all the way through the first-visit trace (10,449 instructions)
before stopping on a benign `s_getpc_b64`. Both can only be true if the divergence happens on
**repeat visits** - later iterations of `k_swin_var`'s loops, which the bisection never inspects.
That is the one hypothesis still standing, and extending the bisection to an (address, occurrence)
pair is what would confirm it.

## The minimal block 0 repro, and a fixture bug that invalidated the first bisection

**Minimal repro: 8x8, one workgroup.** Mismatches scale linearly with workgroup count at a constant
share of written bytes, so nothing cross-workgroup is involved:

| H x W | grid | written | mismatches | share |
|---|---|---|---|---|
| 8x8 | 1x1 | 10690 | 7268 | 68.0% |
| 16x8 | 1x2 | 21381 | 14320 | 67.0% |
| 16x16 | 2x2 | 42812 | 28825 | 67.3% |
| 24x24 | 3x3 | 96229 | 64520 | 67.0% |

Bisect at **8x8** - it is ~4x cheaper than 16x16 and fails identically.

**`statebisect.py` was not bisecting the failing launch.** `rebuild_kernarg` called
`V.make_kernarg(flags=..., grid=...)` with the **defaults**: `H=16, W=16, offy=-4, offx=-4`, no null
`+0x00`/`+0x30`, no float-RGB input at `+0x40`, and none of the `+0x50..0x9f` scalar setup. `arena()`
also never seeded the input, so the pre-block was reading e4m3-shaped bytes as f32.

So the first bisection ran a *different launch* from the one that fails. The `v_lshlrev_b16` defect
it found is real and is fixed, but it was found under the wrong fixture, and it does not move block
0's 28,825 - consistent with having been an unrelated bug that happened to be in the path.

**Anyone re-running the bisection must apply the pre-block kernarg** (flags 0x14, per
`chain_import_preblock_real.py`) and seed `+0x40` with float RGB, or the result describes the wrong
configuration.

**Rebase trap found doing that**: `+0x50`, `+0x58`, `+0x60` and `+0x68` are 4-byte scalars, but
`PTR_FIELDS` lists them, so `make_kernarg` writes full 8-byte pointers there. Clearing only the low
dword - as `difftest_preblock.py` does - leaves stale high bits that the GPU-side rebaser then
shifts, and the checkpoint aborts with "rebased kernarg disagrees with emulator" on `+0x68` alone.
Zero the full 8-byte span, then write the scalar.

## The whole network, end to end, on the captured frame

`net_frame_full.py` runs `k_import` + all 71 blocks + `k_export` at 1707x960, GPU only:

```
net_run OK: 169 dispatches, 807.476 ms GPU, 24.32 s wall, arena 2,259,681,280 B, guards intact
k_export surface : finite=False  nonzero=11.0%  min=-3.755e-06  max=1.875
```

**Every kernel is the real one.** `net_full.py` dispatches `k_ffwd_inpview` in place of block 39 and
block 70 - its own comments call them placeholders "requires DecUpParams kernelspec" / "requires
HeadParams kernelspec". Both real kernels exist and are used here: `_Z14k_dec_upsample11DecUpParams`
and `_Z12k_final_head10HeadParams`. A dispatch count alone does not tell you the right kernels ran.

**The image is garbage** - about 11% of the surface written, confined to the top ~120 rows (1/8 of
960), in RGB stripes. Expected: block 0 is broken and feeds everything. Two things are still worth
noting from it:

* `k_export`'s output range is **-3.8e-06 .. 1.875**, not the +/-65504 f16 saturation every
  intermediate buffer shows. The head and export stages produce image-scale numbers even from
  garbage input, which is weak evidence their format handling is roughly right.
* Only ~1/8 of the rows are written. `k_export`'s grid is (7, 960), which should cover every row, so
  this is an addressing or stride question in `k_export` or in the head's output layout - the first
  thing to look at once block 0 is fixed.

### `net_run.exe` could not read an arena larger than 2 GB

`rd()` used `long n = std::ftell(f)`. `long` is 32-bit on Windows, so at 2.26 GB `ftell` overflowed
and the run died with **no output at all** and a bare non-zero exit - it looked like a GPU or VRAM
failure and was neither. Now `_ftelli64` / `ftello` with `long long`. Any full-resolution arena is
over 2 GB, so nothing at frame scale worked before this.

### Sizing at real geometry

Stage-5 geometry (`H>>4, W>>4` = 60x106) carries the C=512 and ViT stages; the decoder mirrors the
encoder back up to full resolution. Every pointer field in `PTR_FIELDS` must reference real memory -
`make_kernarg` fills them with 1 MiB difftest slots that do not exist in a byte arena, and a kernel
touching one writes outside it. Unused fields share one spare buffer sized for the largest stage.

## Block 0: narrowed to 32 candidate opcodes, not yet fixed

`preblock_firststore.py` hooks `GMem.write` and the execute loop so every emulator store is logged
with the PC that issued it, then diffs against the GPU arena. No GPU instrumentation, so it is
independent of `statebisect`.

**The very first store is already wrong.** At 8x8, seed 1:

```
store #0 of 352   pc=0xb1598   occurrence #1   25/32 bytes wrong   global_store_b8 v1, v2, s[40:41]
```

Not deep in a loop - the first global store the kernel makes. Divergence begins immediately.

Differences measured in the right unit (e4m3 code distance, one byte per value - an earlier pass
compared them as `uint16` words, which made the numbers meaningless):

```
|code delta| <= 1 : 31.3%      sign bit differs : 17.8%
|code delta| <= 2 : 41.3%      exponent differs : 54.1%
mantissa-only differences : 43.9%
```

### Three hypotheses tested and refuted

* **Transcendentals.** `gfx11emu` computes `v_rcp/v_rsq/v_sqrt/v_log/v_exp` exactly; AMD hardware
  approximates them (~1 ULP per the RDNA ISA), which looked decisive. **Refuted by control**: the
  passing `<32,false>` executes 1,664 of them (824 rcp, 816 log, 16 rsq, 8 rcp_iflag) against the
  failing variant's 1,830, and mismatches 0 bytes. Comparable usage, opposite outcome.
* **`MIX_F16_INPUT_FLUSH`.** A toggle left `False` with no evidence, and `v_fma_mix_f32` runs 1024
  times in the failing variant and never in the passing one. Setting it `True` changes the count by
  **zero** - no f16 denormal inputs arise here.
* **`v_perm_b32`, `s_bfe_i32/u32`, `s_bitcmp0_b32`.** All checked against the ISA; all correct.

### The shortlist

Static opcode sets are near-identical (only `s_bitcmp0_b32` differs), so the discriminator is what
each variant *executes*. Opcodes executed by `<32,true>` and **never** by `<32,false>`:

```
v_fma_mix_f32 1024   s_cmp_lt_u32 1024   s_bfe_i32 512   s_bfe_u32 512
s_sext_i32_i16 512   s_sext_i32_i8 512   global_store_b16 128   v_cmpx_o_f32 64
v_cmp_nlt_f32_e64 64   v_mad_i64_i32 18   v_fmaak_f32 16   v_cmp_gt_f32_e64 12
s_bitcmp0_b32 8   v_xad_u32 8   v_cmp_eq_u32_e64 6   v_dual_fmaak_f32 6
v_dual_fmac_f32 6   v_fmamk_f32 6   ds_store_b128 4   global_load_b96 4
v_cmp_class_f32_e64 4   v_cmp_ge_f32_e64 4   v_cmp_lg_f32_e64 4   v_cmp_lt_f32_e64 4
v_sqrt_f32_e32 4   v_dual_sub_f32 4   s_cmp_eq_u64 2   s_or_saveexec_b32 2
v_add_f32_e64 2   v_cmp_ngt_f32_e64 2   v_xor3_b32 2   s_or_saveexec_b32 2
```

That is the whole search space: **32 opcodes**, down from 197 executed and 14,284 instructions. Any
emulator defect in one of these breaks the pre-block while leaving all 22 encoder blocks passing,
which is exactly the observed pattern. Audit them against the ISA; `v_dual_*` (VOPD packed dual-issue)
and `v_cmpx_o_f32` (writes EXEC) are the least-travelled paths and the ones worth reading first.

**Minimal repro**: `difftest_preblock.py 1 8 8` - 7,268 of 10,690 bytes, one workgroup.

## RETRACTED: "Block 0 is non-deterministic on hardware above 4 workgroups"

**This section is wrong. Do not act on it.** The claim came from `preblock_response.py`, a harness
written for the purpose, and does not reproduce in `difftest_preblock.py`, which is the verified
path. Three consecutive runs at the same 64 workgroups:

```
run1  grid=[8,8]  mismatches=457729  sha=b35200d183dac9ede0332fa1
run2  grid=[8,8]  mismatches=457729  sha=b35200d183dac9ede0332fa1
run3  grid=[8,8]  mismatches=457729  sha=b35200d183dac9ede0332fa1
```

Byte-identical output. **The pre-block is deterministic on hardware**, and there is no race, no
missing synchronisation protocol, and nothing here that blocks the full-resolution path. The
contract had already established that `k_flag_set`/`k_flag_wait` are absent from the inference path;
that stands, and proposing them again was a failure to read this file before theorising.

Three candidate explanations for the harness defect were tested and none of them is it: all pointer
fields sharing one scratch buffer (fixed, still non-reproducible), buffer size (raised to 4 MiB,
unchanged), and arena contents (zeros vs e4m3-shaped random, unchanged). The remaining difference is
the runner - `net_run.exe` against `difftest_var.gpu` - and it has not been chased down.

**Rule this cost real time to relearn:** when an ad-hoc harness disagrees with the verified one,
the harness is the suspect. Reproduce a finding in `difftest_*` before writing it down as a property
of the hardware.

The original text follows, retained only so the reasoning can be audited.

## (retracted) Block 0 is non-deterministic on hardware above 4 workgroups

`preblock_response.py` asks a question the emulator cannot: run the real kernel on the GPU several
times with byte-identical input and see whether it agrees with **itself**. It does not.

```
TILE=8   grid 1x1  =  1 workgroup    deterministic (4/4 repeat runs identical)
TILE=16  grid 2x2  =  4 workgroups   deterministic (4/4 repeat runs identical)
TILE=64  grid 8x8  = 64 workgroups   NON-DETERMINISTIC
                                     differing bytes vs run 0: 587, 467, 15406, 233 (of 262144)
```

Same harness, same arena, same kernarg in all three - only the workgroup count changes. So there is
a **cross-workgroup race**, and it is invisible below about 4 workgroups.

**This matters more than any difftest number.** A kernel that does not reproduce itself cannot match
a sequential emulator, and no amount of ALU auditing would ever have fixed it. It also means the
full-resolution frame path cannot work as currently launched: at 1707x960 the pre-block runs
214 x 120 = **25,680 workgroups**, far into the racing regime, so its output there is not a
deterministic function of the input at all.

This is very likely the `k_flag_set` / `k_flag_wait` synchronisation protocol that this file has
listed as outstanding since the schedule was decoded. The pre-block both writes and reads back
through `+0x38` and `+0x48`; without the host-side flag handshake the workgroups have no ordering.

### Two separate problems, not one

Do not conflate them:

1. **The race, above.** Affects every launch at real resolution. Needs the sync protocol.
2. **A deterministic 67% mismatch at ONE workgroup.** `difftest_preblock 1 8 8` is in the
   deterministic regime and still reports 7,268 of 10,690 bytes. Fixing the race will not touch this,
   and it remains unexplained.

### Hypotheses tested and refuted, for the record

Transcendental precision (the passing `<32,false>` executes 1,664 approximate ops to the failing
variant's 1,830 and mismatches nothing); `MIX_F16_INPUT_FLUSH` (zero change); VOP3P `op_sel_hi`
defaulting to `[1,1,1]` (documented default, but **worse**: 7,268 -> 10,695, so `[0,0,0]` stays);
`v_perm_b32`, `s_bfe_i32/u32`, `s_bitcmp0_b32`, the float compare predicates and VOPD parallel-read
semantics (all correct against the ISA).

The `op_sel_hi` experiment is still worth knowing: a 3,400-mismatch swing says `v_fma_mix_f32` is on
the critical path even though that particular default is not the bug.

### (retracted with the above) The race is the pre-block's alone

Same harness, same 64 workgroups, same arena - only the variant and flags differ:

```
pre-block  k_swin_var<32,true>   flags 0x14 : NON-DETERMINISTIC (up to 122,880 of 262,144 bytes)
encoder    k_swin_var<32,false>  flags 5    : deterministic, 4/4 repeat runs identical
```

So blocks 1-70 do **not** need the synchronisation protocol; exactly one block does. That is a much
smaller problem than "the network races", and it is worth checking this way before assuming scope.

## ISA audit of the 32-opcode shortlist: no defect found

Each opcode the failing `<32,true>` executes and the passing `<32,false>` never does, checked against
the RDNA ISA. **Result: no defect.** Recording it so nobody repeats the sweep.

| opcode | checked | verdict |
|---|---|---|
| `s_bfe_i32` / `s_bfe_u32` | offset `S1[4:0]`, width `S1[22:16]`, width 0 -> 0, sign-extend, SCC = (D != 0) | correct |
| `s_sext_i32_i16` / `_i8` | mask to width, subtract 2^w when the sign bit is set | correct |
| `s_cmp_lt_u32` | generic `s_cmp` regex; unsigned, no sign-extension | correct |
| `s_cmp_eq_u64` | `rs()` returns the full 64-bit pair when `o['n'] == 2` | correct |
| `s_bitcmp0_b32` | `(S0 >> (S1 & 31)) & 1` against the expected bit | correct |
| `v_cmpx_o_f32` | `bits()` applies `& s.em`, so EXEC gets the result masked by the *current* EXEC | correct |
| all `v_cmp_*` predicates | `o`/`u`/`nlt`/`ngt`/`nge`/`nle` NaN behaviour | correct |
| `v_xad_u32`, `v_xor3_b32` | `(S0^S1)+S2`, `S0^S1^S2` | correct |
| `v_fmaak_f32` | `S0*S1+K` | correct |
| `v_fmamk_f32` | `S0*K+S1` - operand order matches the assembly form | correct |
| `v_mad_i64_i32` | `P[1]` is the carry operand; sources are `P[2..4]` | correct |
| `v_dual_*` (VOPD) | `w.RV` snapshot before both halves, so sources are read before either writes | correct |
| `global_store_b16`, `global_load_b96`, `ds_store_b128` | `stbytes`/`ldbytes` take `min(4, n-4j)` per dword | correct |
| `v_perm_b32` | selector 0-3 -> `S1` bytes, 4-7 -> `S0`, 12 -> `0x00`, 13-15 -> `0xff` | correct |
| `v_fma_mix_f32` | `{op_sel_hi[i], op_sel[i]}` = f32 / f16-lo / f16-hi | correct |

The translated gfx1030 kernel passes the interesting ones through unchanged - `v_cmpx_o_f32_e32`,
`v_cmpx_o_f16_e32`, `v_cmpx_neq_f32_e32` all appear with `// original:` comments and no rewriting.

**A trap worth naming**: a grep for `\bv_cmpx_o_f32\b` reports **zero** occurrences in that file,
because `_` is a word character and the mnemonic is `v_cmpx_o_f32_e32`. That nearly became a
finding ("the translator dropped the instruction"). Match mnemonics with a trailing `[_ ]` or none.

### What that leaves

The emulator side of the shortlist is clean, which shifts weight toward the translation or toward
something that is not a single opcode - an operand modifier, a literal, or a control-flow difference.
The dynamic approach (`preblock_firststore.py`, tracing back from the first wrong store at
`pc=0xb1598`) does not depend on guessing which opcode matters and is the better remaining lead.

## SOLVED: block 0 was never broken. `+0x68` is the RNG seed, and the fixture corrupted it.

`difftest_preblock.py 1 8 8` now reports **PASS, 0 mismatches**, with no environment override. So do
16x16 and 64x64. The pre-block's translation is correct and always was.

**The mechanism.** `+0x68` is the pre-block's i32 seed (`ctx+0x38`). The kernel loads it
(`0xb04c8 s_load_b256 s[16:23], s[0:1], 0x50`, so s22 = `+0x68`) and immediately mixes it
(`0xb0550 s_mul_i32 s17, s22, 0x9e3779b9` - the golden-ratio constant) into a PCG-style hash feeding
Box-Muller noise.

`run_var.PTR_FIELDS` lists `0x68`, so `make_kernarg` wrote an **8-byte arena pointer** there. Every
fixture then cleared only the **low dword**, leaving arena bits in `+0x6c`. The GPU runner rebases
any in-arena qword, so the kernel ran seeded with the low dword of the device arena address
(`0x04010000`) while the emulator reference used `0`. Different seed, different noise, ~67% of e4m3
bytes differing with sign and exponent flips.

The tell was in the GPU line all along: **"7 pointers rebased"** where only 6 are real. With the
field fully zeroed it reports 6 and the test passes.

**Positive control**: seeding both sides with the value the GPU accidentally used
(`PRE_FIELDS=0x68=i67174400`) reproduces the failing GPU output **bit for bit**. Seeding both with
`12345` also passes, so a nonzero seed is fine - only agreement matters.

Fixed in all 14 fixtures that cleared the low dword only. `for off in (0x50, 0x58, 0x60, 0x68):
struct.pack_into('<Q', ka, off, 0)` then the `<I` scalars.

### What this retires

* **Block 0 is not a defect.** Six hypotheses were raised against it - transcendentals,
  `MIX_F16_INPUT_FLUSH`, VOP3P `op_sel_hi`, an emulator ALU bug, a cross-workgroup race, a
  translation defect - and **all six were wrong**. The cause was a corrupted launch parameter.
* **The "non-determinism above 4 workgroups" retracted earlier is now explained**, not merely
  withdrawn: the seed *was* the device arena address, which changes between allocations.
* **The 32-opcode shortlist and the ISA audit are moot.** Both were sound work aimed at a defect
  that did not exist.

### The lesson worth keeping

Two independent signals pointed at this field hours before it was understood. `statebisect` aborted
with `differing fields (offset, mine, gpu): 0x68: 0x400000000 vs 0x404010000`, and this file already
recorded that low-dword-only zeroing "leaves stale high bits that the GPU-side rebaser then shifts".
Both were filed as harness quirks and neither was carried across to the difftest that was failing.
**When a harness reports a field mismatch, that is a result about the launch, not an obstacle to the
run you wanted to do.**

Also: `"7 pointers rebased"` is a fixture assertion nobody was making. Any launch should state how
many pointers it expects to be rebased and fail when the runner disagrees.

### Registry after the fix

`mean`, `contract2`, `expand`, `qkv`, `conv_res_views`, `ffwd2`, `dec_upsample`, `final_head` and the
pre-block all PASS at 0. `attention` (214) and `attention2` (156) still fail - those are the
random-bytes-as-floats fixture defects recorded earlier, unrelated to this.

## Frame path after the block-0 fix: the kernels are fine, the frame wiring is not

With `+0x68` corrected, the pre-block passes at 8x8, 16x16 and 64x64 with 0 mismatches. In
`net_frame_full.py` at 1707x960 it nonetheless produces `finite=False` output saturating f16 at
+/-65504. **The kernel is verified; the frame's launch of it is not.**

Two hypotheses tested here and refuted:

* **`k_export` dimension order.** `+0x0c` is height and `+0x10` width (launcher tracing, above). The
  runner had them swapped and it is now corrected - but coverage went 11.1% -> 6.0%, i.e. *worse*.
  Coverage is not a validity measure while the data feeding it is wrong, so the contract-derived
  order stays. Do not tune these fields by coverage.
* **HDR input range.** The difftest feeds RGB in [0,1); the capture reaches 65.12, and the ctx
  exposure scalars are passed as zeros. Scaling `k_import` by 1/65 changes block 0's output mean
  from -20.34 to -20.25 and nothing else - still saturating, still non-finite. Not the cause.

That block 0 barely responds to a 65x input change, while passing its difftest, says the frame is
mis-launching it rather than feeding it badly. Prime suspects, none yet tested: the sizes of the
`+0x38` / `+0x48` / `+0xa0` buffers at 25,680 workgroups (the difftest gives each a 1 MiB slot for
64 workgroups, and `act_bytes` may not be the right formula for any of the three), and the per-stage
geometry assumption `h = H >> i`.

### Honest status of the frame

| piece | status |
|---|---|
| `k_import` at 1707x960 | verified, correct |
| pre-block / block 0 | **verified, 0 mismatches at 8x8, 16x16, 64x64** |
| 22 encoder blocks, one C=512 block, one ViT block, a decoder stage | verified as chains |
| `k_dec_upsample`, `k_final_head` | registry PASS |
| `k_export` translation | registry PASS |
| blocks 24-30, 32-38, 40-47, 49-65 individually | **never tested** - same recipes, untested instances |
| the 169-dispatch frame chain | runs, 810 ms, guards intact, **output wrong** |
| ctx scalars `+0x50..+0x68` | unknown values, currently zero |
| `k_export` semantics | never validated - needs correct network output first |

The pattern of this project holds: every defect chased to ground so far has been a fixture or launch
parameter, never a mistranslated kernel. Four kernels were accused and exonerated (`k_attention`,
`k_mean`, `attention2`, and now the pre-block).

## CORRECTION: the network's activations were never garbage - they were read as the wrong format

`net_frame_full.py` reported every intermediate buffer as `finite=False`, saturating f16 at
+/-65504, and that was taken as evidence the chain was producing garbage. **It was a decoding
mistake in the reporting.** Activations are **e4m3**, one byte per value. Decoded correctly, the
same run at 1707x960 reads:

```
block0 out        nan=0.00%  min=-448  max=448  absmean=3.264   nonzero=49.9%
enc s1 pool C=32  nan=0.00%  min=-448  max=448  absmean=26.51   nonzero=25.0%
enc s2 pool C=64  nan=0.00%  min=-448  max=448  absmean=26.63   nonzero=24.9%
enc s3 pool C=128 nan=0.00%  min=-448  max=448  absmean=42.96   nonzero=24.9%
enc s4 pool C=256 nan=0.00%  min=-448  max=448  absmean=30.1    nonzero=24.5%
```

**Zero NaN at every stage**, values inside the e4m3 range (448 is its maximum), plausible means. The
encoder is producing valid activations from the real frame all the way through.

Read the format before concluding from statistics. `finite=False` and "+/-65504 saturation" is what
e4m3 always looks like through an f16 view, and f16 is what `report()` used.

### What is actually still wrong: `k_export`

Everything up to and including the decoder and `k_final_head` now produces finite, in-range data.
The output stage does not:

```
k_export surface : nonzero=6.0%  min=-512 max=512  (read as f16, which IS correct for RGBA16F)
```

Roughly 57 of 960 rows carry data. Tested and refuted here:

* **Dimension order.** `+0x0c`=height / `+0x10`=width per the launcher is now used. Coverage went
  11.1% -> 6.0%, but coverage was not a valid signal while the input was believed to be garbage; it
  is now, and neither order fills the surface.
* **Format/mode at `+0x28`.** Swept 0-3, identical output every time.
* **Aux buffer sizes.** `+0x38`/`+0x48`/`+0xa0` scaled 1x/2x/4x, identical output.

`k_export` has never been validated semantically - the contract flagged that it could not be, until
the network produced real output. It now does, so it can be. That is the next piece of work, and it
is the last one between here and an image.

### `k_export` write pattern: rows 0..69 of 960, then nothing

Measured on the destination surface after a full 169-dispatch run at 1707x960:

```
rows written : 70 of 960, indices 0..69, spacing 1 (contiguous)
cols written : 1707 of 1707 - every written row is COMPLETE
```

A hard stop after 70 rows, not a scatter and not a stride error - each row it writes, it writes
fully. Refuted as causes: the `+0x0c`/`+0x10` dimension order, the format/mode field at `+0x28`
(swept 0-3), the `+0x38`/`+0x48`/`+0xa0` buffer sizes (1x/2x/4x), and the kernarg being 280 B with
`grid_dims` at `+0x80` unset - that was a real bug (the metadata says 320 B) and fixing it changed
nothing.

Still untested: `+0x14`/`+0x18` (from `[rsp+0x308]`/`[rsp+0x310]`, meaning unknown - currently
passed as H,W), `+0x08` (i32 materialised from `xmm10`), the `+0x38`/`+0x3c` job strengths, and
whether `off_head` is the right size and format for what `k_export` expects to read.

## The per-stage recipes generalise: 24 block instances verified, not 2

The C=512 and ViT recipes were each decoded on a single block (23 and 31) and then reused for every
block in their stage on the assumption that a stage shares its recipe. That assumption was never
tested. `net_block512.py` and `net_vit.py` now take a `BLOCK` environment variable, so it can be.

```
C=512 recipe   blocks 23,24,25,26,27,28,29,30   all PASS, 0 mismatches
C=512 recipe   blocks 40,41,42,43,44,45,46,47   all PASS, 0 mismatches
ViT   recipe   blocks 31,32,33,34,35,36,37,38   all PASS, 0 mismatches
```

**24 of 24.** Each is a full 5-dispatch chain against the emulator with that block's own weights, so
the hand-offs are covered too, not just the kernels. `net_encoder_stage1.py` now takes the stage's
channel key as `argv[2]`, which lets the decoder stages (C=256/128/64/32) run through the same
harness rather than only the encoder's C=32.

Coverage after this: blocks 0, 1-22, 23-30, 31-38, 39, 40-47 and 70 are verified, individually or as
chains. That is the whole forward pass except the decoder blocks 48-69, which are running.

## SOLVED: `k_export` writes the whole surface. `+0x14` is a byte pitch, not a dimension.

`+0x14` is the **output row pitch in bytes**. The runner passed `H` (960) there, so each row started
960 bytes after the previous one and overwrote most of it. With `W * 8 = 13656`:

```
+0x14 = 960    :  69 of 960 rows touched
+0x14 = 13656  : 960 of 960 rows touched, all 1707 columns, every row complete
```

The 70 rows reconcile to the byte: the last written byte was 947951, and `947951 / 13656 = 69.42`.
Nothing was stopping early - the writes were landing on top of each other.

Corrected field meanings, read from the kernel at `0xab200..` rather than guessed:

| off | meaning |
|---|---|
| `+0x00` | input, **16 B per pixel**, read with `global_load_b96`, index `(s8*row + x)*16` |
| `+0x08` | input row stride in **elements** (was 0, so every row re-read row 0) |
| `+0x0c` / `+0x10` | height / width, used as the `row < H`, `x < W` guards |
| `+0x14` | **output row pitch in bytes** - the bug |
| `+0x18` | **format/mode** (compared against 3,5,6,4) - the runner had passed `W` here |
| `+0x20` | output; `+0x30` second output (RGB f32, 12 B/pixel) when `+0x38 != 0` |

Bytes per pixel by mode: modes 1,2,3,4,6 write 4 B/pixel; modes 0,5,7,8 write 8 B/pixel (RGBA16F,
which is what the destination is sized for); the stray 1707 fell through to a 16 B/pixel path.

## What is left: `k_final_head` writes a near-constant

The export surface is fully written but still non-finite, and that is now an input problem. The head
buffer that feeds `k_export` at `+0x00` is nearly empty:

```
HEAD_GRID = (1,1)              head nonzero 0.06%, 2 distinct byte values
HEAD_GRID = (ceil(W/256), H)   head nonzero 0.44%, 2 distinct byte values
HEAD_GRID = (ceil(W/8), H/8)   head nonzero 0.00%, 1 distinct byte value
```

The grid changes the result, so the kernel is running - it just writes a constant. `HeadParams` is
only **24 bytes**, three pointers and no dimensions, so the extent must come from the grid, and none
of the three conventions produces real output. `k_final_head` passes its registry difftest at 0, so
this is its wiring at frame scale, not the kernel.

## All 71 blocks are now verified

The decoder stages complete the sweep, each a full chain against the emulator with real weights:

```
blocks 48-55  C=256   0 mismatches -> PASS
blocks 56-61  C=128   0 mismatches -> PASS
blocks 62-65  C=64    0 mismatches -> PASS
blocks 66-69  C=32    0 mismatches -> PASS
```

| blocks | how | result |
|---|---|---|
| 0 | `difftest_preblock` at 8x8, 16x16, 64x64 | PASS |
| 1-22 | encoder chain, 22 dispatches | PASS |
| 23-30 | C=512 recipe, per block | PASS (8/8) |
| 31-38 | ViT recipe, per block | PASS (8/8) |
| 39 | `dec_upsample` registry difftest | PASS |
| 40-47 | C=512 recipe, per block | PASS (8/8) |
| 48-69 | decoder stages, 4 chains | PASS (4/4) |
| 70 | `final_head` registry difftest | PASS |

**Every block in the forward pass computes what the reference computes.** `k_import` is verified at
full resolution and `k_export` now writes its whole surface. Nothing in the network is known to be
mistranslated, and nothing is left unverified except two things, both about *launching* rather than
computing: `k_final_head`'s extent at frame scale (it writes a constant there while passing its own
difftest) and `k_export`'s remaining mode/format selection.

Worth stating plainly, because the earlier sessions' numbers invited the opposite reading: across
this whole project, **no kernel has yet been found to be mistranslated**. Every defect chased to
ground has been a fixture, a launch parameter, or a misread output format.

## `k_final_head`: the input pointer is advanced by 8192 B per y-workgroup

`HeadParams` really is three pointers in 24 bytes - the kernel's own prologue settles it:

```
s_load_b128 s[4:7], s[0:1], null      ; +0x00 and +0x08, two pointers
s_load_b64  s[8:9], s[0:1], 0x10      ; +0x10, the third
s_load_b32  s3,     s[0:1], 0x24      ; hidden group_size_x, masked to 0xffff -> 256
s_mov_b32   s2, s15                   ; s15 = workgroup_id_y
s_lshl_b64  s[12:13], s[2:3], 13      ; << 13 = * 8192
s_add_u32   s1, s4, s12               ; INPUT pointer += wg_id_y * 8192
s_addc_u32  s4, s5, s13
```

So `gy` indexes the **input** in 8192-byte steps, and the extent must cover the input buffer, not the
image height. Measured at 1707x960, head buffer filled:

| grid | head nonzero | distinct bytes |
|---|---|---|
| `(ceil(W/256), H)` = (7, 960) | 0.437% | 2 |
| `(1, input_bytes/8192)` = (1, 12810) | 0.062% | 2 |
| `(ceil(W/256)*H, 1)` flat | 100% | **1** (a single constant) |

0.437% of the 16 B/pixel window is ~7 x 16330 B, and 16330 is exactly what one workgroup writes in
the passing `final_head` difftest. So at (7, 960) the seven x-workgroups land in distinct places and
all 960 y-rows overwrite each other; at (1, 12810) every y-workgroup writes the same place.

**The output address therefore does not derive from `wg_id_y` in the prologue** - only the input
does. Where it comes from is the open question, and it is the last one before an image. Everything
else in the pipeline is verified.

Refuted here: feeding the head from the last decoder block's own output rather than the stage pool
buffer (identical result - the pool wiring was not the problem).

## The frame runner's schedule is wrong at both ends - `k_final_head` is not the tail

The driver's own phase table, recovered from the handle->name mappings and recorded in this file
long before the frame runner was written, says:

```
prologue      0x18002ea60               k_pre_block_1h_32_fp8, k_swin_var<32,true>
encoder loop  0x18002f630-0x18002f929   launcher A, 4 stages / 22 blocks
enc -> mid    0x18002f929-0x18002fd80   k_repack, k_final_head
ViT loop      0x18002fd80-0x1800302d2   k_expand2, k_contract2 x2, k_qkv2, k_attention2
mid -> dec    0x1800302d2-0x180030870   k_repack, k_dec_upsample, launcher B x4
decoder loop  0x180030870-0x180030b18   launcher A x2, launcher B x3
epilogue      0x180030b18-0x180031465   k_post_block_1h_32_fp8, k_swin_var<32,true>
```

`net_frame_full.py` gets four things wrong against it:

1. **`k_final_head` is in the enc -> mid transition, not the tail.** The runner dispatches it as
   "block 70" at the very end. Its name invited that; the schedule does not support it. Chasing why
   it writes a constant at frame scale was chasing the wrong kernel in the wrong place.
2. **The epilogue is `k_post_block_1h_32_fp8` followed by `k_swin_var<32,true>`** - the same kernel
   as the pre-block. Neither is dispatched at all.
3. **`k_repack` is missing from both transitions.** `RepackParams` is 32 B, two pointers plus four
   i32, and this file already describes it as pure index arithmetic with no arithmetic reduction -
   i.e. a layout change. That is exactly the e4m3-tiled to something-else conversion the tail needs.
4. **The prologue dispatches `k_pre_block_1h_32_fp8` before `k_swin_var<32,true>`.** The runner only
   does the latter.

Both missing kernels are built and present:
`_Z22k_post_block_1h_32_fp810PostParams.co`, `_Z8k_repack12RepackParams.co`.

This also resolves the apparent contradiction between `k_final_head` writing single e4m3 bytes
(`global_store_b8`, offsets 0/4/128/132/256/260/384/388) and `k_export` reading 16 B per pixel with
`global_load_b96`: nothing converts between them because the two kernels that do the conversion are
not in the chain.

**Read the schedule table before wiring a runner.** All four errors were avoidable from a table that
had been in this file for a day.

## Grid conventions are per kernel: the translation drops `workgroup_id_y` for some

The translated gfx1030 kernels do **not** share a grid convention. Some disable
`.amdhsa_system_sgpr_workgroup_id_y` and re-materialise the original's `s15` from `s2`, so the
original's y index arrives in the hardware **X** id. Those kernels are 1-D, and a grid in Y gives
every workgroup the same index - they all write the same place.

| kernel | `workgroup_id_y` | extent |
|---|---|---|
| `k_final_head`, `k_ffwd_inpview`, `k_conv_res_views`, `k_dec_upsample`, `k_repack` | **0** | 1-D, in X |
| `k_ffwd2`, `k_qkv_attn`, `k_contract2`, `k_qkv2`, `k_attention2`, `k_expand2`, `k_swin_var`, `k_import`, `k_export`, `k_pre_block`, `k_post_block` | 1 | 2-D |

**Within one C=512 block the kernels differ**: `k_ffwd_inpview` and `k_conv_res_views` are 1-D while
`k_ffwd2` and `k_qkv_attn` are 2-D. At the difftest's 8x8 everything fits in one workgroup and this
is invisible; at frame scale it is the difference between writing the buffer and writing 1/1000th of
it. `net_frame_full.py` now reads the flag per kernel via `wants_2d()` rather than assuming.

Effect on the C=512 buffers at 1707x960: **0.13% -> 48.1% of bytes written**. `k_final_head` with a
1-D grid of `(ceil(out_bytes/16384), 1)` = (1601, 1) goes from 0.44% to 99.999%, and on varying input
produces 255 distinct byte values instead of 2.

### Still constant, and why

Coverage is fixed; content is not. Every mid-network buffer still holds **2 distinct byte values**.
The C=512 stage reads the last encoder pool, which is healthy (+/-448, ~25% nonzero), but it reads it
at C=256 / 120x213 while running at 60x106. Nothing converts between them, because the kernel that
does is missing: the schedule's **enc -> mid** phase is `k_repack, k_final_head`, and the runner has
neither. The same applies at **mid -> dec** (`k_repack, k_dec_upsample, launcher B x4`).

So the grid work was necessary but not sufficient. The remaining gap is the four missing transition
and epilogue dispatches, not the grids.

## `k_repack` wired into enc -> mid: it works, the collapse is downstream of it

`k_repack` is now dispatched between the last encoder stage and the C=512 stage, per the driver's
phase table. Measured in isolation (`STOP_AFTER_REPACK=1`), reading its input and output buffers:

```
input   enc s4 pool  nonzero 24.54%  distinct 254     (healthy encoder output)
output  c512_1 w0    nonzero  3.86%  distinct 253     (real, varying data)
```

**`k_repack` produces valid varying output.** Its extent comes from the four i32 at
`+0x10..+0x1c`, not from the grid - sweeping the grid's per-workgroup span over 256/1024/4096/16384
changes nothing, while the third i32 scales coverage linearly (256 -> 512 doubles 3.86% to 7.81%),
so it is a channel count.

The collapse to 2 distinct byte values therefore happens **inside `c512_stage`**, which turns 253
distinct bytes into 2. That is now the first bad stage, and it is a much smaller target than "the
mid-network is constant".

Caveat on the dimension sweep: the first attempt measured the *downstream* buffer and reported all
four candidates as identical. They are not - measuring `k_repack`'s own output separates them.
Measure the output of the thing you are changing.

## The collapse is at block 25, and `c512_stage` is not at fault

Probing after every dispatch (`C512_TRACE=1`) and after every block (`C512_TRACE=2`) at 1707x960:

```
0 before stage       -> w0  distinct 242      (k_repack output, healthy)
1 ffwd_inpview       -> w1  distinct 254
2 ffwd2              -> w1  distinct 255
3 conv_res_1         -> w2  distinct 255
4 qkv_attn           -> w3  distinct 246
5 conv_res_2         -> w0  distinct 246      block 23 is entirely healthy

after block 23       -> w0  distinct 246
after block 24       -> w0  distinct 246
after block 25       -> w0  distinct   2      <- the collapse
after blocks 26..30, the whole ViT, block 39  distinct 2, then 1
```

**Blocks 23 and 24 work; block 25 destroys the data.** All eight share the recipe and their weight
files are byte-identical in size (L0 524288, L1 263168, L2 917568, L3 263168), so it is not a weight
or recipe difference. Two blocks succeeding and the third failing on the same code path points at
accumulation across blocks rather than anything block 25 does differently.

`k_dec_upsample` (block 39) writes **nothing at all** (0% nonzero), which is a second, independent
defect; the second C=512 stage then starts from an empty buffer.

**Correction to the previous section:** it said the collapse happens "inside `c512_stage`". That was
inferred from the final state of the `c512_1` buffers, which blocks 24-30 and the ViT stage reuse and
overwrite. Measured per dispatch, `c512_stage` is clean for block 23. Do not read a shared buffer's
end state as the output of the first thing that wrote it.

## The C=512 collapse is NaN, and it is not stable run to run

The collapsed byte is **`0x7f`**, not `0x7e`. In e4m3 `0x7f` is exponent 15 / mantissa 7 = **NaN**.
So this is not saturation at 448 - it is NaN filling 48% of the buffer while the rest stays zero.
The "accumulating overflow" reading of the earlier `distinct=2` was wrong.

Worse, the block at which it happens **moves between runs** of the identical command:

```
run A   after 23 distinct 246   after 24 distinct 246   after 25 distinct 2
run B   after 23 distinct 246   after 24 distinct   2
run C   after 23 distinct   2
```

Three runs, three answers. So "block 25 is the culprit" - recorded in the previous section - is
**not supported**; block 25 was simply where that particular run happened to tip over.

Ruled out as the mechanism: an inter-dispatch race. `net_run.cpp` records a `hipEvent` after every
`hipModuleLaunchKernel` and spins on `hipEventQuery` until it completes, so the 169 dispatches are
strictly serialised.

What that leaves, untested: a race *within* a dispatch across workgroups at frame scale (the earlier
non-determinism claim was retracted on the strength of `difftest_preblock` being deterministic, but
that is block 0 at 64 workgroups and says nothing about the C=512 kernels at thousands), and buffer
aliasing in `c512_stage` - for `bi > 0` it passes `work[0]` as **both** `+0x00` and `+0x08` of
`k_ffwd2`, which in the original are presumably distinct input and residual buffers.

NaN from a buffer that is zero-filled at the start needs a 0/0 or inf-inf somewhere; it cannot come
from merely reading uninitialised memory here.

## The C=512 non-determinism was an unwritten buffer; the NaN is `k_qkv_attn`

**Two separate defects, and the first is fixed.**

`k_ffwd2` takes `work[0]` at `+0x00`, but `work[0]` is written only by the block's **last** dispatch.
On the first block of a stage nothing has written it. `net_block512.py` never notices because
`V.build` fills its arena with e4m3-shaped random bytes, while the frame runner zero-fills - and a
normalisation over an all-zero buffer gives 0/0. Passing the stage input there for `bi == 0` makes
the chain **deterministic**: three identical runs now give 246 distinct byte values every time,
where before the same command produced 246, 254, 2, 1 in successive runs.

That also settles the earlier confusion over "which block collapses": it moved between runs (23, 24,
25) because the run was not reproducible, not because any block was special. Both the "block 25 is
the culprit" note and its retraction can now be read as symptoms of this.

**The remaining NaN is localised to one dispatch.** Per-dispatch probe of block 23:

```
0 before stage    distinct 241
1 ffwd_inpview    distinct 254   no 0x7f
2 ffwd2           distinct 255   no 0x7f
3 conv_res_1      distinct 255   no 0x7f
4 qkv_attn        distinct 245   0x7f = 0.99%     <- NaN appears here
5 conv_res_2      distinct 246   0x7f = 0.99%
```

0.99% of block 23's output is NaN, and block 24 spreads it to 48.15% - i.e. every non-zero byte.
NaN is contagious through a convolution, so the entire mid-network collapse traces back to this one
kernel producing ~1% NaN.

Working hypothesis, under test: the stage geometry is `1707>>4, 960>>4` = 106x60, neither a multiple
of the 8-wide attention window. The grid `(ceil(106/8), ceil(60/8))` = (14, 8) therefore covers
112x64, and a softmax over an empty overhanging window is 0/0.

### Two hypotheses for the `k_qkv_attn` NaN, both refuted

**Window tiling.** `106x60` is not a multiple of the 8-wide attention window, so the grid covers
`112x64` and an empty overhanging window would softmax to 0/0. Tested at four geometries:

```
60x106  (does not tile)  NaN 0.99%
56x104  (tiles exactly)  NaN 1.10%
64x112  (tiles exactly)  NaN 0.89%
48x96   (tiles exactly)  NaN 1.39%
```

Exact tiling does not help and the smallest geometry is worst. Not the cause.

**Sparse input.** The stage input is only 3.95% non-zero, which would leave whole attention windows
empty. Raising `k_repack`'s third parameter to widen its output makes things **worse**:

```
repack dim 256   input  3.95%  ->  qkv output 245 distinct, NaN  0.99%
repack dim 512   input  7.90%  ->  qkv output   2 distinct, NaN 48.15%
repack dim 2048  input 31.60%  ->  qkv output   2 distinct, NaN 48.15%
```

So sparsity is not the driver either, and `256` is the best value found rather than an obviously
correct one.

**Method note.** Both of these were parameter sweeps rather than reasoning from the kernel. The
useful next step is to read `k_qkv_attn`'s disassembly and find the division that can produce NaN,
then work out which input makes its denominator zero - not to sweep further.

### `k_qkv_attn` contains 138 real divisions - the NaN is probably honest

The kernel is 16,089 instructions and carries a full IEEE division sequence:

```
v_div_scale_f32  276      (two per division)
v_rcp_f32        138
v_div_fmas_f32   138
v_div_fixup_f32  138      <- the instruction that turns 0/0 into NaN
v_log_f32        137
v_rsq_f32          2
```

138 divisions, and roughly 1% of the output is NaN. `v_div_fixup_f32` producing NaN from 0/0 is
**correct IEEE behaviour**, and this is a GPU-only run with no emulator involved - so the hardware is
doing the right thing with the numbers it was given. The kernel is most likely reporting honestly
that something upstream hands it a zero denominator.

That reframes the search: do not look for a defect inside `k_qkv_attn`. Look for what makes ~1% of
its denominators zero. Candidates, none yet tested:

* the window origin at `+0x20`, passed as `(0,0)` for every C=512 block. The encoder cycles its
  shifted-window origins `(0,0), (-4,-4), (-4,0), (0,-4)` per block; if the C=512 stage does too,
  every block here is running the same window and some are degenerate.
* the weight at `+0x10` - `blockN_layer3` follows `net_full.py`, but that mapping has never been
  checked against the launcher for this stage.
* the stage geometry itself, which is inferred as `SRC>>4` and not read from the `ctx+0x190` tuple.

## RETRACTION: the C=512 chain is still non-deterministic, and several results above are void

Four identical runs of the identical command, probing the same dispatch:

```
run1  distinct  94
run2  distinct 245
run3  distinct 244
run4  distinct   2
```

The `work[0]` fix earlier in this file made three consecutive runs agree at 246 and was recorded as
having made the chain deterministic. **It did not.** Three agreeing runs was luck. The fix itself is
still correct - `k_ffwd2` genuinely read a buffer nothing had written - but it did not close the
only channel.

**Everything derived from comparing runs after that point is void**, and should not be relied on:

* the window-tiling sweep (60x106 / 56x104 / 64x112 / 48x96)
* the `k_repack` dimension sweep (256 / 512 / 2048)
* the shifted-window-origin test (`C512_SHIFT`)
* "block 25 is where it collapses", and the later "block 24", and the NaN percentages attached to
  each

All of those compared two runs and drew a conclusion from the difference. With a non-reproducible
chain that is measuring noise. The giveaway was visible and I missed it: `ENC_MODES[0]` is exactly
`(0, 0)`, so the `C512_SHIFT=0` and `C512_SHIFT=1` runs were bit-identical for block 23 and still
produced 243 and 2.

### What this means

A GPU-only run, dispatches strictly serialised by `net_run`, identical input bytes, and the output
still varies. That requires either a race between workgroups **within** one dispatch, or a read of
memory that is not initialised deterministically. The earlier retraction of the non-determinism
claim (made on the strength of `difftest_preblock` being deterministic) was about block 0 at 64
workgroups and says nothing about these kernels at thousands.

**Establish determinism before measuring anything else in this chain.** Any parameter study here is
worthless until the same command twice gives the same bytes.

### What is NOT affected

These come from single measurements, code reading or emulator difftests, not from run comparisons:
all 71 blocks verified; block 0's `+0x68` seed; `k_export`'s `+0x14` byte pitch and its 960/960 rows;
the per-kernel grid convention from `.amdhsa_system_sgpr_workgroup_id_y`; `k_ffwd2`'s unwritten-buffer
read; `k_qkv_attn`'s 138 IEEE divisions; the four missing schedule dispatches.

## The non-determinism is the PRE-BLOCK, and yesterday's retraction was itself wrong

A bisect over the 170-dispatch chain (`DET_BISECT=1`), hashing the whole arena and comparing two
identical runs per prefix:

```
prefix 85 -> DIFFERS      prefix 5 -> DIFFERS
prefix 42 -> DIFFERS      prefix 2 -> DIFFERS
prefix 21 -> DIFFERS      prefix 1 -> stable
prefix 10 -> DIFFERS
last reproducible prefix : 1        (k_import)
first non-reproducible   : 2
the dispatch that breaks it: #1  k_swin_var<32,true>  grid 214x120
```

**`k_import` is reproducible; the very next dispatch is not.** Everything downstream inherits it, so
every parameter study in this chain - the C=512 sweeps, the NaN percentages, "which block collapses"
- was measuring noise generated by dispatch #1.

**This means the retraction recorded earlier in this file was the error, not the original claim.**
`preblock_response.py` reported the pre-block non-deterministic; that was withdrawn because
`difftest_preblock` gave byte-identical output three times at 64 workgroups. The withdrawal was
wrong: a control at 64 workgroups cannot settle behaviour at 25,680. Scanning the threshold:

```
 16 workgroups  (TILE=32)   deterministic, 4/4 runs identical
 64 workgroups  (TILE=64)   NON-DETERMINISTIC
256 workgroups  (TILE=128)  NON-DETERMINISTIC
1024 workgroups (TILE=256)  NON-DETERMINISTIC
```

Tested and refuted as the cause: the output buffer being undersized. Scaling it 1x/2x/4x
(`ACT_MULT`) leaves dispatch #1 non-reproducible every time.

Open discrepancy: `difftest_preblock` at 64x64 is also 64 workgroups and *was* reproducible across
three runs, where `preblock_response` at the same count is not. The two differ in the runner
(`difftest_var.gpu` vs `net_run.exe`) and in arena contents; arena contents were tested and are not
it. That discrepancy is being re-checked with more runs, since three agreeing runs is exactly the
evidence that misled me once already.

## SOLVED: the pre-block's buffers were under-sized. The whole chain is now reproducible.

`act_bytes(C, H, W)` under-sizes the pre-block's buffers at frame scale. Doubling **all** of them -
the output and the `+0x38` / `+0x48` / `+0xa0` auxiliaries together - makes every one of the 170
dispatches reproducible:

```
ACT_MULT=AUX_MULT=1   first non-reproducible: dispatch #1 (k_swin_var<32,true>, grid 214x120)
ACT_MULT=AUX_MULT=2   the full chain is reproducible over 6 runs
ACT_MULT=AUX_MULT=4   the full chain is reproducible over 6 runs
```

Scaling only the output (`ACT_MULT` alone) does **not** work - the auxiliaries sit next to it and the
overflow lands in them. That is why an earlier 1x/2x/4x sweep of `ACT_MULT` appeared to refute the
under-sizing idea; it was scaling one buffer of four.

**2x is measured, not derived.** The exact per-workgroup requirement is unknown; what is established
is that 1x races and 2x does not, over four runs of all 170 dispatches.

### The control that misled, and why it was not wrong

`difftest_preblock` at 64x64 really is deterministic - five runs, identical `out_sha256`, 0
mismatches. It gives every pointer field its own 1 MiB slot, which is ample at that size.
`preblock_response` at the same 64 workgroups is not, because it sizes the output by `act_bytes`
(262,144 B) while the kernel writes more. Same kernel, same workgroup count, different buffer size.
So the two results were never in conflict; the harness was.

### What this restores

With the chain reproducible, measurements in it mean something again. Re-measured, the `k_qkv_attn`
NaN is real and stable:

```
1 ffwd_inpview  distinct 254   no NaN
2 ffwd2         distinct 255   no NaN
3 conv_res_1    distinct 255   no NaN
4 qkv_attn      distinct 245   0x7f = 0.99%
5 conv_res_2    distinct 246   0x7f = 0.99%
```

The hypotheses voided by the retraction - window tiling, `k_repack` dimensions, shifted-window
origins - can now be re-tested and will mean something.

## The geometry was wrong at every stage: the frame is PADDED to a multiple of 128

This file has recorded since 2026-09-21 that the host pads the frame to a multiple of 128 in both
dimensions (`((x+127)/128)*128`, `0x18002d3b0`) and then builds the stage table from the **padded**
size, with stage k at `C = 32 * 2^k` and `(H, W)` halved **(k+1)** times. `net_frame_full.py`
inferred `SRC >> k` instead - unpadded, and one halving short:

| stage | C | inferred | correct |
|---|---|---|---|
| 0 | 32 | 1707x960 | **896x512** |
| 1 | 64 | 853x480 | 448x256 |
| 2 | 128 | 426x240 | 224x128 |
| 3 | 256 | 213x120 | 112x64 |
| 4 | 512 | 106x60 | **56x32** |
| 5 | 1024 | 106x60 | 28x16 |

Every stage ran at four times the pixel count. `ImportParams` even carries the two separately -
`+0x10/+0x14` source, `+0x18/+0x1c` padded - and both were being given the source values.

**Effect on the pre-block, immediately:**

```
before   block0 out   min -448   max 448   absmean 3.264     (pinned at the e4m3 limit)
after    block0 out   min  -88   max  52   absmean 1.707     (no saturation at all)
```

That is the first time block 0's output has looked like activations rather than clipped noise. It
also explains the buffer under-sizing found this morning: `act_bytes` was not wrong, it was being
fed geometry four times too large, so the 2x multiplier was compensating for the wrong input.

The full chain still runs - 170 dispatches, 364 ms, guards intact - and `k_export` writes all 960
rows. What remains is that the **encoder** saturates: block 0 hands it absmean 1.7 and stage 1 comes
back at absmean 27 with values pinned to +/-448. Everything downstream is constant because of that,
not because of anything in the C=512, ViT or decoder stages.

### After the geometry fix: the encoder is healthy, and the NaN moved

Probing every encoder block (`PROBE_ENC=1`) with the corrected padded geometry:

```
enc block 1   nonzero 100%  distinct 216
enc block 2   nonzero 100%  distinct 253
...
enc block 10  nonzero 100%  distinct 253
```

**The encoder does not collapse.** The earlier reading that "stage 1 returns absmean 27 pinned to
+/-448" was measuring the *pool* buffer over its whole allocation - the pool feeds the next stage,
which is half the size, so most of what was measured was unused space. Measure a buffer over the
region the consumer actually reads, not over its allocation.

`REPACK_DIMS` still defaulted to the literal `60,106` after the geometry moved that stage to `56x32`;
it now derives from the stage table. That alone halved the saturation at the stage entry:

```
stale 60,106   ffwd_inpview  0x7e (=448) 4.31%   0xfe (=-448) 3.80%
from table     ffwd_inpview  0x7e        1.76%   0xfe         1.55%
```

Current state of the C=512 stage, deterministic and with correct geometry:

```
0 before stage   nonzero 13.78%  distinct 243
1 ffwd_inpview   distinct 254    saturation 1.76% / 1.55%, no NaN
2 ffwd2          distinct 255    no NaN
3 conv_res_1     distinct 255    0x7f = 1.79%     <- NaN starts here now
4 qkv_attn       distinct 192    0x7f = 3.57%
5 conv_res_2     distinct 243    0x7f = 3.57%
```

So the NaN source moved from `k_qkv_attn` to `k_conv_res_views`, which means the earlier attribution
was a consequence of the wrong geometry rather than a property of either kernel. Both readings were
taken on a deterministic chain; the difference is the geometry, not noise.

## The C=512 stage is verified at its REAL geometry - so the NaN comes from its input

`net_block512.py` now takes `BH`/`BW`. Run at the geometry the stage actually has in a 1707x960
frame - stage 4 of the padded table, `56x32`, which is only 7x4 workgroups and well within the
emulator's reach:

```
H=8   W=8    0 mismatches -> PASS
H=32  W=56   0 mismatches -> PASS
```

**All five kernels of the C=512 block are correct at the size they really run at.** That is an
elimination against ground truth, not an inference from symptoms - and it retires the whole stage as
a suspect, including both attributions of the NaN made earlier (`k_qkv_attn`, then
`k_conv_res_views`). Neither kernel is defective; they are being handed input that contains a zero
denominator.

### Method note, because three attributions in a row were wrong

Each of those came from reading NaN percentages in the frame chain and pointing at whichever
dispatch showed them first. That is symptom-chasing: in a chain where a dozen launch parameters are
guesses, fixing one moves the symptom to the next and each move looks like a new discovery. Three
"localisations" of the same NaN were three different guessed parameters upstream.

The reference test settles in one run what three rounds of that did not. **Where a difftest can be
run at the real geometry, run it before inferring anything from the frame chain.** Passing at 8x8
says nothing about 56x32, which is why this was worth redoing.

What that leaves: the input to the C=512 stage is `k_repack`'s output, which is 86% zeros. Whether
that is correct is unknown - `RepackParams`' four i32 are still guessed, and `k_repack` is the last
unverified thing between the healthy encoder and the verified C=512 stage.

## `k_repack`: `+0x18` is a work count, and activations are ONE byte per element

Read from the kernel (196 instructions) rather than swept. Its prologue:

```
s_load_b64  s[2:3], s[0:1], 0x14      ; s2 = [+0x14], s3 = [+0x18]
s_lshl_b64  s[8:9], s[4:5], 10        ; [+0x18] << 10  = [+0x18] * 1024
v_mad_u64_u32 v[2:3], null, s14, s15, v[0:1]   ; global_id = 256*wg + tid
v_cmpx_gt_u64_e64 s[8:9], v[2:3]      ; active while global_id < [+0x18]*1024
```

**`+0x18` is the total work in units of 1024 elements**, i.e. `C*ceil(H/4)*ceil(W/4)*16 / 1024`.
The other three: `+0x14` is a divisor (full reciprocal sequence) and is also taken `ceil(/4)`;
`+0x10` multiplies it; `+0x1c` is only ever compared against zero, so it is a flag, and 0 vs 1 makes
no difference here.

Confirmed by measurement - at stage 4 the derived value is 896 where the runner had been passing 256:

```
+0x18 = 256   repack output covers 13.78% of its buffer
+0x18 = 896   repack output covers 48.21%, 249 distinct byte values
```

### Activations are e4m3: one byte, not two

`act_bytes` multiplies by 2, which assumes f16. The activations are **e4m3, one byte per element**,
and with the correct work count that shows up directly:

```
ACT_ELEM=2   repack output 48.21% of the buffer   (the other half is simply unused)
ACT_ELEM=1   repack output 96.43%, 251 distinct   (the buffer is now the right size)
```

917,504 elements in an 1,835,008-byte buffer is exactly the 50% seen at `ACT_ELEM=2`. The earlier
conclusion that the element width is 2 was drawn with the **wrong geometry**, where over-allocation
kept the guards quiet and the unused half was mistaken for data.

**Still open:** `k_conv_res_views` produces NaN either way (1.79% at ELEM=2, 3.57% at ELEM=1), even
though the whole C=512 stage passes its difftest at this exact geometry. So the NaN is still coming
from something about the frame chain's data that the difftest's fixture does not reproduce.

## The NaN was a wrong weight mapping, found by diffing against the verified runner

`net_block512.py` passes its difftest at the real geometry, so it is ground truth for this stage.
Comparing the frame runner's kernargs against it field by field: the buffers all matched, the
**weights did not**.

```
verified (net_block512)   ffwd_inpview L0   ffwd2 L0   conv_res_1 L1   qkv_attn L2   conv_res_2 L3
frame runner (was)        ffwd_inpview L0   ffwd2 L1   conv_res_1 L2   qkv_attn L3   conv_res_2 L2
```

`k_ffwd_inpview` and `k_ffwd2` **share layer0**; the runner had shifted every dispatch from the
second onward, so four of the five kernels ran on the wrong weights. Correcting it removes the NaN
from the stage entirely:

```
before   conv_res_1 0x7f 1.79%   qkv_attn 0x7f 3.57%   conv_res_2 0x7f 3.57%
after    conv_res_1 254 distinct, no 0x7f anywhere in the stage
```

This is what three rounds of symptom-chasing had been calling "the `k_qkv_attn` NaN" and then "the
`k_conv_res_views` NaN". Neither kernel was ever at fault. **Diff against a passing runner before
theorising about a kernel.**

### Two more wiring fixes it exposed

* **`k_dec_upsample` writes through `+0x10`, not `+0x08`.** After block 39, `c512_2[0]` holds 243
  distinct byte values while the buffer at `+0x08` is entirely zero. The second C=512 stage was
  being fed the empty one.
* **The `mid -> dec` transition was missing.** The phase table has `k_repack, k_dec_upsample` there,
  and the C=512 stage ends at stage 4 (56x32, C=512) while the decoder starts at stage 3 (112x64,
  C=256). Both are now dispatched and both produce healthy output (254 and 239 distinct).

### State of the chain

```
block 0            healthy    min -88  max 52   absmean 1.71
encoder 1-22       healthy    100% coverage, 216-254 distinct
k_repack           healthy    96.43% coverage at one byte per element
C=512 23-30        healthy    253 distinct, no NaN
ViT 31-38          healthy    253-254 distinct
block 39           healthy    243 distinct (through +0x10)
C=512 40-47        healthy    252-254 distinct
mid->dec           healthy    repack 254, upsample 239
decoder 48         100% NaN   <- the remaining break
```

**Next**: the decoder blocks are dispatched with the encoder's kernarg convention and **no U-net
skip input**. This file records the skip wiring at `ctx+0x2a8`, "opposite indexing on each side",
and the runner wires none of it. A decoder block reading an unwired skip buffer is the obvious
candidate for 100% NaN from a healthy input.

## Decoder block 48: not the skip connections, and probably not a wiring detail at all

`ctx+0x2a8[r9d]` is passed to launcher A as its `in_ptr`, and the encoder indexes that array as
`[3-stage]` while the decoder indexes it as `[stage]`, so decoder iteration `di` reads the encoder
buffer of matching channel width. The runner wired none of this - it read the previous decoder
stage's output and never touched an encoder buffer.

Wiring the skips **changes nothing**: block 48 returns 100% `0x7f` with `DEC_SKIP=0` and with
`DEC_SKIP=1`. So the missing skip connection is a real omission and worth having fixed, but it is
not the cause.

The input feeding block 48 is healthy - the new `mid -> dec` transition measures 254 and 239 distinct
byte values - and the same kernel (`k_swin_var<256,false>`) at the same geometry (`stage_hw(3)` =
112x64) works correctly as encoder blocks 15-22. Same kernel, same size, healthy input, and it
returns all-NaN only in the decoder.

**The likely reason is structural.** The phase table describes the decoder loop
(`0x180030870`-`0x180030b18`) as **"launcher A x2, launcher B x3"**. Launcher B is the attention/FFN
dispatcher - the five-kernel recipe used by the C=512 blocks - so a decoder stage is *not* 22
`k_swin_var` blocks in a row, which is exactly how the runner dispatches it. Blocks 48-69 are listed
in the block table as "launcher A + launcher B", and that half of it has never been implemented.

**RETRACTED (later re-read, not carried across to this section).** The "Correction + decode: the
real attention/FFN block launcher is 0x180033660" section above (2026-09-22) already established
that the "launcher B" call sites in the phase table are not a kernel dispatcher at all - they are a
label-formatting/logging function (`0x180033600`, five lines, calls a printf-style tag formatter at
`0x18003f194`). The real attention/FFN launcher (`0x180033660`) has exactly **two** call sites, both
inside the two C=512 stages (blocks 23-30, 40-47) - never in the decoder loop. So the decoder was
never supposed to run a five-kernel recipe, and net_frame_full.py's plain `k_swin_var`-only decoder
dispatch is the structurally correct one. Block 48's all-NaN result (this section) had a different,
real cause, fixed later: FRAME_STATE_2026-09-24.md's `DEC_LAST2`/`DEC_SKIP`/skip-parity fixes to the
decoder's flag convention, confirmed by the column-stripe artifact dropping from phase-variance 0.99
to 0.01. Do not re-open "the decoder needs launcher B" - that idea was already dead before this
section was written.

This also puts the earlier decoder-stage verification in context: `net_encoder_stage1.py` passes for
blocks 48-55, 56-61, 62-65 and 66-69 at 0 mismatches - but it dispatches them all as `k_swin_var`,
i.e. it verifies the same wrong structure the frame runner uses. A chain runner can only verify the
schedule it was told to run.

## The decoder runs: its flag convention is not the encoder's

A static audit of the frame runner against the verified runners produced four findings that hold up
on hardware. The decisive one is the decoder's flags.

**The decoder's odd-sized record is the FIRST block of each stage, not the last.** Block 66's record
(22,784 B) matches the `f=8` extent, while block 69 (20,672 B) is the ordinary `f=0/1/2` size - the
mirror of the encoder, where the *last* block carries the larger `f=4` record. So decoder first
blocks take **bit 3**, and decoder last blocks take **no pool bit** at all:

```
flags = (1 if first) | (4 if last)     encoder convention, was used for the decoder too
        block 48  distinct 1, 100% NaN

flags = (8 if first) else 0            decoder convention
        block 48  distinct 245, no NaN
        block 55  distinct 239
```

With `flags=4` a decoder last block runs the fused pool and emits a C-doubled, H/W-halved tensor the
next stage cannot consume; with `flags=1` a first block reads an `f=8` record through the `f=0/1`
layout. Either is enough to produce all-NaN.

Consequence: nothing writes the pool in the decoder, so each stage hands its **last block's own
output** to the next stage and to the head.

### Three further findings from the same audit

* **ViT step 5 `+0x08` was `B270`, should be `B268`** (`net_vit.py:120`). The block's second contract
  input was taking `k_qkv2`'s projection instead of the post-FFN activation, in all eight blocks.
* **The ViT ran every kernel at grid (1,1)** at frame geometry. All five are 2-D kernels, so only one
  workgroup's tile was written and the rest of each buffer kept block 30's output. The "253-254
  distinct" health figure could not see this - most of it was passthrough.
* **The `mid -> dec` transition added earlier is withdrawn.** The phase table has one `k_repack` and
  one `k_dec_upsample` there and block 39 already is that dispatch; the decoder's odd first-block
  records are the upsample plus skip concat, so the upsample happens inside block 48's own
  `k_swin_var` call. The invented second dispatch was also running `k_dec_upsample` on block 48's
  `k_swin_var` weight record.

### Effect end to end

```
head input   0.00% nonzero, distinct 1      ->  11.66% nonzero, distinct 232
head buffer  distinct 2                     ->  distinct 255
```

The head produces real varied output for the first time. `k_export`'s surface is still non-finite
and the rendered image is still mostly black with structure only in the top band, so the output path
is not finished - but everything from `k_import` through `k_final_head` now carries data.
