# Our own frame plan (branch `own-plan`, 2026-10-15)

`tools/build_plan.py` builds the complete `nr_data\<W>x<H>\` from `nvngx_dlssnr.dll` and our kernel dir. It does not use
the installer host, `net_frame_full.py` or any captured manifest. The consumer (`nr_runtime`) and the DLL are unchanged.

## Run it
```
py -I tools/build_plan.py <nvngx_dlssnr.dll> 960x540,1280x720 E:\dlss-runtime\own_plan\nr_data [--kernels DIR] [--no-flex]
py -I tools/build_plan.py --selftest                      # synthetic shrunk models, CPU only, ~5 s
```
* Takes about 1 minute: the FNV hash of the weights for `weights.map` is pure Python.
* `--kernels` defaults to `E:\dlss-runtime\c32fix\nr_data\960x540\kernels` (the v0.3 native and translated `.co`).
* The flex cross-check needs `E:\dlss 5\dlssnr-pytorch\.venv`. Without it the check is skipped with a message.
* For a test run, put `nvngx_dlssnr.dll` in the parent of `nr_data`, as the runtime expects (`E:\dlss-runtime\own_plan\`).

## What it does
1. **Records.** Reads the WEIGHTS_HT records (names and sizes) with `rdna2/emu/weights_ht.py`. The bytes are read only
   for `blend_scale` and the `weights.map` hash. No weights are written: `arena.lzms` is all zero, and the runtime
   copies the weights from the DLL by record name, as in v0.3.
2. **Architecture** (`derive`):
   * Levels are runs of equal-size Swin records. In the encoder an odd record closes a level; in the decoder it opens one.
   * The C_mid run, the ViT run (layer3 = 2 B) and dec_upsample sit between them.
   * Widths come from closed forms that are all checked:
     * C_mid: layer1 = C²+2C
     * heads: layer2 = 3C²+256C+4·heads
     * C_vit: ViT layer4 = C²+2C
     * hidden: ViT layer0 = hidden·C_vit+16
     * head: C_mid·C_vit+16
     * dec_upsample: C_mid·C_vit+2·C_mid
   * Level k has width C_mid >> (n-k), and the decoder level sizes must mirror the encoder's.
   * Cross-check: `Arch.from_weights` of dlssnr-pytorch-flex must give the same result, or the build stops.
3. **Plan** (`plan`). Every buffer is sized from the derived widths and the padded frame, with PAD = 2^(n+3).
   * Dispatch order: import, pre, encoder levels, C_mid enc (4 per block), head, repack, ViT (5 per block), repack,
     dec_upsample, C_mid dec, decoder levels, post, export. That is 156 for 310.8.
   * `meta` carries `vit_lo/vit_hi/bn_lo/bn_hi` from the plan.
4. **Kernarg packing.** The kernarg structs are defined once in **`rdna2/plan/nr_params.h`**: field meanings and
   offsets, with `static_assert`s. The generator parses that header; nothing is hand-offset in Python.
   * Slot: 424 B, then the HIP hidden block (block_count, group_size, grid_dims) after the struct.
   * Field values follow our difftested contracts. The installer's spare/debug pointers are gone.
5. **Kernels** (`choose`). For each op it takes the first native kernel that is present in the kernel dir and supports
   the derived widths; otherwise the translated one, if it exists and was compiled for these widths; otherwise it fails.
   * Native kernels go **straight into the manifest** with their own grids, pre/post included; `overrides.txt` is empty.
     nr_runtime finds pre/post by `meta pre_step/post_step` (needs the own-plan runtime; older runtimes find them by name).
   * `k_post_native` has the 310.8 block70 weights baked in. It is used only if the DLL's block70 record has the same
     FNV (`POST_310`); otherwise the translated post-block is used and a message says so.
   * Every missing kernel is reported in one error, for example:
     `no kernel for var C=24: native k_swin_c24 (needs C % 32 == 0); translated _Z10k_swin_varILi24... (not in <dir>)`.

## Validation (RX 6900 XT, runtime `E:\dlss-runtime\build_v03rc`, reference v0.3 = `E:\dlss-runtime\c32fix\nr_data`)
| check | 960x540 | 1280x720 |
|---|---|---|
| `plan_check.py struct` (dataflow, scalars, weights, buffer bijection) | PASS, 50 buffers | 1 diff: v0.3 crops the post-block grid (160x90), we run the padded 160x96 |
| `plan_check.py run`, history off, 4 frames | **bit-exact** | **bit-exact** |
| `plan_check.py run`, history on (`hist_valid` frames 1-3) | **bit-exact** | 35-37 dB (from the post crop, see below) |
| arena | 623.9 -> **378.5 MB** | 860.9 -> **492.8 MB** |
| vs PyTorch flex (`torch_ref.py`, one frame, clamped linear input) | 31.7 dB, edit corr 0.816 | 30.5 dB, edit corr 0.764 |

* **Post crop.** v0.3's 1280x720 data was generated with `POST_CROP=1` (post grid = only the source tiles); its
  960x540 data was not. With history on, the uncomputed margin changes the output by about 35 dB. That is a lot for
  border pixels and is worth a look on the runtime side, because history may be read with a padded-vs-source stride
  mismatch. We always run the full padded grid. With history off, both forms are bit-exact.
* **PyTorch.** About 31 dB is the known level of our port vs dlssnr-pytorch on mode-0 data: `flicker_torch.py` measured
  31.5 dB for the v0.3 combo data, and v0.3 is bit-identical to this plan. The 45-49 dB figure is PyTorch vs NVIDIA,
  not our port vs PyTorch.
* **Selftest** (`--selftest`), all PASS. The shrunk tables are built from the real record list:

  | case | result |
  |---|---|
  | 310.8 | 156 dispatches |
  | drop one encoder block only | "decoder level sizes ... do not mirror" |
  | drop one block in encoder level 0 and its decoder mirror | 154 |
  | drop a ViT block | 151 |
  | drop one C_mid enc + dec block | 148 |
  | drop the head block | "mid section is not ..." |
  | C_mid 384 / 12 heads | lists every missing kernel (pre/post C0=24, k_swin_c24/48/96/192, c512 natives) |
  | ViT 768/3072 | lists head/repack/expand/... |

## For the new model (2026-10-16)
1. `py -I tools/model_inspect.py <new dll>` (branch new-model-prep) for a first look.
2. `py -I tools/build_plan.py <new dll> 960x540 <dir>`. Three outcomes:
   * It builds. Only block counts changed, and the plan follows them. Test with `plan_check.py run` (output sanity),
     `torch_ref.py` (expect about 31 dB vs flex again) and test_sync.
   * `PlanError: no kernel for ...`. The message lists every op and width that needs a kernel:
     * Swin width C: add `extern "C" ... k_swin_c<C>` to `native-mid/swinvar.hip` (templated, C % 32 == 0).
     * C_mid/heads/ViT changed: the c512/vit/decup natives are hardwired (`native_candidates` says which).
     * Pre/post C0 != 32: no kernel exists.
   * `PlanError` from `derive` (unknown record, non-mirrored levels, a width formula that does not hold): a new
     layer family, so scenario C in `new-model-day1.md`.
3. If its block70 record differs, `k_post_native` drops out automatically. Rebuild it from `post_weights.h`
   (`native-prepost-post/gen_weights.py`), or use the post-noweights kernel when that lands.

## Coordination: branch no-translated
Still translated: import, export, repack, attention2 (ViT), final_head, conv_res_views, and the pre/post anchors.
When native versions land, add one entry per op to `native_candidates()`: `(module, symbol, grid, threads, why-not)`.
The plan then puts them straight into the manifest. Their kernarg must be the struct from `nr_params.h`, so the native
kernel should `#include` it. Not pushed yet when this was written.

## Open
* Pre/post native kernels are still prebuilt binaries (see round 2).
* The activation sizes keep v0.3's generous formula: 2 B/element, plus 2x on the half-resolution pre output. They are
  not minimised.

## Round 2 (same day)
### 1. v0.3 at 1280x720 with history: no stride bug
* Layouts agree. The runtime keeps history as RGB32F over the padded frame (`hist_end`: netout 16 B/px to 12 B/px,
  `pad_w*pad_h` pixels). Pre +0x40 and post +0x40 both index it with the same padded pixel index.
* What differs is the crop. v0.3's 1280 data runs the post-block only over the 160x90 source tiles. The result's
  bottom margin (rows 720-767) is never written, so the history there stays 0. The pre-block reads that zero history,
  and the effect spreads through the global ViT.
* Static 2-frame sequence, frame 1 with history: crop vs full is 56 dB. The difference is spread over the whole
  image and slightly larger toward the bottom (mean |d| 0.0007 at the top, 0.0029 in the last rows). The 35 dB seen
  before came from 4 shifted frames, where history accumulates.
* Against PyTorch with history (`HIST=1 tools/torch_ref.py`, flex with pre lanes 7-9 + post lerp, mirrored padding):
  crop 30.50 dB, full 30.51 dB, 960x540 31.49 dB.
* **Verdict:** v0.3's 720p is not buggy in layout. Its margin history is zero instead of computed. Against the
  reference that is equal within 0.01 dB, so it is invisible next to the port's ~30 dB error. Full compute (this plan)
  is the consistent choice.

### 2. bn_lo/bn_hi and 3. pre/post by index
`nr_runtime.cpp`:
* Bottleneck reuse (`vit_every < -1`) takes the range from `meta bn_lo/bn_hi` when present. Without them it falls back
  to the old ±33 and the name check.
* Pre/post are tagged by `meta pre_step/post_step`, with the name match as the fallback.
* Built with hipcc into `E:\dlss-runtimeuild_ownplan
r_runtime.dll`. Checked at 960x540 against v0.3 data on the
  same runtime, 4 frames each:
  * history on: bit-exact
  * history on with `VIT_EVERY=-2` (bottleneck reuse): bit-exact
* 1280x720 history off: bit-exact.

### 4. Native kernel sources (`rdna2/native/`)
* `swin_var.hip`, `swin_c32.hip` + `swin_common.h`, `c512.hip`, `vit_swz.hip`, `decup.hip` + `nk_common.h`. These are
  the exact sources of the v0.3 modules. They were found by rebuilding every copy in the worktrees and comparing
  `.text/.rodata` with the shipped `.co`.
* The private struct copies are removed. Each file includes `nr_params.h`, and the header now uses the kernels' field
  names (H/W/X0/Y0/T/xp/resp/skip/K/KN/four/out[3]/H5/W5).
* `sh rdna2/native/build.sh <out>` rebuilds all five.
  * `.text/.rodata` are byte-identical to v0.3; only the metadata differs.
  * A plan built with them (`--kernels` = v0.3 dir + rebuilt modules) is bit-exact at 960x540 with history.
  * The generated `kernargs.bin` is byte-identical to the one before the field renames.
* Not brought in: `pre_native.hip` / `post_native.hip`.
  * They need `pre_tables.h` / `post_weights.h`, which are vendor-derived.
  * No source copy in any worktree rebuilds the shipped modules: the shipped post is a plain ELF, probably assembled
    from an edited .s.
  * They stay prebuilt. They only read the first 0x50 B, which equals `PreParams`/`PostParams`.

### 5. no-translated
Not on origin at the time of this round. Add its kernels to `native_candidates()` when it lands.
