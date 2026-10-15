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
   * Native kernels go **straight into the manifest** with their own grids. No translated anchor and no override is needed.
   * Only pre and post keep the override form, because nr_runtime finds them by the names `k_pre_block`/`k_post_block`.
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
* The native kernels still declare private copies of their structs. They match `nr_params.h` (`plan_check` and the
  bit-exact runs show it), but they do not `#include` it yet. Their sources are outside the repo.
* The activation sizes keep v0.3's generous formula: 2 B/element, plus 2x on the half-resolution pre output. They are
  not minimised.
* Two runtime things still keyed by name: pre/post detection, and `vit_every<-1`, which uses ±33 instead of `bn_lo/bn_hi`
  (the new-model-pipeline runtime reads bn_lo/bn_hi from meta). With a different C_mid block count, bottleneck reuse
  needs that runtime change.
