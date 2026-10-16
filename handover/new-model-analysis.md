# DLSS-NR 310.9.16 "new model" (WEIGHTS_CC_SMALL_FP8)

DLL: `nvngx_dlssnr.dll` 310.9.16.0 (sha256 77796c07…f59d, unsigned, from the RenoDX Discord). It is only read as data (PE resources).
Tools: `tools/model_inspect.py --res/--diff-res`, `tools/flex_small.py`, `tools/build_plan.py --res`.

## Format
- Two RCDATA resources: `WEIGHTS_HT` (byte-identical records to 310.8, model id `1ff9997855c80dd5`) and
  `WEIGHTS_CC_SMALL_FP8` (new, 77 records, model id `ca6f64f1dd88b965`).
- Both use the same framing: u64 total resource size (the old "magic" 0x08cda732 was just 310.8's size), then records
  `u64 name_len, name, u64, u64, u64 D, u32 1, data[D], 20-byte trailer`. Matrices e4m3, tables f16/f32, layouts per layer family unchanged.

## Architecture vs 310.8 (scenario B: same building blocks, fewer of them)
| stage | 310.8 (HT) | 310.9 small |
|---|---|---|
| Swin 32 / 64 / 128 / 256 (per side, incl. pre/post) | 5 / 4 / 6 / 8 | 3 / 2 / 2 / 2 |
| split 512 per side | 8 | 2 |
| ViT 1024 (hidden 4096) | 8 | 8 |
| dec_upsample 512 | 1 | 1 |
| parameters | 145.76 M | 113.34 M (-22 %) |
| dispatches (build_plan) | 156 | 80 |

Post-block weights differ, so the baked `k_post_native` (310.8 weights) must not be used; build_plan detects this.
blend_scale 0.767 (310.8: 0.740). dlssnr-pytorch-flex's `Arch.from_weights` agrees with build_plan for both.

## How the DLL chooses HT vs CC_SMALL
Not determined. 310.9 adds strings for a generic `CCNetwork` executor (buffer planner, completion routes,
`cc-fused-swin-grid-v1:MMA_Tiled_NH4W4C`) and the config name `CC_SMALL_FP8`; it drops the
`DLSSNR: enabled=… localTone=…` and `ResizeNetwork` probe strings. Our assumption: CC_SMALL is the default in 310.9,
HT kept for fallback. The port therefore defaults to small when the resource exists; `nr_port.ini [nr] model=large` forces HT.

## Port wiring
- build_plan `--res WEIGHTS_CC_SMALL_FP8` writes meta `model_id` + `weights_res`, and a `res <name>` first line in weights.map.
- nr_runtime reads that line and copies weights from that resource.
- The DLL (model guard, ported from `model-guard` db76cb5) computes the model id of the chosen resource and uses
  `nr_data\<model_id>\<WxH>`; data for a different id turns NR off (pass-through).

## Results (RX 6900 XT, build_plan data, generic kernels, 0 native overrides, 2026-10-16)
Data: `E:\dlss-runtime\newmodel\nr_data\<model_id>\{960x540,1280x720}` (both models built by build_plan from the 310.9 DLL).

| | small (CC_SMALL) | 310.8 (HT) |
|---|---|---|
| nr_bench whole network 960x540 (exclusive, median) | 21.01 ms | 24.49 ms |
| nr_bench 1280x720 | 29.47 ms | 33.28 ms |
| PSNR vs dlssnr-pytorch-flex, 960x540, STR 1/1/1/1 | 39.79 dB (edit corr 0.926) | 40.95 dB (0.980) |
| strict test_sync (default config) | PASS 540p, PASS 720p | PASS 540p |
| flicker 540p, release config (history, vit_every 2, bottleneck) | visible 1.208 %, excess x1.10 | 1.018 %, x1.10 |

Only -14 % ms: the ViT (unchanged, 8 x 1024/4096) dominates. Every kernel fit (no PlanError); the release's native
overrides (c32fix data) are not applied to build_plan data yet, so absolute ms are above v0.3.1's.
Strict test_sync with the release config (history=1) reports mismatches for BOTH models: the reference path does not
model history, so this is the harness, not the new model.
