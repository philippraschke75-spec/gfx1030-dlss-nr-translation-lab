# Accuracy vs dlssnr-pytorch (2026-10-16, branch `accuracy`)

All numbers: one frame, no history, ours vs PyTorch reference output, PSNR peak 1 on the clamped linear RGB, same input
tensor. `tools/layer_cmp.py` dumps/decodes per-dispatch activations (tiles, planes, ViT tokens) and runs the reference
layer on our inputs ("local" error).

| step | 960x540 | 1280x720 | 2560x1440 | cause | fix |
| --- | --- | --- | --- | --- | --- |
| original comparison | 31.7 | 30.5 | 30.1 | setup: our default conditioning is tone 0 / structure 0, the reference's Controls() is 1/1/1/1 | compare with `STR=1,1,1,1` (torch_ref.py, layer_cmp.py) |
| same conditioning | 37.2 | 35.0 | 39.4 | | |
| + window shift axis | | | | bug: build_plan wrote VarParams X0=oy, Y0=ox, so phases 2/3 shifted the wrong axis (grid said the other one). Layers 3,7,11,... were 57-71% exact locally, now 100% | `X0=ox, Y0=oy` |
| + decoder phase | | | | bug: a 6-block decoder level must continue the encoder's phase count (starts at 2) | `start = 2` |
| + ViT extent | | | | bug: ViT field = pooled extent rounded up to 4 (12x16 at 540p, was 10x16); the extra tokens are zero and take part in attention | H6/W6 rounded |
| + pooled cells past the frame left zero | **41.0** | 27.9 | **42.5** | geometry: our padding is real image content, the reference's is near-zero; through the global ViT it moves every pixel | convpl ph/pw = ceil(H/64) |

Local per-layer error after the fixes: pre-block rel 0.02, every encoder/decoder Swin layer 100% bit-exact vs the
reference, ViT layers 99.4-100% exact (attention), 40 ViT layers together rel 0.09 (e4m3 one-step differences).

Remaining gap (why 720p got worse): the padded field. The reference / NVIDIA (OpenDLSS-NR docs/network.md,
Geometry::fromValid) use field = alignUp(valid, 2^reductions) (+align in x if both 4-unit aligned), mirrored padding,
level[k] = alignUp(ceil(level[k-1]/2), 4). We use a 128-multiple field with edge padding: at 720p the field is
768x1280 vs 768x1344, so the ViT sees 12x20 tokens vs 12x24, and the global attention diverges. Before the shift fixes
this happened to score better at 720p; it is luck, not accuracy (layer-exact encoder/decoder now). Fixing it needs the
reference geometry in build_plan; the obstacle is that our pooled outputs use the producer's W/2 as pitch while the
next level would be alignUp(W/2, 4) (e.g. 30 -> 32 at the mid stage): the pool writers need the consumer's pitch.

Precision (class b, not changed): our GEMMs accumulate f16 products in f32 (v_dot2); NVIDIA's tensor cores use F13
truncated fixed point per 16-product group (OpenDLSS-NR numerics.md). PyTorch uses f32 too, so it is invisible against
PyTorch but is the next step towards NVIDIA bit-exactness. Requant order (f16 then e4m3, RNE, saturate 448) matches.

Release config (960x540, history, vit_every=2): flicker x1.23 -> x1.20, visible flicker 1.97% -> 1.77%.
nr_bench 960x540 (exclusive): 24.48 ms before and after.

## Round 2 (geometry, conditioning, NaN)

- Geometry: `build_plan.geometry()` = OpenDLSS-NR Geometry::fromValid (field aligned to 2^reductions, levels
  alignUp(ceil/2, 4)); new native `k_mirror_pad` (rdna2/native/mirror.hip) mirrors the padding after k_import; the swin
  pool writer (+0x38) and decoder upsample reader (+0x30) use the consuming level's pitch alignUp(W/2, 4) (unchanged
  where W/2 is already 4-aligned). Build plan data with `--kernels` pointing at the rebuilt native modules.
- Conditioning: data dirs bake NVIDIA's default (tone 1, structure 1, skin = structure, auto mask, style 0). Runtime
  `nr_set_conditioning`; the DLL passes the host's intensity/style/localStructure/localTone/skinStructure/useAutoMask
  (create + evaluate_v2); intensity is now the output blend (gain x intensity). ini overrides: `intensity`, `style`,
  `local_tone`, `local_structure`, `skin_structure`, `auto_mask` (empty = host value).
- NaN -> +0 in q8 / e4m3_byte of the native modules (pre/post are prebuilt from vendor tables, not rebuilt).

| dB vs PyTorch (default conditioning both sides) | 540p | 720p | 1080p | 1440p |
| --- | --- | --- | --- | --- |
| v0.3.1 | 37.2 | 35.0 | - | 39.4 |
| round 1 | 41.0 | 27.9 | - | 42.5 |
| round 2 | 43.2 | 43.5 | 43.8 | 46.0 |

nr_bench (exclusive, vit_every=1): 540p 25.0 -> 22.0 ms (field 576x960 instead of 640x1024), 720p 33.7 -> 35.9 ms
(768x1344 instead of 768x1280). Strict test_sync 720p: 300/300 frames matched.

Flicker (release config): 540p 1.97 % visible (x1.23) -> 3.12 % (x1.33); 720p 2.13 % (x1.22) -> 3.50 % (x1.41). This is
the conditioning, not the geometry: with `local_tone=0` the 540p run is 1.83 % (x1.21), better than v0.3.1. NVIDIA's
default tone 1 makes the edit about twice as strong (1440p mean |edit| 0.011 -> 0.024).

PNGs (2560x1440, 1080p capture resized, gamma 2.2 for viewing): before = v0.3.1 with its default, after = this branch
with NVIDIA's default, and the PyTorch reference.
