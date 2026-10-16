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

PNGs: 2560x1440 (1080p capture resized), conditioning 1/1/1/1, gamma 2.2 for viewing: before, after, PyTorch.
