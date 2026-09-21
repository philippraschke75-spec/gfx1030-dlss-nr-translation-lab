# Evidence from the NVIDIA host DLL `nvngx_dlssnr.dll` (static, read-only; 2026-09-21)

File: `G:\dlss\Dlss5 2224 1 2026-08-28T18-21Z ZWdvjgs49\dlss5\nvngx_dlssnr.dll`, 165,840,496 B, NGX snippet `rel_310_8`.
Method: `.rdata` strings + `llvm-objdump -d` cross-references (string VA -> referencing code addresses). Everything below is a
string or reference present in the binary; behavior claims are labelled as inference.

## Model inputs/parameters (string names; DLSSNR.* NGX parameter keys)
Textures: Color, MVec, Depth, Output, ControlMask, UI, UIAlpha, Backbuffer, BidirectionalDistortionField (each with
SubrectBaseX/Y/Width/Height). Scalars: MVecScaleX/Y, Intensity, LocalToneStrength, LocalStructureStrength, SkinStructureStrength,
UseAutoMask, Reset, DepthInverted, Enabled, UICorrection, Width/Height, Hint.Render.Preset.
Internal RGBA16F textures: `dlssnr_prev_output` (temporal history), `dlssnr_network_output_scratch`, `dlssnr_original_color`.
=> A colour-only capture is NOT sufficient input: the network needs Color + MVec + Depth (+ history, params, optional masks).
   MVec/depth conventions (scale, inversion, jitter) are host-provided; the game-side values must be captured.

## Weights
Config `CC_Control_History_Blend_Quantize_With_Teacher_honest_tench_2026_07_04_22_30_weights`, preset `CC_SILVER_AARDWOLD`,
loaded by `CG2RLoadWeightBlob` via FindResource("WEIGHTS_HT") (`cg2r_weights.cpp`). Weights go to a device "CG2RWeightHeap"
(aligned per tensor, uploaded with CopyHostToDeviceBuffer); a serialized weight-info map (`deserialize_weight_map`: device,
offset, size) is used. Cubin formats named by the DLL: sE4M3, E5M3, R16F, sE4M3_HWC32, R16F_HWC16, RGBA16F, etc. (=> signed FP8 e4m3 +
f16 payloads, consistent with the WEIGHTS_HT segment analysis in WEIGHTS_HT_FINDINGS.md).

## Network structure (block/layer classes, all named in the binary)
Blocks: cc_tinlayout_pre_block / _fused_pre_block_swin_1h, cc_tinlayout_fused_swin_{1h,2h,4h}_block (+8h), cc_split_swin_16h_block,
cc_vit_block, cc_vit_1d_block, cc_tinlayout_avg_pool_proj_block, cc_tinlayout_upsample_skip_block, cc_tinlayout_post_block,
fused_post_block_swin_1h, single_layer_block, cc_dec_input_upsample.
Specializations: Swin1H Cin=32, Swin2H Cin=64, Swin4H Cin=128, Swin8H Cin=256, SplitSwin16H Cin=512, Vit Cin=1024 (FFN contract 4096),
DecInputUpsample 1024->512. Layer variants per swin kernel: base, `_upsample`, `_ds`, `_inpview`, `_outview`, and sync modes
`_wait`, `_chained`, `_tilesync`, (`_inpview_tilesync`, `_upsample_tilesync`, `_ds_wait`, `_outview_wait`), each with `_fp8`.
Post block variants: `_simple_blend`, `_control_mask`, `_full_rect`, `_rgb`. Pre block: `..._32_1(_ds)(_fp8)`.
Layer tensor names inside blocks: layer0.dw_weight/conv_weight/sin, input_adapter_weight, weight0/1/2, layer1.weight3+ffn_cos_skip,
layer2.qkv_weight+attn_scale+attn_bias, layer3.projection_weight+attn_cos_skip, layer4.dw_weight/weight, inp_upsample_input_scale/sin,
out_gain, out_conv_weight, blend_scale. (matches the 4/5-layer records seen in WEIGHTS_HT: 524288/263168/917568/263168[+layer4]).
Pipeline order in the DLL: pre-block (requires RGB input) -> Swin encoder blocks -> ViT/1D-ViT middle -> decoder upsample-skip blocks ->
post block (blend with original colour / control mask) -> Output.

## Compute path in the DLL (not usable on AMD)
`.data` holds 15 zstd-compressed NVIDIA cubins (ELF with nv.info, sm_XX); launched through NGX cubin backends
(NGXCubinD3D12/CUDA::Dispatch, CubinBackendNGX::launch). The AMD port replaces these with the gfx1100 kernels we translate.

## Still not recovered
Per-launch kernel argument structs (the layer `forward` code that fills them), grid/block dimensions per layer, buffer-pool layout,
tile-sync flag protocol, the exact block graph (source-block indices) and the input packing of Color/MVec/Depth.
Next: read the layer `forward` code (e.g. Swin1H region ~0x180062c94..) for argument construction.
