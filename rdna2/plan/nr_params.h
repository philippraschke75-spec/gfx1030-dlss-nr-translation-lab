// nr_params.h - kernel argument structs of the DLSS-NR frame plan. ONE definition, shared by
//   * tools/build_plan.py (parses this file: field order, types, offsets) to pack kernargs.bin, and
//   * the native kernels (#include it instead of a private copy of the struct).
//
// Rules the parser relies on (keep them):
//   * one field per line, `type name;` or `type name[N];`, natural alignment, explicit `_pad` fields
//   * types: any pointer (8 B), uint64_t, int32_t, uint32_t, float, uint16_t
//   * `static_assert(sizeof(X) == N)` after each struct: the parser checks its own size against it
//
// Kernarg slot = struct, then the HIP hidden-argument block at align8(sizeof struct) (AMDGPU code object v5:
// block_count xyz u32, group_size xyz u16, remainder xyz u16, 16 B reserved, global_offset xyz u64, grid_dims u16).
// The plan fills block_count, group_size and grid_dims, everything else stays 0. Every slot is NR_KA_SLOT bytes.
//
// Notation: "act(C,h,w)" = activation buffer of C channels over an h x w extent, stored as 4x4 tiles
// (C * ceil(h/4) * ceil(w/4) * 16 elements). Stage s of an n-stage encoder has extent PAD >> (s+1)
// (PAD = frame padded to 2^(n+3)); the C_mid stage is stage n, the ViT stage n+1.
// Field meanings were established by difftests of every kernel (rdna2/emu/*FINDINGS.md, VARPARAMS_HOST_CONTRACT.md);
// this file is the contract from here on, not the disassembly.
#pragma once
#include <stdint.h>

#define NR_KA_SLOT 424

// k_import: game colour (RGBA16F) -> padded RGB32F working image.
struct ImportParams {
  const void* src;        // 0x00 input RGBA16F, src_w x src_h
  int32_t src_pitch;      // 0x08 bytes per input row (src_w * 8)
  int32_t src_format;     // 0x0c 0 = RGBA16F
  int32_t src_h;          // 0x10
  int32_t src_w;          // 0x14
  int32_t pad_h;          // 0x18 padded extent (multiple of 2^(n+3))
  int32_t pad_w;          // 0x1c
  float* rgb;             // 0x20 out: pad_w x pad_h RGB32F (12 B/px); padding is written with edge/zero values
  int32_t mode;           // 0x28 0 = raw colour; else max(0,scale*x), x/(1+x), sRGB OETF (the DLL reads this field)
  float scale;            // 0x2c used when mode != 0 (the DLL reads this field)
};
static_assert(sizeof(ImportParams) == 0x30, "ImportParams");

// Mirror padding of the imported image (native, after k_import): pixels past the valid w x h take the colour at
// 2*valid - x - 2 per axis (reflection without edge repeat), as NVIDIA (OpenDLSS-NR network.md, dlssnr-pytorch _mirror).
struct MirrorParams {
  float* rgb;             // 0x00 ImportParams.rgb, pad_w x pad_h RGB32F
  int32_t h, w_;          // 0x08 valid extent
  int32_t pad_h, pad_w;   // 0x10 padded extent
};
static_assert(sizeof(MirrorParams) == 0x18, "MirrorParams");

// Pre-block (block 0): RGB -> C0 features at full and half resolution.
struct PreParams {
  const float* rgb;       // 0x00 ImportParams.rgb
  uint8_t* out_full;      // 0x08 act(C0, pad) - read back by the post-block (outermost skip)
  const uint8_t* w;       // 0x10 record block0.layer0.layer
  int32_t h, w_;          // 0x18 pad_h, pad_w
  float in_scale;         // 0x20 0.0625 (input gain; at 0 the RGB input is ignored)
  int32_t _r24;           // 0x24 0
  float tone, structure;  // 0x28 features 11 local tone, 12 structure (auto mask: 1); NVIDIA default 1, 1 (runtime: nr_set_conditioning)
  int32_t _r30, _pad34;   // 0x30 feature 10 = style / 128 as f32 (0 = default style), 0
  uint8_t* out_half;      // 0x38 act(2*C0, stage 0) - input of encoder stage 0 (C0 channels, 2x headroom)
  const float* hist;      // 0x40 previous output RGB32F (12 B/px, padded) or null; set per frame by the runtime
  float a, b;             // 0x48 features 13 skin, 14 other structure (auto mask; without it -1, -1); default 1, 1
  const uint8_t* src;     // 0x50 fused k_import (native pre): ImportParams.src, or null = rgb comes from a k_import step
  int32_t src_h, src_w;   // 0x58 (pitch src_w * 8, RGBA16F, mode 0); rgb is then unused
};
static_assert(sizeof(PreParams) == 0x60, "PreParams");

// Swin block of width C (encoder and decoder levels), one 8x8 window per workgroup.
struct VarParams {
  const uint8_t* in;      // 0x00 block input act(C, h, w)
  uint8_t* out;           // 0x08 block output act(C, h, w)
  const uint8_t* w;       // 0x10 the block's record
  int32_t H, W;           // 0x18 level extent
  int32_t X0, Y0;         // 0x20 window origin by block index: (0,0) (-4,-4) (0,-4) (-4,0)
  uint32_t flags;         // 0x28 1 first encoder block, 4 last encoder block (also writes p38 = pooled 2C, h/2),
                          //      8 first decoder block (upsamples p30, concat with skip `in`), 2 last decoder block
  uint32_t pad0;
  const uint8_t* p30;     // 0x30 flag 8: the coarser level's output act(2C, h/2, w/2); else null
  uint8_t* p38;           // 0x38 flag 4: pooled output; other blocks get the level's pool buffer (unused)
  uint64_t _r40[12];      // 0x40..0x9f zero
  uint8_t* scratch;       // 0xa0 8 KiB per workgroup; only the translated k_swin_var uses it (null for native)
};
static_assert(sizeof(VarParams) == 0xa8, "VarParams");

// --- C_mid stage (split Swin, 4 dispatches per block). T = ceil(h/4)*ceil(w/4) tiles. ---
struct Ffwd2Params {      // step 1, record layer0
  const uint8_t* x;       // 0x00 block input (null on the stage's first encoder block)
  const uint8_t* xp;      // 0x08 stage input (planes) on the first encoder block, else null
  uint8_t* out;           // 0x10
  const uint8_t* w;       // 0x18 layer0
  int32_t H, W, T;        // 0x20 extent, tiles
  int32_t _pad2c;
};
static_assert(sizeof(Ffwd2Params) == 0x30, "Ffwd2Params");

struct Conv2Params {      // steps 2 and 4 (residual 1x1 conv), records layer1 / layer3
  const uint8_t* x;       // 0x00 branch input (tiles)
  const uint8_t* res;     // 0x08 residual (block input, tiles), null when resp is used
  const uint8_t* resp;    // 0x10 residual (planes) on the stage's first encoder block (step 2), else null
  uint8_t* out;           // 0x18
  uint8_t* out2;          // 0x20 optional planes copy of out (last decoder block: the decoder's input), else null
  const uint8_t* w;       // 0x28
  int32_t H, W, T;        // 0x30
  int32_t _pad3c;
};
static_assert(sizeof(Conv2Params) == 0x40, "Conv2Params");

struct AttnParams {       // step 3, windowed attention, record layer2; grid z = heads
  const uint8_t* x;       // 0x00
  uint8_t* out;           // 0x08
  const uint8_t* w;       // 0x10
  int32_t H, W;           // 0x18
  int32_t ox, oy;         // 0x20 window origin, cycle (0,0) (-4,-4) (-4,0) (0,-4)
};
static_assert(sizeof(AttnParams) == 0x28, "AttnParams");

struct ConvPlParams {     // step 4 of the last encoder block: residual conv + 2x2 pooling to the ViT stage
  const uint8_t* a;       // 0x00
  const uint8_t* _r08;    // 0x08 null
  const uint8_t* res2;    // 0x10
  uint8_t* out;           // 0x18
  int32_t _r20, _pad24;   // 0x20 0
  const uint8_t* w;       // 0x28 layer3
  int32_t h, w_;          // 0x30 C_mid extent
  uint8_t* pooled;        // 0x38 act(C_mid, ViT extent)
  int32_t ph, pw;         // 0x40 ViT extent
};
static_assert(sizeof(ConvPlParams) == 0x48, "ConvPlParams");

struct HeadParams {       // C_mid -> C_vit projection, record <last enc block>.layer4
  const uint8_t* pooled;  // 0x00
  uint8_t* out;           // 0x08 act(C_vit, ViT extent)
  const uint8_t* w;       // 0x10
};
static_assert(sizeof(HeadParams) == 0x18, "HeadParams");

struct RepackParams {     // tiles <-> token rows
  const uint8_t* src;     // 0x00
  uint8_t* dst;           // 0x08
  int32_t h, w_;          // 0x10 ViT extent
  int32_t ntok;           // 0x18 h*w rounded up to 64
  int32_t to_tokens;      // 0x1c 1 tiles->tokens, 0 back
};
static_assert(sizeof(RepackParams) == 0x20, "RepackParams");

// --- ViT block (5 dispatches): expand, contract, qkv, attention, contract ---
struct ExpandParams { const uint8_t* x; uint8_t* out; const uint8_t* w; };   // 0x00/0x08/0x10, record layer0
static_assert(sizeof(ExpandParams) == 0x18, "ExpandParams");

struct ConvParams1d {     // contract (records layer1, layer4)
  const uint8_t* x;       // 0x00
  const uint8_t* skip;    // 0x08 residual
  uint8_t* out;           // 0x10
  const uint8_t* w;       // 0x18
  int32_t K, KN;          // 0x20 inner width, inner*C_vit (hidden or C_vit)
  int32_t four, _pad2c;   // 0x28 4
};
static_assert(sizeof(ConvParams1d) == 0x30, "ConvParams1d");
typedef ConvParams1d ContractParams;

struct QkvParams { const uint8_t* x; uint8_t* out[3]; const uint8_t* w; };   // out = q, k, v; record layer2
static_assert(sizeof(QkvParams) == 0x28, "QkvParams");

struct AttnParams1d {     // global attention over the tokens
  const uint8_t* q;       // 0x00
  const uint8_t* k;       // 0x08
  const uint8_t* v;       // 0x10
  uint8_t* out;           // 0x18
  int32_t ntok, nreal;    // 0x20 padded and real token count
};
static_assert(sizeof(AttnParams1d) == 0x28, "AttnParams1d");

struct DecUpParams {      // ViT result -> C_mid decoder stage (single-record block between ViT and decoder)
  const uint8_t* x;       // 0x00 ViT result (tiles)
  const uint8_t* skip;    // 0x08 C_mid encoder output
  uint8_t* out;           // 0x10 C_mid decoder stage input
  const uint8_t* w;       // 0x18
  int32_t H5, W5;         // 0x20 C_mid extent
};
static_assert(sizeof(DecUpParams) == 0x28, "DecUpParams");

// Post-block (last block): features -> network result.
struct PostParams {
  const uint8_t* x;       // 0x00 last decoder block output
  const uint8_t* skip;    // 0x08 PreParams.out_full
  float* out;             // 0x10 network result, pad_w x pad_h RGBA32F (16 B/px)
  const uint8_t* w;       // 0x18 record <post>.layer0.layer
  int32_t h, w_;          // 0x20 pad_h, pad_w (the runtime reads these for the history layout)
  uint64_t _r28;          // 0x28 0
  float scale;            // 0x30 0.03125
  int32_t _r34;           // 0x34 1
  const float* rgb;       // 0x38 ImportParams.rgb
  const float* hist;      // 0x40 history RGB32F or null (runtime sets it per frame)
  float blend;            // 0x48 value of record <post>.layer0.blend_scale (f16)
  int32_t _pad4c;
  uint8_t* dst;           // 0x50 fused k_export (native post): ExportParams.dst, or null = a k_export step follows
  int32_t dst_h, dst_w;   // 0x58 (pitch dst_w * 8, RGBA16F, mode 0)
  const uint8_t* src;     // 0x60 fused k_import: ImportParams.src (dst_w x dst_h) instead of rgb, or null
};
static_assert(sizeof(PostParams) == 0x68, "PostParams");

// k_export: network result -> game output RGBA16F.
struct ExportParams {
  const float* res;       // 0x00 PostParams.out
  int32_t res_stride;     // 0x08 pixels per result row (pad_w)
  int32_t h, w_;          // 0x0c output extent
  int32_t dst_pitch;      // 0x14 bytes per output row (w * 8)
  int32_t dst_format;     // 0x18 0 = RGBA16F
  int32_t _pad1c;
  void* dst;              // 0x20
  int32_t mode;           // 0x28 inverse of ImportParams.mode (same value)
  int32_t _pad2c;
  const float* rgb;       // 0x30 ImportParams.rgb
  int32_t hist;           // 0x38 0
  float gain;             // 0x3c 1.0
};
static_assert(sizeof(ExportParams) == 0x40, "ExportParams");

// Hybrid resolution (build_plan.py --hybrid F): bilinear resample of a planar e4m3 activation [planes][h][w][16 B]
// (the pre-block's half output / the first encoder block's input, and the last decoder block's output) between the
// pre/post field and a main net run at 1/F of it. Source position of destination pixel x: (x + 0.5) * sx - 0.5, clamped.
struct ResampleParams {
  const uint8_t* src;     // 0x00 [planes][sh][sw][16]
  uint8_t* dst;           // 0x08 [planes][dh][dw][16]
  int32_t sh, sw;         // 0x10
  int32_t dh, dw;         // 0x18
  int32_t planes;         // 0x20 C / 16
  float sx, sy;           // 0x24 source pixels per destination pixel
  int32_t _pad2c;
};
static_assert(sizeof(ResampleParams) == 0x30, "ResampleParams");
