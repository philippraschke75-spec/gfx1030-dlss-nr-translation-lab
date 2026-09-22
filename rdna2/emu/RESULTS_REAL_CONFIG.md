# Real-configuration GPU differential results (RX 6900 XT gfx1030), 2026-09-21

## Method
Real WEIGHTS_HT records (from nvngx_dlssnr.dll) at kernarg +0x10; the real launcher's kernarg layout (VARPARAMS_HOST_CONTRACT.md); origin
offsets, flags, grids from the recovered formulas. Activations are synthetic. Each case: emulator preflight, ONE bounded dispatch inside a
fresh sandbox (`rdna2/build/*-20260921`), reference recomputed at the device's actual allocation base, whole-arena byte comparison, guards intact.
Scope: equality of the translated gfx1030 kernel with my gfx11 emulator on these inputs. It says nothing about network output quality,
real activations, or gfx1100 hardware behaviour.

## Two harness errors found and fixed during this work (kept here so they are not repeated)
1. STALE MODULES: `rdna2/build/kernels-hw-scratch/*.co` were built by older translator states. `_Z10k_swin_varILi32ELb1EEv9VarParams.co` (ca4f0c5d…)
   predates the v_pack_b32_f16 fix; the corrected build is c75137d2…. The four `<N,false>` modules also differ from a fresh translation
   (32: ef05dac0 vs 0c6e4469, 64: 814b868f vs ca6f810e, 128: b39ec36d vs a9834318, 256: 8a67c2cd vs 38c8344c). A first matrix run used the stale ones and
   is therefore SUPERSEDED (log kept as matrix-stale-modules.log). k_import was identical (112560c4). Always rebuild modules with
   `translate_kernels.py --hw-scratch --symbol ... --output <sandbox>` before testing.
2. POINTER-LOOKALIKE PADDING: an unwritten padding dword at VarParams+0x6c that held a stale pointer high word made the harness' pointer-rebasing
   scan rewrite a qword on the GPU only, producing a 28-36k-byte "mismatch". Zeroing the padding (as a clean fixture should) gives 0 mismatches.
   Fixtures must not leave pointer-like bit patterns in fields the real host leaves unused.
The reference must also be recomputed at the device base (difftest_var does; the new import/pre/preblock scripts now do too).

## Verified with FRESH modules
| kernel | config | result |
|---|---|---|
| k_swin_var<32,true> as pre-block | real layout (+0x00 null, flags 0x14, float RGB at +0x40, block0 weights, +0x6c clean) 16x16, grid 2x2 | PASS, 0 / 42,812 bytes |
| k_swin_var<32,true> | synthetic layout flags 16 and 20, 2x2 | PASS, 0 mismatches |
| k_swin_var<N,false> encoder matrix | 13 cases (stages 1-4, real records block1,2,3,4,5,6,8,9,10,14,15,16,22) | **13/13 PASS**, 0 mismatches, guards intact - table below |
| k_import | formats 0-7 x modes 0-3, src 13x21 pad 16x32 | mode 0: formats 1-7 byte-exact, format 0 equal except NaN payload bits; modes 1-3: max 3 ULP (rel 2.3e-7), outputs in [0,1] |

Emulator ops added (ISA-derived unit tests in test_import_ops.py): s_min_i32, s_max_i32, v_minmax_i32, v_cvt_f32_i32, v_cvt_f32_ubyte1,
v_cvt_f64_f32, v_frexp_mant_f32, v_frexp_exp_i32_f64, v_(sub|subrev)_co_ci_u32, v_dot2acc_f32_f16.

## Encoder matrix with fresh modules (all 13 cases complete)
Origins (0,0),(-4,-4),(-4,0),(0,-4) are the four shifted-window modes; modes 0/1/3 were run at widths 64/128/256, all four at width 32 (mode 2 not run at 64+).

| kernel | flags | grid | origin (x,y) | bytes written | mismatches | status |
|---|---|---|---|---|---|---|
| swin_var<32,false> | 1 | 2x2 | (0,0) | 24355 | 0 | PASS |
| swin_var<32,false> | 0 | 3x3 | (-4,-4) | 44761 | 0 | PASS |
| swin_var<32,false> | 0 | 3x2 | (-4,0) | 32514 | 0 | PASS |
| swin_var<32,false> | 4 | 2x3 | (0,-4) | 36570 | 0 | PASS |
| swin_var<64,false> | 1 | 2x2 | (0,0) | 48649 | 0 | PASS |
| swin_var<64,false> | 0 | 3x3 | (-4,-4) | 89525 | 0 | PASS |
| swin_var<64,false> | 4 | 2x3 | (0,-4) | 73139 | 0 | PASS |
| swin_var<128,false> | 1 | 2x2 | (0,0) | 97322 | 0 | PASS |
| swin_var<128,false> | 0 | 3x3 | (-4,-4) | 178999 | 0 | PASS |
| swin_var<128,false> | 4 | 2x3 | (0,-4) | 146214 | 0 | PASS |
| swin_var<256,false> | 1 | 2x2 | (0,0) | 194669 | 0 | PASS |
| swin_var<256,false> | 0 | 3x3 | (-4,-4) | 358012 | 0 | PASS |
| swin_var<256,false> | 4 | 2x3 | (0,-4) | 292581 | 0 | PASS |

## Chained real-pixel run: k_import -> pre-block on actual captured Cyberpunk data (2026-09-22)

`chain_import_preblock_real.py`: crops an NxN tile directly from the captured colour plane (real game pixels,
`RESULTS_REAL_CONFIG` capture from 2026-09-21), runs `k_import` (format=0, mode=0 - see INPUT_CONTRACT_CYBERPUNK_FSR3.md)
on the GPU, takes that GPU-verified float output as the pre-block's input (kernarg +0x40, real layout, block0 real weights),
and runs the pre-block on the GPU. Each stage is independently checked against the emulator at the device's actual
allocation base before being chained; the pre-block never sees synthetic data for this input.

| tile (x,y) | size | import mismatches | pre-block mismatches | pre-block bytes written | status |
|---|---|---|---|---|---|
| (500,300) | 64x64 | 0 | 0 | 685,636 | PASS |
| (0,0) | 64x64 | 0 | 0 | 685,543 | PASS |
| (1600,850) | 64x64 | 0 | 0 | 685,154 | PASS |

Two harness bugs found and fixed while building this (documented so they are not repeated):
1. Kernarg pointers built once against the emulator's synthetic base were reused unchanged after re-targeting the
   emulator at the GPU's real allocation base, causing a read fault. Kernarg construction must be a function called
   fresh on every reference recomputation (the pattern `difftest_var.py` already used; the new chain script had
   inlined it incorrectly).
2. The pre-block's scratch buffer (`+0xa0`) is sized `8 KiB * grid_x * grid_y` by the real host; the fixture's fixed
   1 MiB-per-pointer-field arena only covers this for tiles up to 64x64 (64 workgroups -> 512 KiB). A 128x128 tile
   (256 workgroups -> 2 MiB) overflowed it and produced a write fault that looked like a kernel bug but was not one.

Only format=0 (RGBA16F) real colour data has been chained this way; depth and motion vectors still need their
`k_import` format codes identified (not yet done - see INPUT_CONTRACT_CYBERPUNK_FSR3.md). The rest of the network
(remaining encoder stages, middle blocks, decoder, post block, export) has not been chained onto real pixel data yet.
