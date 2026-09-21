# Differential test: gfx1100 reference emulator vs translated gfx1030 kernel

Date: 2026-09-20. Kernel under test: `_Z16k_swin_1h_32_fp810SwinParams` translated with
`translate_kernels.py --hw-scratch` (helper `swin_layer` linked in), run on an RX 6900 XT.
Reference: `gfx11emu.py`, an independent Python interpreter of the **original gfx1100**
instruction stream (caller + helper, 25,493 instructions, ~130 opcodes; wave32, LDS, scratch,
flat apertures, WMMA, VOPD). Harnesses: `run_emu.py`, `difftest.py`, `swin_gpu_test.cpp`.

## Bug found and fixed by this testing

The first GPU dispatch **hung** (20 s timeout). Bisecting on the emulator's execution trace
(`hangfind.py`) showed the kernel ran fine up to the `s_swappc_b64` and failed on the call
itself: the helper is appended *after* the caller, so the label-relative PC delta is positive,
but the kept `s_addc_u32 sN, sN, -1` came from the original layout (negative delta) and
subtracted 1 from the high address dword. Fixed in `translate_kernels.py` (the high-word add
now carries only). All three helper callers get the fix. The emulator also showed that
kernarg `+40` is a per-thread stride: a zero there is an infinite loop, so it must be the
thread count (256).

## Results (5 fixtures, 1x1 to 3x2 workgroup grids, 256 threads/WG)

| Case | Image (HxW), offsets | Grid | Input magnitude | Emulator steps | Bytes written | Mismatches | Output content |
|---|---|---|---|---|---|---|---|
| 0 | 16x16, (-4,-4) | 2x2 | e4m3 exp 2..7 | 1.23 M | 4,608 | **0** | 165 distinct values |
| 1 | 16x16, (0,0) | 2x2 | e4m3 exp 8..11 | 1.49 M | 8,192 | **0** | saturated (6 values) |
| 2 | 24x16, (-4,-4) | 2x3 | e4m3 exp 2..7 | 1.91 M | 7,680 | **0** | 169 distinct values |
| 3 | 16x16, (-4,-4) | 2x2 | e4m3 exp 4..7 | 1.23 M | 4,608 | **0** | 140 distinct values |
| 4 | 16x24, (-4,-4) | 3x2 | e4m3 exp 0..3 | 1.76 M | 7,680 | **0** | underflow (2 values) |

Every GPU run completed in ~0.5 ms with guard bytes (64 KiB either side of the output)
intact. Cases 0, 2, 3 are the informative ones; cases 1 and 4 match but the output is
saturated/zero, so they prove little. An earlier round with fully random FP8 data also
"passed", but its output was all-NaN (`0x7f`), which the sanity check exposed as vacuous;
those results were discarded.

## How much can the test detect? (mutation testing)

Corrupting one translated instruction at a time and re-running the GPU comparison:

| Corrupted family | Mutants | Detected |
|---|---|---|
| `v_perm_b32` selector | 3 | 3 |
| `v_ldexp_f32` -> `v_mul_f32` | 3 | 3 |
| `v_log_f32`, `v_rcp_f32`, `v_med3_f32`, `v_fma_f32` | 12 | 12 |
| `v_fma_mixlo_f16` operand select | 3 | 3 |
| WMMA lowering (`ds_bpermute` row offset) | 1 | 1 (3,076 bytes differ) |
| `v_div_fixup_f32` -> plain move | 3 | 0 (equivalent for finite normal quotients; special-value handling is not exercised) |
| `v_fma_mix_f32` (sum-of-squares stage) | 6 | 0 (**coverage gap**: this stage sees all-zero data in every fixture, so it cannot influence the output) |

## Limits — read before relying on this

- Only `k_swin_1h_32_fp8` was tested. `k_pre_block_1h_32_fp8` and `k_post_block_1h_32_fp8`
  assemble and load but have unrecovered parameter layouts and are untested.
- The reference is an emulator written from the ISA, not silicon. Agreement shows the
  translation matches my reading of gfx11 semantics; a shared misreading would be invisible.
  Known approximations: fused multiply-add via float64, `v_log/rcp/rsq` via numpy, WMMA via
  float64 matmul.
- Outputs are FP8 (~6 % resolution), so 1-ulp float differences are mostly masked.
- Inputs are random finite FP8, not a trained network's weights; the blob layout is unknown,
  so weights are noise. 74 % of instructions execute (67 % of vector instructions with an
  active lane); ~half the divide sites and a third of the `v_ldexp` sites do not.
- The parameter struct is only partly recovered: `+0x00/+0x08/+0x10` pointers (in/out/blob),
  `+0x18..+0x27` four int32 (H, W, two offsets), `+0x28` thread stride (256), `+0x34` low 16
  bits thread count (256). Bytes `+0x2C..+0x33`, `+0x38..+0x4F` are unused by this kernel.
- This is not a rendered frame, not D3D12/OptiScaler integration, and not proof the network
  produces a correct upscale.
