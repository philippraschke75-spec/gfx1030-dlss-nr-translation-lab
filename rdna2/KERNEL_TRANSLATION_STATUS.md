# Kernel translation status

Date: 2026-09-20. Source ELF SHA-256:
`51a5ac02b52aa2b6f04a0521537833e51b28c96c13d3145a2b45b11d1aa6f1e7`.
Target: RX 6900 XT / `gfx1030`, ROCm 6.4. This report records build and
bounded-test evidence only; it is not a full-network qualification.

## Physical results

| Kernel | Fixture | Result |
|---|---|---|
| `_Z12k_final_head10HeadParams` | Controlled constant, patterned and finite-FP8 basis; 1–2 workgroups | PASS; 32,768 exact bytes in the largest run |
| `_Z13k_align_probePh` | Odd-address byte sentinel probe | PASS; expected bytes and guards |
| `_Z6k_mean10MeanParams` | Independent CPU mean, padded 19×17 RGB input | PASS; exact within declared tolerance, guards and input unchanged |
| `_Z10k_conv_res10ConvParams` | Independent 8,192-byte patterned residual/reference fixture | PASS; zero mismatches, guards and inputs unchanged |
| `lowering_probe` | Synthetic lowering conformance fixture | PASS; 2,560 result words and guards |

Logs are `final-head-basis.log`, `kernel-align.log`, `kernel-mean.log`,
`kernel-conv-res.log`, and `lowering-probe.log`. Each test used one bounded
dispatch and no automatic retry. The logs record that watchdog/TDR settings
were not modified.

## Build coverage

The latest `translate_kernels.py --private-lds` run produced
`build/kernels-private-lds/coverage.json`:

- 29/34 entry points assembled and post-link decoded on `gfx1030`.
- This includes the five `k_swin_var` variants after experimental per-thread
  LDS lowering, but they remain `ASSEMBLED_UNVALIDATED`.
- The three large `k_swin_1h`/pre-block/post-block entry points remain blocked
  by `s_swappc_b64` indirect helper-call relocation and private ABI handling.
- `k_flag_wait` and `k_flag_set` remain blocked by `s_sendmsg_rtn_b64` realtime
  clock semantics and synchronization behavior.

Several other candidates contain WMMA lowering and are only assembled. Their
source descriptors, hidden arguments, LDS/private sizes, global addresses and
full numerical outputs have not yet received authentic fixtures. They must not
be placed in a game bundle or complete network route based on this report.

## Required next gates

1. Build an independent fixture for each selected non-SWIN kernel, including
   authentic argument layout, output shape, input/weight initialization and
   guards.
2. Test the five SWIN variants first with a bounded private-LDS fixture that
   exercises each private offset and each wide global-offset path. Verify LDS
   capacity, workgroup geometry and input/output dependencies.
3. Recover helper-call targets from the source object and implement a complete
   per-kernel call/return relocation strategy before enabling the three blocked
   helper kernels.
4. Qualify realtime flag kernels separately. AMD's RDNA2 ISA documents
   `S_MEMREALTIME` as a constant-frequency 64-bit counter and requires a wait
   for scalar-memory results; the source uses a message-return form, so a
   direct substitution has not been assumed safe.
5. Replay the selected dispatch set through registration, global initialization
   and the real host ABI. Only then attempt an authentic complete job.

## Update: helper linked, hardware scratch, realtime lowering (2026-09-20)

`translate_kernels.py --hw-scratch` (output `build/kernels-hw-scratch/coverage.json`):
**34/34 entry points assemble and decode on `gfx1030`** (`ASSEMBLED_UNVALIDATED`).

- **Helper call.** `swin_layer` (source `0xbd00`) is emitted inside each caller's code
  object; the `s_getpc`/`s_add_u32`/`s_addc_u32`/`s_swappc_b64` sequence is rewritten to a
  label-relative delta, and the helper's `s_setpc_b64` return is kept as is. No separate
  calling convention was needed. `k_swin_1h_32_fp8`, `k_pre_block_1h_32_fp8` and
  `k_post_block_1h_32_fp8` all build this way.
- **Private arrays.** The callers' 64 B/thread private array cannot fit in LDS
  (62,592 B group + 16 KiB > 64 KiB), so `--hw-scratch` uses native gfx10 flat scratch:
  `flat_scratch_init` user SGPRs, the standard `s_setreg HW_REG_FLAT_SCR_*` prologue, and
  `scratch_*` renamed to gfx10 mnemonics. Launch contract is the original one.
- **New lowerings.** `v_dot2acc_f32_f16` -> `v_dot2c_f32_f16`, `ds_store_b96`,
  `s_and_not1_*`/`s_or_not1_*` and flat/global u16/b32 aliases, and
  `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` -> `s_memrealtime` (the source's own
  `s_waitcnt lgkmcnt(0)` is retained). `v_div_*`, `v_fma_mix*`, `v_perm_b32`,
  `v_pk_mul_f16` have direct gfx1030 encodings and are emitted unchanged.

Physical evidence:

| Test | Result |
|---|---|
| Module load of the 3 helper-linked callers | PASS: 210 VGPRs, LDS 62,592 / 64,640 / 62,592 B, 256 threads |
| `flag_test.cpp` on `k_flag_wait` (bounded, guards) | PASS: 1000-spin timeout path stores a nonzero realtime stamp, set-flag path returns immediately, guards intact |

**Still NOT validated:** any neural output. The three helper-linked callers and the ~26
other translated kernels have only been loaded/assembled. `SwinParams` is an 80-byte
by-value struct with no recovered field layout, so no fixture exists yet. Unverified
semantic assumptions to check numerically: `v_dot2c` vs `v_dot2acc` rounding, dropped
`s_delay_alu` hints replaced by `s_waitcnt_depctr 0`/`s_nop 7`, per-lane swizzling of
`scratch_*` accesses of mixed width, and VOPD lowering through scratch VGPRs.

## Update 2: differential test against a gfx1100 emulator (2026-09-20)

See `emu/RESULTS.md`. A new independent interpreter of the original gfx1100 stream was used
as the reference. It exposed a real translator bug (the helper call's high address dword),
now fixed; after the fix `k_swin_1h_32_fp8` runs on the RX 6900 XT and matches the reference
byte-for-byte on 5 fixtures (3 with rich output, 2 saturated/underflowed), and the test
detects corrupted WMMA/perm/ldexp/log/rcp/fma lowerings. It does NOT cover the pre/post-block
callers, real weights, or any rendered frame.
