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
