# SWIN investigation, 2026-09-21

## Allocation/preflight handshake

The numerical runner now requests `--preflight` from var_gpu_test. After
allocation and rebasing, the child writes the actual kernarg bytes and prints
READY with the device base. It retains the allocation and waits for GO on stdin.
The parent emulates that base, compares all kernarg and initial arena bytes,
and sends GO only on success. Exceptions kill the waiting child; the outer
sandbox still bounds the overall lifetime. Legacy checkpoint diagnostics retain
their previous launch path; this upgrade currently covers difftest_var's CLI.

Fresh gated runs: flag 32 PASS, zero mismatches, 0.466 ms; flag 16 FAIL,
35761 mismatching bytes, 0.437 ms. Both guards intact. All 17 host regression
tests pass, including rejection-before-GO and malformed handshake controls.
Subsequently added preflight JSON hashes and reference.bin persistence for future
runs; those evidence writes were not present during the two timings above.

## Resolution of flag 32

Checkpoint 0xc1eac, first arrival, workgroup (0,1), wave 0, lane 23:
GPU and rebased emulator agree on s13=0x4, v5=0, v9=0xb61fbaa2,
v2=0, EXEC=0xffff0000. The original synthetic base produced s13=0x7500
and v5=1. The harness must compare the same rebased kernarg bit patterns.
`difftest_var.py` now obtains arena_dev from the successful GPU log and
recomputes its reference at that base. Preflight still runs before dispatch.
The actual device address is unavailable before allocation; consequently
preflight using the synthetic base is not proof for every rebased field.
Capturing allocation/argument data before launch remains a harness improvement.

A fresh flag-32 differential run passes: seed 1, grid 2x2, 402376 emulator
steps, 0 mismatching bytes, guards intact, 0.472 ms, normal exit 0.
Flag-16 saved-output replay still differs in 35761 bytes at the real base.

## Follow-up native probe

Flag 16 was reproduced after the clamp correction: 35977 mismatching bytes,
0.488 ms, guards intact, 525559 emulator steps. The runner correctly exited 1.

Tracing output offset 0xb00533 identifies workgroup (0,1), wave 0, lane 23.
At source PC 0xc1eac the emulator evaluates FMAC with multiplier bits 0x00000001,
multiplicand bits 0xb61fbaa2, and accumulator +0. It produces -0.
The new standalone native_fmac_probe.cpp executes these exact operands in
one wave on gfx1030, compiled with -fdenormal-fp-math=ieee. All 32 lanes
returned 0x80000000 (-0), agreeing with the emulator for this isolated operation.
Thus a universal FMAC signed-zero correction would be wrong. The full-kernel
+0 output must be investigated upstream (actual input bits or mode state).
The probe is not proof that the full kernel sees identical operands/modes.

Rebuilt `_Z10k_swin_varILi32ELb1EEv9VarParams` with `--hw-scratch` in
`rdna2/build/astra-swin-20260921`, and compiled `var_gpu_test.cpp` with ROCm 6.4.
One bounded differential experiment ran through sandbox.py after emulator
termination: flags 32, seed 1, grid 2x2, 256 threads, original synthetic fixture.

Observed RX 6900 XT result: 0.298 ms, guards intact, 184 mismatching arena bytes.
Emulator: 402376 steps, 35042 changed bytes. First differences were in the slot
for kernarg pointer +0x70, at offsets 0x46, 0x47, 0x68, 0x69, 0x6a.
This reproduces Claude's flag-32 result; it does not resolve it.

Saved GPU arena SHA256:
`75a70ccee5e89831d2c116dc2044b80c256436d0ae4efd506c372b798d77bf9e`.

Host-only hypothesis experiment: flush FP16 subnormal operands of mixed FMA
to signed zero, preserving the default reference behavior unless explicitly
enabled by `probe_mix_denorm.py`. Both baseline and flushing produced 184
mismatches and identical reference SHA256:
`fbae19886ef3d9120b2929abe9ba22089ea2b52444dbb1b9270ec2d82cb8fdf6`.
Therefore this specific input-flushing hypothesis does not explain this
fixture's mismatch. This does not establish actual gfx1100 denormal semantics.

The differential runner previously exited zero after printing FAIL; it now
exits nonzero for failed or empty result sets. The original observed exit zero
must not be interpreted as a successful numerical test.

Next diagnostic: isolate the mixed-FMA operands and destination immediately
before the final differing stores, then test the native operation separately.
The host reference is still a model and must not be assumed correct when it
disagrees with hardware. Flags 16 and 32 remain unqualified. No game or full
neural-job execution has been demonstrated by this work.

## Store trace and clamp correction

`trace_var_store.py` located the first differing write at source PC 0xc166c,
workgroup (1,0), wave 1, lane 0: a global_store_b128 of v2..v5.
The differing v3 is FP32, not FP8. At PC 0xc1d60,
`v_fma_f32 v3, 0x41000000, v3, 0.5 clamp` receives v3=0xbd800005.
The emulator previously ignored clamp and returned 0xb4a00000; the GPU output
is zero. Implementing finite FP32 clamp for FMA and binary FP32 arithmetic
reduces the saved-arena mismatch from 184 bytes to six.

The six remaining byte offsets are 0xb00533, 0xb0053b, 0xb00713, 0xb0071b,
0xb00b27, 0xb00b2b (reference byte 0x80, GPU 0x00). Explicit positive-zero
normalization at clamp did not remove them. Their upstream origin remains
unresolved. No tolerance was used to hide them; the fixture still fails strict
byte equality. NaN clamp/DX10-mode semantics also require separate validation.
Reference SHA256 after correction:
`3b54128ab03186f8b8f6b6f85834368b6ac690a50e28404a2007cf27d36c700e`.

## Subsequent correction: actual allocation and flag-16 provenance (2026-09-21)

The preceding six-byte flag-32 failure is historical. The synthetic arena's
pointer upper word was consumed numerically in the inferred parameter layout.
Replaying at the actual GPU allocation base removes those six differences.
The allocation handshake now holds the allocation, checks rebased kernargs and
input bytes, runs the reference, and only then sends GO to the harness.
Flag 32, seed 1, grid 2x2 passed strict equality for the whole 18 MiB arena
(0.466 ms dispatch, guards intact). This qualifies only that synthetic fixture,
not the inferred ABI or a real neural job. Flag 16 still has 35,761 differing
bytes under the actual-address handshake (0.437 ms, guards intact).

Checkpoint dispatches now also use the allocation handshake. Host negative
controls cover wrong kernargs, changed input arena, unreachable checkpoints,
and protocol rejection. All 18 host tests pass. Instrumentation still poisons
registers and stops waves at first arrival; results are diagnostic, not a proof
of equivalence for later loop visits or uninstrumented scheduling.

Flag-16 seed-1 grid-2x2 investigation:

* Forward and reverse serial emulator workgroup orders produce identical bytes.
  This is not a general race-freedom test.
* At source PC 0xc0fe0, workgroup (0,0), wave 4, lane 0, both sides target arena
  +0x100000. Packed v8 differs: emulator 0x796a7e72, GPU 0x6b737b68.
* At 0xc0794, the same lane uses LDS address v2=0x400 on both sides. This run
  used actual arena base 0x1404010000; s13=0x14 agrees after rebasing.
* At 0xc07a0, immediately after the LDS load, v2 is 0x66165845 versus
  0x5d255604 and v3 is 0x5e2157c6 versus 0x57cf5a8d. Thus the disagreement
  is already in FP16 inputs to the final FP8 conversion.
* `trace_var_lds.py 0x404010000` identifies the final changing write to those
  eight LDS bytes before that checkpoint as ds_store_b16 at 0xbff08, wave 2.
* A GPU checkpoint at 0xbff08, wave 2 lane 0, confirms v64=0x400 on both
  sides, but v7 low half is 0x5845 versus 0x5604. The unused upper half is
  not the cause of this 16-bit store disagreement.
* Earlier at 0xbfea0, before WMMA, wave 2 lane 0 already has different v10
  (0x58005400 versus 0xd2804f80) and residual v7 (0xc0c5a000 versus
  0xc04aa000); sampled v26=0x34803180 and accumulator v42=0 agree.
  This does not establish agreement of all weights or blame the WMMA lowering.

All four new checkpoint dispatches completed with guards intact. Their JSON
reports and instrumented modules are local in
`rdna2/build/astra-swin-20260921/rdna2/build/emu_bisect/operands_<pc>.*`.
The new host provenance script reports changing writes only, not stores that
rewrite identical bytes. Next trace the producers of v10 and residual v7 before
0xbfea0, compare all contributing lanes, and isolate the earliest relevant
operation in a standalone native probe before changing translator semantics.
Flag 16 remains failing; no rendered game frame has been demonstrated.

## Flag 16 resolved in tested fixtures: FP16 pack constant lowering

This entry supersedes the failing status above. Following the inputs backward
reached the first dot-product loop at 0xb1770. Wave 0 lane 0 had identical
accumulator v4=0 and source v5=0xb9a23a09, but v6 differed:
reference 0x3c00302b versus device 0x0000302b. The missing upper half is 1.0h,
written by `v_pack_b32_f16 v11, v5, 1.0` at 0xb0aa8. The translator incorrectly
emitted `v_lshl_or_b32 v11, 1.0, 16, v106`; the integer operation expands that
constant to FP32 0x3f800000, whose shift yields zero. This is an actual
translation defect, independent of the early transcendental approximation noise.

`translate_final_head.lower_pack_f16` now materializes floating inline constants
as binary16 encodings and captures both operands before the destination write.
Both `translate_kernels.py` and `translate_final_head.py` use this helper.
Five new host tests execute the generated integer instruction sequence and
check explicit packed-bit results, including constants in either position,
both constant operands, destination aliases, and raw integer literals.
All 23 host tests pass; corrected SWINvar32true assembles successfully.

Physical validation on RX 6900 XT, 256 threads/workgroup, grid 2x2:

| flags | seed | differing bytes | changed reference bytes | GPU ms |
| --- | --- | --- | --- | --- |
| 16 | 1 | 0 | 37339 | 1.897 |
| 16 | 2 | 0 | 37335 | 0.501 |
| 32 | 1 | 0 | 35044 | 1.721 |
| 31 | 1 | 0 | 39382 | 1.712 |
| 63 | 1 | 0 | 37076 | 0.467 |

All had intact guards and exact whole-arena byte equality, with no tolerance.
Each used emulator preflight at the actual allocation address before GPU GO.
Seed 2 used base 0x904010000; the other runs used 0x404010000.
These individual dispatch times are observations, not performance benchmarks.
Corrected module SHA256:
`c75137d2fc8a69b19e65b0b6d5e1ea0d1cc419af383ad3b7d4867349ddeaa371`.
Seed-1 flag-16 reference SHA256:
`30eda9885103211c2907aae8fd8ddc2b37852e7c1054f9fa7c93fc28cc0e0bcd`.

Reproduction: rebuild with `translate_kernels.py --hw-scratch --symbol
_Z10k_swin_varILi32ELb1EEv9VarParams --output <sandbox>/rdna2/build/kernels-hw-scratch`,
then run `difftest_var.py 32_1 16,32,31,63 1 2 2` and
`difftest_var.py 32_1 16 2 2 2` through `sandbox.py run` with a bounded timeout.
Per-case input, rebased kernargs, reference output and preflight hashes are in
`rdna2/build/astra-swin-20260921/rdna2/build/emu_var/v32_1_f<flags>_s<seed>/`.
The per-seed results JSON contains only the last invocation for that seed;
the per-case evidence folders retain the separate flag cases.

Remaining scope: broader variants/input shapes and authentic runtime parameter
capture, real weights, and game integration. No native GFX1100 oracle or rendered
game frame is established by these synthetic differential passes.

## Next step completed: shape coverage and a conflicting-write fixture

See RESULTS_VAR.md's expanded-shape section for six new physical cases.
Five pass exactly; flags 63 / seed 4 / H=7 W=9 / grid=1x1 fails in 16 bytes,
with intact guards. Original-program host tracing identifies two different
writers for every differing byte at PC 0xc3908: wave 1 lanes 16..31 and wave 5
lanes 0..15 target arena+0x400030..0x40003f. The GPU matches the former values;
the reference matches the latter. This supports write-order dependence in the
inferred fixture, not a proven new translator defect. Dimension/stride legality
is unresolved. Preserve FAIL and establish the actual runtime contract next.

`difftest_var.py` now exposes shape arguments, saves per-case results with output
hashes, and separates evidence by grid and shape. `trace_var_store.py` accepts
`--height`, `--width`, `--seed`, `--gx`, `--gy`, and `--length` for reproducing
the overlapping stores. All 27 host tests pass. No new translator changes or
game deployment were made during this shape-validation step.

## Output contract follow-up

Recovered source-store 0xc3908's exact address expression and offset mapping;
see VAR_OUTPUT_CONTRACT.md. Legacy fixture offy/offx names are reversed for
the +0x20/+0x24 words. The differential runner now has explicit --offset-x/y
and includes offsets in case identities and result metadata. Defaults remain -4.
`var_output_contract.py` models this store path, reproducing the observed
wave-1/wave-5 collision at offset 0x30. This is a scoped static address audit,
not a full race detector. Three added tests bring the host suite to 30 passes.
Physical retest H7/W9/flags63/seed4/grid1x1 with offsets (0,0) passes exactly,
9462 changed reference bytes, intact guards. No translator change was needed.
Full runtime ABI remains unverified; do not equate this synthetic contract
recovery with authentic parameter capture or working game integration.
