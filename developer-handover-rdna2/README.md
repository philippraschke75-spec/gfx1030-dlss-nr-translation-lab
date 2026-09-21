# RDNA2 replacement primitives — local prototype

This directory contains original development work for the user's RX 6900 XT.
It is not an installable DLSS-NR mod and must not be copied into Cyberpunk.

**Update:** the first authentic network kernel, `k_final_head`, has now been
translated and passed isolated numerical tests on this RX 6900 XT, including
two workgroups and 32,768 exact output-byte comparisons. See
[FINAL_HEAD_RESULTS.md](FINAL_HEAD_RESULTS.md) for the successful and failed
experiments, reproduction instructions, and limits.

## What has been verified

`fragment_wmma.h` implements a wave32 FP16-input / FP32-accumulator matrix
operation using RDNA2 cross-lane gathers and dot2 instructions. Its per-lane
input and output layout follows AMD's GFX11 WMMA layout:

- A: lane L holds row L modulo 16 across K, packed as eight pairs.
- B: lane L holds column L modulo 16 across K, packed as eight pairs.
- Both inputs are replicated between lane halves.
- C/D register J in lane L owns row `2*J + floor(L/16)`, column `L modulo 16`.

Register-layout source: https://gpuopen.com/learn/wmma_on_rdna3/

`fragment_test.cpp` supplies raw per-lane fragments rather than making complete
matrices available to the replacement. A separate CPU reference computes ordinary
matrix products. In one bounded dispatch, 32 tiles across 8 workgroups (four
waves each) exercise accumulator preservation, identity matrices, each individual
K position, and deterministic mixed-sign inputs. The values are dyadic and small
enough that all products and sums are exact in FP32. All 8,192 outputs matched
exactly on the RX 6900 XT / HIP 6.4. Output sentinels and guard regions were checked.
Host negative controls reject a wrong value, NaN, and guard overwrite.

Raw evidence: `fragment-run.log` (includes executable SHA-256).

`alignment_test.cpp` uses explicit GFX1030 assembly to test a normal and an
odd-address 16-bit store in one wave. Both wrote the intended bytes, with no
observed guard damage. This does not establish behaviour on another GPU, runtime,
queue type, or in D3D12/HIP interop. Raw evidence: `alignment-run.log`.

## Build and reproduction

Existing local dependencies: ROCm 6.4, Visual Studio 2022 C++ Build Tools.
Run in a separate PowerShell process so compiler environment changes are scoped:

```powershell
powershell -NoProfile -File .\rdna2\run_fragment_test.ps1
powershell -NoProfile -File .\rdna2\run_fragment_test.ps1 -Test alignment
```

Each invocation compiles and runs one bounded experiment, with no retries.
Neither test modifies installed runtime files, game files, or GPU settings.

`build/fragment-gfx1030.s` is compiler-generated assembly for the tested matrix
wrapper. It uses dot2 and ds_bpermute, 28 VGPRs, no private spill storage and no
allocated group memory. These resource counts apply to this standalone wrapper,
not to an eventual inlined network kernel.

## Limits and integration work

- All 32 lanes must participate; partial EXEC masks are not qualified.
- Input replication is a caller requirement.
- Passing exact dyadic cases does not establish bitwise agreement with native
  GFX11 WMMA for rounding, denormals, infinities or NaNs. No native GFX11 comparison
  device was used.
- This is a C++ primitive, not a binary instruction patch. Live register values,
  register allocation, calling conventions, barriers and machine-code addresses
  must be preserved when integrating into an existing kernel.
- Static inventory of this user's installer found 203 F32/F16 WMMA sites across
  24 functions in the GFX1100 code object. The code also includes PC-relative
  references and indirect calls. Expanding instructions without fixing those
  references is not a valid translation.
- The published research emulator's WMMA handler in `src/emulator/emu.py` is still
  a value stub. Its 44 passing host tests are not a full numerical network oracle.
- Complete network output, D3D12/HIP interop, Cyberpunk presentation, temporal
  correctness and performance remain unverified.

The first authentic-kernel milestone has been reached for controlled inputs in
`k_final_head`. The next integration work is translating and checking the other
kernel families and their cross-function references, then assembling a full
network. Rebuilding the public HIP bridge alone is insufficient.

## Follow-up kernel-family work

`translate_kernels.py` now performs a source-hash-checked, per-symbol assembly
pass over the remaining GFX1100 entry points. It regenerates metadata from the
same ELF, allocates lowering scratch registers from the observed source
register footprint, lowers the tested WMMA form, expands supported VOPD pairs,
handles GFX10 wide global offsets, and can lower the observed small private
arrays into per-thread LDS for an explicitly selected build-only experiment.

The current coverage report is `build/kernels-private-lds/coverage.json`.
At the latest run, 29 of 34 entry points assembled as
`ASSEMBLED_UNVALIDATED`; the five remaining SWIN/helper entry points are
blocked by indirect helper-call relocation. The flag wait/set pair remains
blocked because it uses a realtime-clock message whose exact synchronization
contract has not yet been physically qualified. These are honest build states,
not claims of network execution.

Three additional isolated GPU checks have passed on the RX 6900 XT / gfx1030:

- `k_align_probe`: expected odd-address 16-bit store bytes, guards intact.
- `k_mean`: padded 19x17 RGB fixture matched an independent CPU mean exactly.
- `k_conv_res`: 8,192 output bytes matched a CPU-predicted matrix product plus
  patterned residual; inputs and guards were unchanged.

The synthetic `lowering_probe` also passed 2,560 result-word checks covering
per-thread private LDS isolation, dynamic byte/word loads, positive and
negative wide offsets, condition-code preservation, and zero/nonzero CLZ.
It validates the lowering mechanisms, not any authentic neural tensor.

The next required work is per-kernel authentic fixtures and physical tests,
then exact helper-call relocation and registration. The SWIN kernels must not
be enabled in a complete job merely because their objects assemble.
