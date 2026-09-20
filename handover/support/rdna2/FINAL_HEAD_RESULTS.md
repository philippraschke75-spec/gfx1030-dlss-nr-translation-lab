# First translated network kernel: results

Date: 2026-09-20. Device: AMD Radeon RX 6900 XT, gfx1030, HIP 6.4.

**Result: the translated `k_final_head` kernel passes isolated numerical tests
on this GPU, including two workgroups. The complete DLSS-NR network is not yet
ported and Cyberpunk has not been tested.**

## What was translated

Source function: `_Z12k_final_head10HeadParams` in the GFX1100 code object
extracted from the user's installer. Source object SHA-256:

`51a5ac02b52aa2b6f04a0521537833e51b28c96c13d3145a2b45b11d1aa6f1e7`

The corresponding generic RDNA2 function writes zeros. This experiment instead
translates the numerical GFX1100 implementation, retaining its input staging,
barrier, repeated matrix products, intermediate FP16 accumulation, FP8 output
conversion and tensor addressing.

`translate_final_head.py`:

- Reassembles instructions for GFX1030 rather than relabeling an incompatible
  binary.
- Expands both WMMA sites into cross-lane gathers and dot2 operations using
  scratch VGPRs disjoint from the original register allocation.
- Uses temporary results when splitting dual instructions, preserving their
  simultaneous reads even where a destination aliases another operand.
- Converts direct branch targets to assembler labels and relocates the
  PC-relative FP8 lookup-table reference.
- Maps the GFX1030 workgroup-ID register to the register expected by the original
  body and emits a matching kernel descriptor and argument metadata.
- Uses conservative waits/delays; no performance claim is made.

Output code-object SHA-256, tested in the passing runs:

`8b0243689bf2a400459d9d988b8dbad361c614fa563538ff1ec12a96c8bac7c3`

## Failures encountered and resolved

1. Assembly initially stalled while padding a 2,722-byte constant-data block in
   the executable section. Explicit zero-fill alignment resolved the stall. The
   assembler call now has a timeout. No GPU was involved in this failure.
2. The first harness used `hipFuncGetAttributes` with a module-function handle.
   That returned `invalid device symbol` before any launch. It was corrected to
   `hipFuncGetAttribute`, the module-handle API.
3. The first numerical launch returned all zeros, failing all 16,384 output-byte
   comparisons. The extracted `g_e4m3_lut` was an all-zero, runtime-initialized
   table. The isolated harness had omitted that initialization. Materializing its
   256 E4M3FN-to-FP16 entries in the standalone module fixed this failure.
   This was a defect in the isolated experiment, not evidence that the original
   mod fails to initialize its table.

The failed numerical log is retained as `final-head-constant.log`; the passing
initialized run is `final-head-initialized.log`.

## Numerical evidence

| Test | Workgroups | Output bytes compared | Mismatches | Distinct output bytes |
|---|---:|---:|---:|---:|
| Constant inputs/weights | 1 | 16,384 | 0 | 1 |
| Patterned signed inputs/weights | 1 | 16,384 | 0 | 87 |
| Patterned signed inputs/weights | 2 | 32,768 | 0 | 88 |

All test dispatches use 256 threads per workgroup. Allocation guards remained
intact; inputs and weights were checked for unexpected writes in the numerical
tests. An earlier argument probe verified that the runtime supplies the expected
hidden workgroup-size and grid fields.

For the patterned tests, the CPU computes an ordinary integer matrix product:
16 or 32 rows by 512 inner elements by 1,024 columns. Input and weight values
are independently seeded integers from -2 through 2, scaled by 1/16. Their
products and intermediate sums are exactly representable over this test range.
The CPU then quantizes the result to E4M3 using nearest-even rounding.

Input, output and weight address mappings were reconstructed from the original
kernel. Their bijectivity and bounds are checked before dispatch. The reference
does not execute the instruction emulator or reuse the replacement's lane
shuffle implementation. Both the reference and GPU use the same test matrices,
but different computation implementations.

Logs: `final-head-abi-fixed.log`, `final-head-initialized.log`,
`final-head-patterned.log`, `final-head-multigroup.log`.

## Reproduction

Use the bundled Python interpreter or Python 3.10+, and the installed ROCm 6.4
and Visual Studio 2022 Build Tools. Commands run from the workspace root:

```powershell
python rdna2/translate_final_head.py --abi-probe
powershell -NoProfile -File rdna2/run_fragment_test.ps1 -Test final_head -Mode abi
python rdna2/translate_final_head.py
powershell -NoProfile -File rdna2/run_fragment_test.ps1 -Test final_head -Mode patterned
powershell -NoProfile -File rdna2/run_fragment_test.ps1 -Test final_head -Mode multigroup
```

The translator requires the locally extracted, hash-matched source object and
its metadata dump. Each harness invocation performs one dispatch and does not
retry failed GPU work. Keep the module hash with the log.

## Limits

This validates one kernel under controlled inputs. It does not establish native
GFX1100 bitwise equivalence for all values, NaNs, extreme magnitudes or rounding
edge cases. There was no RDNA3 reference device. It does not validate the other
network kernels, a full neural job, D3D12/HIP sharing, a presented image or game
performance. The reported synchronization duration is not an FPS benchmark.

The generated module is an isolated research artifact, not a game DLL. Its
constant-table handling and symbol registration still need production integration.
No game installation, driver, GPU clock or watchdog setting was changed.

The next integration work is to apply the translation approach to the remaining
kernel families, preserving their descriptors, scratch/LDS requirements and
cross-function references, and validate each before assembling a full network.
