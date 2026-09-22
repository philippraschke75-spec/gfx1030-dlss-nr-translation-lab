# k_swin_var differential results (dispatched-path kernels)

## Offset contract follow-up

The 7x9/flags-63 mismatch below is explained by the negative-origin synthetic
fixture. The recovered output address expression and trace reproduce overlapping
writes. Retesting seed 4, grid 1x1 with explicit X/Y offsets both zero passes
with zero differing bytes and intact guards, without changing the translator.
See [VAR_OUTPUT_CONTRACT.md](VAR_OUTPUT_CONTRACT.md) for the exact formula,
field mapping, evidence and remaining limits. All 30 host tests pass.

## Expanded shape validation (2026-09-21)

The runner now accepts `--height` and `--width`. Evidence directories include
variant, flags, seed, grid, height and width, and contain `result.json` with an
output SHA256 alongside the preflight/input/reference evidence. Summary filenames
also include shape and flag list. Changing shapes no longer overwrites another
shape's artifacts. All 27 host tests pass.

New `<32,true>` cases, all with intact guards:

| H x W | Grid | Seed | Flags | Mismatching bytes |
| --- | --- | --- | --- | --- |
| 17 x 19 | 2 x 1 | 3 | 16 | 0 |
| 17 x 19 | 2 x 1 | 3 | 63 | 0 |
| 17 x 19 | 1 x 2 | 3 | 16 | 0 |
| 17 x 19 | 1 x 2 | 3 | 63 | 0 |
| 7 x 9 | 1 x 1 | 4 | 16 | 0 |
| 7 x 9 | 1 x 1 | 4 | 63 | **16, FAIL** |

For the failing case, host tracing of the original program finds conflicting
stores at PC 0xc3908 to arena offsets 0x400030..0x40003f (slot 4, pointer field
0x38). Wave 1 lanes 16..31 and wave 5 lanes 0..15 write different bytes to the
same addresses. All 16 GPU output bytes equal the wave-1 candidates; all 16
reference output bytes equal the wave-5 candidates. This is consistent with
different write ordering, not evidence for another arithmetic translation fix.
The legality of this inferred shape/stride contract remains unknown. Do not
mark this case passing, impose a wave order, or mask these mismatches.

Local trace: `rdna2/build/edge63-trace.json`; byte-by-byte comparison:
`rdna2/build/edge63-conflicting-writes.json`. The tracing script now accepts
shape, grid, seed and target-length options. Next establish authentic runtime
dimensions, strides and output ownership before using arbitrary small shapes
as correctness fixtures. These remain synthetic tests, not game rendering.

## Flag-16 translation fix (2026-09-21, supersedes historical failures below)

`v_pack_b32_f16` lowering incorrectly passed floating inline constants directly
to integer instructions. At source PC 0xb0aa8, `1.0` became FP32 0x3f800000;
shifting it by 16 produced zero instead of the required FP16 upper half 0x3c00.
At 0xb1770, wave 0 lane 0, the reference loaded v6=0x3c00302b but hardware
loaded 0x0000302b. That missing term corrupted subsequent accumulation.

Both translators now share `lower_pack_f16`, which materializes floating
constants as binary16 bit patterns before packing. Integer literals remain raw.
The corrected `<32,true>` module passes flag 16 on seeds 1 and 2, grid 2x2:
zero differing bytes across 18 MiB, non-degenerate output, guards intact.
Seed 1 changed 37,339 bytes; seed 2 changed 37,335. Flag 32 seed 1 also passes.
Combined flags 31 and 63, seed 1 grid 2x2, also pass with zero differing bytes
and guards intact (39,382 and 37,076 changed bytes respectively).
The original 35,761-byte flag-16 failure is resolved for these synthetic cases.
All 23 host regression tests pass, including constant and aliasing pack cases.

Module SHA256: `c75137d2fc8a69b19e65b0b6d5e1ea0d1cc419af383ad3b7d4867349ddeaa371`.
This is interpreter-versus-GFX1030 validation, not a native GFX1100 comparison
or an in-game rendering result. Authentic runtime parameters remain unvalidated.

## 2026-09-21 correction

Flag 32, `<32,true>`, seed 1, grid 2x2 now passes a fresh bounded hardware
comparison: 0 mismatching bytes across the 18 MiB arena, guards intact,
0.472 ms. The corrections are in the reference/harness: finite FP32 clamp
was ignored, and the reference used synthetic pointer bits instead of the
actual rebased device pointer bits. No kernel translation change was required.
The historical table below predates these corrections.

The fixture's pointer/scalar contract remains inferred. In particular, s13
consumes 0x7500 from the synthetic pointer high word, versus 0x4 on the observed
device allocation. This is a synthetic-fixture pass, not authentic runtime
parameter validation. Before the translation fix above, rebased reference replay
still left 35761 mismatching bytes for flag 16 against its saved device output.

Reference: `gfx11emu.py` (gfx1100 interpreter). Device: RX 6900 XT (gfx1030), translation built with
`translate_kernels.py --hw-scratch`. Fixture: `run_var.py` (guessed VarParams, finite small e4m3 inputs,
grid 2x2, 256 threads). Comparison: the **entire 18 MiB arena** (all parameter buffers plus 64 KiB guards),
byte for byte. Every case ran in the sandbox (`sandbox.py`) with a hard timeout; the GPU is reached only
after the emulator has terminated without fault on the same bytes.

| Variant | flags 0 | 1 | 2 | 4 | 8 | 16 | 32 | 31 | 63 |
|---|---|---|---|---|---|---|---|---|---|
| `<32,true>`  | PASS | PASS | PASS | PASS | PASS | FAIL 35,977 B | FAIL 184 B | FAIL | FAIL |
| `<32,false>` | PASS | PASS | PASS | PASS | invalid (LDS need exceeds static size) | – | – | – | – |
| `<64,false>` | PASS | PASS | PASS | PASS | PASS | – | – | – | – |
| `<128,false>`| PASS | PASS | PASS | PASS | PASS | – | – | – | – |
| `<256,false>`| PASS | PASS | PASS | PASS | PASS (was FAIL 19,626 B until the emulator's `s_or_saveexec_b32 s6,s6` bug was fixed) | – | – | – | – |

PASS = 0 mismatching bytes, non-degenerate output (187-256 distinct byte values, 21-186 KB written).
"–" = not run. Flags are the low bits of the u32 at kernarg +0x28 (bit tests at 0xB0248..0xB0344).

## Facts established while getting here
- Explicit params end at kernarg +0xA8. `+0xA8/+0xAC/+0xB0` are HIP hidden block counts and `+0xB4` the
  hidden group size; the kernel computes `wgy * gridDim.x + wgx`. A fixture that hard-codes them is wrong
  (it made every multi-row grid fail until fixed).
- The first GPU dispatch of these kernels completes (0.3-0.5 ms); no hang was observed for any variant/flag.
- 42 further gfx11 instruction forms were implemented in the emulator; each divergence found by
  `statebisect.py` in that earlier investigation was an emulator/fixture defect.
  The later flag-16 pack defect above is a confirmed translation defect.

## Open
- Broader flags/inputs/variants still need qualification; the historical failures
  for flags 16 and 32 above have been superseded for the explicitly retested cases.
- gfx1030 preserves the upper 16 bits of a 16-bit VALU result; the emulator zero-extends. Unresolved whether
  gfx11 does either; upper-half filtering is now opt-in exploratory behavior,
  not the default strict comparison and not a correctness criterion.
- Reference is an interpreter written from the ISA, not silicon. Inputs are noise, not trained weights.
