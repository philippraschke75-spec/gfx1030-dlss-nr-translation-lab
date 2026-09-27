# pre_block Sites 1–4: Collapse Gates (2026-09-27)

Status: **no rewrite implemented**. The collapse remains gated on empirical
validation of the intermediate representation and a zero-mismatch GPU diff.

## Evidence checked in this checkout

- `handover/pre_block-rewrite-sketch.md` records E positional independence as
  measured for several rounding modes. That closes the spatial-independence
  question under the tested modes; it does not close intermediate requantization.
- In `handover/perf-optimization-2026-09-26/kernels-current/_Z21k_pre_block_1h_32_fp89PreParams.s`,
  the first WMMA is at `0x2b5c4`. The emitted replacement sequence then contains
  `v_add_f32 v9, 0, v9`, `v_cvt_f16_f32 v2, v9`, and
  `v_cvt_f32_f16 v2, v2` near `0x2b5cc–0x2b5d8`. This demonstrates at least
  f16 conversion after the WMMA result in this older lowering, but is not proof
  of the original kernel's downstream values or of e4m3 requantization. Later
  stores in the helper also use f16. The generated lowering is not a substitute
  for tracing what Site 3/4 actually reads.
- `rdna2/emu/trace_wmma_pre.py` can snapshot D immediately after an emulated
  WMMA (`mat`/`probe ... D`), but its current implementation stops at the
  instruction and does not pair those values with the later Site 3/4 A operand.
- `rdna2/emu/difftest_pre.py` defines the GPU differential harness. Its module
  and executable are expected under `rdna2/build/kernels-hw-scratch` and
  `rdna2/build`; those paths are absent in this checkout. The existing
  `rdna2/build` tree has historic artifacts in other build subdirectories, but
  not the configured current pre-block GPU module / runner / required Python
  command, so the required verification could not be run here.
- The user-specified full-frame arena hash is a mandatory gate. No new full-frame
  hash was measured in this session.

## Next measurement

Extend the trace harness to continue each workgroup from immediately after Site
1/2 through the next Site 3/4 WMMA, preserving lane/register values. For the same
wave and output channel, compare Site 1/2 D bits with the Site 3/4 A values at
the actual operand read. Include the intervening instructions and report whether
the value is carried as f32, rounded to f16, converted to e4m3 (and with which
scale/mode), or otherwise transformed. Run representative positions and finite
inputs that exercise normal, subnormal, sign, and quantization-boundary values.
Do not infer e4m3 merely from the fact that the Site 3/4 A operand itself is
e4m3-shaped: the intervening transformation must be identified.

Once both gates are closed, implement in an isolated lowering path and require
`difftest_pre.py` with zero mismatches plus `net_frame_full.py` arena hash
`d54273b81de7cf88` before retaining or committing the change. If the full-frame
spec cannot print PASS, document the GPU output identity and arena hash as the
specified substitute proof.

## Update 2026-09-27: requantization is not a simple scale

Direct comparison, same impulse (4,4), wave 6: Site 1's D output shows exactly
one non-zero channel delta (v8, delta 0.349 vs baseline -0.927). Site 3's A
operand for the same impulse shows multiple non-zero deltas of inconsistent
magnitude (0.5, 2.125, -1.0, 3.0, -0.234, 0.344, -0.125, -0.031...) - no single
scalar multiplies 0.349 into all of these. This rules out "D -> scale -> A" as
a simple per-channel relationship. The transformation between site 1/2's D and
site 3/4's A operand is likely a genuine additional linear mix across channels
(possibly another small matrix, not just a scale+requantize step), or A draws
from a wider combination (e.g. site1+site2 D concatenated then mixed) than a
1:1 channel copy. This does not close gate 2; it narrows what "closing" it
actually requires - the dual-stop harness needs to solve for a matrix, not a
scalar, between the two stages.

## Update 2026-09-27 (2): amplitude sweep contradicts pure-affine collapse - real nonlinearity present

Impulse-amplitude sweep at a fixed pixel (4,4), wave 6, reading Site 1's own D
output directly (not Site 3): amp=1 -> delta 0.349, amp=2 -> 0.975, amp=4 ->
1.446, amp=-1 -> -0.173, amp=-2 -> -0.780, amp=-4 -> -1.822. This is NOT
proportional to amplitude (ratios 2.79x then 1.48x for successive doublings,
not the required 2.0x), and is asymmetric in sign (|delta(-1)| != |delta(1)|).
This is the classic signature of a saturating nonlinearity (tanh/GELU/sigmoid-
like), not the pure 1x1 linear convolution the 2026-09-26 sketch assumed.
Site 3's A operand shows the same signature (amp 1/2/4/8 -> delta 0.5/2/4/8,
amp -1/-2 -> -1.5/-2.5, not linear either).

This means: "no spatial taps" (confirmed, still holds - impulses do not
propagate to neighbouring pixels) is NOT the same claim as "affine in RGB"
(does NOT hold - there is a real per-pixel nonlinearity, most likely an
activation function, between the raw RGB->1x1-conv step and the value actually
read as the WMMA operand). The G_k = E^T . B_k collapse formula in
handover/pre_block-rewrite-sketch.md section 1 assumes linearity and is
INVALIDATED by this measurement as stated. It does not mean no collapse is
possible - a per-channel scalar nonlinearity (if it is a KNOWN, simple,
closed-form function applied per-element) can still be composed with the
matrix multiply and reproduced exactly - but the rewrite can no longer skip
straight from RGB to output via one linear matrix; it must apply the correct
nonlinearity at the right point.

Next measurement: densely sample amp in [-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8]
for one fixed output channel, fit against candidate closed-form nonlinearities
(tanh, GELU, softplus, leaky-relu-with-known-slope) instead of assuming
linearity. This is a real, not-yet-closed gate; do not implement the linear
collapse from the 2026-09-26 sketch without correcting for this.
