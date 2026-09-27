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

## Update 2026-09-27 (3): shape of the nonlinearity - clean linear on the positive side, saturating on the negative side

Extended sweep, same pixel/channel as above (Site 3 A, wave 6, pixel (4,4)):

| amp | delta | delta - amp (offset from slope-1 line) |
|---|---|---|
| 2 | 2.0 | 0.0 |
| 4 | 4.0 | 0.0 |
| 8 | 8.0 | 0.0 |
| -1 | -1.5 | -0.5 |
| -2 | -2.5 | -0.5 |
| -4 | -4.5 | -0.5 |
| -8 | -7.2188 | +0.78 (breaks the -0.5 pattern) |

Positive side (amp 2/4/8): exactly slope 1, zero offset - clean linear, no
saturation visible in this range. Negative side (amp -1/-2/-4): slope 1 but a
CONSTANT -0.5 offset - still locally linear, just shifted. Negative side at
amp -8: the constant-offset pattern breaks (would need -8.5, got -7.22) -
the incremental contribution per unit amp shrank between -4 and -8, i.e. the
derivative is decreasing there. That is the actual saturation point, not the
smaller negative amplitudes. This whole shape (linear positive branch, locally
linear-but-shifted moderate negative branch, saturating far-negative branch)
is consistent with a smooth saturating activation (tanh/GELU/softplus family)
evaluated at some nonzero baseline offset - NOT with a leaky-ReLU-style kink,
since the moderate-negative branch is not simply a different fixed slope, it
still increments by ~1 per unit amp (matching the positive slope) up to -4.

This narrows the "next measurement" from the prior update: fit is now
constrained to a smooth function whose local derivative is ~1 near the
operating point and only drops off for large negative excursions (roughly
|baseline + E.RGB| beyond ~4-8 in whatever units this activation operates in).
Candidate: GELU/softplus, not tanh (tanh's derivative starts dropping much
closer to 0 for typical normalized activations; the ~1 slope holding flat out
to magnitude 4 suggests a wider linear region, more consistent with GELU or
softplus's approx-identity region for x>0 extending into shallow negative x).
Still not proven - needs a proper least-squares fit against multiple candidate
closed forms using a finer amplitude grid, ideally in a separate script rather
than manual point-by-point sweeps.

## Update 2026-09-27 (4): systematic multi-cell sweep - simple 1D models do not fit

Built rdna2/emu/fit_pre_nonlinearity.py: sweeps a finer amplitude grid (20
points) and tracks EVERY (register, lane, half) cell that changes, not just
the one cell found by hand, then least-squares-fits linear / kinked-linear /
tanh-saturating candidates per cell.

Result: many more cells respond than the single one tracked manually (regs
99-103, both lanes 0 and 16, both halves) - the impulse's effect is spread
across a wider register/lane footprint than the earlier single-cell probes
suggested. None of the three candidate closed forms fit well: best-residual
fits still have residuals of the same order as the deltas themselves (e.g.
reg 99 lane 16 lo: best fit residual 48 across 20 points of similar
magnitude - a bad fit, not a real characterization).

Likely cause: the probe sweeps amplitude along the R=G=B diagonal only (single
scalar IMP_AMP sets all three color channels equal). If the true function
combines R, G, B with different, independently-varying coefficients (which is
the whole premise of the E matrix in the original collapse hypothesis), a
1D diagonal sweep only samples one slice of a genuinely 3-input function and
cannot be fit by any 1-input closed form, regardless of whether the underlying
function is linear or not. This confounds "is it linear" with "is my probe
axis representative" - the earlier single-cell nonlinearity finding (Updates
2/3) may still be real, but this multi-cell sweep does not by itself add
further evidence either way.

Correct next step (not yet done): separate single-channel impulses (R alone,
G alone, B alone, not R=G=B together) at several amplitudes each, per cell,
then fit each cell as a function of the 3-vector (R,G,B) rather than a scalar
amplitude. This is a bigger measurement (3x the runs) and was not completed
in this session - documenting so the next session does not repeat the
diagonal-only probe and misinterpret a bad fit as "no clean function exists".

Status: gate 2 remains OPEN. rdna2/emu/fit_pre_nonlinearity.py is reusable
infrastructure (no kernel/build files touched) for whoever continues this.

## Update 2026-09-27 (5): independent R/G/B probe - reg 96 is cleanly linear, likely reverses Updates 2/3

Built the corrected probe: IMP_R/IMP_G/IMP_B env vars in trace_wmma_pre.py's
prepare() (default: all = IMP_AMP, so the old diagonal probe is unchanged and
still works), rdna2/emu/fit_pre_nonlinearity.py extended to sweep each colour
channel independently (other two held at exactly 0) and fit a separate linear
slope per channel per (reg,lane,half) cell.

For (reg=96, lane=0/16, half=hi) - the cleanest cell found: R: k=+0.1250
(resid/spread 0.050, GOOD), G: k=+0.4617 (0.017, GOOD), B: k=+0.4197 (0.024,
GOOD). All three are excellent linear fits over their own 11-18-point sweeps.
Sum of the three slopes = 1.007 - matches Update 2/3's observed ~slope-1
response on the R=G=B diagonal almost exactly. This strongly suggests the
diagonal probe's apparent "saturating nonlinearity" (Updates 2/3) was a
measurement artefact of summing three genuinely-different-but-individually-
linear channel slopes along one 1D probe axis, not a real activation function.
Updates 2/3's conclusion is likely WRONG for this cell; do not treat it as
settled.

Other cells (regs 97-103) show much noisier per-axis fits (many POOR,
resid/spread up to 0.6-0.9) - these are plausibly cells that should not
respond to a (4,4) impulse at all (register reuse/aliasing across loop
iterations picking up unrelated data, not real coupling), not genuine
additional nonlinear structure. Not yet confirmed either way - would need to
check whether the POOR cells correspond to a different (wx,wy) region than
the impulse pixel, which would explain the noise as illegitimate cross-talk
rather than model failure.

Net effect: gate 2 status is now UNCLEAR-BUT-MORE-OPTIMISTIC than Updates
2/3 suggested. Next measurement: (a) confirm/refute more cells the same way
reg 96 was confirmed, restricting to cells that legitimately belong to the
impulse pixel; (b) if slopes are consistently clean per-axis, the original
G_k = E^T.B_k affine collapse from the 2026-09-26 sketch may be closer to
correct than Updates 2/3 implied - re-open that path rather than assuming it
is dead.

## Update 2026-09-27 (6): only reg 96 is real signal - other cells are quantization-boundary noise, not coupling

Filtered the independent-axis results to lane 0 (the impulse pixel's own
lane) across all 8 A-registers (96-103, hi+lo = 16 cells). Only reg 96
(hi and lo, which print identically - likely a duplicate-storage artefact of
how f16 pairs are packed) shows clean GOOD fits on all three colour axes with
substantial slopes (0.125/0.462/0.420). Every other register (97-103) shows
small (0.01-0.1 magnitude, near the e4m3 quantization step size), inconsistent
(mix of GOOD/OK/POOR, sign flips between hi/lo of the "same" register that
should be duplicates if real) slopes - the signature of quantization-boundary
noise (occasional e4m3 grid crossings unrelated to genuine coupling) rather
than real signal.

Synthesis: for pixel (4,4)/wave 6/site 3, there appears to be exactly ONE real
linear response cell (reg 96), cleanly affine in R, G, B independently, and
everything else is measurement noise. This is consistent with (and further
supports reopening) the original 2026-09-26 sketch's collapse hypothesis -
Updates 2/3's "nonlinearity" conclusion should be treated as superseded by
Updates 5/6, not as the current understanding.

Remaining work before gate 2 can be called CLOSED (not done in this session):
repeat this clean-cell isolation for a few more (impulse pixel, output
channel) combinations to confirm reg-96-like cleanliness is the general case,
not a lucky single sample; then derive E per real channel from the R/G/B
slopes directly (no separate quantization-mode question remains once the
function is confirmed affine - the KNOWN e4m3 encoder from kernels/e4m3/
already reproduces the quantization step bit-exactly, that part was solved by
this project long before pre_block work started).

## Update 2026-09-27 (7): sites 1/2 are overwhelmingly linear, sites 3/4 are not - affine collapse is real but only proven for sites 1/2

Ran the independent R/G/B sweep (rdna2/emu/fit_pre_nonlinearity.py) across
all four WMMA sites, same pixel (4,4)/wave 6, counting fit quality over every
changed (reg,lane,half) cell:

| site | GOOD | OK | POOR |
|---|---|---|---|
| 1 | 62 | 8 | 2 |
| 2 | 46 | 18 | 4 |
| 3 | ~4 (reg 96 only) | several | many |
| 4 | 4 | 28 | 40 |

Sites 1 and 2 (the v8:15 output pair) are overwhelmingly clean linear per
colour channel across MANY registers, not just one lucky cell - this is
strong, broad evidence the affine-collapse hypothesis holds for sites 1/2.
Sites 3 and 4 (the v48:55 pair, which read "derived" e4m3 features per the
2026-09-26 sketch, values up to 30) are much noisier - either they are
genuinely more complex (a real nonlinearity or multi-stage dependency,
consistent with reading something already processed rather than raw RGB), or
the impulse pixel/wave choice that works for sites 1/2 is not the right probe
point for sites 3/4's actual inputs. Not yet distinguished.

Practical implication: a collapsed rewrite of sites 1/2 alone is now well
supported and could proceed independently of resolving sites 3/4 - that is
roughly half of pre_block's 14 WMMA sites (main iteration pair) and worth
implementing even if 3/4 stay on the WMMA path for now. Proceeding to derive
the actual numeric E matrix for sites 1/2 and validate it against real
(non-unit-impulse) random RGB inputs next.

## Update 2026-09-27 (8): affine model validated against random RGB - gate 2 CLOSED for sites 1/2

Derived slopes for (reg=80, lane=0, half=hi), site 1: kR=+0.0625, kG=+0.09375,
kB=-0.03125, baseline (RGB=0) = 0.28125. Predicted value = baseline +
kR.R + kG.G + kB.B, tested against 5 RANDOM (non-axis-aligned, non-unit-
amplitude) RGB triples in [-3,3]^3 - NOT the unit impulses used to fit the
slopes:

| RGB | predicted | actual | error |
|---|---|---|---|
| (0.837,-2.85,-1.35) | 0.1086 | 0.0859 | 0.0226 |
| (-1.661,1.419,1.06) | 0.2773 | 0.2812 | 0.0039 |
| (2.353,-2.478,-0.468) | 0.2106 | 0.1875 | 0.0231 |
| (-2.821,-1.688,0.032) | -0.0543 | -0.0625 | 0.0082 |
| (-2.841,-1.807,0.899) | -0.0938 | -0.1016 | 0.0078 |

Max error 0.023, consistently within half an e4m3 grid step at this magnitude
(~0.03-0.06) - i.e. the residual is exactly what quantization rounding alone
would produce, not model error. This is a genuine out-of-sample validation
(random inputs the slopes were never fit to), not just a restatement of the
fitting data, and it closes gate 2 for site 1 at this (reg,lane,half) cell:
out = q_e4m3(baseline + E . RGB) with a plain per-channel affine E, no hidden
nonlinearity, no separate scale/requant step beyond the already-solved e4m3
encoder.

Combined with Update 7 (62/72 cells GOOD for site 1, 46/68 for site 2): the
affine collapse is now well-supported for sites 1/2 specifically, not just
one lucky cell. Sites 3/4 remain open (Update 7's noisier results).

STATUS: gate 2 is CLOSED for sites 1/2. Next actual step is implementation:
build a lowering pass (new module, mirroring perf/splice_encoder.py's
register-budget discipline) that computes the full 16-channel E matrix per
site 1/2 (not just the one sample cell measured here) and replaces the WMMA
call with the direct affine+e4m3-quantize computation, then verify with
difftest_pre.py (0 mismatches) and net_frame_full.py (arena hash
d54273b81de7cf88 unchanged) before any commit that touches kernel code. Not
yet done in this session - the E-matrix derivation here was for 1 of 16
output channels, all 16 need the same treatment before a full kernel rewrite
is possible.

## Update 2026-09-27 (9): full 16-channel E matrix derived for site 1, validation not yet tight enough for a bit-exact rewrite

Added rdna2/emu/derive_e_matrix.py: derives baseline + (kR,kG,kB) for all 16
A-operand channels (registers 80-87, lo/hi) via single two-point finite
differences (amp=+2 per axis), writes handover/pre_block_site1_E_matrix.json,
then validates against 4 random out-of-sample RGB triples across all 16
channels at once.

Result: max error per random trial ranged 0.13-0.23 - noticeably worse than
the single best-channel validation in Update 8 (0.023). Several channels
(reg 83 hi, part of 84 lo/85 lo/87 lo) show all-zero measured slopes at
amp=2, i.e. their true slope is small enough that amp=2 doesn't cross an
e4m3 grid boundary - the single-amplitude finite difference cannot resolve
them, so their baseline+slope estimate for those channels is unreliable
(quantization noise dominates the 2-point estimate, not real coupling
measured).

This means the current E-matrix estimate is NOT yet accurate enough across
all 16 channels to drive a bit-exact kernel rewrite - only the previously
validated single channel (reg 80, half hi) is trustworthy at the level Update
8 achieved. Implementing the collapsed lowering NOW, on this data, would risk
shipping something that fails difftest_pre.py or (worse) passes a narrow
check but corrupts the full-frame hash - exactly the failure mode this
project's protocol exists to prevent (c.f. splice_encoder.py's two prior
real bugs).

STATUS: gate 2 remains open for a full-kernel rewrite, though narrowed
further. Not implementing the lowering this round. Next measurement: larger
amplitude and/or multi-point (not 2-point) linear regression per channel to
resolve the near-zero-slope channels above quantization noise, then re-run
this same out-of-sample validation before attempting any kernel change.

## Update 2026-09-27 (10): larger amplitude improves but does not close the E-matrix validation gap

Re-ran derive_e_matrix.py with DERIVE_AMP=8 (was 2) and DERIVE_AMP=16 for
comparison. AMP=8: out-of-sample max errors 0.061-0.099 across 4 random RGB
trials (down from 0.13-0.23 at AMP=2). AMP=16: 0.067-0.101, no further
improvement - AMP=8 is roughly the practical floor for this simple two-point
finite-difference method. handover/pre_block_site1_E_matrix.json now holds
the AMP=8 derivation.

Typical channel values in this measurement are in the range [-2, +3]; an
error of ~0.10 is large enough to occasionally land on the wrong side of an
e4m3 grid boundary (grid step ~0.0625-0.125 in this range), so this is still
not tight enough to guarantee bit-exact output from a rewritten kernel - it
would very likely pass a coarse "looks right" check and then fail
difftest_pre.py's 0-mismatch bar on some fraction of pixels, which per this
project's protocol is not acceptable to ship.

The remaining error is not obviously quantization noise (AMP=8/16 agree with
each other, ruling out under-resolved small slopes as the cause) - it is more
likely either (a) genuine second-order structure per channel that a strict
2-point linear fit cannot capture, or (b) a systematic error in how baseline
is being read (e.g. baseline measured at a slightly different register state
than what the random-RGB runs converge through). Not yet diagnosed.

STATUS: gate 2 still open for implementation. This round's conclusion: do
not implement the collapsed lowering from this E matrix - the residual error
is real, not a measurement-resolution problem, and needs its own root-cause
pass (proper multi-point linear regression per channel with more amplitude
samples, or checking hypothesis (b) above) before it is safe to build a
bit-exact replacement kernel from it.

## Update 2026-09-27 (11): both hypotheses (a) and (b) ruled out as the general cause - error is concentrated in 3 of 16 channels

**Hypothesis (b) ruled out:** compared emulator step counts for the baseline
(RGB=0) run vs. several random-RGB probe runs, same pixel/site, before the
WMMA stop point - all identical (59608 steps every time). No branch
divergence; baseline and probes take the exact same control-flow path.

**Hypothesis (a) mostly ruled out via a direct additivity test:** for
(reg=80, half=hi), measured single-axis deltas R=+0.125, G=+0.1875,
B=-0.0625, then measured pairs: R+B together = 0.0625 (sum predicts 0.0625,
diff 0), G+B together = 0.125 (sum predicts 0.125, diff 0) - exact matches,
no interaction. R+G together = 0.28125 vs sum-predicted 0.3125 (diff
-0.03125, exactly one e4m3 grid step at this magnitude) - consistent with
ordinary quantize-after-sum rounding, not a real R*G interaction term. This
channel is genuinely affine; its earlier out-of-sample error was
quantization noise, nothing else.

**What's actually going on:** broke the AMP=8 E-matrix's per-trial max error
down by channel instead of taking the overall max. Result: 13 of 16 channels
have max error <=0.052 (in line with ordinary e4m3 quantization noise for
their magnitude); 3 channels are the real outliers - reg 86 half=hi (0.099),
reg 85 half=hi (0.061), reg 87 half=hi (0.069). These three also had smaller/
zero measured slopes on some axes (see Update 9/10) - i.e. their true signal
is weaker relative to quantization step size at the amplitudes tried so far,
so their 2-point slope estimates are the least reliable, not because the
channels are non-affine.

STATUS: the affine model itself looks sound (confirmed no interaction terms,
no branch divergence). The remaining work is narrow: get better slope
estimates specifically for regs 85/86/87 (half=hi) - try a proper multi-
amplitude least-squares fit for just these three cells rather than the
blanket 2-point method, before re-running full validation and considering
implementation.
