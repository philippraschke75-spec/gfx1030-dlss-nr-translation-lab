# Frame runner state, 2026-09-24

The chain renders a full 1707x960 frame: 170 dispatches, ~620 ms, guards intact,
`k_export` finite over all 960 rows and 1707 columns. The scene is sharp and
legible. What remains is one colour defect and a set of parameters the capture
never recorded.

## Fixed this session, with the evidence

| fix | env | evidence |
|---|---|---|
| The tail is the post-block, not `k_final_head` | always on | R6/R6b/R6c: epilogue dispatches handle `0x180066350`; kernarg layout read from the stores at `0x180030e13`-`0x180030ea4`; grid `(PAD_W//8, PAD_H//8)` from the same idiom as the prologue's |
| `ctx+0x100` is the export input | always on | allocated `ctx[0x18]*ctx[0x1c]*16` at `0x18002ded7`-`0x18002df37`; `k_export` +0x00 built at `0x18002d7ca` |
| `k_export` +0x08 = `PAD_W` | always on | no shear at the padded pitch |
| Encoder skip leak removed | always on | `enc_stage_out.get(3 - di, ...)` sat in the encoder loop with `di` leaking from the allocation loop, so stages 2-4 read stage 1's output |
| U-net skip wired in the decoder | `DEC_SKIP=1` | `head input` went from 11.66% to 100% of bytes |
| Pre-block at the padded full frame | `PRE_FULL=1` | R9: grid and H/W come from `ctx+0x18` (`0x18002ee72`, `0x18002edbd`), same idiom as the post-block. Removed the hard edge at row 256 |
| `k_pre_block_1h_32_fp8` as block 0 | `PRE_1H=1` | R10: the prologue branches are exclusive and C:67 had them inverted. `byte [0x18009b1f0] == 1` pairs pre_block_1h with post_block_1h |
| **Decoder last block takes flag 2** | `DEC_LAST2=1` | **R5, `0x180030a0e`-`0x180030a22`. This removed the column stripe: dec s1 phase-var 0.992 -> 0.008, dec s4 0.986 -> 0.001** |
| Previous stage at +0x30, not +0x38 | `DEC_PTRA=1` | E3: `trace_reads 256_0 8` shows flag 8 reads +0x30 (32,768 B) and never touches +0x38 |
| Skip parity | `SKIP_PARITY=1` | encoder stage s ping-pongs `ctx+0x1d8[s]` / `ctx+0x2a8[3-s]`; the first block writes `ctx+0x1d8[s]`, so with even block counts the decoder's skip is the second-to-last output |

## Corrections to the contract that this session established

* The activation layout is **NCHW16c**: `(c//16)*(H*W*16) + (y*W + x)*16 + (c%16)`.
  Proven by a 16-impulse probe, 16 of 16 landing at the predicted positions. The
  "4x4 tile" reading of C:22 was inferred and is refuted.
* The U-net skip array is **`ctx+0x1d8`**, indexed by `s` in the encoder and
  `3-d` in the decoder. `ctx+0x2a8[d]` is the decoder's first-block output and
  half of its ping-pong pair. The skip is passed by **buffer aliasing**, never as
  a pointer argument.
* C:67 had the prologue branches inverted.
* `difftest`s ran at `H = W = 16`, two windows per axis, and were blind to
  column-periodic structure. `net_encoder_stage1.py` now takes `DIFF_H`/`DIFF_W`.
  Block 1 passes at W=64 with 0 mismatches, and the **reference emulator shows the
  same period-4 structure**, so that structure is intrinsic to the kernel and not
  a translation defect.

## The one remaining defect

A's output anti-correlates with the input: block-mean luminance correlation
-0.2574 / -0.3875 / -0.5463 at n = 1 / 8 / 32, and mid-band structural
correlation S_mid ~ 0 in every configuration tried. Green is specifically
suppressed: A's channel means are R 0.3133, G 0.0827, B 0.2631 against an input
of R 0.1218, G 0.2307, B 0.2721.

No scalar moved it. `POST_F30`, `POST_F48`, `EXP_S0`, `EXP_S1`, every channel
permutation and every input substitution left it in place.

**The open reading:** if A is a residual-form post-block, i.e. it produces a
correction rather than an image, then anti-correlation with the input is expected
and something should add the input back. That would make `k_export`'s +0x38 term
an add with the wrong sign rather than a spurious subtract, and would change what
the right value for it is.

## Parameters that cannot be read from the binary

Three remaining unknowns are **application-supplied per frame**, not constants:

* `ctx+0x30` (post-block +0x30). Never written anywhere in the frame, setup or
  driver code; only read twice, both in the epilogue.
* `k_export` +0x38 = `xmm6` = job param `+0x40`, loaded unconditionally at
  `0x18002d47b`.
* `k_export` +0x3c = `xmm7` = job param `+0x3c` when positive, else the constant
  at `0x18006d2c8`.

`EXP_S0=0` removes a subtraction that otherwise puts -65.125 into an RGBA16F
display surface, and 0 is an ordinary value for a strength control. That is a
parameter choice, not a fitted constant.

`POST_F30=0.125` is different: it was fitted against a frame-wide R:G:B ratio
that has since been retired as a correctness proxy. It should not be treated as
derived.

**Consequence for what "finished" means:** the kernels and the schedule can be
fully correct while the output still looks wrong, because the per-frame job
parameters were never captured. Closing that gap needs a capture that records
the job struct at the API boundary (the block at `rax` in the frame function,
fields +0x08 through +0x40), not more disassembly.

## Method note

Six rounds of hypotheses moved nothing. One literal read of the host's flag
computation moved everything. Where a host read is available, take it before
reasoning from symptoms.

Two metrics were used and both had to be discarded: frame-wide R:G:B against the
input (a denoiser need not match its input's channel balance) and "sharper by
eye" (amplified noise also looks sharp). The surviving measure is the band-passed
structural correlation in `smid.py`, with the column phase-variance as a guard.
