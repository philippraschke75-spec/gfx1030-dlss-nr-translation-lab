# Full-frame GPU target: 20 ms on RX 6900 XT

Status 2026-09-27: **target not reached**. This note records measurements and
the resulting implementation priority. The source for the 172.799 ms reference
is the user's `Kernel Time Breakdown.html` (1707×960, 156 dispatches, arena
SHA-256 prefix `d54273b81de7cf88`). The file is a GPU event timing report;
the `net_frame_full.py` harness also performs disk I/O and is not a 20 ms wall
time benchmark.

## Budget

| Kernel family | Reference GPU time | Share |
|---|---:|---:|
| pre_block | 39.697 ms | 23.0% |
| post_block | 33.458 ms | 19.4% |
| all four active `swin_var` widths | 62.227 ms | 36.0% |
| every other kernel | 37.417 ms | 21.7% |
| **total** | **172.799 ms** | **100%** |

Pre and post alone cost **73.155 ms**. Even an ideal elimination of every other
dispatch would leave more than 20 ms. Those two kernels need a combined speedup
of at least 3.66× under that impossible best-case assumption, while the rest of
the network still needs its own budget. The target therefore requires major
rewrites of the common `swin_layer` computation and the `swin_var` family.

## Reproduced correctness baseline

The local `net_frame_full.py --stop=full` run used the saved real
`color.bin`, 1707×960, and 156 dispatches. The complete arena SHA-256 was
`d54273b81de7cf8831762757f2c85e0fac6b0aba65b52fb20c85b7acc0106f2e`;
guards were intact. This exactly matches the reported prefix. Its GPU time was
294.018 ms; repeated runs in this session ranged far lower, so this number is
not a replacement for the 172.799 ms clean-clock reference. Full-frame raw
logs and arenas are in `rdna2/build/perf_20ms_*` (local build artifacts).

## Dynamic work, measured on the existing translated kernels

`perf/exec_hist.py` counted original instructions in the interpreter, and
`perf/kernel_profile.py` mapped the executed addresses to the translated
assembly. These are **instruction counts**, not proportional GPU time.

| Representative workgroup | Estimated translated instructions per wave | WMMA lowering per wave | WMMA share |
|---|---:|---:|---:|
| pre_block, 8×8 test | 39,137 | 3,168 | 8.1% |
| post_block, constant-input test | 39,725 | 3,168 | 8.0% |
| `swin_var<256>`, 8×8 flags=1 test | 326,978 | 126,720 | 38.8% |

The HTML footnote says pre/post are “mostly from WMMA-emulation”. The dynamic
instruction evidence does not support that as a claim about the instruction
mix. Optimizing *only* WMMA cannot yield the needed pre/post speedup. In the
pre_block fixture, 268,320 of 311,394 emitted wave-instructions executed
inside the shared `swin_layer` helper; its wider normalization, quantization,
addressing and control flow require profiling and a structured rewrite.
WMMA remains a large target for `swin_var<256>`.

## Rejected low-level candidate

An isolated build lowered independent GFX11 `v_dual_*` halves directly to two
gfx1030 instructions instead of two temporary writes plus two copies. A static
dependency check found 39/39 pre_block and 158/159 `swin_var<256>` sites safe
for this form. The candidate compiled, passed pre_block and `swin_var<256>`
GPU/emulator difftests with 0 mismatches, and reproduced the full-frame arena
SHA-256 above. It did **not** improve a repeated pre_block benchmark:
32 launches averaged 35.938 ms for the existing build and 36.003 ms for the
candidate. The source change was removed. This is consistent with only 142
dynamic `v_dual_*` operations per pre_block wave (under 1% of translated
instructions). The candidate's build files remain under ignored `rdna2/build/`
for investigation, but are not installed.

## Next implementation gates

1. Profile the shared `swin_layer` helper at the level of loops and basic
   blocks, then replace one complete expensive mathematical section with
   RDNA2-native code. Pre and post both call it, so a proven common change
   can benefit 42% of the frame.
2. For `swin_var`, prioritize the 256/128 variants and the dynamic WMMA
   groups documented in `handover/swin_var-rewrite-sketch.md`. Prove exact
   math, including quantization and attention interaction, before collapsing
   operations. The 64/32 variants require separate alignment.
3. Every candidate must pass its relevant 0-mismatch kernel or chain
   difftests and reproduce the full arena hash
   `d54273b81de7cf88` before installation. Measure multiple interleaved
   full-frame runs because GPU-clock variation is large on this machine.

No source kernel rewrite is currently verified as faster. The measured target
gap remains 152.799 ms relative to the clean 172.799 ms reference.
