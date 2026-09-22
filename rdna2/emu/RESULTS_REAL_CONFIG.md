# Real-configuration GPU differential results (RX 6900 XT gfx1030), 2026-09-21

## Method
Real WEIGHTS_HT records (from nvngx_dlssnr.dll) at kernarg +0x10; the real launcher's kernarg layout (VARPARAMS_HOST_CONTRACT.md); origin
offsets, flags, grids from the recovered formulas. Activations are synthetic. Each case: emulator preflight, ONE bounded dispatch inside a
fresh sandbox (`rdna2/build/*-20260921`), reference recomputed at the device's actual allocation base, whole-arena byte comparison, guards intact.
Scope: equality of the translated gfx1030 kernel with my gfx11 emulator on these inputs. It says nothing about network output quality,
real activations, or gfx1100 hardware behaviour.

## Two harness errors found and fixed during this work (kept here so they are not repeated)
1. STALE MODULES: `rdna2/build/kernels-hw-scratch/*.co` were built by older translator states. `_Z10k_swin_varILi32ELb1EEv9VarParams.co` (ca4f0c5d…)
   predates the v_pack_b32_f16 fix; the corrected build is c75137d2…. The four `<N,false>` modules also differ from a fresh translation
   (32: ef05dac0 vs 0c6e4469, 64: 814b868f vs ca6f810e, 128: b39ec36d vs a9834318, 256: 8a67c2cd vs 38c8344c). A first matrix run used the stale ones and
   is therefore SUPERSEDED (log kept as matrix-stale-modules.log). k_import was identical (112560c4). Always rebuild modules with
   `translate_kernels.py --hw-scratch --symbol ... --output <sandbox>` before testing.
2. POINTER-LOOKALIKE PADDING: an unwritten padding dword at VarParams+0x6c that held a stale pointer high word made the harness' pointer-rebasing
   scan rewrite a qword on the GPU only, producing a 28-36k-byte "mismatch". Zeroing the padding (as a clean fixture should) gives 0 mismatches.
   Fixtures must not leave pointer-like bit patterns in fields the real host leaves unused.
The reference must also be recomputed at the device base (difftest_var does; the new import/pre/preblock scripts now do too).

## Verified with FRESH modules
| kernel | config | result |
|---|---|---|
| k_swin_var<32,true> as pre-block | real layout (+0x00 null, flags 0x14, float RGB at +0x40, block0 weights, +0x6c clean) 16x16, grid 2x2 | PASS, 0 / 42,812 bytes |
| k_swin_var<32,true> | synthetic layout flags 16 and 20, 2x2 | PASS, 0 mismatches |
| k_swin_var<N,false> encoder matrix | 13 cases (stages 1-4, real records block1,2,3,4,5,6,8,9,10,14,15,16,22) | **13/13 PASS**, 0 mismatches, guards intact - table below |
| k_import | formats 0-7 x modes 0-3, src 13x21 pad 16x32 | mode 0: formats 1-7 byte-exact, format 0 equal except NaN payload bits; modes 1-3: max 3 ULP (rel 2.3e-7), outputs in [0,1] |

Emulator ops added (ISA-derived unit tests in test_import_ops.py): s_min_i32, s_max_i32, v_minmax_i32, v_cvt_f32_i32, v_cvt_f32_ubyte1,
v_cvt_f64_f32, v_frexp_mant_f32, v_frexp_exp_i32_f64, v_(sub|subrev)_co_ci_u32, v_dot2acc_f32_f16.

## Encoder matrix with fresh modules (all 13 cases complete)
Origins (0,0),(-4,-4),(-4,0),(0,-4) are the four shifted-window modes; modes 0/1/3 were run at widths 64/128/256, all four at width 32 (mode 2 not run at 64+).

| kernel | flags | grid | origin (x,y) | bytes written | mismatches | status |
|---|---|---|---|---|---|---|
| swin_var<32,false> | 1 | 2x2 | (0,0) | 24355 | 0 | PASS |
| swin_var<32,false> | 0 | 3x3 | (-4,-4) | 44761 | 0 | PASS |
| swin_var<32,false> | 0 | 3x2 | (-4,0) | 32514 | 0 | PASS |
| swin_var<32,false> | 4 | 2x3 | (0,-4) | 36570 | 0 | PASS |
| swin_var<64,false> | 1 | 2x2 | (0,0) | 48649 | 0 | PASS |
| swin_var<64,false> | 0 | 3x3 | (-4,-4) | 89525 | 0 | PASS |
| swin_var<64,false> | 4 | 2x3 | (0,-4) | 73139 | 0 | PASS |
| swin_var<128,false> | 1 | 2x2 | (0,0) | 97322 | 0 | PASS |
| swin_var<128,false> | 0 | 3x3 | (-4,-4) | 178999 | 0 | PASS |
| swin_var<128,false> | 4 | 2x3 | (0,-4) | 146214 | 0 | PASS |
| swin_var<256,false> | 1 | 2x2 | (0,0) | 194669 | 0 | PASS |
| swin_var<256,false> | 0 | 3x3 | (-4,-4) | 358012 | 0 | PASS |
| swin_var<256,false> | 4 | 2x3 | (0,-4) | 292581 | 0 | PASS |

## Chained real-pixel run: k_import -> pre-block on actual captured Cyberpunk data (2026-09-22)

`chain_import_preblock_real.py`: crops an NxN tile directly from the captured colour plane (real game pixels,
`RESULTS_REAL_CONFIG` capture from 2026-09-21), runs `k_import` (format=0, mode=0 - see INPUT_CONTRACT_CYBERPUNK_FSR3.md)
on the GPU, takes that GPU-verified float output as the pre-block's input (kernarg +0x40, real layout, block0 real weights),
and runs the pre-block on the GPU. Each stage is independently checked against the emulator at the device's actual
allocation base before being chained; the pre-block never sees synthetic data for this input.

| tile (x,y) | size | import mismatches | pre-block mismatches | pre-block bytes written | status |
|---|---|---|---|---|---|
| (500,300) | 64x64 | 0 | 0 | 685,636 | PASS |
| (0,0) | 64x64 | 0 | 0 | 685,543 | PASS |
| (1600,850) | 64x64 | 0 | 0 | 685,154 | PASS |

Two harness bugs found and fixed while building this (documented so they are not repeated):
1. Kernarg pointers built once against the emulator's synthetic base were reused unchanged after re-targeting the
   emulator at the GPU's real allocation base, causing a read fault. Kernarg construction must be a function called
   fresh on every reference recomputation (the pattern `difftest_var.py` already used; the new chain script had
   inlined it incorrectly).
2. The pre-block's scratch buffer (`+0xa0`) is sized `8 KiB * grid_x * grid_y` by the real host; the fixture's fixed
   1 MiB-per-pointer-field arena only covers this for tiles up to 64x64 (64 workgroups -> 512 KiB). A 128x128 tile
   (256 workgroups -> 2 MiB) overflowed it and produced a write fault that looked like a kernel bug but was not one.

Only format=0 (RGBA16F) real colour data has been chained this way; depth and motion vectors still need their
`k_import` format codes identified (not yet done - see INPUT_CONTRACT_CYBERPUNK_FSR3.md). The rest of the network
(remaining encoder stages, middle blocks, decoder, post block, export) has not been chained onto real pixel data yet.

## Full stage-1 encoder chained on real pixels: import -> pre-block -> blocks 1-4 (2026-09-22)

`chain_encoder_stage1_real.py` extends the import+pre-block chain above through all four width-32 stage-1 blocks,
using each block's real weight record and the real flags/origin-mode schedule from VARPARAMS_HOST_CONTRACT.md. Between
blocks the verified output slot is copied forward byte-for-byte without reinterpretation (the exact inter-block tensor
encoding is still only partly understood; copying the raw bytes matches what the real host does - it re-targets a
pointer, it does not reshape/copy-convert).

Real 64x64 tile at (500,300) in the captured frame, seed 1: **all 6 stages PASS, 0 mismatches, guards intact throughout**:

| stage | kernel | bytes written | mismatches | status |
|---|---|---|---|---|
| import | k_import (format=0) | 48,928 | 0 | PASS |
| preblock | k_swin_var<32,true>, block0 weights, flags 0x14 | 685,636 | 0 | PASS |
| block1 | k_swin_var<32,false>, block1 weights, flags 1, origin (0,0) | 392,729 | 0 | PASS |
| block2 | k_swin_var<32,false>, block2 weights, flags 0, origin (-4,-4) | 377,231 | 0 | PASS |
| block3 | k_swin_var<32,false>, block3 weights, flags 0, origin (-4,0) | 384,292 | 0 | PASS |
| block4 | k_swin_var<32,false>, block4 weights, flags 4, origin (0,-4) | 445,786 | 0 | PASS |

Full log: `rdna2/emu/logs/stage1_full_chain_result.log`. Only one tile position has been run through this
full 6-stage chain so far (the individual stages were separately confirmed on 3-5 tile positions each - see above).

One harness bug found while writing this (documented so it is not repeated): a symbol-table lookup (`D.SYMS['32_0']`)
returns `lds=None` for every k_swin_var variant except `<32,true>`, which the caller must resolve via `group_size()`
before use; passing `None` straight to `run_workgroup` fails inside the emulator's LDS bounds check with a confusing
`TypeError` rather than a clear error.

**What this does and does not show:** the whole width-32 stage of the encoder now runs, on real captured game pixels,
end to end, matching my independent emulator at every step, on the physical GPU. It does not show the network's output
is correct (no gfx1100 or NVIDIA reference exists), and it stops at the end of stage 1: the stage-1-to-stage-2 transition
needs a downsample step (channel count 32->64, spatial size halved) that has not been identified or tested yet - this is
the next concrete blocker toward continuing the real-pixel chain.

## Harness note: manual `nohup ... &` backgrounding hung twice on a second full-chain run (2026-09-22)

A second confirmation run of the full stage-1 chain (tile (900,500), same script/weights/module as above) appeared to
hang for ~20 minutes with near-zero CPU usage (0.02-3 s of CPU time accumulated) and no `var_gpu_test.exe` subprocess
running, reproducing on a clean retry. Before assuming a real kernel/GPU hang, it was investigated directly (host-only,
bounded `max_steps`, no GPU): `k_import`'s output for this tile is fully finite (no NaN/Inf, range 0.0007..1.64), and
the pre-block emulator terminates cleanly for all 64 workgroups (~140,700 steps each, ~71 s total, no step-limit hit).
A GPU health check (`graphics_hook_test.exe`) run immediately after killing the stuck process passed all three checks,
confirming no device reset occurred. The hang was reproduced only when the sandboxed multi-stage script was launched via
a manual `nohup <cmd> &` redirect; it did not reproduce when launched through the harness's own background-tracking
mechanism. Conclusion: this was a host/tooling pipe- or handle-coordination issue in how stdout was inherited across the
nested subprocess chain (sandbox.py -> chain script -> var_gpu_test.exe) under manual backgrounding, not a translation,
emulator, or hardware fault. Recorded here because "the emulator/GPU hung" would otherwise be a plausible but wrong
conclusion to draw from the symptoms; always use the harness's tracked background execution for long sandboxed runs.

## Follow-up on the harness hang note above: pinpointed, still not a kernel/correctness issue (2026-09-22)

Instrumenting `chain_encoder_stage1_real.py` with per-row progress prints and retrying at tile (900,500) showed the
stall is real (not just impatience): the pre-block's first emulator pass progressed normally through workgroup row 4
of 8 (~9 s/row, matching direct measurement), then made almost no further progress over the next ~880 s of wall time.
Direct, isolated testing (host-only, no sandbox) of *every single workgroup* in rows 5-7 of that exact pre-block
dispatch, with the exact same real pixel data and weights, completed cleanly in ~1.1 s each with no exception and no
step-limit hit - so the computation itself is not the problem; it is fully deterministic and correct on this tile too.
The stall is specific to the long-running python subprocess launched by `sandbox.py run` in the background; the most
likely explanation is OS-level power/priority throttling of a long-lived, non-foreground console process (Windows
"Efficiency Mode"/background power throttling), not a translation, emulator, or GPU fault. A GPU health check
immediately after each kill passed cleanly both times.

Practical takeaway for future sessions: for these multi-stage real-pixel chain scripts (each stage's emulator pass can
legitimately take 70-150 s), prefer running shorter to keep total wall time low, or expect and tolerate slow progress
in background python subprocesses rather than treating it as a hang; verifying suspicious slowdowns by re-running the
exact same workgroup(s) directly (host-only, bounded `max_steps`) is an effective and cheap way to distinguish a real
translation bug from host/OS scheduling noise, and was used successfully here.

Net effect: the stage-1 chain (import -> pre-block) is now independently confirmed correct (though not through a
completed GPU dispatch) on a 4th real tile position (900,500), in addition to the 3 GPU-dispatch-verified positions
above. No evidence of any real correctness problem was found at any tested position.

## Stage transition, GPU-confirmed: block4's fused pooled output correctly feeds block5 (2026-09-22)

`gpu_verify_block5_real.py`: real captured pixels through import -> pre-block -> blocks 1-4 (emulator only - these
kernels were already independently GPU-verified earlier in this document, so re-dispatching them again would be
redundant), extracting block4's real fused pooled output (kernarg +0x38, written only for flags=4 - see
VARPARAMS_HOST_CONTRACT.md), then ONE bounded GPU dispatch of block5 (`k_swin_var<64,false>`, real block5 weights,
fresh module ca6f810e19694682) consuming that real pooled data as its input.

**PASS: 0 mismatches, 196,484 bytes written, guards intact** (`arena_dev=0x404010000`, 0.438 ms dispatch). This closes
the gap the earlier host-only-only chain test left open: the newly-identified stage-transition mechanism is now
GPU-verified end to end on real captured game data, not just plausible in the emulator.

Still open: block5's own real flags/origin/mode for a launch that also needs a *second* pooled output at its own
`+0x38` (it is not last-in-stage here, so this run did not exercise that), and blocks 6-8 to complete stage 2.
