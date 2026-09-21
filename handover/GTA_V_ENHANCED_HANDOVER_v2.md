# Handover v2 — closing the gap to the first manually testable GTA V Enhanced frame on gfx1030

Prepared 20 September 2026 for the supervisor of **Sakushey/gfx1030-dlss-nr-research**.
Target hardware: RX 6900 XT / Navi 21 / `gfx1030`, ROCm 6.4 on Windows.
Audited upstream state: `main` @ [`7215992`](https://github.com/Sakushey/gfx1030-dlss-nr-research/commit/7215992065fef4eb9979b6cdcfe0ebdb1a518cf8) ("Finalize publication sync and repository hardening"), one commit after the state audited in handover v1 (`dcc39b8`).

This document replaces v1's status statements. It keeps v1's stage structure where it still holds, and it adds new, executed evidence. **It lists only defects and vacancies that were proven** — by reading the pinned code and, where feasible, by running it. Suspicions are kept out of the findings and are marked as hypotheses in the one place they matter (section 5).

---

## 0. Reading guide and evidence labels

| Label | Meaning |
|---|---|
| **Public source** | Read directly in the pinned checkout (file:line given) |
| **Executed** | Run in this workspace on 20 Sep 2026; command and result given |
| **Project-reported** | The repository's own claim about a private artifact or physical run; not re-run here |
| **Inferred** | Follows from Public source + Executed by arithmetic or lookup, but rests on an identity assumption stated next to it |
| **Proposed** | A prescription; the named tool/file does **not** exist yet unless stated |

Inspection method: `handover-upstream/` (a clean clone) was fast-forwarded to `7215992`; `python -m unittest discover -s tests/host -t tests/host` (the project's CI command) was run: **72 tests, OK**. The GPU used for all Executed hardware results is the local RX 6900 XT (`gcnArchName = gfx1030`, ROCm 6.4). No GTA process, D3D12 route or OptiScaler build was run for this document; **no game frame exists**.

---

## 1. Bottom line

1. **What changed upstream since v1:** only publication/CI hardening (workflows, `CITATION.cff`, publication-sync scripts, two new test modules) plus a refined status line. Every technical vacancy F1–F7 from v1 is still present at `7215992` and was re-proven (section 3).
2. **The stated blocker is unchanged:** a one-workgroup translation of the "J3" job launches on gfx1030 and never completes before the Windows watchdog fires. The latest diagnostic (`DIAG_B2_TDR`) narrows the failure to `[dispatch, 0xB3A58]` = the first 3,547 modeled per-wave steps (25.9 % of the dispatch). No mechanism is identified.
3. **New, and it changes prioritisation:** the project's own census says the `swin_layer` helper and the three `*_1h_32_fp8` kernels are **not on the dispatched path** of the authentic GTA frame (`src/isa/p16i_isa.py:359-362` and `:382-383`). Address `0xB3A58` lies inside `_Z10k_swin_varILi32ELb1EEv9VarParams` (source-image range `0xB0200–0xC3B00`). So the first-frame critical path runs through the five `k_swin_var` variants, not through the helper-linked kernels that section 4 reports on. *(Inferred — see 5.1 for the identity check.)*
4. **A failure of exactly the J3 kind was reproduced independently, root-caused and fixed** on the same source image, in a different translation of a different kernel: launch accepted, no completion, root cause = a stale `s_addc_u32` high dword after a PC-relative retarget. It was located in 14 bounded GPU runs (4 s limit each) by trace-order bisection over a 7,209-instruction execution trace. The **method** transfers directly to J3; the **cause** is a hypothesis for J3, not a finding (5.2).
5. **A numerical reference that the repo lacks now exists as a candidate:** an independent wave32 gfx1100 interpreter with a real WMMA implementation, differential-tested against the GPU (5 fixtures, 0 mismatching bytes) and mutation-tested. It should be adopted or ported to close V6 (section 4).
6. **What is entirely absent from the public repository** (grep-proven): any D3D12/DXGI code, any capture of launch arguments, any per-kernel parameter/weight-layout description, any composition math. These are the largest part of the remaining distance (V10, V11).
7. **Honest distance to the first testable frame:** the physical J3 gate has not been passed by anything in the repo; one other kernel passes a numerical test against an emulator on real hardware; ~33 kernels have no numerical evidence. See section 6 for the ordered gates.

---

## 2. Verified repository state at `7215992`

| Area | Fact | Label / proof |
|---|---|---|
| Host tests | 72 tests pass (CI command) | Executed |
| Milestone position | "between M0 and M1"; M1 (one-workgroup physical completion) in progress | Public source: `ROADMAP.md`, `STATUS.md` |
| Physical diagnostic | `DIAG_B2_TDR`, interval `[dispatch, 0xB3A58]`, 3,547 per-wave steps, 25.9 % | Project-reported: `STATUS.md` |
| Candidate-F defect | "not identified"; two of three candidate noncanonical `s_addc` sites eliminated by measurement, one remains in the interval | Project-reported: `STATUS.md`, `docs/current-state.md:68` |
| GPU-touching code | only `tests/soft_wmma_test.cpp` (+ bench) is a published GPU program | Public source: `STATUS.md` component table |
| Bridge exports | exactly 29 symbols incl. `hipLaunchKernel`, `hipMalloc`, `hipMemcpy(Async)`, `hipMemcpyToSymbol`, `__hipRegisterFatBinary/Function/Var`, `hipImportExternalMemory`, `hipExternalMemoryGetMappedBuffer` | Public source: `src/bridge/exports.def` (counted) |
| Launch gate | `DLSSNR_GFX1030_ALLOW_KERNEL_LAUNCH` defaults off; blocked launches return a real error | Public source: `amdhip64_7.cpp:157-175` |
| Launch logging | logs function pointer, grid, block, shared bytes, stream — **not argument bytes** | Public source: `amdhip64_7.cpp:366-392` |
| Source-image identity | 82 `s_getpc_b64` sites in the gfx1100 original, all `s_addc … -1`: matches the project's "82 sites" figure | Executed (own scan) vs Public source `p16h_lib.py:227-231` |

The last row is the strongest evidence that the repository's private "Candidate F" and the ELF used in this workspace (`51a5ac02b52aa2b6f04a0521537833e51b28c96c13d3145a2b45b11d1aa6f1e7`, gfx1100 member of installer `cf7ada14…ad63c`) are the same code.

---

## 3. Proven implementation vacancies

Each item: **Claim → Proof → Repair → Acceptance (with negative control)**. Line numbers refer to `7215992`.

### V1. The documented GPU smoke command cannot find its source file

- **Proof (Public source + Executed):** `scripts/run_test.ps1:5` sets `$Source = Join-Path $Here "soft_wmma_test.cpp"` with `$Here` = the script's own directory (`scripts/`). `ls scripts` lists no such file; the source is `tests/soft_wmma_test.cpp`. The script's own check at lines 26–28 therefore throws before any compile.
- **Repair:** derive the repository root from `$PSScriptRoot`; use `tests/soft_wmma_test.cpp`; write outputs to a declared build directory; add `-BuildOnly` (compile, no launch); resolve HIP root from a parameter with `C:\Program Files\AMD\ROCm\6.4` as default and record the selected version. Do not rely on a CIM class for Visual Studio discovery.
- **Acceptance:** from a clean checkout and a different working directory, `-BuildOnly` produces the executable or a precise prerequisite error, without touching the GPU.

### V2. The smoke test and the benchmark pass on all-NaN output

- **Proof (Public source + Executed):** `tests/soft_wmma_test.cpp:163-181` updates `max_abs` only when `e > max_abs`, then passes on `max_abs <= 1e-3`. `tests/soft_wmma_bench.cpp:102` uses `std::max(max_abs, …)`. Both ignore NaN. A standalone compile of exactly these two comparison loops on ROCm's `clang++` with a NaN-only GPU result printed `max_abs=0 -> PASS` for **both**.
- **Repair:** one shared comparator: check length; check every element finite before any tolerance arithmetic; initialise output with a sentinel that cannot equal any expected value; verify guards separately; retain mismatching indices with actual/expected values. Any NaN/Inf is a failure for these finite fixtures.
- **Acceptance (host-only negative controls):** injected NaN, injected +Inf, untouched sentinel, one wrong element and a corrupted guard byte must each fail. A correct result must pass.

### V3. The smoke test's stated accumulator ownership contradicts AMD's documented WMMA layout

- **Proof (Public source + AMD documentation):** `tests/soft_wmma_test.cpp:35-37` says lanes 0–15 own rows 0–7 and lanes 16–31 rows 8–15 and calls this "GFX11 wave32 WMMA C ownership"; the code uses `row_base = (lane >> 4) * 8` (line 60). AMD's WMMA article ([gpuopen.com/learn/wmma_on_rdna3](https://gpuopen.com/learn/wmma_on_rdna3/)) gives `r = ele*2 + (lIdx/16)` — interleaved rows — and states that A/B fragments must be replicated between lanes 0–15 and 16–31. The dense test can still compute a correct matrix (it stores with its own layout); it says nothing about native register ownership.
- **Repair:** keep the dense test under an accurate description; add a **fragment** test that builds native A/B register fragments (replicated half-waves) and checks each D register against `D[lane][j] = C[2j + (lane>>4)][lane & 15]`.
- **Acceptance:** identity, one-hot, signed non-symmetric and non-zero-accumulator cases; a deliberately transposed or half-wave-swapped implementation must fail.

### V4. The bundle parser treats the entry count as a fixed "version" and scans for a terminator

- **Proof (Public source + Executed):** `registry_ident.h:33` names the u64 after the 24-byte magic `kBundleVersion` and requires `5`; `registry_ident.cpp:135-175` rejects any other value, then reads 24-byte descriptors until an all-zero triple. In the Clang offload-bundle format that field is the **number of entries**. Reading the supplied installer (`cf7ada14…`, bundle at `0xE3200`): the field is **9**; the nine entries are a host entry and gfx10-3-generic, gfx11-generic, gfx1100, gfx1101, gfx1102, gfx1200, gfx1201, gfx9-generic; the 24 bytes after the counted table are zero. `kExpectedArch` (`registry_ident.h:41`) lists four architectures.
- **Scope of the defect (be precise):** for the repository's *own* target bundle, whose table evidently has five entries followed by a zero triple, the current parser can work. It is a defect against the format, and it makes the identity gate unusable for any other input, including this installer (which would be reported as "bundle version mismatch" before hashing).
- **Repair:** (1) parse exactly `entry_count` descriptors within a validated byte span with overflow-safe checks; (2) do not require a terminator; (3) validate payload ranges, duplicate IDs, id lengths; (4) keep structural parsing separate from an allow-listed **profile manifest** (bundle hash, target IDs, source-ELF hash, replacement hash); (5) unknown registrations stay untouched; a profile mismatch keeps neural execution disarmed and says so.
- **Acceptance:** synthetic bundles with 1, 5 and 9 entries parse; non-zero bytes directly after the table are not consumed as descriptors; truncated table, truncated payload, huge count, duplicate IDs, overflowed spans and a changed hash all fail predictably. **Do not** change `5` to `9` or weaken the hash check.

### V5. No clean-checkout recipe exists for the replacement payload

- **Proof (Public source + Executed):** `src/bridge/README.md` requires the intentionally unpublished `bridge_gfx1030_fatbin.h`. `src/isa/p14d_rebuild.py` operates on `phase9_final_module/asm` and `phase5_exact_fragment/…`; neither directory exists in the checkout (`ls` fails). It is a descriptor-only normaliser, not a translator.
- **Repair:** publish original generation tooling and a dependency graph while continuing to exclude proprietary inputs and generated bytes: *user-supplied installer → extract + hash → metadata → disassembly → per-symbol translation → assemble/link → descriptor/relocation validation → bundle → generated header + manifest*. Each step names its producer and tool version; missing inputs fail before the bridge compiles.
- **Acceptance:** a second developer regenerates identical qualified outputs from documented local inputs without private phase directories.
- **Available input to this task:** section 4 lists a working translator (`translate_kernels.py`) that already performs extraction-to-linked-object for all 34 entry points; it is a starting point, not a drop-in.

### V6. The emulator's native WMMA is a value stub, so its numerical verdicts cannot cover WMMA

- **Proof (Public source):** `src/emulator/emu.py:1250-1256` — `op_v_wmma_f32_16x16x16_f16` leaves accumulators unchanged; its own comment says "value-stub". The oracle-base copy has the same stub. (The emulator does contain `op_v_dot2c_f32_f16`, the software-WMMA building block.)
- **Repair:** add an explicit **control-only** mode whose reports can never receive a numerical verdict; in numerical mode, raise `UNSUPPORTED_NUMERIC` for any stub. Implement the fragment operation per V3's layout with an independently expressed matrix reference.
- **Acceptance:** executing WMMA with a known non-zero product changes the accumulators as expected; a stub trips `UNSUPPORTED_NUMERIC`.
- **Available input:** `support/rdna2-v2/emu/gfx11emu.py` implements this instruction (A/B per-lane rows/columns, replicated halves, `D[j] = rows 2j + lane/16`), and a corrupted WMMA lowering was detected by it against real hardware (section 4).

### V7. The readiness tool cannot advance from new evidence

- **Proof (Public source):** `tools/k_readiness.py:127-141` assigns `F_output_dependency_slice`, `G_j3v2_input_expected_harness`, `J_adversarial_release_tests` the literal `"NOT_BUILT"` and then requires each to be `!= "NOT_BUILT"` for `j3_ready`; `I_final_package` is a literal string. No new result can flip it.
- **Repair:** an evidence-derived manifest keyed by hashes, device, runtime, fixture, shape, launch configuration and tool version; typed stage results (`NOT_RUN/PASS/FAIL/UNSUPPORTED`) whose referenced artifacts are verified. Track host, isolated physical, authentic dispatch, complete job, transport and presentation separately.
- **Acceptance:** missing, failed, stale and wrong-profile records cannot yield readiness; a complete matching synthetic set can. A final-head record must not advance J3. Do not "fix" it by editing literals.

### V8. Twenty-four instruction forms are measured-unsupported by the emulator and live in the rest of the pipeline

- **Proof (Public source + Executed):** `src/isa/p16i_isa.py:349-357` (`CLAIMED`) lists 24 mnemonics the project's census measured as unsupported. A scan of the source image's disassembly shows every one of them occurs in non-swin kernels: `k_import` / `k_export` / `k_reproject` (`v_cvt_f64_f32`, `v_frexp_*`, `v_subrev_co_ci_u32`, `v_min_f32`, `s_max_i32`, `global_store_dwordx3`, …), `k_mean` (`v_add_f64`, `v_cvt_f32_f64`, `s_bcnt1_i32_b32`, `global_atomic_cmpswap`), `k_conv_res2` (`v_readlane_b32` ×75, `v_writelane_b32` ×43), `k_attention`/`k_attention2` (`ds_read2st64_b32`, `ds_write2st64_b32`, `v_lshrrev_b64`), `k_repack` (`v_cmpx_gt_u64`, `v_cmp_le_u64`), `k_expand`, `k_dec_upsample`, `k_conv_res_views`. None occurs in any of the six swin kernels (five `k_swin_var` variants and `k_swin_1h_32_fp8`; checked without truncation).
- **Consequence:** the swin path can be host-validated with the current emulator, but the input/output stages (`k_import`, `k_export`, `k_reproject`, `k_mean`) and the attention/conv kernels cannot until these forms are implemented.
- **Repair:** implement each form with an oracle test and a negative control; report coverage as a *population* (which forms) per kernel, as the project's own guidance requires.

### V9. The physical liveness blocker (J3) is unresolved

- **Proof (Project-reported):** `STATUS.md`. The mechanism, the failing instruction and even the exact kernel are not named in the public repository. See section 5 for the evidence this document adds and the exact experiments to run.

### V10. No capture, dispatch graph, parameter layout or weight layout is available

- **Proof (Public source + Executed):** the bridge records only function pointer, grid, block, shared size and stream for a launch (`amdhip64_7.cpp:366-392`); it never touches `args`. `kernel_policy_map.h` gives a stem→policy table for 33 kernels (policy + `dispatch_ptr` flag) and nothing about their argument structs. Each kernel takes its own by-value parameter struct (source-image metadata: e.g. `SwinParams` 80 B explicit + hidden, `VarParams` 424 B kernarg). No file in the public tree describes any of them, nor a weights/blob layout, nor kernel order.
- **Why it blocks the frame:** a numerically meaningful dispatch needs real parameter bytes. Section 4 shows a guessed value is not merely wrong but dangerous: in the kernel examined, the word at kernarg `+0x28` is a loop stride, and zero makes the kernel spin forever.
- **Repair:** section 6, gate G2 (capture at the bridge boundary; static + emulator recovery as the fallback).

### V11. No D3D12/HIP interop code and no composition code exist in the repository

- **Proof (Executed):** `grep -rli "d3d12\|dxgi\|ID3D12" src tools tests scripts` returns nothing; no FFX/OptiScaler code either. `STATUS.md` marks rung 7 "not reached".
- **Note:** the *vendor runtime* evidently performs interop itself — the bridge's export list contains `hipImportExternalMemory` and `hipExternalMemoryGetMappedBuffer` (Public source). Whether ROCm 6.4 on this device supports importing D3D12 resources is **untested** (section 6, G6).

---

## 4. Reference work executed in this workspace (adoptable)

All of this is in `support/rdna2-v2/` (source, harnesses, logs; no vendor binaries, no disassembly). Nothing here was executed inside GTA.

### 4.1 Translation techniques verified on the RX 6900 XT

| Technique | Evidence |
|---|---|
| Emit the shared helper `swin_layer` inside each caller's code object; keep `s_swappc_b64`/`s_setpc_b64` as a real call | 3 callers assemble, load (210 VGPRs, 62,592–64,640 B LDS, 256 threads); `k_swin_1h_32_fp8` runs to completion and matches the reference |
| **Relocation rule:** after rewriting the low-word `s_add_u32` of a `getpc` triple, the following `s_addc_u32` high dword must be re-derived from the new delta's sign | Without it the kernel hangs at the call; with it, correct. Root cause of the first hang (section 4.3) |
| 64 B/thread private array cannot live in LDS next to a 62,592 B group segment (16 KiB more > 64 KiB); use **native gfx10 flat scratch**: `flat_scratch_init` user SGPRs after kernarg, wave offset after workgroup IDs, `s_add_u32/s_addc_u32` + `s_setreg HW_REG_FLAT_SCR_LO/HI`, `scratch_*` renamed to gfx10 mnemonics | Assembled and executed |
| `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` → `s_memrealtime` (keep the source's `s_waitcnt lgkmcnt(0)`) | `k_flag_wait` bounded test: timeout path stores a non-zero stamp, flag-set path returns immediately, guards intact |
| `src_shared_base`/`src_private_base` on gfx1030 return `{lo = 0, hi = aperture}` identical to a real generic pointer | `aperture_probe.cpp` output: shared `0x2000000000000000`, private `0x1000000000000000`, equal to `&lds[0]`/`&priv[0]` |
| Direct gfx1030 equivalents: `v_dot2acc_f32_f16→v_dot2c_f32_f16`, `ds_store_b96→ds_write_b96`, `s_and_not1_*→s_andn2_*`, `s_or_not1→s_orn2`, `flat/global_load_u16/b32` aliases; `v_div_scale/fmas/fixup`, `v_fma_mix*`, `v_perm_b32`, `v_pk_mul_f16` assemble unchanged | Assembled; numerically covered where the fixture reaches them (below) |

Result of the full build: **34/34 entry points assemble** (`ASSEMBLED_UNVALIDATED`), after the flag kernels and helper callers were lowered.

### 4.2 Independent gfx1100 emulator and differential result

`emu/gfx11emu.py` (~700 lines, wave32, LDS, scratch, flat apertures, WMMA, VOPD) executes the **original** instruction stream of `k_swin_1h_32_fp8` and its helper. `emu/difftest.py` runs the same fixture on the GPU through `swin_gpu_test.cpp` and compares bytes.

| Case | Image, offsets | Grid | Input magnitude | Bytes written | Mismatches | Output content |
|---|---|---|---|---|---|---|
| 0 | 16×16, (−4,−4) | 2×2 | e4m3 exp 2..7 | 4,608 | **0** | 165 distinct values |
| 1 | 16×16, (0,0) | 2×2 | exp 8..11 | 8,192 | **0** | saturated (6 values) |
| 2 | 24×16, (−4,−4) | 2×3 | exp 2..7 | 7,680 | **0** | 169 distinct values |
| 3 | 16×16, (−4,−4) | 2×2 | exp 4..7 | 4,608 | **0** | 140 distinct values |
| 4 | 16×24, (−4,−4) | 3×2 | exp 0..3 | 7,680 | **0** | underflow (2 values) |

Guards (64 KiB each side of the output) stayed intact. An earlier round using fully random FP8 data also "passed" but produced an all-NaN output; a sanity check exposed it as vacuous and it was discarded. **Only cases 0, 2, 3 are informative.**

Mutation testing (corrupt one executed translated instruction, expect a mismatch): every corruption of `v_perm_b32`, `v_ldexp_f32`, `v_log_f32`, `v_rcp_f32`, `v_med3_f32`, `v_fma_f32`, `v_fma_mixlo_f16` and of the WMMA lowering was detected (25/25). Not detected: `v_div_fixup_f32` (equivalent for finite normal quotients) and `v_fma_mix_f32` in the sum-of-squares stage (that stage sees all-zero data in every fixture — a **fixture coverage gap**, proven by tracing the values in the emulator). Lane-aware coverage: 67 % of vector instructions execute with an active lane.

### 4.3 The bug this uncovered (a worked example of the J3 failure class)

The first GPU dispatch of the translated `k_swin_1h_32_fp8` timed out. Steps taken, each bounded to 4 s:

1. Emulator gave a first-visit execution order for wave 0 (7,209 distinct instructions).
2. Truncate the translated `.s` with `s_endpgm` after the label of the *k*-th first-visited instruction, re-assemble, dispatch; binary-search for the first *k* that fails to finish.
3. Result: the last passing point was the instruction before `s_swappc_b64`; truncating at the first helper instruction hung. So the call target was wrong.
4. Cause: helper appended after the caller ⇒ positive delta, but the kept `s_addc_u32 s5, s5, -1` (correct only for the original negative delta) subtracted 1 from the high dword.

This is a real failure of the symptom the repository reports (launch accepted, no completion), found by a method the repository can apply to J3 without any new hardware risk beyond bounded runs. It is **not** shown to be J3's cause.

### 4.4 Limits of section 4 (must travel with the results)

- The reference is an interpreter written by the same author as the translator from the ISA description. Agreement shows consistency with that reading of gfx11 semantics, not with silicon; a shared misreading would be invisible. The project's independent oracle remains necessary.
- `k_swin_1h_32_fp8` is, by the project's census, **off** the dispatched path. The results prove the technique, not the first-frame path.
- Inputs are random finite FP8, not trained weights; the blob layout is unknown. FP8 output quantisation (~6 %) masks 1-ulp float differences. `fma` is emulated through float64; `v_log/rcp/rsq` through NumPy.
- `k_pre_block` and `k_post_block` assemble and load only. The five `k_swin_var` variants were load-tested only (an earlier private-LDS build); they have **not** been dispatched by this work.

---

## 5. Closing J3 (V9): what to do with the evidence

### 5.1 Establish the kernel identity first (1 hour, host-only)

`0xB3A58` and the `v_cmpx_neq_f32 0, v5` at `0xB5124` (`emu.py:927`) are original gfx1100 addresses. In the source image they fall in `_Z10k_swin_varILi32ELb1EEv9VarParams` (`0xB0200–0xC3B00`); (0xB3A58 − 0xB0200)/4 ≈ 3.6 k instructions, consistent with 3,547 modeled per-wave steps. **Have the maintainer confirm**: (a) that J3 is `k_swin_var<32,true>`, and (b) the address base of their diagnostic. If (a) is false, everything in 5.2 still applies as method but not as candidate list.

Inside that interval the original module has exactly two PC-relative triples: `0xB2F4C` (`s_add_u32 s0,s0,0xfff57b30`) and `0xB3310` (`…0xfff5776c`), both `s_addc … -1`. Many other `s_addc_u32` instructions in the first 300 bytes are ordinary 64-bit pointer arithmetic (e.g. `0xB0324`, `0xB0334`, `0xB0340`) whose correctness depends on **SCC** surviving from the preceding `s_add_u32`.

### 5.2 Static audit that costs nothing (run before any further physical experiment)

Implement `tools/pcrel_target_audit.py` (Proposed): for every `s_getpc_b64` triple in the *translated* module, compute `effective = pc_next + sext32(lo) + (hi32 << 32)` and compare with the intended target (rodata offset or function) in the new layout; also verify each `s_addc` that follows an `s_add_u32` still sees the SCC that `s_add_u32` produced (no inserted or rewritten instruction between them clobbers SCC). `find_sites` in `src/isa/p16h_lib.py:212-290` already returns `target`; the missing piece is the comparison against the intended target and the SCC check. **Hypothesis (H-J3-A), not a finding:** the surviving candidate site is an instance of the relocation defect in 4.3. The audit either confirms it in minutes or removes it.

### 5.3 If the audit is clean: bounded trace-order bisection

Use the 4.3 method on the real J3 artifact and fixture; every experiment answers one question and is authorised individually.
- Build variants that end at first-visit checkpoints. **Every participating wave must be able to reach the checkpoint**: truncating with `s_endpgm` is safe with respect to barriers on gfx10 (an ended wave leaves the barrier count), but a checkpoint *inside* a barrier region reached by only some waves is not.
- Add a negative control for every checkpoint scheme: a variant with a known-bad injected loop must hang under the same harness, a known-good tiny kernel must finish.
- Stop after a timeout or device loss; never alter TDR; never retry without a new hypothesis. A host timeout is not a way to cancel a stuck kernel.
- Use a fixture whose parameters were made safe by the host model first (section 6, G2): a stride/count field of zero in a loop is enough to reproduce the symptom with a **correct** translation.

That last point matters: **a non-completing kernel is not by itself evidence of a translation defect.** The kernel examined in section 4 spins forever on a zero stride parameter. Before attributing J3's non-completion to hardware or translation, prove that the host model terminates on the exact bytes sent to the GPU, with the same kernarg block (including hidden arguments), grid, and initial memory.

---

## 6. Critical path to the first manually testable rendered frame

Gates are ordered by dependency. Each gate needs its own raw artifacts and at least one negative control; passing one never implies the next. "Agent task" text for each gate is in `CODING_AGENT_BRIEF_v2.md`.

### G0 — Evidence and build foundation (host-only; 1–2 sessions)
Fix V1, V2, V3, V6 (interface), V7; add the synthetic tests named above; keep `verify_publication.py` and the publication manifest valid (regenerate through the project's own script; do not delete checks). **Exit:** CI green; clean-checkout `-BuildOnly` works; negative controls fail as specified.

### G1 — Freeze one profile and fix identity (host-only)
Fix V4. Record: installer SHA-256 (`cf7ada1486b499700a84846b342ca2b1defdb4db622843f812151f255f2ad63c`), the counted bundle (9 entries), the gfx1100 source ELF (`51a5ac02…f1e7`), model/weights identity, ROCm 6.4 build, driver, MSVC/SDK, GTA V Enhanced build/hash, route (FFX/OptiScaler or other). **Exit:** a machine-readable profile; parser tests pass.

### G2 — Recover every kernel's contract (parallel with G0/G1)
Preferred: **capture at the bridge boundary** (Proposed, extends `amdhip64_7.cpp`; launches remain suppressed on unqualified hardware):
- `__hipRegisterFunction` → map function pointer → full mangled name; `__hipRegisterVar` → globals and sizes.
- `hipMalloc`/`hipFree` → allocation IDs, sizes, lifetimes; log every pointer argument as *(allocation ID, offset)*, never as a raw address.
- `hipLaunchKernel` → copy the **argument bytes** (sizes from code-object metadata: `.args` `by_value` sizes and pointer arguments), grid, block, shared bytes, stream, and order/dependencies (`hipEvent*`, stream identity).
- `hipMemcpy/Async` and `hipMemcpyToSymbol` → hash and, locally, store host→device payloads (weights, LUT such as `g_e4m3_lut`), with destination and offset.
- `hipImportExternalMemory` → handle type, size, flags (identifies the D3D12 resources and formats).
Fallback (when capture is impossible): static kernarg census (`tools/p16g_kernarg.py`) + emulator-guarded dispatch to recover field types, exactly as in section 4: classify each loaded SGPR group as pointer or integer from its uses, run the emulator on an arena to obtain read/write extents and to prove termination, then test the GPU. This was partially done for `SwinParams` (inferred from code, not verified against the real runtime; fields `+0x00/+0x08/+0x10` pointers, `+0x18..+0x27` four int32, `+0x28` thread stride, `+0x34` low 16 bits = threads); it is **slow and unverifiable against the real runtime**, which is why capture is preferred.
**Exit:** `job.json` with the ordered dispatch list, argument schemas, allocation table, initial-content hashes; every field of every executed kernel's arguments classified; the set of kernels actually launched.

### G3 — Host reference for the whole selected job
Close V8 for the launched kernel set; extend the emulator/oracle in numerical mode; run the captured job on the host to obtain the **golden intermediate and final tensors**. Where the emulator is too slow, add a checkpoint/replay mode that runs one dispatch at a time from captured buffer states. **Exit:** golden tensors at every kernel boundary, with a stated numerical tolerance rationale (FP8 outputs need exact or ±1-code criteria; float intermediates need justified relative tolerances) and an independent second implementation for at least the WMMA-bearing kernels.

### G4 — Physical qualification, kernel by kernel
Per-symbol ledger (translation status, descriptor, registers/scratch/LDS, relocations, globals, coverage, fixture, GPU result). For each launched kernel: the smallest meaningful checked fixture → the captured shape → boundary tiles. **J3 first**, using section 5. Reuse the section 4 harness pattern (emulator reference, guard bytes, 4–10 s bounded launch, sentinel-filled outputs, mutation controls). Generalise `swin_gpu_test.cpp` to take a captured kernarg block with a relocation list (allocation IDs → device pointers). **Exit:** every launched kernel has matching GPU-vs-reference evidence on captured shapes, with mutation results reported per family and uncovered sites listed (as in 4.2).

### G5 — Complete job, standalone
A one-shot replay runner (Proposed `src/harness/replay_job.cpp`): rebuild allocations/constants from `job.json`, run the dependency graph on the GPU, save named intermediates and the final tensor; compare to G3's tensors; check input immutability, output coverage, guards, fresh-process repeatability, and an input-dependency perturbation whose effect is known from the reference. **Exit:** one authentic job, correct at every boundary. Still not a game frame.

### G6 — Registration path and transport
1. Same job through the real bridge registration path (counted parser from V4; `__hipRegisterVar` + `hipMemcpyToSymbol` for the LUT; launch gate enabled only for the named profile). Confirm the compiled object — not the generic stub — is what launches.
2. Transport: first a standalone probe (Proposed `src/harness/d3d12_hip_probe.cpp`): create a D3D12 shared-heap resource, import via `hipImportExternalMemory`, write a known pattern from a kernel, read back through D3D12. It answers whether ROCm 6.4 supports this route on the device. If it does not, implement the **CPU-staged diagnostic route**: fenced D3D12 readback (`GetCopyableFootprints`, `CopyTextureRegion`) → HIP job → D3D12 upload, with bounded waits and never on the Present critical path. Verify D3D12/HIP device identity, not device index 0. Test transport with labelled asymmetric diagnostic patterns before any neural data.

### G7 — Composition contract
Obtain the residual/replacement convention, exposure, colour domain, crop, format and clamping from the host implementation or an observable boundary; the log line "residual on" does not define the math. Do not assume the network output is standalone RGB. **Exit:** a written contract with the captured constants and a host-side composition reference reproducing the baseline image from captured inputs before any neural data is applied.

### G8 — The manual single-frame test (the "desperately awaited" stage)
Preconditions: G5, G6, G7 pass; profile from G1; a controlled single-player scene; normal rendering as baseline; exactly one job per arming; launch gate enabled for that process only.
Procedure the human tester follows, and the artifacts to save:
1. Start the profile build; confirm the log shows the substituted counted bundle hash and the qualified module hashes (not the generic stub).
2. Arm one capture by explicit action; the system records job/frame ID and freezes inputs (jitter, exposure, history/reset state, resources).
3. Save the **baseline** frame and the composition inputs.
4. Run the one qualified job; verify tensor hash and G5-style checks before applying.
5. Compose per G7 and present via the game's controlled path; save the raw tensor, the pre-UI composed target, the displayed capture and a difference image, all keyed by the same job/frame ID.
6. On any failure: keep the baseline, print the first failing boundary, disable further submission automatically.
**Pass criteria:** identities and hashes match the profile; dispatch list equals the captured graph; numerical checks pass; the presented frame is identifiable. Image-quality judgement is reported **separately** from execution correctness; a visible difference alone proves neither.

### G9 — After the first frame
Temporal stability (history handling), sustained play, performance — not reachable before G8 and not addressed here.

**Parallelism:** G0, G1, G2 can run concurrently; G3 and the J3 work of G4 start once G2's first capture (or the fallback recovery for the swin variants) exists; G6.2 (transport probe) is independent of everything after G0 and can start at any time.

---

## 7. What only the supervisor can supply (do not let the agent guess)

1. Confirmation of the J3 identity and diagnostic address base (5.1), and access to the failing artifact + fixture (a percentage in a status report is not reproducible).
2. The host integration used for the first frame (the modified executable behind the Discord log was not available for this audit) and source or an observable boundary for its scheduler and composition math.
3. Matching model weights and initialisation, supplied locally.
4. A machine/session for bounded GPU runs and a GTA V Enhanced installation. **Compliance:** the repository's non-goals exclude anti-cheat circumvention. Run the test only in a mode in which the game permits the modified component to load; if that requires disabling any protection, stop and report — do not work around it.

---

## 8. Package contents and provenance

`support/rdna2-v2/` contains original, project-authored code and logs only: `translate_kernels.py` (changes since v1: helper linking, relocation fix, `--hw-scratch`, new lowerings), `emu/` (interpreter, harness, difftest, mutation, coverage, bisect tools, `RESULTS.md`, logs), `swin_gpu_test.cpp`, `flag_test.cpp`, `aperture_probe.cpp`, and `helper_inventory.py`. It does **not** contain vendor ELFs, code objects, disassembly, weights or game files. Tools consume user-supplied local inputs (the installer and its extracted gfx1100 ELF); running them requires the installer with SHA-256 `cf7ada14…ad63c`. Per `docs/reverse-engineering-boundaries.md`, generated translated kernels and raw disassembly remain local and unpublished.
