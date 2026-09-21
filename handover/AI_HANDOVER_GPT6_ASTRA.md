# AI-to-AI handover: RDNA2 (gfx1030) DLSS-NR translation — everything done so far

Written 21 Sep 2026 by Claude (Sonnet 5) for the next AI assistant (ChatGPT "GPT 6 astra") continuing this work for the user
(Philipp). Read section 0 first; it tells you what is safe to claim.

## 0. Ground rules you must inherit

**Goal:** get a DLSS Neural Rendering (DLSS-NR) workload, whose GPU kernels were compiled for AMD RDNA3 (`gfx1100`), running on the
user's **RX 6900 XT (`gfx1030`, RDNA2)**, with the long-term aim of one manually testable rendered **GTA V Enhanced** frame.
Nothing renders yet. **No game frame, no D3D12/HIP interop, no OptiScaler integration, no real weights have been touched.**
Do not claim otherwise. The honest distance to the end is large (see section 8).

**How the user wants you to work** (learned this session):
- Honest, short status; state limits. "Just get it working" is the mood, but never overclaim.
- **Run tests in an isolated environment.** GPU runs go through `rdna2/emu/sandbox.py` (isolated copy + hard timeout that kills the
  process tree). A GPU cannot be truly walled off: a hung kernel can still reset the driver. **Never change TDR/watchdog, clocks,
  voltage, BIOS or registry.** One bounded dispatch per experiment; only reach the GPU after the emulator has proven the exact same
  bytes terminate.
- Commit only when asked. No global git identity is configured: commit with
  `git -c user.name="Philipp Raschke" -c user.email="philipp.raschke75@gmail.com" commit ...` and end messages with
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (or your own attribution).
- Project boundaries (from the upstream repo's `docs/reverse-engineering-boundaries.md`): do not publish translated proprietary
  kernels or raw disassembly; **no anti-cheat circumvention**; no watchdog changes. Vendor-derived files are gitignored
  (`analysis/*-disassembly.txt`, `analysis/installer-bundles/`, `*.co`, `*.o`, `*.exe`, `rdna2/build/`).
  **Caveat:** commit `2d6ec85` accidentally contains `analysis/gfx1100-notes.txt` and `analysis/selected-kernel.txt` (metadata derived from the
  vendor ELF). Fine locally; scrub from history before pushing anywhere public.

## 1. Environment (Windows 11, Git Bash + PowerShell)

| Item | Value |
|---|---|
| Repo (git, branch `master`, no remote) | `C:\Users\tes43\Documents\ChatGPT\dlss 5` |
| Python | `C:\Users\tes43\AppData\Local\Programs\Python\Python310\python.exe` (`python` is **not** on PATH in Git Bash; numpy 2.2.6 present, pytest absent) |
| ROCm 6.4 | `C:\Program Files\AMD\ROCm\6.4\bin` (`hipcc.exe`, `clang.exe`, `llvm-mc.exe`, `ld.lld.exe`, `llvm-objdump.exe`, `llvm-readobj.exe`, `hipInfo.exe`) |
| GPU | RX 6900 XT, `gcnArchName = gfx1030`, wave32 |
| Source installer | `G:\dlss\dlssnr_on_amd_setup.exe`, SHA-256 `cf7ada1486b499700a84846b342ca2b1defdb4db622843f812151f255f2ad63c` |
| Source ELF (gfx1100 member, 34 kernels) | SHA-256 `51a5ac02b52aa2b6f04a0521537833e51b28c96c13d3145a2b45b11d1aa6f1e7`, extracted to `analysis/installer-bundles/000e3200-3-hipv4-amdgcn-amd-amdhsa--gfx1100.elf` |
| Original disassembly (local only) | `analysis/gfx1100-disassembly.txt` (197k instructions, 35 functions) |
| Upstream research repo (clones, gitignored) | `research/` (dev checkout, has local edits) and `handover-upstream/` (clean, at `7215992`), from `github.com/Sakushey/gfx1030-dlss-nr-research` |

Shell quirks that cost time — avoid them:
- Huge heredocs fail (`ENAMETOOLONG`): create big files with the Write tool.
- Python source written through a bash heredoc repeatedly turned `\n`/`\1` into real newlines/control characters. For edits containing
  backslashes use the Edit tool or write a `.py` patch file with the Write tool.
- Foreground `sleep` is blocked: wait with `timeout 110 bash -c "until [ -f X.done ]; do sleep 3; done"` or run long jobs in the
  background (`nohup bash script.sh &`) and poll a marker file.
- Bare `sed -n "A,Bp"` needs numeric A,B; the disassembly line format is `\t<mnemonic> <ops>   // <ADDR12hex>: <encoding words>`.

## 2. Repository map (what each file is)

```
analysis/                         (mostly gitignored, local inputs)
rdna2/
  translate_kernels.py            MAIN translator: gfx1100 kernel -> gfx1030 .s/.o/.co (see 3)
  translate_final_head.py         earlier single-kernel translator + WMMA lowering (ds_bpermute + v_dot2c)
  helper_inventory.py             instruction-family census of the shared helper swin_layer
  kernel_inventory.py, inspect_kernel.py, *_test.cpp   earlier bounded hardware tests (final head, mean, conv_res, ...)
  module_smoke.cpp                loads a .co and reports regs/LDS/threads (no dispatch)
  flag_test.cpp                   bounded test of k_flag_wait (s_memrealtime lowering)
  aperture_probe.cpp              proves src_shared_base/src_private_base semantics on gfx1030
  swin_gpu_test.cpp               bounded dispatch harness for k_swin_1h_32_fp8 (3-pointer fixture)
  var_gpu_test.cpp                bounded dispatch harness for k_swin_var*: guarded arena + kernarg pointer rebasing
  KERNEL_TRANSLATION_STATUS.md    running status (tables + updates)
  emu/
    gfx11emu.py                   independent wave32 gfx1100 interpreter (the reference) — see 4
    run_emu.py / difftest.py      k_swin_1h_32_fp8 fixture + emulator-vs-GPU differential test
    sanity.py, mutation.py, coverage.py, coverage2.py, hangfind.py   test-power tools (see 5.2)
    run_var.py, difftest_var.py, sweep_var.py       k_swin_var* fixture, differential test, flag sweep
    statebisect.py                dumps VGPR/SGPR state on GPU + emulator, bisects first divergence (see 6)
    sandbox.py                    isolated sandbox builder + bounded process runner
    missing_ops.py, patch_ops.py  list undecodable instructions / one-shot emulator patch
    ulp_analysis.py               measures GPU-vs-emulator differences in FP8 code steps
    RESULTS.md, RESULTS_VAR.md, logs/   results documents and raw logs
handover/
  GTA_V_ENHANCED_HANDOVER_v2.md   supervisor-facing document about the upstream GitHub repo (proven vacancies V1-V11, gated path to a frame)
  GFX1030_DLSS_NR_GREEN_STATE_TECHNICAL_AUDIT.md   a document found in the tree (not written by me)
  CODING_AGENT_BRIEF.md, ...      v1 material from an earlier session
  AI_HANDOVER_GPT6_ASTRA.md       this file
```
`CODING_AGENT_BRIEF_v2.md` (copy-ready instructions for the upstream maintainer's coding agent, promised in handover v2 section 6)
**was never written** (session interrupted). It is an open deliverable.

## 3. The translator (`rdna2/translate_kernels.py`) — what it does now

Usage: `python translate_kernels.py --hw-scratch --output <dir> [--symbol <mangled>]...` (no `--symbol` = all 34 entry points).
It reads the pinned ELF, checks its SHA-256, regenerates metadata, initialises the E4M3 lookup table `g_e4m3_lut` into a copy of
`.rodata` (`initialized-rodata.bin`), lowers each kernel and assembles for `gfx1030` with `llvm-mc` + `ld.lld`, then decodes the result.
Output per kernel: `.s`, `.o`, `.co`, `-changes.json`; plus `coverage.json` (status `ASSEMBLED_UNVALIDATED`, `source_vgprs`, `vgprs`, `lds`).

Achieved and hardware-verified techniques (all on the RX 6900 XT):
1. **Shared helper linking.** The three large callers (`k_swin_1h_32_fp8`, `k_pre_block_1h_32_fp8`, `k_post_block_1h_32_fp8`) call an internal
   GFX11 helper `swin_layer` (source `0xbd00`, 24,694 instructions, 14 WMMA sites) via `s_getpc_b64 / s_add_u32 / s_addc_u32 / s_swappc_b64`.
   The helper is emitted inside each caller's code object; `s_swappc_b64`/`s_setpc_b64` remain a real call.
2. **Relocation rule (real bug found and fixed).** The helper is appended *after* the caller, so the label-relative delta is positive; the
   kept `s_addc_u32 sN, sN, -1` (right for the source's negative delta) subtracted 1 from the high address dword → jump to garbage →
   kernel launches, never completes (watchdog symptom). Fix: rewrite that `s_addc_u32` to `..., 0`. Found by trace-order bisection.
3. **`--hw-scratch`.** The callers' private array (64 B/thread) cannot go in LDS next to a 62,592 B group segment (>64 KiB). Use native gfx10
   flat scratch: `flat_scratch_init` user SGPRs after kernarg, wave offset after workgroup IDs, `s_add_u32/s_addc_u32` + `s_setreg
   HW_REG_FLAT_SCR_LO/HI`, `scratch_*` renamed to gfx10 mnemonics. (Without the flag: experimental private-LDS lowering.)
4. `s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` → `s_memrealtime` (source's `s_waitcnt lgkmcnt(0)` kept).
5. Direct equivalents: `v_dot2acc_f32_f16→v_dot2c_f32_f16`, `ds_store_b96→ds_write_b96`, `s_and_not1_*→s_andn2_*`, `s_or_not1→s_orn2`, several
   flat/global/ds aliases. `v_div_scale/fmas/fixup`, `v_fma_mix*`, `v_perm_b32`, `v_pk_*` assemble unchanged.
6. WMMA `v_wmma_f32_16x16x16_f16` → `v_mbcnt` + `ds_bpermute_b32` + `v_dot2c_f32_f16` per accumulator row (layout: A per-lane row, B per-lane
   column, replicated across lane halves; D VGPR j holds rows `2j + lane/16`, column `lane%16` — matches AMD's WMMA article).
7. VOPD (`v_dual_*`) split via scratch VGPRs; `s_delay_alu`/`s_waitcnt_depctr` → `s_waitcnt_depctr 0` + `s_nop 7`; `s_waitcnt` widened to
   `vmcnt(0) lgkmcnt(0)` + `s_waitcnt_vscnt null,0`.
Result: **34/34 entry points assemble**; all assemble ≠ numerically correct.

Known translator hazards **not yet examined** (hypotheses, unproven):
- `v_clz_i32_u32` is lowered as `v_ffbh_u32` + `v_min_u32 d,32,x`. The gfx11 instruction returns **-1** for a zero input; the lowering returns 32.
  The emulator implements the -1 behaviour and the tests still passed, so it is either not exercised or harmless — verify before relying on it.
- 16-bit VALU results: gfx1030 **preserves** the upper 16 bits of the destination VGPR (observed on hardware for `v_cvt_f16_f32`); the emulator
  zero-extends. Whether gfx11 zeroes or preserves is **unresolved** (AMD's PDF fetch timed out). The bisect tool treats upper-half-only
  differences as "soft".

## 4. The reference emulator (`rdna2/emu/gfx11emu.py`)

An independent Python/NumPy wave32 interpreter of the **original gfx1100 disassembly text** (not of my translation). Models: 256 VGPRs, SGPRs,
EXEC/VCC/SCC, LDS (shared per workgroup, 32-bit address wrap), per-lane scratch, flat apertures (`SH_HI`/`PR_HI`), global memory as guarded regions
(`GMem`, faults on any out-of-region access, records touched extents), barriers, WMMA (float64 matmul), VOPD, packed f16 math with `op_sel`/`op_sel_hi`,
`v_div_scale/fmas/fixup`, `v_fma_mix*`, `v_perm_b32`, `d16` loads, `ds_bpermute`, `v_cmp_class`, `s_bfe*`, etc. ~130 opcodes.
Unknown opcodes raise `NotImplementedError` (never guessed); memory/LDS assertions are annotated with `@addr wave op args`.
Hooks: `run_workgroup(..., trace=, stop_at=, snap=, poison=)` (first-visit trace of wave 0, snapshot at first arrival, poisoned initial VGPR/SGPR/VCC/SCC).
Semantics I got wrong and fixed while testing against hardware (useful to know where bugs hide): DS address wrap; `s_add_i32/s_sub_i32` SCC = signed overflow;
float literals in scalar ops; `v_pk_*` inline float constants are f16; `ds_load_2addr_b64` writes 2×2 dwords; a stale older `v_pk_mul_f16` handler shadowing the `op_sel` one;
`s_or_saveexec_b32 s6, s6` must read the source before writing the destination.
**Limits:** it is my reading of the ISA, not silicon; `fma` via float64; `v_log/rcp/rsq/exp/sqrt` via NumPy (they differ from hardware by ULPs); f16 denormal/flush modes not modelled;
`v_rcp/log` differences are expected to show up as 1-ULP divergences.

## 5. Verified results (all on the RX 6900 XT, host emulator as reference)

### 5.1 `k_swin_1h_32_fp8` (with helper) — `emu/RESULTS.md`
5 fixtures (16×16…24×16, grids 2×2/2×3/3×2), 0 mismatching bytes; only cases 0,2,3 are informative (cases 1/4 saturate/underflow). Note: by the upstream project's own
census this kernel and `swin_layer` are **off the dispatched path** of the authentic GTA frame.

### 5.2 Test-power checks (important — an earlier round was vacuous)
Fully random FP8 data produced an **all-NaN output** and even deliberately broken translations "passed". Always check output entropy (distinct values, NaN count) and run
mutation tests. Mutation testing (corrupt one executed translated instruction, expect mismatch): 25/25 detected for `v_perm`, `v_ldexp`, `v_log`, `v_rcp`, `v_med3`, `v_fma`,
`v_fma_mixlo`, WMMA lowering. Undetected: `v_div_fixup` (equivalent for normal quotients) and `v_fma_mix_f32` in an RMS-norm stage that sees all-zero data in every fixture (fixture coverage gap;
proven by tracing values). Lane-aware coverage was 67 % of vector instructions. **Mutation testing has not been repeated for the `swin_var` kernels.**

### 5.3 The dispatched-path `k_swin_var` kernels — `emu/RESULTS_VAR.md`
Fixture: `run_var.py` (guessed VarParams, finite small e4m3 inputs, grid 2×2, 256 threads); comparison = **whole 18 MiB arena** byte-for-byte + 64 KiB guards.

| Variant | flags 0 | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| `<32,true>` | PASS | PASS | PASS | PASS | PASS | FAIL 35,977 B | FAIL 184 B |
| `<32,false>` | PASS | PASS | PASS | PASS | invalid (LDS need > static size) | – | – |
| `<64,false>` | PASS | PASS | PASS | PASS | PASS | – | – |
| `<128,false>` | PASS | PASS | PASS | PASS | PASS | – | – |
| `<256,false>` | PASS | PASS | PASS | PASS | PASS (after emulator fix `s_or_saveexec`) | – | – |

(PASS = 0 mismatches, 187–256 distinct output byte values, 21–186 KB written.) Flags `31`/`63` on `<32,true>` also fail (they include bits 16/32).
Every GPU dispatch of these kernels **completed** (0.3–0.5 ms, guards intact): none reproduced the upstream project's hang.

### 5.4 Other hardware facts established
- `src_shared_base`/`src_private_base` on gfx1030 = `{lo=0, hi=aperture}`, identical to a real generic LDS/scratch pointer (`aperture_probe.cpp`).
- `k_flag_wait` bounded test passes (timeout path stores a non-zero `s_memrealtime` stamp; set-flag path returns immediately). `k_flag_set` has no hardware test.
- Module-load test passes for the 3 helper-linked callers (210 VGPRs, 62,592–64,640 B LDS, 256 threads).

## 6. Debugging method that worked (reuse it)

1. **Trace-order bisection for hangs** (`hangfind.py`): emulator gives the first-visit order of instructions; truncate the translated `.s` with `s_endpgm`
   after `.Lpc_<addr>:` labels, re-assemble, run with a 4–8 s bound; binary-search the first point that fails to finish.
2. **State-dump bisection for wrong answers** (`statebisect.py`, `python statebisect.py <32_1|32_0|64_0|128_0|256_0> <flags> [seed] [wgx wgy]`):
   at a chosen instruction every wave saves EXEC/SCC/VCC, forces EXEC=-1, dumps `v0..v(NREG-1)` and `s0..s63`, writes a marker and ends; VGPRs are poisoned (0xDEADBEEF) at entry
   on both sides. The emulator snapshots the same point; binary search finds the first instruction whose *pre-state* differs. Classes deliberately treated as "soft":
   differences only in the upper 16 bits (16-bit-op dst), pointer-like values (code/rodata addresses `0x8000..0x140000` differ by construction), stale code-pointer high dwords, and
   (with `ULP=n` env) float values within n ULPs. The bisect adopts the GPU's real arena base so data pointers compare equal. **Every "divergence" so far was an emulator/fixture
   defect** — do not assume the translation is wrong first.
3. **Fixture pitfalls learned the hard way:** the HIP runtime overwrites the *hidden-argument* area of the kernarg at launch — never stash your own data there. For `VarParams` the explicit
   struct ends at `+0xA8`; `+0xA8/+0xAC/+0xB0` are `hidden_block_count_x/y/z` and `+0xB4` is `hidden_group_size_x` (the kernel computes `wgy * gridDim.x + wgx`). A fixture that hard-codes them
   breaks every multi-row grid. A **zero in a loop-stride field is an infinite loop on a correct translation** (e.g. `SwinParams+0x28` must be the thread count 256): prove termination in the emulator on the exact
   bytes before any GPU run.

## 7. Recovered parameter contracts (partial, inferred from disassembly — not verified against the real runtime)

`SwinParams` (`k_swin_1h_32_fp8`, 80 B explicit): `+0x00` ptr (input), `+0x08` ptr (output), `+0x10` ptr (weight blob, FP8), `+0x18..+0x27` four int32 (H, W, two offsets; `(16,16,-4,-4)`-style),
`+0x28` u32 thread stride (read by the helper; 256), `+0x34` low 16 bits = threads (256). Reads 512 B/WG of input, ~20.6 KB of blob.
`VarParams` (`k_swin_var*`, explicit 168 B): pointers at `+0x00,+0x08,+0x10,+0x70,+0x78,+0x80,+0x98,+0xA0` (+ possibly `+0x30..+0x68`, `+0x88`, `+0x90` — pointer-vs-int unresolved, the fixture treats them as arena pointers and the
kernel terminates cleanly); `+0x18` four int32; `+0x28` **flags** (bit tests: `&12==8`, `&16`, `&48`, `&32`, `bit0`, `bit3`); different flag bits select different output buffers (`+0x08`, `+0xA0` by default; `+0x30`,`+0x38`,`+0x70` with other bits).
Nothing is known about the trained weights' blob layout, tensor shapes, kernel order or the composition math — that information lives in the vendor host runtime.

## 8. Open items, in priority order

A. **`<32,true>` flag 16 (35,977 B) and flag 32 (184 B).** Flag 16: first visible difference is a **1-ULP `v_log_f32`** (`0xbf4a6f98` vs `…99`) at `0xB070C`, but ~92 % of written bytes then differ by ≥8 e4m3 code steps
   (`ulp_analysis.py`) — that is *not* ULP noise, so something downstream is wrong or is amplifying it. Next step: `ULP=4 python statebisect.py 32_1 16 1 0 0` to skip the log noise and find the first large divergence
   (that patch is in `statebisect.py`; the command was rejected by the user right before this handover, so it has **not** been run). Flag 32: divergence only in the final per-element f16 dot-product loop
   (`v_fma_mix_f32` with `op_sel_hi:[1,1,0]`, `global_load_u16`/`ds_load_u16`), sparse lanes, GPU=0 where the emulator has a value (`+0x70` output slot). Suspects to test on hardware with tiny native-op kernels: f16 denormal handling, `v_fma_mix` semantics, gfx10-vs-gfx11 16-bit high-half behaviour.
B. Verify `v_clz_i32_u32` lowering and the 16-bit high-half question (section 3); build a native-op conformance harness (tiny `.s` kernels executing one gfx10-native instruction on random inputs vs the emulator) — the cleanest way to validate emulator semantics against real silicon.
C. Repeat mutation testing for the `swin_var` kernels; add fixtures with larger/adversarial magnitudes and more seeds (currently seed 1, 'small' e4m3 range 2..7 only).
D. `k_pre_block_1h`/`k_post_block_1h` (params unrecovered; assemble/load only); `k_flag_set` hardware test; the ~28 remaining kernels (import/export/reproject/mean/attention/conv_res2/expand/repack/dec_upsample/ffwd/qkv…) — none numerically verified.
   The upstream census lists 24 instruction forms its own emulator cannot execute; all of them live in those kernels (`v_add_f64`, `v_frexp_*`, `v_readlane/writelane` ×118 in `k_conv_res2`, `ds_*2st64`, `global_atomic_cmpswap`, `s_bcnt1`, …) — mine needs them too.
E. **Capture at the HIP bridge** (upstream `src/bridge/amdhip64_7.cpp`): it forwards 29 HIP symbols incl. `hipLaunchKernel`, `hipMalloc`, `hipMemcpyToSymbol`, `__hipRegister*`, `hipImportExternalMemory`, but logs only function pointer/grid/block/shared/stream —
   not argument bytes, allocation IDs or memcpy payloads. Adding capture (launch args from code-object metadata, allocation table, weights/LUT payload hashes, launch order/streams) is the biggest shrinker of the reverse-engineering effort.
F. Not started: standalone complete-job replay vs an independent reference; D3D12↔HIP transport probe (does ROCm 6.4 support `hipImportExternalMemory` from D3D12 on this device?); composition contract; the single-frame manual test protocol (see `GTA_V_ENHANCED_HANDOVER_v2.md` section 6, gates G0–G9).
G. Write `handover/CODING_AGENT_BRIEF_v2.md` (task prompts per gate for the upstream maintainer's coding agent).
H. The upstream repo's own blocker ("J3", `DIAG_B2_TDR`, interval `[dispatch, 0xB3A58]`, 3,547 per-wave steps) lies inside `k_swin_var<32,true>` (source range `0xB0200–0xC3B00`) — **inferred, confirm with the maintainer**. My independent translation of that kernel does **not** hang for any tested flag,
   which is evidence that the kernel is translatable/runnable and that their hang is a translation/fixture/host-model issue, not an inherent hardware limit. Their candidates are "noncanonical `s_addc`" sites; my relocation bug (section 3.2) is the same failure class but is only a hypothesis for theirs.

## 9. How to run things (isolated)

```bash
PY=/c/Users/tes43/AppData/Local/Programs/Python/Python310/python.exe
REPO="/c/Users/tes43/Documents/ChatGPT/dlss 5"
SB="<any scratch dir>/sandbox"
cd "$REPO/rdna2/emu" && $PY sandbox.py make "$SB"          # copies committed sources + local inputs; re-run/copy emu/*.py after edits
cd "$SB/rdna2" && $PY translate_kernels.py --hw-scratch --output build/kernels-hw-scratch \
    --symbol _Z10k_swin_varILi32ELb1EEv9VarParams          # add more --symbol (or none for all 34)
"/c/Program Files/AMD/ROCm/6.4/bin/hipcc.exe" -O1 --offload-arch=gfx1030 var_gpu_test.cpp -o build/var_gpu_test.exe
cd "$SB" && SWIN_TIMEOUT=8 $PY "$REPO/rdna2/emu/sandbox.py" run "$SB" 900 -- $PY -u rdna2/emu/difftest_var.py 32_1 0,1,2,4,8 1 [gx gy]
                                                            # keys: 32_1 32_0 64_0 128_0 256_0 ; results in <sb>/rdna2/build/emu_var/
                                                            # state bisect: ... -- $PY -u rdna2/emu/statebisect.py 32_1 <flags> 1 0 0   (env ULP=4 optional)
```
Emulation cost: ~1–10 s per workgroup (`k_swin_var<256>` ~1 min per flag). The upstream host tests: `cd handover-upstream && $PY -m unittest discover -s tests/host -t tests/host` (72 tests pass at `7215992`).

## 10. Upstream repository facts (Sakushey/gfx1030-dlss-nr-research, audited at `7215992`)
Proven vacancies V1–V11 with file:line and repairs are in `handover/GTA_V_ENHANCED_HANDOVER_v2.md`: smoke launcher path bug; NaN-accepting comparators (reproduced: all-NaN output passes); WMMA ownership claim contradicts AMD's layout; bundle parser treats the entry
count as a fixed "version 5" (the user's installer has 9 entries); no clean-checkout build recipe; WMMA is a value stub in the emulator; readiness script cannot advance; 24 unsupported instruction forms; J3 unresolved; no capture/param/weights layout; no D3D12/HIP/composition code.
The repo's evidence discipline (proof ladder, negative controls, host-valid ≠ hardware-valid) is the standard to hold yourself to.

## 11. Git state
`2d6ec85` (translator, tools, first handover) → `409139f` (relocation fix, emulator, GPU differential tests, handover v2) → `b7498a1` (swin_var results, sandbox, statebisect).
The next commit (made right after this file was written) adds: the `s_or_saveexec_b32` fix, `ULP` tolerance in `statebisect.py`, `ulp_analysis.py`, and this handover. `rdna2/emu/logs/mutation_results.json` is a stale partial result and is intentionally not committed.
