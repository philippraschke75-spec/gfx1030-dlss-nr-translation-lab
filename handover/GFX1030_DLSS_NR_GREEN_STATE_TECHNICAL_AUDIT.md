# Technical Audit: Requirements for `GREEN FOR PHYSICAL TEST LAUNCHING`

**Project:** [Sakushey/gfx1030-dlss-nr-research](https://github.com/Sakushey/gfx1030-dlss-nr-research)

**Target:** AMD Radeon RX 6900 XT, Navi 21, `gfx1030`, Windows HIP/ROCm runtime

**Objective:** define the information and implementation work required before the project can honestly claim a robust physical test state and progress toward a rendered DLSS-NR frame.

## Audit conclusion

The project has a strong host-side research framework, but it is not physically green. The decisive implementation gap is the internal GFX11 device helper `swin_layer`. Three large SWIN callers still depend on that helper, and it has not yet been translated into a relocatable `gfx1030` function and linked with its callers. The five smaller SWIN variants currently load on the RX 6900 XT, but module loading proves only code-object and ABI acceptance; it does not prove neural correctness.

The public repository also reports that its J3 path reaches HIP but fails liveness before the Windows watchdog, that a complete neural job is not qualified, that D3D12/HIP interop is not demonstrated, and that no GTA V Enhanced frame has been presented. These are the correct status boundaries. The repository's proof ladder must be preserved: host agreement cannot be promoted to hardware validity, and hardware validity cannot be promoted to a game frame.

For this audit, **GREEN FOR PHYSICAL TEST LAUNCHING** means that one authentic translated kernel, with its real ABI and bounded test data, completes on `gfx1030` before timeout, passes guard and read-only checks, and matches an independently implemented CPU/reference result. Assembly and module loading alone are lower gates.

## Verified progress

The local translation pipeline accepts only the pinned GFX11 ELF and checks its exact hash. It regenerates metadata from that image, initializes the target FP8 lookup data, emits wave32 `gfx1030` metadata, and records source/translator/output hashes. The translator handles the observed WMMA form, selected VOPD forms, branches, PC-relative read-only data, wide global offsets, selected integer operations, and an experimental private-scratch-to-LDS mapping. Unsupported forms are rejected and recorded as `BLOCKED`.

The private-LDS build produces 29 `.co` candidates marked `ASSEMBLED_UNVALIDATED`. Five SWIN variants pass a real RX 6900 XT module smoke test with valid target architecture, register metadata, LDS allocation, and thread limits. Several smaller kernels pass bounded tests for alignment, mean reduction, convolution/residual processing, and synthetic lowering. The final-head basis test covers all finite FP8 encodings against an independent CPU reference. These are useful lower-rung results, but they are not a complete neural or game result.

The indirect-call resolver is now correct. The three large callers contain a `s_getpc_b64` plus signed-add sequence followed by waits/barriers and `s_swappc_b64`; scanning only two preceding instructions previously reported an unknown target. The corrected scan resolves all three calls to:

```text
_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti
source address: 0xbd00
```

This fixes the diagnostic ambiguity, not the helper implementation.

## Missing information

### Authoritative ABI ledger

Every kernel and internal function needs a machine-readable ledger containing source/target symbol names, source hash and address, kernarg size and offsets, user/system SGPR mapping, work-item VGPR mapping, wave size, workgroup geometry, maximum flat workgroup size, fixed/dynamic LDS use, scratch alignment, live-in/live-out SGPR/VGPR ranges, EXEC/VCC/SCC assumptions, memory access ranges, and helper relocations.

`swin_layer` has no standalone kernel metadata. Its live register contract must be inferred from all three callers and checked with liveness analysis. Treating it as a normal kernel will corrupt caller state.

### Authentic SWIN data contract

The project needs the exact `SwinParams`, `BlobLayout`, tensor dimensions, strides, padding rules, weight layout, FP8 conversion rules, and output contract. The fixture must document every argument byte and buffer range. It must allocate guard regions and preserve copies of read-only inputs.

A guessed or zero-filled parameter block is not an authentic test. It can skip control flow, produce an accidental zero result, or access invalid memory. Numerical testing must use real layouts and nontrivial data.

### Relocatable helper ABI

The kernel translator is not yet a general device-function translator. The helper path must emit relocatable code while preserving caller state, including registers, EXEC/VCC/SCC, scratch/private memory, LDS offsets, barriers, and PC-relative data. It needs local-label relocation, symbol visibility, and a link step that proves the helper's live ranges do not overlap caller-live registers.

## Missing implementations

### Helper translation and linking

The helper is approximately 24,694 instructions long and contains 14 WMMA sites. Its inventory includes roughly 134 operation forms, including thousands of dependency-control and EXEC-sensitive instructions. It must be translated as a function, not copied as a kernel. The first link target should be `_Z16k_swin_1h_32_fp810SwinParams`; only after that passes should the pre-block and post-block callers be added.

### Instruction lowering

The helper still requires proven RDNA2 lowerings for forms including `v_fma_mix_f32`, `v_perm_b32`, `v_pk_mul_f16`, dual-issue multiply/max operations, divide-scale sequences, scratch forms, GFX11 dependency control, and all helper-local PC-relative/indirect control. Each lowering needs an ISA semantic rule, an assembly test, and an independent numerical conformance test. A superficially similar instruction is not acceptable where FP16 rounding, packed lanes, denormals, saturation, compare flags, or EXEC behavior differ.

### WMMA validation

The existing WMMA lowering needs an independent matrix conformance suite. Test lane mapping, accumulator ordering, FP16 conversion, rounding, zero/NaN behavior, and edge values against a scalar or independently written matrix reference. Do not compare a lowering against itself.

### Scratch, LDS, and barrier model

The experimental private-to-LDS map must be proven for disjoint per-thread regions, bounds, alignment, 64 KiB LDS limits, helper/caller layout agreement, and barrier participation. Test multiple waves and multiple workgroups; a one-workgroup zero-input test is insufficient.

### Hardware waitcnt and wave semantics

The J3 watchdog result has localized the first physical failure but has not identified its cause. Bounded diagnostics must distinguish waitcnt ordering, divergent barriers, changed VCC/SCC, incorrect wave32 EXEC behavior, initialization, scheduling, and code-object ABI defects. Preserve module hashes, metadata, runtime/driver versions, arguments, geometry, and checkpoint buffers. Do not disable TDR/watchdog settings or use unbounded retries.

### Realtime flag operations

`k_flag_wait` and `k_flag_set` use `MSG_RTN_GET_REALTIME` and remain blocked. The RDNA2 implementation must preserve 64-bit constant-frequency counter semantics and wait behavior. Replacing it with an arbitrary clock or deleting waits is not a valid fix.

## Required physical gates

1. **Reproducible artifact:** source ELF hash, translator revision, ROCm/LLVM version, target triple, metadata, and output code-object hash are recorded.
2. **Host semantic agreement:** emulator, independent ISA oracle, and lowering tests agree for every form in the selected kernel, with negative controls.
3. **Module acceptance:** HIP loads the exact object on `gfx1030` and reports expected symbol attributes.
4. **Authentic one-workgroup launch:** real ABI and real test data complete before timeout; guards and read-only inputs remain intact.
5. **Numerical result:** output matches an independent CPU/reference implementation under a documented tolerance.
6. **Multi-workgroup result:** multiple waves, workgroups, non-aligned dimensions, and representative sizes pass without corruption or timeout.
7. **Complete neural job:** all required kernels, helper-linked callers, and flag operations run in the correct sequence.
8. **D3D12/HIP interop:** resources, descriptors, formats, fences, and lifetimes are demonstrated on the target system.
9. **First frame:** a captured game frame proves that the translated neural output was consumed by the final render.

## Recommended implementation order

1. Add the ABI ledger and helper live-register report.
2. Add a helper instruction inventory that freezes operation counts and unsupported forms.
3. Implement relocatable helper emission without changing instruction semantics.
4. Add direct RDNA2 lowerings and conformance fixtures one family at a time.
5. Validate packed/mixed-precision and divide-scale behavior.
6. Validate all 14 WMMA sites independently.
7. Link `swin_layer` with `k_swin_1h_32_fp8`.
8. Run module loading, then one guarded authentic dispatch with a hard timeout.
9. Compare the output with the independent reference.
10. Extend to pre-block and post-block callers.
11. Implement and test realtime flag semantics.
12. Qualify the complete neural job, then D3D12/HIP interop.
13. Attempt Cyberpunk/GTA V Enhanced only after all earlier gates pass.

## Coding-agent brief

> Work only from the pinned source ELF and preserve its hash. Do not invent an ABI or call an assembled object functional. First implement relocatable-function mode for `_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti`, deriving its live-in/live-out register and memory contract from all three callers. Assemble the helper alone, then link it with `_Z16k_swin_1h_32_fp810SwinParams`. For every unsupported instruction family, add a semantic lowering and independent conformance fixture. Preserve wave32 EXEC/VCC/SCC, LDS barriers, waitcnt ordering, FP8/FP16 behavior, scratch remapping, and PC-relative relocations. Use only bounded RX 6900 XT dispatches after host and module gates pass. Capture hashes, metadata, versions, arguments, geometry, guards, completion, and numerical comparison. Leave status red when a gate is missing.

The agent must not guess `SwinParams`, remove waits/barriers to avoid a hang, substitute an unverified clock, run an unbounded neural/game dispatch, modify watchdog/TDR/clock settings, redistribute vendor binaries, or promote assembly/module loading to functional status.

## Required evidence package

Every green claim must include the source ELF hash, translator commit, output code-object hash, exact ROCm/LLVM/runtime/driver versions, metadata/disassembly, ABI ledger entry, argument dump, launch geometry, register/LDS/scratch requirements, raw runtime log, guard/read-only results, CPU/reference output and tolerance calculation, GPU identity, completion time, and— for a frame claim—resource/fence traces and before/after captures.

## Final audit assessment

The work has achieved meaningful lower-rung results: a disciplined proof framework, host emulator/oracle infrastructure, reproducible translation metadata, 29 assembled candidates, five RX 6900 XT module-load passes, and several bounded physical/numerical tests. The decisive gap is the translated and numerically validated `swin_layer` helper plus the authentic SWIN data contract. Until that helper is emitted as a correct relocatable `gfx1030` function, linked with a caller, launched with real parameters, and compared with an independent reference, the project remains an advanced research prototype rather than a physically green DLSS-NR implementation.
