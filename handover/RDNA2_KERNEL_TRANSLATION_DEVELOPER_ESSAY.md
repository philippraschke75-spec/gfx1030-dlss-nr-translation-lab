# RDNA2 DLSS-NR Kernel Translation: Current State and Next Work

## Purpose

This document describes the current state of the experimental GFX11-to-GFX10.3 kernel translation work for an AMD RX 6900 XT (`gfx1030`). It is intended for developers who want to continue the implementation and testing. Every completed item below is based on a local build or a bounded test; unresolved items are deliberately kept separate from the verified results.

## What has been achieved

The project accepts only the pinned GFX11 source ELF. The translator checks its SHA-256 before using it, regenerates the ELF metadata from the source image, initializes the FP8 lookup table for the translated target, and records source, translator, and output hashes in the build report. This prevents accidental testing against a different vendor binary.

The target code objects are assembled for AMD GFX10.3 (`amdgcn-amd-amdhsa--gfx1030`) with wave32 metadata. The translation layer currently handles the observed WMMA form (`v_wmma_f32_16x16x16_f16`), several GFX11 aliases, selected VOPD forms, branch relocation, PC-relative read-only data references, wide global offsets, selected integer min/max and CLZ operations, and an experimental lowering of private scratch arrays into per-thread LDS. Unsupported forms are rejected and recorded as `BLOCKED`; they are not silently emitted.

The current private-LDS build produces 29 assembled, unvalidated kernel candidates. Five SWIN variants are included:

```text
_Z10k_swin_varILi32ELb1EEv9VarParams
_Z10k_swin_varILi32ELb0EEv9VarParams
_Z10k_swin_varILi64ELb0EEv9VarParams
_Z10k_swin_varILi128ELb0EEv9VarParams
_Z10k_swin_varILi256ELb0EEv9VarParams
```

All five load successfully on the RX 6900 XT. The module smoke test reports valid `gfx1030` architecture, register usage, LDS allocation, and a 256-thread launch limit for each variant. The measured LDS allocations are 23,808, 23,824, 40,192, 36,096, and 27,392 bytes respectively. This proves driver/module acceptance only; it does not prove correct neural output.

Several smaller translated kernels have passed bounded physical tests with guard buffers and read-only input checks. The tests cover alignment, mean reduction, convolution/residual processing, and a synthetic lowering probe. The probe validates private-LDS isolation, dynamic byte/word loads, positive and negative wide offsets, SCC handling, and CLZ lowering. The final-head basis test exercised all 254 finite FP8 encodings and produced zero mismatches against its independent CPU reference.

The indirect-call analysis is now precise. The three larger SWIN callers resolve their `s_getpc_b64`/signed-add/`s_swappc_b64` sequences to the shared helper at source address `0xbd00`:

```text
_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti
```

The earlier `unknown` relocation report was caused by scanning only two instructions before `s_swappc_b64`; the real sequence contains waits and barriers. The resolver now searches the preceding setup window and accounts for the GFX11 PC value semantics.

## What remains

The three large callers remain blocked because the shared helper is not translated or linked. The helper is an internal device function, not an independently launchable kernel, so it has no standalone launch ABI. It begins with caller-provided registers and shared/private state and must be emitted as a relocatable function that preserves the caller contract.

The helper contains approximately 24,694 instructions, 14 WMMA sites, 857 EXEC-dependent branches, and many instruction forms that the current lowering layer does not yet cover. The important families include `v_fma_mix_f32`, `v_perm_b32`, `v_pk_mul_f16`, dual-issue multiply/max operations, divide-scale sequences, and the original dependency-control scheduling. Replacing these blindly with approximate instructions would invalidate the numerical result.

The next implementation must therefore:

1. Extract the helper’s register, scratch, LDS, and calling convention from the three callers and the source metadata.
2. Add a relocatable-function translation path separate from the kernel metadata path.
3. Lower each unsupported instruction family with an explicit semantic rule and an assembly/conformance test.
4. Relocate the helper’s internal branches and read-only data references.
5. Emit the helper object and link it with one caller first.
6. Load-test the combined module on `gfx1030` before attempting a dispatch.

Only after the combined module loads should a numerical fixture be built. That fixture needs the real `SwinParams`, `BlobLayout`, tensor dimensions, strides, weights, input/output pointers, and expected output contract. A zero-filled or guessed parameter block is not a valid neural test and may cause invalid memory access or an unbounded dispatch.

The flag kernels remain blocked for a separate reason. They use `s_sendmsg_rtn_b64` with `MSG_RTN_GET_REALTIME`. A correct RDNA2 implementation must preserve the intended 64-bit counter semantics and wait behavior; substituting an arbitrary clock or removing the wait is not acceptable.

## Recommended contribution order

Developers should work in small, reviewable steps:

1. Add a helper disassembler/inventory report and freeze its instruction-family counts.
2. Implement one lowering family at a time, starting with forms that have direct RDNA2 equivalents.
3. Add a synthetic register/LDS conformance kernel for every lowering that changes operands or addressing.
4. Assemble the helper as a relocatable object and compare its register and LDS requirements with the caller assumptions.
5. Link only the `k_swin_1h_32_fp8` caller first.
6. Run module loading and guarded dispatch tests with timeout protection.
7. Add a CPU reference for a small, fully specified SWIN tile before testing larger shapes.
8. Extend the same procedure to the pre-block and post-block callers.
9. Address realtime flag semantics.
10. Integrate with OptiScaler/DLSS-NR and attempt a single rendered frame only after the standalone numerical tests pass.

## Evidence and important limitations

The physical tests were bounded and did not change watchdog, TDR, driver, or clock settings. They demonstrate module acceptance and correctness for the specific fixtures listed above. They do not demonstrate a complete DLSS-NR neural pass, D3D12 integration, OptiScaler compatibility, Cyberpunk rendering, or GTA V Enhanced rendering. No developer should describe the project as producing a working DLSS frame until the helper-linked numerical fixture and an actual application frame have both passed.

Useful implementation files are `rdna2/translate_kernels.py`, `rdna2/translate_final_head.py`, `rdna2/kernels_test.cpp`, `rdna2/module_smoke.cpp`, `rdna2/lowering_probe_test.cpp`, and `rdna2/KERNEL_TRANSLATION_STATUS.md`. The generated `coverage.json` files record the per-symbol status and should accompany any future report.
