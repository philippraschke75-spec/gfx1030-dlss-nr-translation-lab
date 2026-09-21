# RX 6900 XT investigation — 2026-09-20

## Verified locally

- GPU: AMD Radeon RX 6900 XT, gfx1030, wave32, reported by HIP 6.4.
- ROCm 6.4 and Visual Studio 2022 Build Tools are installed.
- Bundled Python successfully ran all 44 research host tests.
- Corrected research/scripts/run_test.ps1 source path and VS discovery.
- Hardened tests/soft_wmma_test.cpp: architecture check, NaN output sentinel,
  rejection of nonfinite results.
- One 32-thread GPU block computed one 16x16x16 matrix tile. All 256 outputs
  matched the CPU reference exactly; zero nonfinite outputs. Raw log:
  ../research/soft-wmma-run.log. This does not qualify a DLSS-NR kernel.

## Static inspection of user's installer

Input: G:/dlss/dlssnr_on_amd_setup.exe
SHA256: cf7ada1486b499700a84846b342ca2b1defdb4db622843f812151f255f2ad63c

inspect_bundles.py extracted embedded ELF payloads without running the installer.
Hashes and targets are in installer-bundles/manifest.json.
ROCm 6.4 llvm-objdump decoded the gfx10-3-generic object with --mcpu=gfx1030
and gfx1100 object with --mcpu=gfx1100. Disassembly is saved alongside this file.

The generic RDNA2 package is not an entirely empty object. It contains substantial
code, but critical functions are incomplete:

- _Z13k_align_probePh at 0x5d500 starts with s_endpgm; remaining bytes are padding.
  It performs no store. Thus an unchanged zero test buffer does not establish a
  physical alignment/store fault when this code object is used.
- _Z12k_final_head10HeadParams at 0x57500 initializes v1 to zero and writes v1
  with global_store_byte through its output loop. It contains no matrix inference
  computation. Compare the much larger gfx1100 implementation at 0xa3c00.

These findings apply to this installer, not automatically to AnarchyRat's absent
modified DLL. His last supplied log reports completed jobs, but is insufficient
to establish valid neural output.

## Remaining work

A working network needs implementations of the missing numerical paths, matching
kernel arguments, thread/lane layouts, synchronization, and numerical validation.
The standalone dot2 tile test is not a drop-in WMMA replacement. The published
research repository omits generated translation inputs/payloads, so its bridge
cannot simply be built and installed from this checkout. Upstream research commit:
dcc39b815f9d8f0ccd7b77ab1e5393f32a2645c8.

No game files, drivers, clocks, watchdog settings or installed runtime files were
modified. There is no working Cyberpunk mod produced by this investigation yet.

## Follow-up: register-layout-compatible primitive

Implemented ../rdna2/fragment_wmma.h and tested its GFX11-style wave32 register
layout on the RX 6900 XT. A single dispatch checked 32 tiles / 8,192 outputs
against an independent scalar matrix reference: exact agreement, no nonfinite
outputs or guard damage. This covers exactly representable dyadic data only.
The original small smoke test used contiguous-row ownership; its misleading
comment claiming native GFX11 ownership was corrected. Native WMMA uses even/odd
row ownership across the two half-waves instead.

A separate bounded explicit-assembly store probe confirmed both aligned and
odd-address 16-bit writes on this device/runtime. This supports diagnosing the
original no-op alignment kernel as a stub rather than inferring a hardware store
failure from its unchanged buffer.

Static census: kernel-inventory.json records 203 F32/F16 WMMA sites in 24 of the
35 functions disassembled from the GFX1100 object. Integration requires address
relocation and control-flow handling as well as the arithmetic primitive.
See ../rdna2/README.md for reproduction and limitations.

## Follow-up: authentic final-head kernel

The GFX1100 `_Z12k_final_head10HeadParams` function was translated to GFX1030,
including its matrix operations, tensor layout, FP8 conversion and lookup-table
reference. The first constant-data run failed because the isolated experiment
omitted initialization of `g_e4m3_lut`. After fixing that harness omission, both
constant and patterned inputs passed; a two-workgroup patterned run matched all
32,768 output bytes against an independent scalar reference. This is an isolated
kernel result, not a complete network or game result. See
../rdna2/FINAL_HEAD_RESULTS.md for evidence and remaining work.
