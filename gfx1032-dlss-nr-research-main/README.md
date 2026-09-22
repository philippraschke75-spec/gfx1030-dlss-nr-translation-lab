# gfx1032-dlss-nr-research
# DLSS-NR on AMD RDNA2 / gfx1032

Experimental research into running NVIDIA DLSS Neural Rendering (DLSS-NR) workloads on AMD RDNA2 hardware, specifically **gfx1032 / Radeon RX 6600**, using AMD HIP/ROCm and modified AMDGPU code objects.

> **Status: Experimental / active reverse-engineering**
>
> The project has progressed from extracting the original AMD/HIP code object to producing a gfx1032-targeted experimental object and successfully loading the resulting DLL far enough to run inside GTA V Enhanced. Correctness, kernel execution, and performance are still being validated.

---

## Hardware / Software Environment

### Target GPU

* GPU: AMD Radeon RX 6600
* Architecture: RDNA2
* LLVM target: `gfx1032`
* VRAM: ~8 GB
* CPU: Ryzen 5 5500
* System RAM: 32 GB DDR4
* Storage: 1 TB M.2 SSD

AMD documentation identifies `gfx1032` as an RDNA2 target, and current ROCm documentation includes gfx1032 among supported GPU targets.

### ROCm / HIP

Installed under:

```text
C:\Program Files\AMD\ROCm\7.2\
```

Important tools used:

```text
llvm-objdump.exe
hipInfo.exe
clang++.exe
llvm-objdump.exe
```

Some expected LLVM utilities such as `llvm-readelf.exe` are not present in the installed ROCm 7.2 `bin` directory, so ELF inspection has primarily been performed with `llvm-objdump`.

HIP eventually confirmed the RX 6600 as:

```text
gfx1032
```

### OS / Driver

* Windows
* AMD Adrenalin 26.8.1
* Windows build observed during research: `10.0.19044.3031`
* Debloated Windows installation referred to locally as "SaphireOS"

---

# Project Goal

The ultimate goal is to determine whether the DLSS-NR workload used by GTA V Enhanced can be made to execute correctly and efficiently on AMD RDNA2/gfx1032 hardware.

The research path is:

```text
Original DLSS-NR DLL
        ↓
Extract embedded AMDGPU/HIP code
        ↓
Analyze ELF/code-object structure
        ↓
Translate/adapt target architecture
        ↓
Produce gfx1032 code object
        ↓
Reinsert into DLL
        ↓
Load in GTA V Enhanced
        ↓
Prove RX 6600 kernel execution
        ↓
Validate image/output correctness
        ↓
Optimize performance
```

---

# Important Project Files

The exact contents of the folder should still be verified with a directory listing before treating this as a literal complete inventory. The following files/artifacts are confirmed from the research history.

## Original / extracted material

### `gfx1032_embedded_original.elf`

Extracted AMDGPU ELF/code object from the original DLSS-NR material.

Important properties observed:

* ELF AMDGPU shared object/code object
* `.text` present
* `.rodata` present
* AMDGPU-specific metadata
* Original ELF SHA recorded during research:
  `EF7B4ED9...`
* Original AMDGPU ELF flags:

```text
0x01000038
```

This file is one of the most important reference artifacts because it represents the original code before our gfx1032 modifications.

---

## V3 research artifacts

### `version.dll`

Current/working research DLL at various stages of the project.

Previously recorded:

```text
Size:   7,304,459 bytes
SHA256: FA616204DD68521217875FB41835D3189C89E7C87C3050E069DF16BAFC458A05
```

The DLL contains the DLSS-NR integration and embedded AMD HIP/code-object material.

---

### `gfx10-3-generic.elf`

Earlier extracted AMDGPU object associated with the original embedded material.

Previously observed:

```text
Offset: 0x750
Size:   1,170,280 bytes
```

This was used during the earlier ELF/code-object analysis.

---

# V4

V4 is currently the most important experimental generation.

## `version_v4.dll`

V4 research DLL.

Previously recorded size:

```text
7,304,459 bytes
```

V4 successfully reached GTA V Enhanced and produced a brief black-screen period before the game continued loading.

This is significant because the modified DLL is not simply failing immediately at DLL loading.

---

## `v4_gfx1032_object.bin`

Current experimental gfx1032 AMDGPU object.

This is the main code object being analyzed.

It successfully disassembles using:

```powershell
llvm-objdump.exe -d --triple=amdgcn-amd-amdhsa --mcpu=gfx1032
```

The executable `LOAD` region begins at:

```text
File offset: 0xAD00
Virtual address: 0xBD00
```

---

# V4 ELF Structure

Program headers observed:

```text
PHDR
LOAD r--
LOAD r-x
LOAD rw-
LOAD rw-
DYNAMIC
RELRO
STACK
NOTE
```

The executable region:

```text
LOAD
offset:  0xAD00
vaddr:   0xBD00
filesz:  0x10D918
memsz:   0x10D918
flags:   r-x
```

The start of this region corresponds exactly with the beginning of the major `swin_layer` function.

---

# Important Kernel: `swin_layer`

Symbol:

```text
_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti
```

V4 location:

```text
Start: 0xBD00
Size:  0x20FB4
End:   0x2CCB4
```

This is one of the primary pieces of the neural workload being investigated.

The kernel successfully disassembles under the gfx1032 target.

Examples of recognized instructions include:

```text
s_cbranch_execnz
v_add_nc_u32_e32
s_load_dword
v_mov_b32_e32
s_cmp_lt_u32
s_lshl3_add_u32
s_add_u32
s_addc_u32
v_sub_f32_e32
v_lshlrev_b32_e32
v_mul_hi_u32_u24_e32
v_cvt_f32_u32_e32
```

---

# Other Important Kernels

## `k_pre_block_1h_32_fp8`

Symbol:

```text
_Z21k_pre_block_1h_32_fp89PreParams
```

V4:

```text
Start: 0x2DE00
Size:  0x2624
End:   0x30424
```

Metadata observed:

```text
private_seg_size: 0x40
num_agpr:          0
num_vgpr:          0xC2
numbered_sgpr:     0x2B
uses_vcc:          1
uses_flat_scratch: 1
has_dyn_sized_stack: 0
has_recursion:       0
```

---

## `k_post_block_1h_32_fp8`

Symbol:

```text
_Z22k_post_block_1h_32_fp810PostParams
```

V4:

```text
Start: 0x30500
Size:  0x1C1C
End:   0x3211C
```

Metadata observed:

```text
private_seg_size: 0x40
num_agpr:          0
num_vgpr:          0xC2
numbered_sgpr:     0x29
uses_vcc:          1
uses_flat_scratch: 1
has_dyn_sized_stack: 0
has_recursion:       0
```

---

# Architecture / ELF Flag Investigation

One major unresolved difference is:

### Original

```text
e_flags = 0x01000038
```

### V4

```text
e_flags = 0x01000054
```

The exact significance of every changed bit has not yet been fully established.

This is a priority investigation because AMDGPU ELF metadata can affect how code objects are interpreted and loaded.

---

# Instruction Analysis

The V4 object contains many instructions that LLVM recognizes normally.

However, several regions are displayed as raw `.long` values instead of symbolic instructions.

Example:

```text
.long 0xd7007c21
.long 0x000200ff
.long 0x00006080
.long 0xdc510000
```

Other unusual instruction patterns have also been observed.

A significant unresolved area is the presence/count of `v_dual_*` instructions and whether those instructions represent:

1. valid gfx1032 instructions,
2. architecture-specific encodings,
3. instructions LLVM 7.2 does not decode cleanly under gfx1032,
4. or instructions that require actual translation.

---

# WMMA / MFMA Investigation

Searches were performed for:

```text
wmma
mfma
v_wmma
v_mfma
```

The expected obvious matches were not found in the V4 disassembly.

This means we cannot currently characterize the workload simply as a straightforward collection of RDNA3/RDNA4 matrix instructions.

Further instruction-level analysis is required.

---

# Kernel Termination Investigation

An initial search for:

```text
BF 81 00 00
```

(the plain `s_endpgm` encoding being investigated) produced no matches.

However, this initially led to an incorrect conclusion because the inspected address was incorrectly assumed to be the end of `swin_layer`.

The symbol table subsequently established:

```text
swin_layer end = 0x2CCB4
```

while:

```text
k_pre_block begins = 0x2DE00
```

Therefore the earlier bytes around `0x2EDB4` belonged to `k_pre_block`, not `swin_layer`.

### Current next step

Inspect the actual final bytes/instructions from:

```text
0x2CC70 → 0x2CCB4
```

using LLVM disassembly.

This remains unfinished.

---

# GTA V Enhanced Integration

Game location used during research:

```text
C:\Program Files (x86)\Steam\steamapps\common\
Grand Theft Auto V Enhanced
```

The research DLL is used as:

```text
version.dll
```

The project intentionally tries to minimize changes to normal driver/runtime files and concentrate modifications in the research DLL/code object where possible.

---

# Original DLSS-NR Runtime Behavior

Original DLSS-NR AMD configuration observed:

```ini
[DlssNrOnAmd]
Enabled=1
LocalStructure=2.000
SkinStructure=-1.000
UseAutoMask=1
Scale=0.03125
Temporal=0
Tonemap=-1
UseFsrInputs=1
UseDepth=1
Interop=1
PreUpscale=0
PreHistory=0
Async=0
Inline=1
SpinDraw=0
InlineWaitMs=200
HipDevice=-1
```

Observed runtime status included:

```text
Post-upscale
Structure intensity: 2.0
FSR 3.1.4 active
HIP job
Async zero-copy
```

---

# Performance Observations

Earlier working DLSS-NR tests showed significant overhead.

Observed approximate ranges:

```text
History OFF:
~30–33 ms

History ON:
~63–65 ms
```

More recent observations from the current setup have shown approximately:

```text
~16–30 ms under load
```

These numbers are **not yet considered an optimized gfx1032 baseline**.

We still need to determine exactly how much time is spent in:

* GPU kernel execution
* memory copies
* capture
* synchronization
* kernel launch overhead
* waiting
* other host-side processing

Therefore current timing should be treated as a prototype measurement rather than a hard performance ceiling.

---

# Raw Output / Debugging Artifacts

Earlier DLSS-NR runtime captures included files such as:

```text
dlssnr_in
dlssnr_out
outraw
imgenc
hist
```

Observed sizes included approximately:

```text
dlssnr_in/out:  16,588,800 bytes
outraw:         35,389,440 bytes
imgenc:         26,542,080 bytes
hist:                    0 bytes
```

These captures were used to investigate the actual data flowing through the neural-rendering path.

---

# WinDbg Investigation

WinDbg has been used to investigate actual HIP kernel launch behavior.

Important breakpoint target:

```text
amdhip64_7!hipHccModuleLaunchKernel
```

This is intended to establish whether the modified code object is actually reaching HIP kernel launch and, ultimately, the RX 6600.

Some breakpoint sessions reached the target but required further investigation because `g`/execution did not immediately provide useful additional output.

This remains one of the most important runtime-validation paths.

---

# Current Research Questions

The project currently needs to answer five major questions.

## 1. Does the V4 code execute correctly on gfx1032?

Not merely:

> Does the DLL load?

but:

> Does the RX 6600 actually execute the converted code object?

---

## 2. Is the V4 AMDGPU ELF metadata correct?

Especially:

```text
0x01000038
vs
0x01000054
```

---

## 3. Are the unusual instruction encodings valid for gfx1032?

Particularly:

* `.long` sequences
* `v_dual_*`
* unusual vector/matrix-related instructions

---

## 4. Is the kernel ABI correct?

Need to validate:

* VGPR allocation
* SGPR allocation
* private segment size
* LDS requirements
* workgroup dimensions
* kernarg layout
* launch parameters
* code-object metadata

---

## 5. What is actually consuming the 16–30+ ms?

Potential optimization targets include:

* register pressure
* occupancy
* LDS usage
* memory traffic
* synchronization
* launch overhead
* history handling
* host/GPU copies
* instruction selection

---

# Current Status

### Completed

* [x] RX 6600 identified as gfx1032
* [x] ROCm/HIP environment established
* [x] Original DLSS-NR DLL investigated
* [x] Embedded AMDGPU object extracted
* [x] ELF structure analyzed
* [x] V3/V4 research generations produced
* [x] V4 gfx1032 object produced
* [x] V4 ELF program headers verified
* [x] V4 executable region identified
* [x] `swin_layer` located
* [x] Major kernels mapped
* [x] Kernel metadata inspected
* [x] V4 disassembly confirmed
* [x] GTA V Enhanced successfully loads the V4 DLL far enough for meaningful testing
* [x] Initial runtime/performance observations collected

### In progress

* [ ] Inspect actual end of `swin_layer`
* [ ] Explain `e_flags` difference
* [ ] Decode `.long` instruction regions
* [ ] Investigate `v_dual_*`
* [ ] Validate kernel ABI
* [ ] Prove actual RX 6600 kernel execution
* [ ] Validate neural output
* [ ] Profile actual GPU vs synchronization/copy cost
* [ ] Optimize performance

---

# Current V4 Assessment

V4 should currently be considered:

> **A structurally valid, disassemblable experimental gfx1032 code object that can be integrated into the DLSS-NR DLL and loaded far enough into GTA V Enhanced for runtime investigation.**

It is **not yet proven** to be a fully correct native gfx1032 implementation.

The next objective is therefore not blindly modifying more instructions.

The next objective is:

```text
PROVE WHAT V4 IS ACTUALLY EXECUTING
             ↓
IDENTIFY THE FIRST INCORRECT/EXPENSIVE STAGE
             ↓
FIX THAT STAGE
             ↓
MEASURE AGAIN
```

---

# Performance Outlook

The current observed ~16–30 ms under load should **not** yet be treated as the final performance limit.

Potential optimization areas include:

* occupancy
* VGPR pressure
* synchronization
* memory traffic
* kernel launch behavior
* history processing
* architecture-specific instruction selection

A successful gfx1032 implementation could potentially improve substantially over the current prototype, but no specific final latency should be assumed until profiling identifies where the current time is going.

---

# Research Philosophy

The project favors:

1. **Preserving the original DLL whenever possible**
2. **Changing the smallest possible amount of binary data**
3. **Keeping the target specifically on gfx1032 / RX 6600**
4. **Verifying every modification against the original**
5. **Using reproducible PowerShell/LLVM commands**
6. **Proving runtime behavior rather than assuming it**
7. **Separating structural validity from functional correctness**
8. **Measuring performance before optimizing**

---

# Important Historical Note

The project has explored both gfx1030/RX 6950 XT research and gfx1032/RX 6600 research.

The **current primary target is gfx1032**.

The gfx1030 work is useful as comparative RDNA2 research, but it should not replace the gfx1032 path unless evidence shows that doing so is technically necessary.

---

# Next Session — Exact Starting Point

Resume from:

```text
V4 gfx1032 object
```

and inspect:

```text
0x2CC70 → 0x2CCB4
```

for the true end of:

```text
_Z10swin_layerR7SwinLDSPKhRK10BlobLayouti
```

Then continue with:

```text
1. swin_layer termination
2. e_flags 0x01000038 → 0x01000054
3. unknown instruction encodings
4. v_dual_* investigation
5. HIP kernel-launch proof
6. ABI validation
7. performance profiling
```

---

## Project Motto

**ROCm → SOCM → Robots**

If this eventually becomes a public research project, this belongs in the README somewhere. 🤖

---

## Disclaimer

This is experimental reverse-engineering and compatibility research. No claim is currently made that the resulting implementation is equivalent to NVIDIA's native DLSS-NR implementation, fully correct, production-ready, or performant.

The goal is to establish reproducible evidence for what is technically possible on AMD RDNA2/gfx1032 hardware.
