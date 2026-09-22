# External reference material (public docs, downloaded 2026-09-22)

Public, freely-redistributable AMD/LLVM documentation pulled in to stop re-deriving well-documented facts
from scratch via disassembly and bisection. None of this is proprietary or game-derived; safe to keep in the
repo. Read the relevant piece here *before* spending hours on disassembly/bisection for a new question - check
this folder first.

## Files

- **`AMDGPUUsage.html`** - the full LLVM AMDGPU backend user guide (llvm.org/docs/AMDGPUUsage.html). Authoritative
  source for kernel calling convention, kernarg layout, hidden/implicit args, kernel descriptor fields, address
  spaces, and target-specific behavior. Large (1.4 MB); grep it rather than reading whole.
- **`hidden_args_section.txt`** - extracted table of every `hidden_*` kernarg field (`hidden_block_count_x/y/z`,
  `hidden_group_size_x/y/z`, `hidden_remainder_x/y/z`, `hidden_grid_dims`, `hidden_heap_v1`,
  `hidden_dynamic_lds_size`, `hidden_private_base`, ...) with their exact meaning. **This is the table that
  would have immediately explained k_qkv_attn's "+0x34 scalar" mystery** (see
  `rdna2/emu/VARPARAMS_HOST_CONTRACT.md`'s k_qkv_attn section) - it was rediscovered by hours of bisection
  instead of a five-minute read here. Check this file FIRST any time a kernarg offset past the end of a
  kernel's own declared explicit-args size (found in its `.s`'s `.args: [.offset 0, .size N, .value_kind
  by_value]` metadata) doesn't make sense as a "real" field.
- **`sgpr_order_section.txt`** - the SGPR Register Set Up Order table: the fixed, densely-packed order system
  SGPRs appear in when their `enable_sgpr_*` kernel-descriptor bits are set (private_segment_buffer,
  dispatch_ptr, queue_ptr, kernarg_segment_ptr, dispatch_id, flat_scratch_init, private_segment_size, ...,
  then workgroup_id_x/y/z). Explains why some kernels see the kernarg pointer at `s[0:1]` (only
  kernarg_segment_ptr enabled) and others at `s[2:3]` (dispatch_ptr enabled first) - this is exactly the ABI
  distinction that mattered for `k_qkv_attn` (dual pointer) vs `k_ffwd`/`k_conv_res`/`k_qkv_attn2` (single
  pointer). Check this BEFORE assuming which SGPR holds which pointer for a newly-investigated kernel.

## ISA reference guides (`isa/`)

Real PDFs (verified `%PDF` headers, not landing-page HTML - AMD's own developer.amd.com/docs.amd.com links now
redirect to a JS-rendered portal that doesn't serve raw PDFs to curl; these came from a GitHub mirror,
`azhirnov/cpu-gpu-arch`, and were spot-checked against the same repo's file listing before trusting them).

- **`RDNA2_ISA.pdf`** (4.4 MB) - the gfx1030 target's own instruction set architecture. Confirmed via
  `pdftotext` + grep: **zero mentions of "wmma"** - independently confirms RDNA2 has no hardware matrix
  instruction, which is why `translate_final_head.py`'s `wmma()` function has to synthesize
  `v_wmma_f32_16x16x16_f16` in software (via `ds_bpermute_b32` + `v_dot2c_f32_f16`) for every translated kernel
  that uses it.
- **`RDNA3_ISA.pdf`** (3.3 MB) - the gfx1100 source target's ISA, for comparison. 40 "wmma" mentions.
- **`RDNA3.5_ISA.pdf`** (3.4 MB) - grabbed alongside RDNA3 for completeness; not yet used for anything specific.

**External validation of existing work**: AMD GPUOpen's WMMA article
(gpuopen.com/learn/wmma_on_rdna3/, not saved locally, just read) describes the real hardware's WMMA data layout:
`A`/`B` fragment data must be replicated identically between lanes 0-15 and lanes 16-31 of a wave32 wavefront.
Our own `wmma()` lowering computes `lane = (mbcnt_lo(-1,0) >> 4) << 2` before its `ds_bpermute_b32` calls -
exactly encoding "which half of the wave" a lane belongs to, matching this real convention. This is
independent, after-the-fact confirmation that the existing WMMA software lowering (already proven correct via
GPU hardware verification across the whole 22-block encoder chain) implements the real semantics, not a
coincidence that happened to pass tests.

## A fabricated search result, for the record

A web search for prior art on "RDNA3-to-RDNA2 kernel translation" produced a search-summarizer claim that a
GitHub repo `Maxritz/rocm_sdk_builder` implements WMMA-on-RDNA2 emulation via native dot instructions. Checked
via `gh api repos/Maxritz/rocm_sdk_builder` - **404, the repo does not exist**. The summarizer invented it; it
was never in the actual returned link list. Documented here so it isn't treated as real by a future session.

## Not yet checked

FP8/E4M3 instruction support across RDNA generations (neither ISA PDF mentions "fp8"/"e4m3"/"e5m2" - this
project's E4M3 handling is all software bit-tricks in the game's own kernel code either way, so this doesn't
block anything, just noting it wasn't confirmed either way from these PDFs specifically).
