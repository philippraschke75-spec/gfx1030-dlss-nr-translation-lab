# gfx1030 DLSS-NR translation lab

Experimental AMD RDNA2 (gfx1030, RX 6900 XT) translation and runtime research for the DLSS-NR workload. **This is a research log, not a mod:
no neural-enhanced frame has been produced, and nothing here renders or enhances anything in a game.** Every claim below is scoped to what was
actually measured; open questions are listed as such.

## Status - September 21, 2026 (evening)

**Now measured (real data, this machine):**

* **The real model inputs exist and are captured.** Cyberpunk 2077 uses the native FSR3 upscaler (`ffxFsr3UpscalerContextDispatch`) on this GPU.
  A one-shot capture at that call produced one frame of *linear HDR RGBA16F colour*, *32-bit depth*, *RGBA16F motion vectors* (all 1707x960 render
  resolution, 2560x1440 output) plus per-frame jitter, motion-vector scale and render size. See
  [input contract](rdna2/emu/INPUT_CONTRACT_CYBERPUNK_FSR3.md). The earlier 8-bit backbuffer capture is *not* what the network consumes.
* **The weights and the network layout were recovered from the vendor files.** The DLSS-NR DLL embeds a 147.7 MB `WEIGHTS_HT` resource: the
  container framing is decoded and consistent for all 153 tensor records ([findings](rdna2/emu/WEIGHTS_HT_FINDINGS.md), parser
  `rdna2/emu/weights_ht.py`, index of offsets/statistics only). The tensor *encoding* (mixed FP8-e4m3 matrices and f16 tables) is only partly understood.
* **The launch contract was read from the AMD port's own host code.** The 71-block schedule (4 encoder stages of 4/4/6/8 blocks at widths
  32/64/128/256, per-block shifted-window modes and flags), the `k_swin_var` argument layout, the input padding rule (multiples of 128) and the
  `k_import` / pre-block layouts are documented in [VARPARAMS_HOST_CONTRACT.md](rdna2/emu/VARPARAMS_HOST_CONTRACT.md) and
  [DLL_HOST_EVIDENCE.md](rdna2/emu/DLL_HOST_EVIDENCE.md). Static reading only; the model's numerics were not compared with NVIDIA output.
* **Translated kernels verified against an independent emulator on the physical GPU** ([results](rdna2/emu/RESULTS_REAL_CONFIG.md)): the whole
  encoder (13/13 real-weight cases: all four shifted-window modes at width 32, modes 0/1/3 at widths 64/128/256), the real first block (`k_swin_var<32,true>` as pre-block),
  and `k_import` for source formats 0-7 (mode 0 byte-exact apart from NaN payload bits; other modes within 3 ULP).
  The reference is my own gfx1100 interpreter (`rdna2/emu/gfx11emu.py`), **not** a real gfx1100 GPU, so this shows translation equality on those inputs, not
  correctness of the network. Two harness errors found on the way (stale kernel modules, pointer-like padding) are documented so they are not repeated.

**Not done:** the attention/ViT middle blocks, decoder/upsampling, post block and export kernels, the inter-block sync flags, a full offline run on the captured
frame, real-frame-format handling in the import stage, any comparison against real network output, and everything live-in-game (D3D12/HIP transport,
per-frame speed, fallback). Roughly a third to a half of the offline work remains; the live path has not been started.

## Capture tooling (Cyberpunk, diagnostic only)

A small MinHook-based DXGI/D3D12 hook layer ([rdna2/rebuild](rdna2/rebuild)) replaces the earlier research runtime that stalled during hook installation
([hook audit](rdna2/HOOK_SUSPENSION_AUDIT.md), [bundle header crash](rdna2/BUNDLE_HEADER_CRASH_FINDING.md)). It observes Present, copies a frame with
D3D12 fencing (F8), logs capture timing, probes the upscaler dispatch read-only, and can record a one-shot copy of the upscaler's colour/depth/motion-vector
inputs (F9). Each build was tested in isolation on the GPU before use ([tests](rdna2/rebuild)); the in-game result is in
[rdna2/diagnostics/2026-09-21](rdna2/diagnostics/2026-09-21). Measured cost of one 2560x1440 capture: about 18-22 ms inside one Present, one dropped-frame hitch,
frame time about 12.5 ms (about 80 fps) otherwise. This is not a performance figure for the network.

## Developer entry points

* [Real inputs](rdna2/emu/INPUT_CONTRACT_CYBERPUNK_FSR3.md), [launch contract](rdna2/emu/VARPARAMS_HOST_CONTRACT.md), [weights](rdna2/emu/WEIGHTS_HT_FINDINGS.md), [host-code evidence](rdna2/emu/DLL_HOST_EVIDENCE.md)
* [Verification results](rdna2/emu/RESULTS_REAL_CONFIG.md), [earlier synthetic results](rdna2/emu/RESULTS_VAR.md), [continuation notes](rdna2/emu/ASTRA_CONTINUATION.md)
* Emulator and tools: `rdna2/emu/` (`gfx11emu.py`, `difftest_var.py`, `difftest_import.py`, `difftest_preblock.py`, `trace_reads.py`, `sandbox.py`); host tests: `python -m unittest discover -s rdna2/emu -p "test_*.py"`
* [AI handover](handover/AI_HANDOVER_GPT6_ASTRA.md) and [continuation brief](handover/CLAUDE_CONTINUE_2026-09-21.md)
* Capture: [launch capture](rdna2/LAUNCH_CAPTURE.md), [preflight history](rdna2/CYBERPUNK_CAPTURE_PREFLIGHT.md)

## Method and limits

* GPU tests run one bounded dispatch at a time in a throwaway sandbox after the emulator proves termination on the same bytes, with guard regions and a
  hard timeout. No driver, watchdog or clock settings are modified. `sandbox.py` isolates files and process lifetime, not the GPU itself.
* Kernel modules must be rebuilt with the current translator before testing; a stale build produced misleading results once.
* Results are labelled by evidence level (measured / read from code / inferred). Treat "inferred" items as hypotheses.

## What is not in this repository

Vendor DLLs, installers, extracted payloads, GPU-code disassemblies, the extracted weight tensors, captured game frames/planes, compiled binaries, local build
and sandbox directories, and the MinHook dependency (clone [TsudaKageyu/minhook](https://github.com/TsudaKageyu/minhook) v1.3.4, commit
`c3fcafdc10146beb5919319d0683e44e3c30d537`, into `rdna2/rebuild/third_party/minhook`). The `gfx1032-dlss-nr-research-main` folder was supplied by another
contributor and is kept as received; its historical claims are not validated by this project. Local launch scripts need privately supplied files and
are not a ready-to-install mod. The test machine is an RX 6900 XT (gfx1030); a `gfx1032` label does not establish compatible translated kernels. This project
does not circumvent anti-cheat and does not distribute proprietary kernels.
