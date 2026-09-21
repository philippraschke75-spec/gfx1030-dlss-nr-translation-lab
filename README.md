# gfx1030 DLSS-NR translation lab

Experimental AMD RDNA2 translation and runtime research. **Not ready for
in-game rendering. No successful DLSS-NR game frame has been demonstrated here.**

## Current status — September 21, 2026

* Translation/emulator work includes FP16 packing corrections and additional
  SWIN flag-16, output-contract, and preflight checks. See
  [translation results](rdna2/emu/RESULTS_VAR.md).
* Capture-only HIP forwarding and bootstrap diagnostics are implemented.
  Research kernel launches are deliberately blocked. Host mock tests do not
  establish real workload or rendering correctness.
* A malformed descriptor header in the imported research runtime was reproduced
  as a HIP initialization `bad allocation`. A corrected copy passes isolated
  registration/device enumeration. See the [bundle audit](rdna2/BUNDLE_HEADER_CRASH_FINDING.md).
* Cyberpunk startup remains blocked. Live stacks show allocation during a
  thread-suspension path; the implementation hazard and required source changes
  are detailed in the [hook audit](rdna2/HOOK_SUSPENSION_AUDIT.md). Exact lock
  ownership and an in-game fix are not yet established.

## Developer entry points

* [Continuation notes](rdna2/emu/ASTRA_CONTINUATION.md)
* [Capture implementation](rdna2/LAUNCH_CAPTURE.md)
* [Cyberpunk test history](rdna2/CYBERPUNK_CAPTURE_PREFLIGHT.md)
* [Selected diagnostic evidence](rdna2/diagnostics/2026-09-21/README.md)

The `gfx1032-dlss-nr-research-main` source/notes were supplied by another
contributor. Their historical claims are not automatically validated by this
project. The test machine is an RX 6900 XT (`gfx1030`); a `gfx1032` label does
not establish compatible translated kernels.

Vendor DLLs, extracted payloads/disassemblies, compiled executables and local
build directories are not distributed in this update. Local launch scripts
require generated configuration and privately supplied runtime files; this
repository is not a ready-to-install game mod.
