# First-launch capture preparation

The second diagnostic attempt is staged after the user confirmed a clean game
reaches the main menu. See CYBERPUNK_CAPTURE_PREFLIGHT.md for the standalone
loader provenance, current hashes and launch instructions. The bridge now logs
entry and return for forwarded HIP error-returning APIs. Hash preflight and
mock forwarding/blocking tests pass; the second in-game attempt is pending.

## Build blockers resolved for capture-only mode

`prepare_launch_capture.py --capture-only --build` now builds
`rdna2/build/launch-capture-bridge/amdhip64_7.dll` against the fixed absolute
HIP 6.4 backend path. Four typed wrappers were added using the installed HIP
header declarations: hipEventCreateWithFlags, hipEventQuery,
hipStreamCreateWithFlags, hipStreamSynchronize. The build checks all 33 exports
against its export list and the selected backend DLL.

Capture-only mode removes substitution code and its generated-payload-header
dependency from the staged source. It forwards original registration unchanged
and unconditionally blocks hipLaunchKernel, even with the capture environment
variable absent or the old allow-launch variable set to 1. This is a distinct
diagnostic variant; it does not fake payload identification or provide a
rendering-capable replacement module. The original substitution-capable build
still needs its genuine generated payload header.

An integrated mock-backend DLL test passes: exact flag/pointer forwarding and
sentinel return values for all four added functions; forced launch rejection;
registered SWIN argument capture. The mock launch function was never called.
The final diagnostic DLL imports only KERNEL32.dll and dynamically resolves the
fixed HIP backend when a forwarded API is called. It has not been installed in
the game or used to collect an authentic capture.

The older dependency status below describes the original substitution build.

No authentic replayable workload was found in the inspected local files.
The imported `gfx1032-dlss-nr-research-main/runtime_capture.txt` explicitly
identifies itself as a placeholder. The existing bridge only logs launch
dimensions and handles; it does not capture `args` or allocation contents.

## Implemented

`launch_capture.h` is an opt-in Windows host-side hook. Function registration
maps the five exact SWINvar symbols to host entry points. When the process has
`DLSSNR_CAPTURE_FILE` set, every intercepted launch is blocked regardless of
the ordinary launch-enable setting. The first recognized SWIN launch saves:

* Symbol, process ID, grid/block dimensions, shared-memory size and stream handle.
* The 168 explicit VarParams bytes from `args[0]`, read with ReadProcessMemory.
* Explicit flags saying device buffers were not captured, launch was not
  forwarded and payload identity was not independently verified by this hook.

The destination is created with CREATE_NEW and is never overwritten. Only one
recognized capture is attempted per process. Invalid pointers produce a failed
read record without launching. Unregistering a module removes its mappings.
If an earlier unknown launch is blocked and the application aborts, the SWIN
capture may never be reached; consult the bridge's BLOCKED_KERNEL_LAUNCH log.
Do not enable unvalidated earlier kernels just to force the capture to happen.

`prepare_launch_capture.py` checks the upstream bridge source SHA256 and stages
an instrumented copy in `rdna2/build/launch-capture-bridge`. It changes neither
the upstream checkout nor the game directory. Capture runs before backend
launch resolution/forwarding. Normal behavior is retained when the environment
variable is absent. `inspect_launch_capture.py` decodes the saved words without
dereferencing pointers and explicitly reports `replay_ready=false`.

## Validation performed

Compiled `launch_capture_test.cpp` with the local ROCm compiler and executed
the host-only test. Checks passed for disabled mode, unknown function, exact
byte capture, dimensions, one-shot behavior, module removal and invalid address.
Its generated `capture-test-*.json` files are **mock records**, not game captures.

## Deployment blockers and next actions

The bridge's required generated `bridge_gfx1030_fatbin.h` is absent. This hook
has been compiled/tested independently; an integrated bridge DLL has not been
built or installed. No replacement payload or dummy identity header was created.
Generate the real header using the project's verified payload pipeline before
building this staged bridge, or implement and separately validate a dedicated
capture-only bridge without substitution. Keep those identities distinct.

The user must identify the local game installation before deployment can be
targeted. Then verify the actual imported runtime and bridge build configuration,
preserve existing files, and use a unique absolute capture destination. Capture
mode deliberately returns a launch error, so this is not a game-rendering test.
The hook does not write the environment variable globally or enable kernels.

A record alone is insufficient for replay: verify the source module hash;
trace allocation/import/mapping lifetimes and pointer views; capture device
buffers at a synchronized point; record actual stream/event dependencies and
the complete sequence. Current work has not implemented those buffer captures
or collected an authentic input/weights packet. Do not treat pointer values as
portable allocation IDs or guess unresolved +0x30..+0x68 field types.
