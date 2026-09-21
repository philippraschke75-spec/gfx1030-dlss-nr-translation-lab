# Rebuilt graphics observation layer

## Fenced readback and queue conflict gate

Current installed SHA256:
4b20feb5f1eea4cfc341aa429aa72e7e9e4c1fcbf2d096486ac7730fe6cead4f.
The earlier metadata build is retained as metadata-known-working.asi.

Queue binding now lives in swapchain private data with COM lifetime management.
It accepts only a DIRECT queue from the same device, rejects conflicting
canonical queue identities, and invalidates capture after ResizeBuffers1.
Unknown/conflicting associations never submit a copy. Creation traces include
API name, nested-call depth and the candidate queue's ExecuteCommandLists module.
The cause of the two queue identities seen in Cyberpunk remains unproven; the
new trace is needed to resolve that integration question. No queue is guessed.

With the new launcher, F8 requests one synchronous bounded readback at a real
Present call. The validated queue receives PRESENT -> COPY_SOURCE, a texture
copy to readback memory, and COPY_SOURCE -> PRESENT. A fence must complete
before mapping or freeing submitted objects. Timeout/device-loss retains GPU
objects rather than freeing potentially live resources and blocks further
captures for that process. A timeout can therefore prevent resize; restart the
diagnostic session if it occurs. The CPU wait limit is 2 seconds.

Supported formats: RGBA8, BGRA8, R10G10B10A2, single-sample 2D backbuffers.
The preview is a tightly packed PPM beside the session log. R10 channels are
scaled to 8 bits; no HDR tone mapping or color-space correction is claimed.
Capture assumes the application's backbuffer is in PRESENT state at the hook
and DXGI operations for that swapchain are correctly serialized by the caller.
Third-party wrappers and unobserved creation paths remain integration risks.

Validation: WARP and physical AMD vendor 0x1002/device 0x73bf tests create a
96x80 red render target, queue its clear, capture it through the hook's exported
path, and verify every RGB byte. Row pitch removal, subsequent ResizeBuffers,
all three creation paths, and unknown/conflicting queue rejection pass.
The physical test's completed readback was uploaded and downloaded through HIP
on RX6900XT gfx1030: all 23,040 bytes match after stream synchronization.
This is CPU-staged transfer across separate test processes, not shared-resource
D3D12/HIP interop, neural dispatch, or GPU output returned to Cyberpunk.
hip_frame_stage_test.cpp is a standalone known-red-fixture test; it is not loaded
by the game. The next game attempt establishes whether queue binding permits
capture or supplies the missing conflict trace. No in-game frame copy has yet
been demonstrated with this build.

## Device/queue/backbuffer metadata extension

The installed build now observes CreateSwapChain, CreateSwapChainForHwnd and
CreateSwapChainForComposition as well as Present/Present1. Successful D3D12
creation records canonical swapchain, creation queue and device identities;
the queue's device must match the swapchain's device. Present queries the
current backbuffer index and resource description, releasing COM references
immediately. Logged fields include width, height, format, buffer count, queue
type and resource-query success. No GPU copy or resource barrier is submitted.

Three fresh-process WARP tests pass with allocator workers active. They verify
all three creation paths, exact queue/device identity, pre-existing swapchains
with unknown queue, and ResizeBuffers followed by refreshed dimensions. The
test demonstrates no backbuffer references are retained across resize.
Installed SHA256: d6ef312e736cdbf8a561b8c8696cf79abfe0cb1db77381b45e309ca1450be78e.
The earlier successful presentation-only binary is backed up under
../build/rebuild/presentation-only-known-working.asi. Full launcher preflight
passes; the metadata extension's in-game test is pending.

Limits: this is a bounded 32-entry diagnostic table of scalar identities, not
a resource ownership registry. Object destruction/address reuse and
ResizeBuffers1 queue reassignment are not tracked. Queue identity describes
observed creation, not proof of the queue currently executing each frame.
CoreWindow creation is not intercepted. Do not use this table to submit GPU
work; lifetime/queue tracking must be strengthened first. Logging queries run
on the presenting thread and performance has not been qualified. Frame pixels,
depth, motion vectors, synchronization and HIP interop remain uncaptured.

## First in-game observation confirmed

Session 2026-09-21 19:40:51: the user reports being in-game. The session log
records `Observed Present; forwarding unchanged`, after successful activation.
Inspection of Cyberpunk PID 1452 finds the rebuilt probe and the inert legacy
capture ASI, with no amdhip module or runtime from gfx1032-dlss-nr-research-main
in its enumerated modules. Evidence is saved in
../diagnostics/2026-09-21/rebuilt-hook-game-test/.
This establishes in-game presentation interception for this attempt. It does
not establish long-run stability, device/resource capture, HIP interop or any
neural rendering. The old runtime's hook code has been bypassed, not repaired.

This is a source-built diagnostic replacement for graphics observation, not a
reconstruction of the complete proprietary research runtime. It does not load
that runtime, register its GPU bundle, invoke HIP, or perform neural rendering.

## Implementation

`graphics_hook.cpp` discovers Present/Present1 implementation addresses using
a hidden D3D12 WARP swapchain, prepares two MinHook hooks, and enables them as
one queued batch. Each callback records its first call and forwards arguments
and the original HRESULT unchanged. Ordinary loads are inert. ASI startup is
opt-in via DLSSNR_REBUILD_PROBE=1 and restricted to Cyberpunk2077.exe.
The module is pinned for process lifetime; live unloading is unsupported.
Initialization runs on a worker after DllMain returns.

Dependency: MinHook v1.3.4, commit
`c3fcafdc10146beb5919319d0683e44e3c30d537`, cloned in third_party/minhook.
Source: https://github.com/TsudaKageyu/minhook/tree/v1.3.4 . Retain its LICENSE.txt.
Its inspected thread-enumeration allocation completes before its suspension
loop, unlike the imported runtime's per-thread container/node allocation path.
This removes that specific pattern from this replacement. It does not prove
absence of every deadlock, thread-creation race or hook conflict. MinHook can
skip threads it cannot suspend; additional production qualification is needed.

## Build and validation

Run `build.ps1` with the installed AMD ROCm 6.4 clang toolchain and Windows SDK.
Output: ../build/rebuild/DLSSNRGraphicsProbe.dll and graphics_hook_test.exe.
The standalone test creates a D3D12 WARP swapchain, activates hooks while eight
threads repeatedly allocate/free process-heap memory, joins those workers,
and invokes both actual DXGI methods with DXGI_PRESENT_TEST. It checks callback
counts, successful forwarded HRESULTs, inert initialization and rejection of
duplicate initialization. Five fresh-process runs passed within 20-second
timeouts. These are test presents, not rendered neural frames or image-quality
validation. The test does not establish the exact lock-owner cycle in the old DLL.

Installed diagnostic plugin SHA256:
`0ecbf7f256d5e661f6ece344534b973ef42305fb11c98ffac22e84ff52e2577d`.
Installed separately as bin/x64/plugins/DLSSNRGraphicsProbe.asi. The original
capture plugin remains inert when its environment switch is cleared.

## Next manual test

Close Cyberpunk, its error reporter and Vortex completely. Run
Start-Rebuilt-Hook-Test.cmd and press Play in the fresh Vortex window. The new
launcher clears the old capture switches, sets only the rebuilt probe switches,
checks installed file hashes, and records a unique log and last-session.json.
It restores the parent environment after launching Vortex. Exit Vortex after
the test so future game launches do not inherit the probe switch.

At installation time the earlier Cyberpunk process was still running, and
preflight correctly refused to start another test. No new game run occurred.
The current process has not been modified by copying the new plugin to disk.

Success criterion: the game reaches its menu and the new log records an actual
Present/Present1 callback. A hook-active message alone is insufficient. WARP
entry points may differ from game/overlay entry points; no observation requires
further integration work. Device/queue/resource capture and the entire neural
pipeline are subsequent work. Do not call this a working DLSS replacement.
