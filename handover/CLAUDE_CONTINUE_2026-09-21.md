# Claude Code continuation — verified state, September 21, 2026

## Start here

You are continuing Philipp's AMD RDNA2 DLSS-NR translation research, not starting
over. Work and communicate in English. The user wants practical implementation
and tests, minimal repeated permission questions, and honest distinctions between
synthetic kernel correctness, capture, inference and rendered output. They switched
agents because Codex usage was nearly exhausted. Do not spend another session
rediscovering the bootstrap or asking for the unavailable original DLL source.

Workspace: `C:\Users\<user>\Documents\ChatGPT\dlss 5`.
GPU: **RX 6900 XT, gfx1030**, PCI vendor 1002/device 73bf.
Game: `C:\Program Files (x86)\Steam\steamapps\common\Cyberpunk 2077\bin\x64`.
Launch through Vortex: `C:\Program Files\Vortex\Vortex.exe`.
The user removed their mods and verified ordinary Cyberpunk startup. We then
installed our own diagnostic setup. Do not assume the old mod inventory remains
active. At handover inspection Cyberpunk, its crash reporter and Vortex were all
absent from the process list. Recheck before installing anything.

**Most recent success: a real 2560x1440 Cyberpunk backbuffer was copied to CPU
memory with D3D12 fencing and saved as a PPM. There is NO neural enhancement.**
The separately tested translated SWIN kernel still passes on real gfx1030.
These two paths are NOT connected into a model execution pipeline.

## Immediate objective and what not to assume

User asked how to apply the neural link. We explained that complete inference
requires exact model weights, tensor/parameter layouts, kernel ordering and
dependencies, verified input requirements, output validation and integration.
Proposed next direction: build an **offline inference/replay path** using the
saved capture, before adding inference into the live game.

This proposal is NOT evidence that a color-only PPM suffices for the model.
Depth, motion vectors, exposure, history, etc. are possible requirements that
must be recovered, not invented. No validated complete graph or weights loading
path was established. First inventory existing assets and evidence, determine
the actual missing contract, and implement the smallest independently testable
graph segment. Do not manufacture weights, tensor shapes or kernel arguments.

## Verified milestones

1. Original research DLL initialization threw `bad allocation` inside
   `amdhip64_6.dll` during hipGetDeviceCount. We reproduced this without the game
   by registering its embedded bundle and then enumerating devices.
2. The bundle had an eight-byte gap in the descriptor stream. Compacting only
   the header preserved all payload bytes and offsets. The same isolated test
   then returned success and one device. This fix is real and independent of
   later hook problems.
3. The old DLL's hook setup stalls in heap allocation after suspending peers.
   Live stacks and disassembly establish the allocation-after-suspension hazard.
   Exact heap-lock ownership was NOT established; do not overclaim the lock cycle.
4. We built a new source-controlled diagnostic graphics layer using MinHook,
   avoiding the old DLL entirely. Cyberpunk reached gameplay and actual Present
   calls were observed. The old HIP/research runtime was absent from module lists.
5. Metadata extension captured D3D12 device, creation queue, backbuffer dimensions,
   format and count. An earlier run showed conflicting scalar queue observations.
   We replaced raw-pointer association with swapchain-owned COM private data and
   added ambiguity gating and source/depth logging.
6. Latest run: two distinct swapchains, each accepted with its own DIRECT queue
   on the same device. F8 copied one frame successfully, fence completed and
   PRESENT resource state was restored. This run had no queue conflict. It does
   NOT prove the earlier same-pointer observation's cause.
7. Standalone physical D3D12 readback test on RX6900XT rendered solid red and
   verified every RGB byte, including row-pitch removal. Subsequent resize passed.
   The saved test pixels also survived HIP upload/download with stream sync.
   That is **CPU-staged transfer in separate test processes**, not shared-resource
   D3D12/HIP interop, neural dispatch, or returning an enhanced frame to the game.

## Exact successful game capture

Session: 2026-09-21 20:04:25, Vortex PID 12752 (historical).
Image:
`rdna2/build/rebuild/20260921-200425-381e34246bb64b3e9fc54c3c977ae5d5.log.ppm`
Log: same path without `.ppm`.
Image SHA256: `0cc7c393fd6d1dda686023ed54c548e9d985aa62ffd7fa729d5d663396c7217f`.
P6 PPM, 2560x1440, 11,059,200 RGB payload bytes, 167 distinct byte values.
Header and exact payload size were checked. No visual-quality claim was made.
Source DXGI format 24 = R10G10B10A2_UNORM, three backbuffers.
Capture converts packed 10-bit RGB to 8-bit RGB. It is NOT a lossless copy of
the original HDR/color pipeline. There is no tone mapping/color-space correction.
For inference fidelity, prefer extending capture to preserve raw native-format
bytes, footprint, color-space metadata and corresponding model inputs.

Durable selected evidence:
`rdna2/diagnostics/2026-09-21/first-game-frame/capture.log`
`rdna2/diagnostics/2026-09-21/first-game-frame/result.json`
`rdna2/diagnostics/2026-09-21/frame-readback-tests/`
Earlier milestones: `rebuilt-hook-game-test/`, `rebuilt-metadata-game-test/`.
Do not confuse known-red standalone test PPM with the real game frame.

## Current installed game setup

All hashes SHA256, verified at handover:

* `bin/x64/version.dll`: standalone Ultimate ASI Loader v9.7.4,
  `031a3e5576d91dce1e438d36b9a3d462c7334ab4791990a8ff1e3ddc0e132daf`.
* `bin/x64/plugins/DLSSNRGraphicsProbe.asi`: current rebuilt layer,
  `4b20feb5f1eea4cfc341aa429aa72e7e9e4c1fcbf2d096486ac7730fe6cead4f`.
* `bin/x64/plugins/DLSSNRCapture.asi`: old opt-in diagnostic bootstrap,
  `861d27b450ae180812642165eb12a9d0721a8a64e42cbd9a0f637f7794d37a4a`.
  It loads as an ASI but remains inert with the old capture switch unset.

**Use `rdna2/rebuild/Start-Rebuilt-Hook-Test.cmd`, NOT the old
`rdna2/Start-Capture-With-Vortex.cmd`.** The old launcher loads the problematic
research runtime. The rebuilt launcher clears DLSSNR_RESEARCH_CAPTURE and
DLSSNR_CAPTURE_FILE, then sets DLSSNR_REBUILD_PROBE=1, a unique
DLSSNR_REBUILD_LOG, and DLSSNR_REBUILD_FRAME (log path plus .ppm).
User presses Play in the newly opened Vortex, enters gameplay, then presses
F8 once. Each process allows one capture request. Close Vortex completely
between runs so its child processes inherit the intended environment.

Manifest: `rdna2/build/rebuild/installed.json` pins all three installed files.
Session pointer: `rdna2/build/rebuild/last-session.json`.
Launcher supports `-CheckOnly`; refuses game/reporter/Vortex already running.
Installed plugin is inert during normal launches without the opt-in variable.
The user reported faster loading but we did NOT measure or attribute it.

Known-working rollback binaries:
`rdna2/build/rebuild/presentation-only-known-working.asi`
`rdna2/build/rebuild/metadata-known-working.asi`.
Use verified hashes/manifests when replacing; never alter a loaded plugin.

## Rebuilt source map

`rdna2/rebuild/graphics_hook.cpp`
: Explicit RebuildStart initialization outside DllMain; optional ASI worker
  only for Cyberpunk2077.exe. Hooks Present, Present1, CreateSwapChain,
  CreateSwapChainForHwnd, CreateSwapChainForComposition and ResizeBuffers1.
  Uses a hidden WARP D3D12 swapchain to discover method entry points. Module is
  pinned for lifetime; live unloading unsupported. Callbacks forward original
  arguments and HRESULTs. F8 calls RebuildCaptureFrame once. No old runtime/HIP
  initialization is included in this DLL. Metadata logs only on changes.

`queue_binding.h`
: Private GUID data stores a COM queue reference on the swapchain. Only DIRECT
  queues on the same canonical D3D12 device are accepted. Multiple distinct
  canonical queue identities invalidate binding permanently for that object.
  ResizeBuffers1 invalidates binding; it does not yet support per-buffer queues.
  Unknown/conflicting queues return E_ACCESSDENIED before GPU submission.
  Logs API name, nesting depth and ExecuteCommandLists implementation module.

`frame_readback.h`
: Dedicated allocator/list/readback heap/fence. Submits PRESENT->COPY_SOURCE,
  CopyTextureRegion, COPY_SOURCE->PRESENT on the bound queue. Signals and waits
  for fence before mapping. Removes padded row pitch. 256MiB allocation cap;
  supported single-sample, single-mip 2D RGBA8/BGRA8/R10G10B10A2 only.
  Two-second CPU fence wait. Timeout/failed signal/device loss preserves GPU
  references and stops additional captures, rather than freeing in-flight objects.
  Retained backbuffer on timeout can prevent resize; restart diagnostic process.

`graphics_hook_test.cpp`
: Loads DLL inertly, calls explicit start, tests actual WARP or `--amd` DXGI calls
  while eight allocation workers run. Tests all three creation paths, exact
  queue/device identity, unknown-queue handling, resize refresh, queued red clear,
  exact 96x80 PPM pixels, post-copy resize, conflict invalidation. DXGI_PRESENT_TEST
  calls alone are not rendered frames; the separate queued clear/readback is.

`hip_frame_stage_test.cpp`
: Standalone known-red-fixture upload/download test, not a model. Expects 96x80
  PPM and gfx1030. Uses hipMemcpyAsync and hipStreamSynchronize. 23,040 matching
  bytes. Not loaded into the game and not a zero-copy/shared-handle solution.

`dxgi_probe.h`, `frame_metadata.h`, `build.ps1`
: Build/test support. `DxgiProbe.create(true)` chooses an AMD hardware adapter;
  default is WARP. Metadata's 32-entry scalar table is diagnostic only and must
  not be treated as an ownership registry; capture uses private-data binding.

Dependency: `rdna2/rebuild/third_party/minhook`, git clone v1.3.4 commit
`c3fcafdc10146beb5919319d0683e44e3c30d537`; keep LICENSE.txt. Its inspected
enumeration allocations precede the suspension loop. This avoids the specific
old allocation pattern, but does not prove every hook race impossible. It may
skip unsuspendable threads. Preserve bounded testing and rollback.

## Build and test commands

PowerShell cwd = workspace. Compiler and SDK already work locally.

```powershell
& ./rdna2/rebuild/build.ps1
# Run test with WorkingDirectory rdna2/build/rebuild:
./graphics_hook_test.exe DLSSNRGraphicsProbe.dll --amd
# Omit --amd for WARP.
```

Use Start-Process -WindowStyle Hidden with RedirectStandardOutput/Error and
WaitForExit(20000). If that child test times out, terminate ONLY that child.
Never blindly kill arbitrary game/system processes. The helper must run in
the output directory to locate its DLL and write fixtures.

HIP standalone build:
```powershell
& 'C:\Program Files\AMD\ROCm\6.4\bin\hipcc.exe' --offload-host-only -std=c++17 rdna2/rebuild/hip_frame_stage_test.cpp -o rdna2/build/rebuild/hip_frame_stage_test.exe
# In rdna2/build/rebuild, after graphics_hook_test generated the red fixture:
./hip_frame_stage_test.exe readback-test.ppm
```

Bundled Python:
`C:\Users\<user>\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
Host regression command: `python -m unittest discover -s rdna2/emu -p 'test_*.py'`.
Latest suite: 30 passes. Native outputs/logs live in rdna2/build/rebuild.
Use --offload-host-only for host-only HIP tools; avoid needless device compilation.

## Translation/kernel status

Read `rdna2/emu/ASTRA_CONTINUATION.md`, `RESULTS_VAR.md`, `VAR_OUTPUT_CONTRACT.md`.
Prior headline count (29 versus 34) is not a qualified whole-model graph. There
are 34 assembled kernels in the broader work, and five SWIN-var variants were
investigated. Scope each correctness claim to the tested symbol/fixture.

Confirmed translation bug fixed: v_pack_b32_f16 floating constants were fed into
integer lowering as FP32 bits; shared lower_pack_f16 materializes FP16 bits.
Flag16/32/31/63 synthetic tests for SWINvar32true subsequently passed.
Some unusual shapes with negative offsets produced overlapping writes in the
inferred fixture, not a proven translation defect. Zero-offset retest passed.
Do not erase the historical failure or assume the whole ABI is recovered.

Most recent physical rerun, requested by user while budget was low:
symbol `_Z10k_swin_varILi32ELb1EEv9VarParams`, flag16, seed1, 16x16 fixture,
grid2x2, offsets -4/-4, 256 threads/workgroup.
GPU dispatch 0.623ms; zero mismatches across arena, guards intact;
37,339 changed reference bytes, 525,687 modeled steps.
Module SHA256 `c75137d2fc8a69b19e65b0b6d5e1ea0d1cc419af383ad3b7d4867349ddeaa371`.
Output SHA256 `30eda9885103211c2907aae8fd8ddc2b37852e7c1054f9fa7c93fc28cc0e0bcd`.
Evidence sandbox: `rdna2/build/gpu-recheck-20260921/rdna2/build/emu_var/`.
Earlier module/input sandbox: `rdna2/build/astra-swin-20260921/`.

Run translated GPU tests through `rdna2/emu/sandbox.py run <dir> <timeout> -- ...`.
This limits files/process lifetime, NOT GPU isolation. Keep emulator preflight
at the actual allocation address, exact comparison and guards. Do not alter TDR.
Fresh sandbox needs initialized-rodata.bin as well as the tested .co and
var_gpu_test.exe. sandbox.py make deletes an existing target: only use a checked
new workspace path. The successful recheck used difftest_var.py 32_1 16 1 2 2.

## Old runtime and forensic artifacts

`gfx1032-dlss-nr-research-main/version.dll` is currently header-repaired SHA256
`c206fb29a1265200b12dcb46b6123154fc549506ba4849c76b0742e5d7fe2979`.
Original preserved as `version.before-bundle-header-repair.dll`, SHA256
`fa616204dd68521217875fb41835d3189c89e7c87c3050e069df16bafc458a05`.
User says DLL came from TripleZer000 and was built by AI; exact original source
cannot be obtained. Do not equate this provenance with a matching source build.

Bundle descriptor stream at file offset 0x9b600 declares nine entries. Eight
zero bytes at bundle offset142 precede the real next descriptor at150. Standard
parse reads bogus name_length1172152 and HIP throws bad allocation. Repair
compacts header only; payload offsets/data unchanged. Scripts:
`extract_capture_probe_bundle.py`, `repair_capture_bundle.py`,
`capture_registration_probe.cpp`. Scripts pin original hash and use backup if
present. Original/repaired probes and AMD offload-bundler validation succeeded.

Audits: `rdna2/BUNDLE_HEADER_CRASH_FINDING.md`, `HOOK_SUSPENSION_AUDIT.md`,
`CYBERPUNK_CAPTURE_PREFLIGHT.md`, `LAUNCH_CAPTURE.md`.
Important RVAs: OpenThread IAT8da28, SuspendThread IAT8daf8/thunk65c10,
HeapAlloc IAT8d988. Loop9d24 calls helper36370, helper allocates16 bytes then
suspends, caller container grows at9d48->21311->4e824/HeapAlloc. Live worker
TID9380 blocked there while peers had pre-existing suspend count1.
Old early GfnRuntimeSdk C++ exceptions were first-chance observations, NOT proven
fatal. A later game breakpoint 0x80000003 was also seen before HIP enumeration.

`inspect_process_stacks.cpp` externally uses DbgHelp/StackWalk64 and balances
each added suspension exactly. Run elevated if sandbox SymInitialize fails.
Do not indiscriminately resume peers in an active hook transaction.
Evidence: `rdna2/build/closed-20260921-1845/`, selected copies in diagnostics/.

## Partner updates already downloaded and compared

`partner-updates/Sakushey-gfx1030` at
7215992065fef4eb9979b6cdcfe0ebdb1a518cf8.
`partner-updates/TripleZer000-gfx1032` at
be225412da5b05b33b279d0c0792ae65169992d9.
Read `partner-updates/COMPARISON.md`.
TripleZer source/notes match existing import after line-ending normalization;
its DLL is the same malformed original. Sakushey changes mostly docs/publication,
CI and test differences; bridge source is unchanged. Neither supplies the old
graphics-hook transaction source. Do not overwrite our changes with either clone.
Sakushey reports unresolved physical J3 liveness and no neural game frame;
that is separate from our synthetic SWIN passes and graphics-hook milestone.

## GitHub and outstanding publication request

User repo: https://github.com/philippraschke75-spec/gfx1030-dlss-nr-translation-lab
Latest published main commit verified by push:
`7305f67356d2cc400f1bf30b585534f86b31931d` (83-file source/diagnostics update).
This predates the rebuild/frame-capture work. **Current workspace is newer.**
Local branch remains master at older history. Publication used an alternate
index `rdna2/build/github-publish.index`, read-tree origin/main, commit-tree
parented to remote main, and a fast-forward push. Do not assume local HEAD is
the latest remote or force-push it. Fetch/check before updating.

The user subsequently requested uploading literally everything, including files
omitted previously, then redirected work to partner repos/rebuilding. **That
full-folder upload remains unfinished.** Current .gitignore excludes build/vendor
binaries, nested research repos, and most imported generated files. If continuing
publication, inventory size, repository limits and credentials/private data,
explain precisely what can be published, and avoid silently claiming everything
was uploaded. Do not automatically publish this handover or captures merely
because earlier code updates were authorized without assessing the current scope.

## Recommended continuation order

1. Preserve this successful frame and installed plugin; no immediate game run is
   needed. Read the current rebuilt code and evidence, not all historic dumps.
2. Inventory available model/runtime assets and recovered ABI/graph evidence.
   Clearly list what is known vs absent. The color PPM is an initial fixture,
   not proof of complete model inputs. Preserve native 10-bit data next if needed.
3. Design an offline replay artifact format with dimensions, format/color space,
   tensor layout, weights provenance, strides, synchronization and hashes.
   Connect only an exactly understood graph segment to its verified translated
   kernel and compare against an independent reference. Use synthetic tests for
   development but never describe their output as learned neural enhancement.
4. Complete graph/weights/input requirements and output checks before any game
   output substitution. Shared-resource D3D12/HIP interop is still unimplemented;
   the CPU-staged path is a valid diagnostic starting point, not a performance plan.
5. Later integrate output upload/composition with explicit fences, resource
   states, resize/destruction/device-loss handling and pass-through fallback.
   Do not replace the game backbuffer using an unvalidated kernel result.

Keep final updates short and concrete: what changed, what passed, what still
blocks the neural path. The user's successful gameplay is valuable evidence
of hook integration, not a reason to claim DLSS5 is operational.
