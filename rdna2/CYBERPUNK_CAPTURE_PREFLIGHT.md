# Cyberpunk capture preflight, 2026-09-21

## Breakpoint diagnostics installed

The exception observer now includes EXCEPTION_BREAKPOINT (0x80000003), UTC
timestamps, and a separate 16-event breakpoint budget so handled C++ exceptions
cannot exhaust it. It still returns CONTINUE_SEARCH and does not suppress faults.
A host test exhausts ordinary logging, raises a synthetic breakpoint, verifies
its stack is recorded, and confirms a later test handler receives it. Test passes.
Installed bootstrap SHA256:
861d27b450ae180812642165eb12a9d0721a8a64e42cbd9a0f637f7794d37a4a.
Both manifest bootstrap hashes updated; corrected runtime and blocked-launch
bridge remain unchanged. No existing report was found in the inspected local
REDEngine/CrashDumps locations. A new game attempt is needed to collect the
previously filtered breakpoint; this is instrumentation, not a crash fix.

## Repaired-runtime attempt terminated before enumeration

The user confirmed the 18:37 attempt eventually crashed. Both game and reporter
have exited. The final runtime log adds FAULT exception 0x80000003 at
Cyberpunk2077.exe+0x2a52f36 on thread 19760, last job -1. Archived as
build/crash-20260921-1837/dlssnr_on_amd-final.log. This is a breakpoint exception;
its reason is not established. No hipGetDeviceCount entry occurred in this run.
The earlier observed GfnRuntimeSdk C++ exception is not proven causal.
The current exception observer excludes breakpoint exceptions, so it did not
capture this final fault's stack. Further diagnosis requires the breakpoint
stack or game's assertion/crash report. The isolated bundle repair remains
validated, but this run does not establish successful in-game initialization.

## Bundle-header repair deployed for next capture attempt

The active workspace research version.dll now has corrected descriptor bytes,
SHA256 c206fb29a1265200b12dcb46b6123154fc549506ba4849c76b0742e5d7fe2979.
Original bytes are preserved as version.before-bundle-header-repair.dll in the
same folder (SHA256 fa616204dd68521217875fb41835d3189c89e7c87c3050e069df16bafc458a05).
The active filename and containing directory are unchanged, retaining the same
settings and resource lookup location. No bootstrap rebuild is needed.
Updated the runtime manifest hash; full Vortex CheckOnly preflight passes with
game/reporter/Vortex closed. All research kernel launches remain blocked.
The isolated registration test succeeds after the repair; in-game confirmation
is pending. See BUNDLE_HEADER_CRASH_FINDING.md for reproduction and limits.

## 18:28 attempt: exception stack reaches HIP backend

Evidence preserved in build/crash-20260921-1828. Despite the Vortex Electron
NODE_OPTIONS warning, Cyberpunk PID 5212 loaded the runtime and reached device
enumeration. Thread 19756 entered hipGetDeviceCount at 16:28:18.638 UTC and
recorded a C++ first-chance exception with frames inside amdhip64_6.dll
(offsets 0x8388ff, 0x837e77, 0x8368b5, 0x3539, 0x1605f, 0x8d088,
0x15cd9, 0x2a4701, 0x43323, 0x5998b), followed by the bridge at
0x2724 and research runtime at 0xa814/0x9c04. No return is logged.
This localizes an exception to the HIP initialization call chain; it does not
establish its type/message, whether it was handled, or the ultimate termination
cause. A separate first-chance exception in GfnRuntimeSdk.dll occurred on thread
25960; do not call it fatal merely because the observer logged it.
Next diagnostic work should resolve backend stack offsets / exception details
and compare initialization after original fatbinary registration with the
successful standalone enumeration. The Electron warning alone is not evidence
of the game failure. No translated kernel execution is recorded.

## Third attempt: same API boundary; exception observer installed

Session 20260921-180825 again stops after ENTER hipGetDeviceCount (thread
25548). The latest runtime section has no new exception line and no new
REDEngineErrorReporter load entry. Do not attribute the older C++ exception
line to this run. Evidence is preserved in build/crash-20260921-1808.

Added capture_exception_trace.h: an observer installed by the bootstrap worker
before loading the runtime. It logs selected first-chance exceptions, process
and thread IDs, and stack addresses with module paths/offsets to the session's
capture filename plus .exceptions.log. It always returns CONTINUE_SEARCH;
handled exceptions must not be treated as fatal by themselves. Logging is capped
at 16 events, guarded against reentrancy, and skips stack-overflow stack walking.
Other handlers or termination without an exception may prevent useful output.

The host test verifies a thrown C++ exception is logged with stack frames and
still reaches its normal catch handler. Enabled, disabled, and wrong-process
bootstrap tests pass. Installed plugin SHA256:
`f55e8521185d40b44dd7f4fb7b8546efec47706b41c134cbe681440338ed2b6d`.
Manifest updated; file preflight passes. This adds diagnosis, not a crash fix.
Research launches remain blocked. Close Vortex fully before the next attempt.

## Post-reboot installation completed

After the user's reboot, neither Cyberpunk nor its crash reporter was running.
Installed the tested process-restricted bootstrap and verified SHA256
`9496ff82e3a1cc021640c9329f09ceabb3852a491f892c9e3dd01d377919f596`.
Updated only the two bootstrap entries in launch-manifest.json; the other
expected hashes were preserved. Full Vortex CheckOnly preflight passes.
Post-reboot isolated hipGetDeviceCount again returns rc=0, count=1.
The game exception is still unresolved. A subsequent controlled capture attempt
can verify game-only loading without contaminating the crash reporter; it must
not be described as a rendering test or a confirmed crash fix.

## Second attempt failed; process restriction built, deployment pending

The 17:59:31 Vortex session crashed. Evidence is preserved in
build/crash-20260921-1759. Bridge logging records ENTER hipGetDeviceCount on
thread 39676 at 15:59:47.559 UTC without a return; the runtime records C++
exception 0xe06d7363 on that thread at KERNELBASE.dll. This identifies the
observed call boundary, not the throwing module or root cause. No capture JSON
was produced. An isolated enumeration through the same bridge subsequently
returned success and one device; this does not reproduce the game's context.

The runtime also reports loading into REDEngineErrorReporter.exe. Capture
environment inheritance allowed the ASI to initialize in that child process,
mixing logs from multiple processes. capture_bootstrap.cpp now checks the
executable basename before starting its worker (Cyberpunk2077.exe only).
Enabled, disabled, and wrong-process-with-enabled-environment mock tests pass.
The production plugin was rebuilt, but installation was stopped by the running
game/reporter guard. Close both before replacing the installed ASI and refreshing
its manifest hash. The launcher currently rejects the changed workspace plugin
hash; do not bypass that check. No further game test is ready yet.

## Second diagnostic attempt prepared after clean baseline

The user removed the mods and confirmed an ordinary launch reaches the main
menu. Inspection found the old loaders absent. This supersedes the historical
mod inventory below; it does not establish what caused the first crash.

Installed only standalone Ultimate ASI Loader v9.7.4 as bin/x64/version.dll
and plugins/DLSSNRCapture.asi. The latter is the only active ASI found under
plugins. CET and RED4ext were not reinstalled.

Official source: https://github.com/ThirteenAG/Ultimate-ASI-Loader/releases/tag/v9.7.4
(`Ultimate-ASI-Loader-NoPDB_x64.zip`, dinput8.dll renamed to version.dll).
The archive matched the GitHub release asset SHA256:
`e5860e7d9a1805267535b65749575b5e406cc6ea3325c7392189c578815045d1`.
Installed loader SHA256:
`031a3e5576d91dce1e438d36b9a3d462c7334ab4791990a8ff1e3ddc0e132daf`.

The bridge now writes ENTER before each BRIDGE_FWD_ERR backend call, retaining
return logging, to locate a call that does not return. Rebuilt production and
mock bridges, verified all 33 exports, and passed the integrated host-only
forwarding/launch-blocking test. Production bridge SHA256:
`05e3655412a8f893931059328e79533a8d2e6df699ce531fdea78c431df07570`.

Refreshed launch-manifest.json, including the installed loader. File hash
preflight passes. The Vortex preflight correctly refuses the currently running
Vortex: exit it completely, including the tray, before using
Start-Capture-With-Vortex.cmd, then press Play in the newly opened Vortex.
No second game attempt has been performed yet. Research kernel launches remain
unconditionally blocked; this is an initialization/parameter capture test,
not rendered-frame validation. Startup failure remains possible.

## Vortex mod audit after crash

Read-only inspection of the supplied staging directory found:

* CET 1.37.1 supplies the installed Ultimate ASI Loader version.dll/global.ini
  and cyber_engine_tweaks.asi; RED4ext supplies winmm.dll. They were not leftover
  OptiScaler files and must not be removed as such.
* FrameGen Ghosting Fix 5.2.7 includes a native RED4ext plugin at
  red4ext/plugins/DLSSEnablerBridge2077/dlss-enabler-bridge-2077.dll as well as
  CET Lua, redscript and archive content. The native plugin is present in the
  game installation. It is a relevant graphics integration to isolate later,
  not a proven cause of this crash.
* `predefined configuration.-24628-0-1-1759070897` contains OptiScaler.ini only
  in the inspected staging tree. Redeployment may restore that removed config;
  the staged package does not contain an OptiScaler executable DLL.
* A FrameGenGhostingFix warning about missing/incompatible DLSS Enabler belongs
  to September 14, not the September 21 crash attempt. Do not attribute this
  crash to that stale warning. The September 21 redscript log reports successful
  compilation at 17:43:24 and output saved at 17:43:25. CET logged successful
  hook setup; these facts do not prove overall mod compatibility.

No existing mods were changed during the audit. Pending: an ordinary Vortex
launch with the research capture ASI disabled. This isolates the newly added
runtime before considering any graphics-mod disable/redeploy experiment.

## First real attempt: startup crash, plugin disabled

The 17:42:57 Vortex capture session reached Cyberpunk. Bootstrap logging confirms
the capture bridge and research runtime loaded. Bridge logging records all 34
kernel registrations and g_e4m3_lut registration. The runtime log confirms
graphics hooks, a 2560x1440 swapchain and the RX6900XT device, then ends after
the D3D12/HAGS environment report. No kernel launch or captured arguments are
recorded. This does not identify the faulting instruction or establish that
HIP initialization is the crash cause. No useful Windows crash event/dump was
found in the locations checked.

Evidence preserved under rdna2/build/crash-20260921-1743; use
original-game-bridge.log for the untouched game log (phase10_hip_bridge.log in
that evidence directory also contains the later isolated backend probe).
The isolated capture_backend_probe.exe successfully loaded the same diagnostic
bridge and returned hipGetDeviceCount rc=0/count=1, without loading the research
runtime or launching kernels. Basic backend enumeration alone is not broken;
in-process registration, hooks, initialization and other mod interactions remain
unresolved.

Our added plugin was renamed to plugins/DLSSNRCapture.asi.disabled, with its
hash checked first. Existing loaders were retained. The capture launcher's hash
preflight now refuses startup because the active .asi path is absent. Next verify
an ordinary Vortex game launch with the plugin disabled before another capture
attempt; acquire a crash stack or instrument API entry/return boundaries rather
than assuming the translated kernels caused this startup failure.

## Opt-in bootstrap installed

### Vortex startup

Use `rdna2/Start-Capture-With-Vortex.cmd`. It invokes the hash-checked PowerShell
launcher with `-ViaVortex` and starts the located executable
`C:\Program Files\Vortex\Vortex.exe`. Cyberpunk and Vortex must both be closed
first, including the Vortex tray process. The script refuses a running Vortex
instead of silently sending a new window request to a process without the
capture environment. Use the usual Play button in the newly opened Vortex.
Exit that Vortex instance after the test; its descendants inherit capture mode.
No persistent environment settings or Vortex configuration were changed.

Validated: PowerShell parsing, file/hash/path preflight with the process check
mocked absent, and actual rejection of an existing Vortex instance. Neither
Vortex nor the game was started in these checks. Actual Vortex-to-game variable
inheritance remains to be observed: launch redirection through a pre-existing
Steam or REDlauncher process may lose the environment. Bootstrap logging and
the capture record determine whether the diagnostic route really ran.

Installed only `bin/x64/plugins/DLSSNRCapture.asi` (SHA256
e8d75bbe9c28066171985d2e6753400f3a92dc112a171ab26b4d4a45119d489d).
Existing ASI/RED4ext loaders and Cyber Engine Tweaks were retained. global.ini
has LoadPlugins=1 and LoadFromScriptsOnly=1; Ultimate ASI Loader's documented
implementation searches both scripts and plugins with that setting.
Reference: https://github.com/ThirteenAG/Ultimate-ASI-Loader/blob/master/source/dllmain.cpp

The bootstrap is inert unless DLSSNR_RESEARCH_CAPTURE=1 and a capture output
path is present. In a worker thread it refuses an already loaded HIP7 module,
then loads the fixed absolute capture-only bridge and imported research runtime
paths. It does not replace version.dll or winmm.dll. Bootstrap mock tests pass
for disabled and enabled modes. The actual imported runtime is not yet tested
in Cyberpunk and may reject the adapter or abort at an earlier blocked launch.

`rdna2/Start-Cyberpunk-Capture.ps1` verifies hashes of the plugin, bridge and
runtime, sets environment only for the launched process, and restores the
parent environment. Its `-CheckOnly` preflight passes against the installed
plugin. The game has NOT been launched by this preparation step.

Logs: `rdna2/build/capture-bootstrap/bootstrap.log`, bridge log in the game's
working directory, and an optional unique JSON under `rdna2/build/game-captures`.
The launcher records last-session.json even when no recognized SWIN launch
occurs. A successful LoadLibrary log is not evidence of initialization, capture
or rendering. Normal Steam launches leave the diagnostic plugin inactive.
Removal consists only of removing this added DLSSNRCapture.asi file while the
game is closed. No existing proxy must be deleted to undo this installation.

The earlier status below is retained as the inspection history.

Update: the four missing exports are now implemented in the staged bridge
builder. The independent capture-only variant builds without the generated
payload header, has all 33 exports, and passes integrated host-only forwarding
and launch-blocking tests. See LAUNCH_CAPTURE.md. No DLL has been installed;
the remaining proxy-loading and game invocation path still needs verification.

Inspected installation: `C:\Program Files (x86)\Steam\steamapps\common\Cyberpunk 2077\bin\x64`.

With explicit user authorization, removed the identified OptiScaler proxy
`dxgi.dll` (version resource 0.9.4.0), OptiScaler.ini/log, fakenvapi.dll/ini/log,
dlssg_to_fsr3_amd_is_better.dll and its log, Remove_OptiScaler.bat, and
D3D12_Optiscaler including D3D12Core.dll. No backup was requested. The removal
used checked absolute paths and rejected reparse points. The old log identified
0.6.7-pre11, so it must not be treated as the installed DLL's version.

Remaining proxies were identified using version resources and imports:

* version.dll: Ultimate ASI Loader, SHA256
  67565DD7F899C0E4EE3903B1CC8078FF20CA80B8100497C6AC813392E431D443.
* winmm.dll: RED4ext loader, SHA256
  22690F3AC5EFBFA825B758028F114D0E7D83833160716F199C56056DB01FC8BE.

These are distinct from OptiScaler and were retained. No new runtime or capture
DLL was installed, and the game was not launched.

The imported research version.dll statically imports amdhip64_7.dll and 33 HIP
symbols. The existing research bridge exports 29. These four imports are absent
from its export list and wrappers:

* hipEventCreateWithFlags
* hipEventQuery
* hipStreamCreateWithFlags
* hipStreamSynchronize

Consequently that runtime cannot be paired with the current bridge unchanged.
The missing generated bridge_gfx1030_fatbin.h also remains a build dependency.
Do not replace the ASI-loader version.dll with this experimental runtime before
resolving dependency compatibility, payload identity and proxy loading strategy.

The supplied G:\dlss\message.txt describes someone else's RX6600/GTA Enhanced
run. It is not evidence of Cyberpunk integration or a local RX6900XT capture.
The G:\dlss package contains NVIDIA DLSS-NR/Streamline DLLs, but their presence
does not establish a game invocation path. An authentic Cyberpunk capture has
not been collected.
