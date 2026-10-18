# Rdna2NrRuntime: an RX 6000 backend for AMDNR

`Rdna2NrRuntime.dll` runs our RDNA2 port of DLSS-NR (HIP, gfx1030-1036) behind the **lmxxf runtime C interface**
(`LmxxfNrApi.h`, ABI v1). AMDNR's `Lmxxf::Backend` can drive it like `LmxxfNrRuntime.dll`, so AMDNR's interleave,
placement, residual and temporal passes, composition and passes UI all keep working.

## Folder

| File | What |
|---|---|
| `Rdna2NrRuntime.dll` | Exports `LmxxfNrGetApi` (ABI 1, all 19 table entries), plus the optional `LmxxfNrGetImportPoolStats` and `LmxxfNrSetTierPolicy`. |
| `nr_runtime.dll` | Our HIP runtime. Rdna2NrRuntime.dll loads it from the same folder. |
| `nr_data\<model id>\<W>x<H>\` | Kernels and plans. `ca6f64f1dd88b965` is the new small model (310.9.16 CC_SMALL), `1ff9997855c80dd5` the old one (310.8). Steps run from 640x360 to 2560x1440. No NVIDIA weights. |
| `Rdna2NrRuntime.ini` | Optional settings. The defaults are fine. |
| `NOTICE`, `LmxxfNrApi.h` | Licences and the interface header (MIT, lmxxf / Kien; 3zwr1's changes). |

What the runtime needs at run time:
- The user's own `nvngx_dlssnr.dll` (310.8 or 310.9.16) for the weights. It is searched in this order: ini `weights_dll`, this folder, above
  `nr_data`, `assets_directory`, the game folder.
- The AMD driver's HIP runtime (`amdhip64_6.dll`, which ships with Adrenalin).

## Interface behaviour

- **Create**
  - Refuses cleanly (`LMXXF_NR_UNAVAILABLE` plus a `GetLastError` text) when:
    - the device's adapter is not an RDNA2 dGPU (PCI 0x73A0-0x73FF or 0x7420-0x743F). That rules out APUs, WARP and other vendors;
    - no HIP device has the D3D12 adapter's LUID (no fallback to another GPU);
    - `nvngx_dlssnr.dll` is missing or holds a model we have no data for;
    - the queue belongs to another device.
  - `assets_directory` is used as a second place to look for `nr_data` and the weights.
- **PrepareFrame**
  - Picks the `nr_data` step nearest in height to `color_height x model_scale` (or ini `model_res`).
  - Builds the network on a thread, which takes 1.3-2.7 s. Until it is ready, every job is a **passthrough job**: `private_output` = the
    colour, and `OutputReady` = 1. PrepareFrame never fails just because the network is still loading.
  - A change of colour size, `passes` or `FULL_NETWORK` rebuilds the network. As with lmxxf, abandon the old job first.
  - `private_output` is RGBA16F at colour size, left in `NON_PIXEL_SHADER_RESOURCE`, and holds the answer in fed units (fed + edit).
- **RecordInputs**
  - Records into the host's list: a copy of the colour, the network input (resampled when the colour size differs from the step), the
    warped history, then `WriteBufferImmediate(flag)`.
  - It needs `ID3D12GraphicsCommandList2`.
  - It sets its own descriptor heap, root signature and PSO on the list, so the host rebinds its own afterwards, as it already does after
    lmxxf's calls.
- **EnqueueHipAsync**
  - The runtime's thread waits, bounded at 500 ms, for the flag from the GPU, then runs the network on its own HIP stream.
  - No D3D12 queue wait is ever issued, and the queue is never drained.
  - `EnqueueHip` and `ExecuteAfterProducer` behave the same.
- **OutputReady** never waits.
- **RecordOutputs**
  - If the job has finished, it only records the copy and compose.
  - Otherwise it records a bounded one-thread GPU spin on the flag first (default 250 ms, TDR-safe). A timeout gives passthrough.
- **AbandonJob** waits on the CPU until the job ran or was dropped. Measured: 21 ms in flight, 496 ms for a list that was never submitted.
  Nothing is recorded.
- **CancelUnsubmitted, Poll, Retire, ResetHistory, Drain, GetStatus, GetLastError** are implemented. `GetStatus` reports
  `rdna2-hip model=<id> net=<WxH> passes= hist= split= jobs= pass= skip=none`.
- **Frame flags**
  - `STRENGTH`: Detail = the luma part of the edit, Colour = the rest.
  - `TEMPORAL`: motion, history and `history_reset` feed our history.
  - `passes` 1..3: one network context per pass, chained on the HIP stream, each fed the previous pass's output.
  - `CONTROLS`: tone, structure, skin, other and style go to our conditioning, applied live with no rebuild.
  - `DEBUG_VIEW` 0..4.
  - `CODEC_PASSTHROUGH` is supported.
  - `FULL_NETWORK` turns off our middle-stage reuse.
  - `VULKAN_BRIDGE` is accepted. Nothing changes, because we never drain or wait on the queue.
  - `output_smooth` is ignored: our history does that job.
- **No frame skipping in the runtime.** AMDNR's interleave decides. Our `nr_split` (one network spread over N jobs) is an ini option and is
  off by default.

## Changes needed in AMDNR (small)

1. **Backend choice.** Add `[DlssNr] NrBackend=rdna2`, or let the lmxxf backend take a configurable DLL name, e.g.
   `LmxxfRuntimeDll=Rdna2NrRuntime.dll`. The rest of `Lmxxf::Backend` stays as it is.
2. **GPU table.** Allow gfx103x (RX 6000 dGPUs) for this backend when `Rdna2NrRuntime.dll` is present. Today the table stops lmxxf on
   RDNA2 before its first frame. APUs can stay refused; `Create` refuses them anyway.
3. **Runtime path.** Point `Lmxxf::RuntimePath()` / `AmdRuntimePath` at `Rdna2NrRuntime.dll` in this folder, and skip the lmxxf pak/assets
   check for it. Either pass this folder as `assets_directory` or leave it empty; `nr_data` sits beside the DLL.

## Measured (RX 6900 XT, Windows 11, AMD HIP 6.4, test_amdnr.exe)

| Test | Result |
|---|---|
| Output against our own pipeline (nr_run2, same input), 960x540 and 1920x1080, 10 jobs each | **bit-exact** (100% of values) |
| 2 and 3 passes against a chain of 2 / 3 reference contexts | **bit-exact** |
| History (TEMPORAL) against nr_run2's history, 12 jobs | worst job 56.5 dB (≥ 50 dB) |
| AbandonJob mid-flight / never-submitted list, then 8 more jobs | safe, the following jobs are bit-exact |
| 50 x Create / load / run / Destroy (640x360, every 3rd ends on an abandoned job) | VRAM +0 MB, private bytes −7 MB, handles −1 |
| Unknown nvngx_dlssnr.dll / ABI 2 / WARP / queue of another device / LUID mismatch | clean refusal with a message |
| Colour 1280x720 into the 960x540 network (resampling) | finite output, mean edit 0.016 |

Timings (new model, history on, the data dir's default middle-stage reuse). The time runs from the start of the host frame (record,
submit, the GPU copy of the colour) to `OutputReady` = 1, which is essentially the network time:

| Network | 1 pass | 2 passes | 3 passes |
|---|---|---|---|
| 960x540 | 6.9 ms (p90 10.4) | 13.5 ms | 19.4 ms |
| 1280x720 | 12.2 ms (p90 16.8) | | |
| 1920x1080 | 23.6 ms (p90 33.7) | | |

Network load at first use: 1.4 s (540p), 1.9 s (720p), 2.7 s (1080p).

## Licence

- `Rdna2NrRuntime.dll`, `nr_runtime.dll` and `nr_data`: GPL-3.0-or-later. Source is on the `amdnr-runtime` branch of
  https://github.com/philippraschke75-spec/gfx1030-dlss-nr-translation-lab, in `handover/game-ready-sync/amdnr/` and `runtime/`.
- `LmxxfNrApi.h`: MIT, Copyright (c) 2026 Kien (lmxxf), with GPL-3 changes by 3zwr1. The header comment is kept and the credit is in
  `NOTICE`.
- No NVIDIA or AMD files are included.
