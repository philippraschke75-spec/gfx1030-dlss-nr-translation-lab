# DLSS-NR "Preset" (`DLSSNR.Hint.Render.Preset`) — what it does, and the port (branch `nr-preset`, 2026-10-17)

## Answer
The preset is the **weight selector**. It picks which embedded network NVIDIA's DLL runs; it changes no conditioning,
no constants and no compose step.

| host preset | 310.8 (`c32fix`, WEIGHTS_HT only) | 310.9.16 (`newmodel`, HT + CC_SMALL) |
|---|---|---|
| 0 "Default" | HT (fallback) | **CC_SMALL_FP8** (fallback = shipping default id 2) |
| 1 "Preset #1" | HT | **WEIGHTS_HT** (the 310.8 model, "large") |
| 2 "Preset #2" | HT (fallback) | CC_SMALL_FP8 |
| 3 "Preset #3" / anything else | HT (fallback) | CC_SMALL_FP8 (fallback) |

So in RenoDX's UI "Preset #3" is the same as Default on both DLLs. Only **Preset #1 on 310.9** changes the picture:
it runs the old/large model (145.8 M params) instead of the new small one (113.3 M). On 310.8 no preset changes anything,
which is what janblade's INSTALL-DLSSNR.md already says ("runtime 310.8 contains a single built-in preset").

## Evidence (static, read-only; no vendor code or data copied)
* Both DLLs read `DLSSNR.Hint.Render.Preset` at create through the NGX parameter getter (unsigned; missing key -> 0; bit 31
  cleared) and pass it to `CG2RFindWeightByPreset` (`cg2r_weights.cpp`, `CG2RNetworkManager::ResolvePresetToDescriptor`).
* There is a table of weight descriptors. Each entry holds a config name, an integer **preset id**, the RCDATA resource
  name and a short name. 310.8: one entry, id 1 = `WEIGHTS_HT` / `CC_SILVER_AARDWOLD` (stride 0x288). 310.9.16: two entries
  (stride 0x290): id 1 = `WEIGHTS_HT` / `CC_SILVER_AARDWOLD` (config `CC_Control_History_..._honest_tench_..._weights`),
  id 2 = `WEIGHTS_CC_SMALL_FP8` / `CC_SMALL_FP8` (config `hnet-crazy-cuckoo-small-fp8`). Apart from those fields the two
  310.9 entries hold the same small header values (the per-model data differs only in the weight resource).
* Lookup: an exact id match wins; otherwise the code searches for the shipping default id and logs
  `DLSSNR: preset %d is not available in this DLL build; falling back to shipping default preset %d ('%s')`.
  The hard-coded default id is **1 in 310.8** and **2 in 310.9**.
* An independent measurement on NVIDIA hardware (Likely7/Veyra-NRVideo `docs/NR_SF_310916_REPORT_2026-10-17.md`, 1080p
  RTX 5070, 310.9.16) agrees: preset 0 = 3.155 ms, 1 = 5.533 ms, 2 = 3.160 ms. In their words, 1 = old model A, 2 = the
  faster model B, and 0 auto-selects (= B).

## Hosts
* janblade OptiScaler (`janblade-src`): ini `[DlssNr] Preset` (+ `Pass2Preset`, `Pass3Preset`, absent = inherit pass 1),
  clamped to 0..3 (`PassProfiles.h PassPreset`). Menu: "Advanced preset hints (effect unverified)", entries Default /
  Preset 1..3. Changing it rebuilds the feature. Passed as the `preset` argument of `dlssnr_call_create` and set as
  `DLSSNR.Hint.Render.Preset` (UInt) on the NVIDIA path (`DlssNr_Proxy.cpp`, `forwarder/dlssnr_forwarder.cpp`).
* RenoDX DLSS 5 addon: clshortfuse/renodx does **not** publish the dlss5 addon (only `dlssfix`). The public MIT
  derivative Yukikaze20170315/RenoDX-DLSS5-Generic (`dlssnr.hpp`) has setting `NRPreset` (default 0, effect "recreate"),
  UI "Default / Preset #1 / #2 / #3", and sets the raw value as `DLSSNR.Hint.Render.Preset`.
* OpenDLSS-NR (maanHimself, MIT): no render preset. Its "preset" means Style (natural / cinematic).
* dlssnr-pytorch(-flex): no preset. The model is picked by the weights you load (`Arch.from_weights`).

## Port change (`handover/game-ready-sync/dll/dlssnr_nr.cpp`)
* `dlssnr_call_create` now uses its `preset` argument. With `nr_port.ini [nr] model=auto` (the default), preset 1 selects
  the large model (WEIGHTS_HT id + `nr_data\<HT id>\<step>`), and every other preset selects small when the DLL has it,
  as NVIDIA does. `model=small|large` still forces the model and ignores the preset.
* The choice is per feature, so janblade's per-pass presets work (pass 1 small, pass 2 large, ...). The model id is cached
  per choice (`g_dllIds[small|large]`).
* If preset 1 asks for large but the install has no data for the HT model, create logs that and uses the default model.
  That matches NVIDIA's fallback, and it means a user preset never turns NR off.
* Preset 0 (janblade's default) takes exactly the old code path, so output is unchanged.
* `test_sync`: `NR_TEST_PRESET=<n>` passes the preset to create.
* The runtime/plan side needs no change: each data dir already names its weight resource (weights.map `res` line).

## Verification
Setup: RX 6900 XT, ROCm 6.4 build (`E:\dlss-runtime\nrpreset\b.bat`, a copy of `v042\src\b.bat` pointed at this
worktree). The DLL is 310.9.16 with the v0.4.3 package data (`v042\nr_data`, both model ids), 960x540, `vit_every=1`.
Baseline = the `origin/v0.4.3-rc` dlssnr_nr.cpp (`nrpreset\bbase.bat`). Scripts: `nrpreset\run.sh`, `all.sh`, `hash.sh`.

Strict test_sync (60 frames, DLL vs reference `nr_run` of the picked dir):

| run | data dir picked | result |
|---|---|---|
| base, preset 0 | ca6f64f1dd88b965 (small) | PASS 60/60 |
| new, preset 0 / 2 / 3 | ca6f64f1dd88b965 (small) | PASS 60/60 each |
| new, preset 1 | **1ff9997855c80dd5 (HT)** | PASS 60/60 |
| new, preset 1 + `model=small` | small (the ini wins) | PASS |
| new, preset 0 + `model=large` | HT (the ini wins) | PASS |

Flicker mode (40 frames, OUTHASH of the shown output):

| run | OUTHASH | HF out/in | flicker excess | worst frame |
|---|---|---|---|---|
| base preset 0 | 4c874aac48692383 | x1.213 | x1.34 | ~20 ms |
| new preset 0, 2, 3 | **4c874aac48692383** (bit-exact with base) | x1.213 | x1.34 | ~20 ms |
| new preset 1 (HT) | b9783e39f1bf6383 | x1.217 | x1.49 | ~36 ms |

So Preset #1 gives a slightly stronger edit (mean luma edit +0.0047 vs -0.0004) and more flicker, at about 1.8x the
frame cost. That is the same ratio NVIDIA shows (5.5 vs 3.2 ms). The default output is bit-exact with v0.4.3.

## For the transfer/colour agent (RenoDX "HDR Transfer Strength", paper white, colour strength)
* The official RenoDX dlss5 addon source is not public (clshortfuse/renodx has only `src/addons/dlssfix`).
* Public MIT code with the same controls: Yukikaze20170315/RenoDX-DLSS5-Generic. `dlssnr.hpp` holds `transfer_strength`
  (default 1.0), `transfer_mode`, `paper_white_scale` (there is a comment block at its definition), `color_strength`, and
  per-pass `pass_transfer_strength` / `pass_color_strength`. Its shaders are in `shaders/legacy_encode|decode|apply*.hlsl`
  and `codec_math.hlsli`.
* OpenDLSS-NR (MIT): `demo/nr_pass.h` has `paperWhite`, `colorStrength`; see `demo/shaders/nr_preprocess.comp` /
  `nr_composite.comp`.
* A fork credits its colour composition to RenoDX's DLSS 5 addon (MIT), per its release notes (Dagherbou/OptiScaler_DLSSNR).
