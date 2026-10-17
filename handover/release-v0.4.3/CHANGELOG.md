# Changelog (DLSS 5 Neural Rendering, AMD RDNA2 port)

## v0.4.3

- **Looks to choose from.** `presets\` now holds full `nr_port.ini` files, each with a short note on look, cost and when to use it: Standard (the default, = `nr_port.ini`), Strong, Cinematic, HighQuality_720p, HiRes_1080p, Native_1440p, Performance, Community. To switch, copy one over `nr_port.ini`. See "Choosing a look" in the README.
- **Intensity above 1 keeps the colours.** NVIDIA's rule: the brightness ratio is raised to the intensity (at most 2x either way) and the hue stays. Before, the edit was scaled per channel, which shifted colours (about 2x the chroma shift at intensity 2). Intensity up to 1 gives the v0.4.2 image.
- **Strong** (`local_tone=2`, `intensity=1.3`): about +45 % effect, shadows +41 %, faces +35-85 %, no extra GPU time. **Cinematic** = Strong + NVIDIA's Cinematic style: the strongest, graded look.
- **New `nr_port.ini` keys** `compose=add|nv|ratio` (default `add` = as before; `nv` = NVIDIA's own blend, for users copying NVIDIA multipass setups; `ratio` = OkLab residual after janblade's Transfer 2) and `detail=` (live-frame detail above the model resolution, 1 = as is). The **Community** preset uses `compose=nv` and `pass2_gain=1`.
- **Much less flicker: `edit_temporal=0.7`** (new default in `nr_port.ini` and every preset). The shown NR edit is blended with the previous one, moved by the game's motion vectors, depth-gated and clamped to the local range of the current edit (two small model-size passes, no measurable cost). Flicker harness (2560x1440 frame, settled frames, 960x540 / 1920x1080 model): Standard 0.83 -> 0.075 % / 0.68 -> 0.090 %, Strong 1.48 -> 0.20 % / 1.05 -> 0.16 %, Cinematic 4.26 -> 1.46 % / 2.71 -> 0.83 % visible flicker. `edit_temporal=0` gives the v0.4.2 image. Also new: `edit_limit` (cap the edit against local contrast, 0 = off) and `edit_warp=bicubic`.
- Same kernels and `nr_data` as v0.4.2.

## v0.4.2

- **Faster, same image.** New c32 swin kernel (one window per 64-thread workgroup, weights from a load-time f16 shadow, packed q8/e4m3; about 2.1x faster) and new ViT GEMMs (8x4 register tile, 128 threads, LDS prefetch). Bit-exact with v0.4.1 (both models, 960x540 / 1280x720 / 832x468 / 640x360). The c32 kernel is also deterministic run to run now. Network at 960x540 / 1280x720 (RX 6900 XT, median, new model): 11.02 -> 9.67 ms / 17.55 -> 15.03 ms; old model 20.26 -> 18.04 ms / 32.65 -> 28.36 ms.
- **`nr_split=N` (2, 3 or 4).** The network runs in N balanced parts over N frames (split points measured at load); the shown edit is the last finished one, warped by the game's motion vectors. New default `nr_split=3` (`nr_split=1` still means 2 parts, `0` = off); the Performance preset uses 4. Synthetic pacing test (2560x1440 frame, 960x540, new model, median fps): CPU 8 / GPU 12 ms per frame: 53 / 58 / 60 fps at N = 2 / 3 / 4; CPU 6 / GPU 6 ms: 79 / 88 / 110 fps.
- **Model resolutions 1920x1080 and 2560x1440** (new model; old model 1920x1080) for image-quality comparison: presets `HiRes_1080p.ini` / `Native_1440p.ini`. Network (RX 6900 XT, vit_every=1): 9.7 / 14.9 / 29.5 / 51.2 ms at 540p / 720p / 1080p / 1440p (old model 1080p 54.7 ms); VRAM about 0.86 GB at 1080p, 1.42 GB at 1440p. `model_res=auto` never picks them (only an explicit `model_res`). 1080p data uses NVIDIA's field 1088 rows (was 1152: +0.4 dB vs the reference).
- **Fix: block artifacts when the game resolution equals the model resolution** (e.g. 1280x720 game at 1280x720, or the 1440p preset): the network input is now clamped to [0,1] there too (as for every other size), and highlights above 1 keep their brightness (frame + NR edit).
- The data dirs are new (c32 workgroup size changed): use the `nr_data` from this package, not an older one.

## v0.4.1

- **Faster, same image.** The pre and post blocks now read the game's colour and write the result themselves (k_import, the mirror padding and k_export are fused in), and the post block passes its arguments by value. Bit-exact with v0.4. Network at 960x540 / 1280x720 (RX 6900 XT, median): new model 11.33 -> 10.97 ms / 18.03 -> 17.47 ms, old model 20.68 -> 20.31 ms / 33.25 -> 32.70 ms.
- **ViT reuse default per model.** `vit_every=0` (new default in `nr_port.ini` and the Quality preset) uses the model's own default: every 3rd job for the new model, every 2nd for the old one (it flickers more at 3). Any other value still overrides it.
- New opt-in `int8_prepost=1`: post-block MLP on int8 dot products (about -0.1 ms, tiny image change). Default 0.

## v0.4

- **NVIDIA's new model.** nvngx_dlssnr.dll 310.9.16 carries a new, ~2x lighter model (`WEIGHTS_CC_SMALL_FP8`). It is picked automatically; `model=large` in `nr_port.ini` forces the old one. 310.8 keeps working with the old model. Network at 960x540 (RX 6900 XT): **11.1 ms** (new model) vs 24.3 ms in v0.3.1; with the default settings about 7 ms.
- **Much closer to NVIDIA's image** (+6-8 dB vs the reference implementation): window-shift axis, decoder phase, ViT extent, NVIDIA's geometry (`Geometry::fromValid`), mirror padding and pooling layout fixed; NaN outputs become +0. The window-shift fix also makes every swin step cover the full plane, which removes the old rare (about 1 in 420 runs) nondeterminism of the c32/swin kernels (stale rows 508-511).
- **NVIDIA's default look**: intensity 1, style 0, local tone 1, local structure 1, skin -1, auto mask on.
- **The OptiScaler NR sliders work** (intensity, style, local tone, local structure, skin, auto mask), live. `nr_port.ini` keys of the same name override them.
- **Faster kernels**: int8 swin MLP/QKV (`int8_swin=1`), faster pre/post blocks and softmax, post weights built at load (so the post block also serves the new model), ViT QKV register tiling.
- **Presets** in `presets\`: Quality (960x540, ViT every 3rd job, new default), Balanced (832x468), Performance (640x360).
- `nr_data` now ships per model: `nr_data\<model id>\<WxH>\` for 960x540, 1280x720, 832x468 and 640x360.

## v0.3.1

- **Uses the correct GPU.** NR runs on the GPU the game renders on (matched by adapter LUID), never on an iGPU.
- **VRAM leak fix** (MSFS 2024): features the game abandons are freed, and recreating NR no longer grows VRAM use.
- **HDR auto-exposure** (RE Requiem): scene-linear HDR input is exposed automatically before the network.
- **HDR with 2 passes**: the second pass follows the first pass's HDR decision, and its brightening is capped.
- **Unknown or new `nvngx_dlssnr.dll`**: NR turns itself off cleanly and the game keeps its normal image; `dlssnr_nr.log` says why.
- **All kernels are our own**, with no NVIDIA data inside; the weights are read from your `nvngx_dlssnr.dll` at load.
- **Faster**: about −0.6 ms per network run.
- **Less VRAM**: our own frame plan needs about 380 MB instead of 620 MB at 960x540 (490 MB instead of 860 MB at 1280x720).
- New `nr_port.ini` keys (defaults): `hdr_key=0.18`, `hdr_adapt=0.05`, `hdr_exposure_bias=1.0`, `hdr_exposure=0`, `allow_any_gpu=0`, `evict_idle_ms=2000`.

## v0.3

- **Up to 60 FPS without frame generation** on an RX 6900 XT (Fallout 4, 1440p, 960x540 model): about 40-45 FPS in heavy scenes, up to 60 in lighter ones.
- **Even frame pacing.** `nr_split=1` splits the network over two frames, so the frame time stays even and no longer alternates long/short.
- **Faster.** The c32 kernels now run natively.
- **Less flicker.** The temporal history follows the game's motion vectors.
- **Fast games no longer stutter.** v0.2.3 dropped to a native frame whenever the GPU was still busy (Assetto Corsa). Now it waits for the GPU instead (`busy_mode=wait`, `busy_wait_ms=100`). In the pacing test this gives 0 native frames.
- **HDR fix.** Characters and highlights no longer turn white in HDR games (RE Engine). `hdr_mode=auto` detects HDR input and converts it for the network.
- **Clean 2 passes.** With OptiScaler's "2 passes" the effect is no longer applied twice. The second pass is damped (`pass2_gain=0.5`).
- **Safe setup and uninstall.** `setup_windows.bat` backs up every file it replaces, and `Uninstall_DLSS5.bat` restores them. The setup text is now AMD-specific.
- **Wrong-folder warning.** Setup warns when the folder has no 64-bit game exe, for example a 32-bit launcher folder.
- **Hardware log and self-check.** `dlssnr_nr.log` records the GPU, driver and VRAM, warns when VRAM is low, and runs a self-check at start. If the self-check fails, NR switches off cleanly instead of crashing.
- **ShortFuse's nvngx_dlssnr.dll works.** The weights are found by name, so any build that carries the same weights works.
- New `nr_port.ini` defaults: `nr_split=1`, `busy_mode=wait`, `hdr_mode=auto`, `pass2_gain=0.5`.
- `nr_data` ships 960x540 (recommended) and 1280x720.
