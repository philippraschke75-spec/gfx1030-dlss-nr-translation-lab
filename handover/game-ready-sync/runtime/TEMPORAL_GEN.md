# GEN.md - runtime / generator side (agent temporal-runtime), 2026-10-11

## Findings (from source, before GPU runs)
* SPEC unknown #1 RESOLVED: post-block +0x40 IS the history lerp. native-prepost-post/post_native.hip compose():
  `if (aux && gate > 0) res = res + clamp(gate * sigmoid(head.a), 0, 1) * (aux[pix*3+k] - res)` with aux = PostParams +0x40
  (const float*, 12 B/px RGB, same pix index/padding as +0x38), gate = +0x48 = 0.7397 (blend_scale) in all our data dirs.
  The release data (make_release_dir.py) nulls post +0x40, and pre +0x40 is null too -> no history anywhere today.
* So history = TWO pointers: pre-block +0x40 (lanes 7-9) and post-block +0x40 (lerp), both -> the same 12 B/px Phist buffer.
  k_export +0x38 stays 0. No extra dispatches (k_reproject/k_mean are not in our graph; the warp is the DLL's job).
* Phist for the next frame = post-block output rgb (ctx+0x100 "net result", RGBA f32 16 B/px padded, P space [0,1]).
* Arena: nr_ext/nr_run do NOT re-upload the arena per frame (only at create / nr_reset); per frame only src is copied and dst zeroed.
* No data regeneration needed: the runtime allocates its own 12 B/px history buffer and patches both +0x40 pointers per frame.
  The combo data dirs are used as is (release ctx118 is only 4 B/px, unused).

## Data dirs
E:\dlss-runtime\temporal\nr_data\{960x540,1280x720} are byte copies of the combo dirs: same kernels and overrides, no vendor weights,
relative paths, and nvngx_dlssnr.dll in E:\dlss-runtime\temporal supplies the weights. No PRE_40/CTX118_RGB regeneration is needed.
The runtime owns the 12 B/px history buffer and patches pre/post +0x40 itself, so history on/off is a per-frame choice on the same data.

## Runtime API (nr_runtime.h "ABI v2", additive: old DLL + new runtime = history off, bit-exact)
* `nr_hist_layout(c, &pad_w, &pad_h)`: 960x540 gives 1024x640. The values are read from post-block +0x20, so query them; do not hardcode.
* History image: RGB32F, pad_w*pad_h*12 bytes, row stride pad_w*12, image at the top-left (W x H), padding don't-care.
  - Colour space: the network's own. With release data (import mode 0) that is the input's linear RGB.
  - The post-block output is in that space, clamped to [0,1].
* `nr_run2(c, in, out, ms, hist_valid)`: host path. History = the previous frame's output, unwarped.
* `nr_ext_history(c, hist_in_off, hist_out_off)`: call after nr_ext_open. Both are offsets into the shared io buffer.
  - Each must be 4-aligned, >= nr_bytes, and have room for pad_w*pad_h*12 bytes. 0 = unused.
  - io layout:

    | offset | contents | written by |
    |---|---|---|
    | 0 | input RGBA16F (nr_bytes) | DLL |
    | out_off | output | runtime |
    | flag_off | u32 flag | runtime |
    | hist_out_off | P_t | runtime, before flag_out = seq |
    | hist_in_off | warped P_{t-1} | DLL, before flag_in = seq |
* `nr_ext_enqueue2(c, seq, max_inflight, hist_valid)`:
  - hist_valid=0: pre/post +0x40 are null (today's path).
  - hist_valid=1: copies io+hist_in_off into the history buffer and runs with history. With hist_in_off=0 it uses the runtime's own unwarped previous output instead.
  - The DLL sends hist_valid=0 on the first frame, on reset or resize, and when the previous frame was skipped or late. Where the warp is invalid for a pixel, it writes the current colour.
* The old `nr_ext_enqueue` and `nr_run` are enqueue2 and run2 with hist_valid=0.
* Debug: env NR_HIST_MASK = 1 (pre-block lanes only), 2 (post lerp only), 3 (default, both).

## Verification (RX 6900 XT; hist_test.py + test_sync via E:\dlss-runtime\temporal\verify.sh)
* History off:
  - test_sync PASS 300/300 at 960x540 and 1280x720 (old DLL API, combo and temporal dirs).
  - nr_run2(...,0) is bit-exact to the saved combo output (4 frames), also after history frames have run in between, so no state leaks.
* Static image, history on: does NOT converge.
  - mean |o_t - o_{t-1}| stays at ~0.009-0.012 every frame (max ~0.3). Same with the pre-block alone.
  - Post lerp alone (mask 2) is stable (0.0), as expected: aux == res on a static frame.
* ±1 LSB input noise, frame-to-frame change:

  | resolution | no history | history | post lerp only |
  |---|---|---|---|
  | 960x540 | 1.03e-2 | 1.03e-2 | 0.95e-2 |
  | 1280x720 | 1.00e-2 | 0.90e-2 | not run |

  - SPEC criterion F_hist < 0.5 F_nohist: FAIL.
  - The network itself turns 1 LSB of input into 1e-2 of output.
* Moving input vs PyTorch: not run (no HARNESS.md).

## Update 2: mode 1 data, noise amplification, hist_valid=2
* New option hist_valid=2: post-block lerp only, with pre-block +0x40 null. It is stable on static frames.
  hist_valid=3 is pre-block only (debug). Debug calls nr_dbg_steps / nr_dbg_read are for stage tracing.
* make_release_dir.py takes env IO_MODE (default 0). mkdirs.sh builds both data dirs from one `POST_CROP=1 gen_patched.py` run:
  - E:\dlss-runtime\temporal\m0\960x540 (mode 0) is byte-identical to the combo dir. This confirms the route.
  - E:\dlss-runtime\temporal\m1\960x540 is mode 1: k_import/k_export +0x28 = 1.
* Results at 960x540 from amp_test.py via E:\dlss-runtime\temporal\amp.sh, history off unless noted. All values are in P (proxy) space.

  | | mode 0 (shipped) | mode 1 |
  |---|---|---|
  | noise amplification for ±1 LSB on all pixels (mean\|dP_out\| / mean\|dP_in\|) | **205x** (1.46e-2) | **42x** (2.97e-3) |
  | output pixels changed > 1/255 | 92 % | 40 % |
  | static, hist 1: \|o_t - o_{t-1}\| | ~1.4e-2, no convergence | 5.2e-3 -> 8e-4 after 10 frames (converges, like U2 E2b) |
  | noise flicker, hist 0 / 1 / 2 | 1.46e-2 / 1.50e-2 / 1.48e-2 | 2.99e-3 / **0.98e-3** / 2.98e-3 |

* Mode 1 + hist 1 passes the SPEC criterion: 3.0x less flicker. Mode 0 gets no benefit from history.
* Mode 0 alone has 5x the noise amplification of mode 1. This is the likely root cause of the in-game flicker.
* Stage trace (fraction of bytes that differ between a clean and a noisy run, after the first n dispatches):

  | after | bytes changed |
  |---|---|
  | import | 19-34 % of the import RGB |
  | pre-block (act A, fp8) | 1-1.5 % (fp8 rounding absorbs most of it) |
  | first swin C32 dispatch | 5-6 % |
  | encoder stage 1/2 (dispatches 3-11) | 30-41 % |
  | everything later | flat at ~43-46 % |

  The spread happens in the first two encoder swin stages, where fp8 rounding flips propagate. It is the same in both modes.
  The difference between the modes is in magnitude, not reach: mode 0 feeds linear HDR (unbounded) into a network trained on proxy [0,1].
* Not checked: whether mode 1 looks right in game. Commit 26bce36 noted dark blocks with mode != host value; that commit's own finding was host value 1.

## Open
* With release data (import mode 0: linear HDR input, output clamped to [0,1]), pre-block history forms an unstable feedback loop.
  U2's E2b converged (0.0008) with import/export MODE 1 (proxy P). Next: a mode-1 data dir (IMPORT_MODE=1 via gen_patched.py) + nr_run2.
* The vendor truncates history to half; we keep f32.
