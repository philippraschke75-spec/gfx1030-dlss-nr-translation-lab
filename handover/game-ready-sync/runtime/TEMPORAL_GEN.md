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
