# k_export writes only 70 of 960 rows: cause and fix

## Cause

`+0x14` of the ExportParams kernarg is the **output row pitch in bytes**, not a dimension. `net_frame_full.py`
passed `(H, W)` there, so the pitch was 960 bytes instead of `W * 8 = 13656`. The kernel does
`dst = +0x20 + s6 * row` with `s6 = +0x14` (`0xab2b8 s_mul_hi_i32 / 0xab2bc s_mul_i32 s6, s6, s15`), so row `y`
starts 960 bytes after row `y-1` and every row overwrites most of the previous one. The whole grid runs, but it
only spans ~950 KB, which the reader (true pitch 13656) sees as ~70 rows. It is not a hard stop.

## Numbers (1707x960, `--stop=full`, GPU only)

| run | rows touched | fully written |
|---|---|---|
| before (`+0x14=960`) | 70 | 69 |
| `+0x14 = 13656` only | 960 | 960 |
| `+0x08 = 1707` only | 70 | 69 (no effect on the count) |
| both (now the default) | 960 | 960 |

The 70 checks out to the byte. Baseline last written byte = **947951**. With the mode that was actually
selected (see below) the kernel writes 16 B/pixel, so `959*960 + 1707*16 - 1 = 947951`. Divided by the true pitch
that is row 69.42, i.e. rows 0..69.

## Kernarg fields as k_export reads them (from the disassembly, `0xab200..`)

* `+0x00` ptr: input, **16 B per pixel**, read with `global_load_b96` (RGB f32), index `(s8*row + x)*16`
* `+0x08` i32: input row stride in **elements** (I passed 0, so every row read row 0). Now `W`.
* `+0x0c` H, `+0x10` W: guards `row < H`, `x < W`.
* `+0x14` i32: output row **pitch in bytes**. This was the bug.
* `+0x18` i32: **format/mode** (`s7`, compared against 3, 5, 6, 4). I had passed W (1707) here. Now 0.
* `+0x20` ptr: output; `+0x30` ptr: second (RGB f32, 12 B/pixel) output, used when `+0x38 != 0`.
* `+0x28`, `+0x38`, `+0x3c`, `+0x4c` are also read.

So your hypotheses map as: (1) `+0x14/+0x18` are pitch and mode, **not** output dimensions, and the buffer is
not too small. (2) and (3) not needed for the row count.

## Output bytes per pixel by `+0x18` mode (sentinel-filled destination, row 0 window)

* modes 1, 2, 3, 4, 6: 6828 bytes per row = **4 B/pixel** (packed)
* modes 0, 5, 7, 8: 13656 = **8 B/pixel** (RGBA16F, what the buffer is sized for)
* 1707 (the old value): fell through to a **16 B/pixel** path

## Not solved

The surface is still not finite (mode 0: min -65.1, max inf; modes 5/7/8: +-512, 70.8% nonzero). That is now
about the *input* at `+0x00`: k_export expects 16 B/pixel float RGB(A), while `off_head` is sized
`act_bytes(32,H,W)` and the format `k_final_head` writes there has **not been checked**. I did not test this.
It is your item 2 and is the next thing to look at. Which 8 B mode is right is also untested.

## Changes (net_frame_full.py only)

* defaults `+0x08 = W`, `+0x14 = 8*W`, `+0x18 = 0`; env overrides `EXPORT_F08/F14/F18`
* `EXPORT_SENTINEL=1` fills the destination with 0xA5 so "written" is measured, not inferred from nonzero
* reports rows touched, rows fully written, and last written byte

Reproduce: `py rdna2/emu/net_frame_full.py rdna2/build/real-input/color.bin 1707 960 --stop=full`
(add `EXPORT_F14=960` to see the 70 again).
