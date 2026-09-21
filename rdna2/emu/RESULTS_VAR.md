# k_swin_var differential results (dispatched-path kernels)

Reference: `gfx11emu.py` (gfx1100 interpreter). Device: RX 6900 XT (gfx1030), translation built with
`translate_kernels.py --hw-scratch`. Fixture: `run_var.py` (guessed VarParams, finite small e4m3 inputs,
grid 2x2, 256 threads). Comparison: the **entire 18 MiB arena** (all parameter buffers plus 64 KiB guards),
byte for byte. Every case ran in the sandbox (`sandbox.py`) with a hard timeout; the GPU is reached only
after the emulator has terminated without fault on the same bytes.

| Variant | flags 0 | 1 | 2 | 4 | 8 | 16 | 32 | 31 | 63 |
|---|---|---|---|---|---|---|---|---|---|
| `<32,true>`  | PASS | PASS | PASS | PASS | PASS | FAIL 35,977 B | FAIL 184 B | FAIL | FAIL |
| `<32,false>` | PASS | PASS | PASS | PASS | invalid (LDS need exceeds static size) | – | – | – | – |
| `<64,false>` | PASS | PASS | PASS | PASS | PASS | – | – | – | – |
| `<128,false>`| PASS | PASS | PASS | PASS | PASS | – | – | – | – |
| `<256,false>`| PASS | PASS | PASS | PASS | PASS (was FAIL 19,626 B until the emulator's `s_or_saveexec_b32 s6,s6` bug was fixed) | – | – | – | – |

PASS = 0 mismatching bytes, non-degenerate output (187-256 distinct byte values, 21-186 KB written).
"–" = not run. Flags are the low bits of the u32 at kernarg +0x28 (bit tests at 0xB0248..0xB0344).

## Facts established while getting here
- Explicit params end at kernarg +0xA8. `+0xA8/+0xAC/+0xB0` are HIP hidden block counts and `+0xB4` the
  hidden group size; the kernel computes `wgy * gridDim.x + wgx`. A fixture that hard-codes them is wrong
  (it made every multi-row grid fail until fixed).
- The first GPU dispatch of these kernels completes (0.3-0.5 ms); no hang was observed for any variant/flag.
- 42 further gfx11 instruction forms were implemented in the emulator; each divergence found by
  `statebisect.py` so far was an emulator/fixture defect, not a translation defect.

## Open
- flags 16 / 32 on `<32,true>`: not yet root-caused (flag 16 begins with a 1-ULP `v_log_f32` difference but ~92 % of written bytes differ by >=8 e4m3 code steps, so it is not just ULP noise). `statebisect.py` shows the
  flag-32 divergence only in the final per-element f16 dot-product loop of one wave (sparse lanes).
- gfx1030 preserves the upper 16 bits of a 16-bit VALU result; the emulator zero-extends. Unresolved whether
  gfx11 does either; the bisect classifies upper-half-only differences as "soft".
- Reference is an interpreter written from the ISA, not silicon. Inputs are noise, not trained weights.
