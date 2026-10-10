# k_swin_var<32> arithmetic recipe (decoded on the CPU emulator, step 3 window (5,5) of g_A)

Tools: `emu_trace32.py STEP WGX WGY out.npz` captures every WMMA visit (A, B, C, D); `emu_log32.py` logs the arithmetic
of wave 0 between two WMMA site/visit pairs; `a2.py`..`a5.py` check the hypotheses below against the capture.
WMMA = translator lowering: 8 sequential v_dot2c per output, k pairs (2k, 2k+1) in A-column order, C carried in f32.

The release c64/c128/c256 natives are NOT bit-exact either (commit 95a7261, native-mid/RESULTS.md: 53-56 dB, 66-84 %
exact). No exact recipe exists to copy; this one comes from the translated kernel.

## Verified (exact against the emulator capture)
- K order = MEMORY channel order (A column j = x memory channel j); B rows/columns are the record matrix under
  rm = pi32 restricted to 16 = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]. Expand: one 32-k chain from 0.
- Contract: four partials (hidden 0-31, 32-63, 64-95, 96-127), each a 32-k chain starting at 0 (site 5 then 6).
- Residual is an **f16 chain**, not f32: y = f16(x*rs); then for v = 1..4: y = f16(f32(P_v) + f32(y)).
  Yq = q8(y) matches the qkv A operand 128/128 (a5.py). Old native: f32 y, which is why it is ~34 dB end to end.
- Hidden (99.2-100 %, residue from the f16 activation below): h = f16(D); activation in packed f16:
  t = clamp(h, -4, 4); u = pk_fma(|t|, s9, 0.4473 (0x3728)); p = pk_fma(t, u, 0.8945 (0x3b28)); h*p (pk_mul f16); q8.
  (s9 is an SGPR constant, still to read; the native uses f32 fmaf here.)
- q8 = RNE e4m3 with 448 saturation (the log2/div path is exact).

## Seen, not yet verified
- qkv epilogue: f16(D), stored as both the e4m3 byte and f16.
- q/k L2 norm: squares in f32 of f16 values, summed in two fixed chains of 8, then 2 cross-lane bpermute adds;
  f16(sum), max(., 0x388205ff), **v_rsq_f32** (approximate: use __builtin_amdgcn_rsqf), f16, then pk_mul_f16
  (scale * value) and pk_mul_f16 by the head scale (f16).
- Still undecoded: logits/bias/exp ramp, denominator order, context, out_proj, final residual, downsample (flags 4),
  upsample (flags 8; astra-work also found an LDS-overrun emulator case there).
