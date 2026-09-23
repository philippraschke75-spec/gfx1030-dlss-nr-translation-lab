# Block 0 pre-block: root cause of the 8x8 mismatch

## Verdict

The 7268-of-10690-byte mismatch of `difftest_preblock.py 1 8 8` is **not an emulator or translation defect**.
It is a launch-fixture bug: the GPU and the emulator are run with **different RNG seeds**.

* Kernarg `+0x68` is the pre-block's i32 seed (`ctx+0x38`). The kernel does
  `0xb04c8 s_load_b256 s[16:23], s[0:1], 0x50` (s22 = `+0x68`) then `0xb0550 s_mul_i32 s17, s22, 0x9e3779b9`
  (golden-ratio seed mix), feeding a PCG-style hash -> uniform float -> `log` -> Box-Muller noise.
* `run_var.PTR_FIELDS` lists `0x68`, so `make_kernarg` writes an 8-byte arena pointer there. The fixture zeroes
  only the low dword, leaving arena bits in `+0x6c`. `var_gpu_test.cpp` rebases any in-arena qword, so the
  GPU reports **7 pointers rebased** (6 are real) and the kernel is seeded with the low dword of the device
  arena address (`0x04010000` for `arena_dev=0x404010000`). The emulator reference is seeded with `0`.
* Different seed -> different noise everywhere -> ~67% of e4m3 bytes differ, with sign/exponent flips.

## Evidence (8x8, seed 1, real block0 weights, sha256 5fe2ab86289f7b21)

| run | result | GPU out hash |
|---|---|---|
| repro as given (x4) | FAIL 7268, 7 rebased | e4bea7c6d234 (every time) |
| `PRE_ZERO=0x68` (x2) | **PASS 0**, 6 rebased | b7cb8af16fbc |
| `PRE_ZERO=0x68 PRE_FIELDS=0x68=i67174400` (both sides seeded 0x04010000) | PASS 0 | **e4bea7c6d234** = the failing GPU output |
| `PRE_ZERO=0x68 PRE_FIELDS=0x68=i12345` | PASS 0 | 15c615cd07a6 |

Row 3 is the positive control: given the seed the GPU accidentally saw, the emulator reproduces the failing GPU
output bit for bit. `BISECT_LEGACY_KERNARG=1 statebisect.py 32_1 0x14 1` aborts in the allocation preflight with
`differing fields (offset, mine, gpu): 0x68: 0x400000000 vs 0x404010000`.

## The premise "store #0 at 0xb1598 is 25/32 wrong, divergence is immediate" does not hold

* `0xb1598` (and `0xb38c0`) **own 0 final bytes**; every byte they wrote is overwritten later
  (`preblock_lastwriter.py`). "Earliest store whose final bytes differ" is not a causal criterion.
* At the first arrival of `0xb1598`, strict compare: `v1` equal and the stored byte `v2 & 0xff` equal on
  **32/32 lanes**. The store wrote correct data.
* Attributed by last writer, the wrong bytes belong to `0xc0fe0`, `0xb852c`, `0xb8464..0xb849c`, ... (the whole
  noise-dependent output), earliest `0xb8464` (store #128).

## Strict-mode instruction-level result (no heuristic filters)

With identical seeds (fixed fixture), the first register divergence is
`0xb070c v_log_f32_e32 v12, v12`: identical input (lane3 `3ee0693c`, lane5 `3ef5647e`, lane8 `3f1f511e` on both
sides), output `emu=bf9851eb gpu=bf9851ec`, `bf87d090/bf87d091`, `bf2f2aab/bf2f2aac`; 10 of 32 lanes in wave 0,
1 ULP. This is the emulator's exact log2 versus hardware's approximation. It is real but **benign here**: the
arena is bit-identical at the last instruction (memory predicate) and `PRE_ZERO=0x68` gives 0 mismatches. So
transcendental precision is refuted as the cause, consistent with the earlier control, but the emulator is not
bit-exact for `v_log_f32`; that only matters if a quantizer sits on a boundary.

Register bisection cannot find the seed bug: registers "heal", and it converged on PC-relative pointers
(`s_getpc_b64` then `s_add_u32`, e.g. `0xbf838`/`0xbf844`, `s2/s3` at `0xb8464`) that differ by construction.

## Changes to `statebisect.py`

* Kept the pre-block kernarg fix; dump base now derives from the output pointer (`+0x08`), because the fix nulls
  `+0x00` (the old base) and HIP overwrites `+0xa8` and up. Before this the GPU dump was silently empty.
* Real per-occurrence probing on both sides (SGPR counter `s97` on the GPU, compile hook in the emulator),
  bisecting wave 0's dynamic trace; the old `occ=` was inert. `BISECT_UPTO=<pc>` limits the range.
* `BISECT_PREDICATE=mem` bisects on arena bytes instead of registers.
* Refuses `flags 0x14` without `DLSSNR_WEIGHT_BLOB` (random weights = a different launch); prints blob hash.
* Removed the strict-mode blanket skip of every `s_getpc_b64` destination (it hid `s2..s5`, `s18/19`, `s38/39`
  at every checkpoint). Now only a pair equal to an exact getpc return address with high dword 0 is explained,
  and counted in the output.
* Asserts the kernel does not use the reserved SGPRs `s96+`.
* Tests: `test_preblock_kernarg.py` (new, 12 tests) plus the existing 9 all pass.

## Same fixture bug elsewhere (not edited)

Low-dword-only zeroing of `0x50..0x68`: `difftest_preblock.py:23`, `preblock_firststore.py:69`,
`preblock_diffshape.py:27`, `chain_import_preblock_real.py:67`, `chain_encoder_stage1_real.py:94`,
`chain_encoder_stage1_2_real.py:96`, `gpu_verify_block23_real.py:45`, `gpu_verify_block5_real.py:39`,
`gpu_verify_stage2/3/4_real.py`, `probe_real_transition.py:38`, `probe_stage_transition.py:40`,
`count_transcendentals.py:42`, `net_full.py:209`. Fix: `for off in (0x50,0x58,0x60,0x68): pack '<Q' 0`.
Any GPU-vs-emulator conclusion drawn from these on flags-0x14 launches is suspect until re-run.

## Not tested / open

* Only 8x8, seed 1 (input seed). Not tested at 64 workgroups. Hypothesis for the non-determinism you are
  chasing there: the seed is the device arena address, so it varies with allocation. Cheap check: run 64 WG twice
  with `PRE_ZERO=0x68`.
* The real host's `ctx+0x38` value is not known here; `12345` passes, so a nonzero seed is fine.

## Retraction of the earlier text of this file

It claimed `v_lshl_or_b32` at `0xb0568` does not write `v2`. That was a misreading: the snapshot is taken
*before* the instruction at that PC, and by then `v5 = 0x01010000`, so `(v5 << 16) | 0 = 0` is correct.

## Reproduce

    cd rdna2/build/import-real-20260922
    export DLSSNR_WEIGHT_BLOB="$(pwd)/rdna2/build/weights/block0.bin"
    PRE_ZERO=0x68 py rdna2/emu/difftest_preblock.py 1 8 8          # PASS 0
    py rdna2/emu/preblock_lastwriter.py 1 8 8                      # 0 wrong
    BISECT_UPTO=b1598 py rdna2/emu/statebisect.py 32_1 0x14 1      # v_log_f32 transition
