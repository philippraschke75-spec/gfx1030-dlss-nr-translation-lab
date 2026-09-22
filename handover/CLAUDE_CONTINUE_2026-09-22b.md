# Handover — 2026-09-22 (second session of the day)

Supersedes `CLAUDE_CONTINUE_2026-09-22.md` for everything except its §3 (the Sakushey ask) and §4
(conventions), which still stand. Read those two sections of the older file too.

The goal the user cares about is **rendering a frame**. This session moved from "kernels mostly
work" to "we know what the network actually does". Read §2 first if you want the shortest path.

## 1. Git state — clean, nothing in flight

Branch `master`, everything committed, working tree clean apart from long-standing untracked
scratch files. Ten commits this session, `fa8b304`..`5cd82f4`.

**`origin/main` is still at `152fdfe` — nothing has been pushed.** The user explicitly said to
ignore the GitHub repo for now ("we ignore the github repo for now"), so do **not** push unless
asked. The older handover's §1 describes the merge state if it comes up again.

**Do not re-run `git add -u`** or other broad staging: it trips the harness's "Out-of-Place
Publication" classifier. Explicit `git add <path> <path>` works fine.

## 2. The big result: the network schedule is recovered

This is the part that matters for a frame. Full detail is in `rdna2/emu/VARPARAMS_HOST_CONTRACT.md`
under "The whole-network driver: schedule recovered" — read that section before doing anything else.

The entire network is one driver function, `0x18002ea60`-`0x180031465` (~2,050 instructions). Its
backward jumps give the stage skeleton: prologue, **encoder loop**, **ViT loop**, **decoder loop**,
epilogue, with transitions between. Handle→kernel names were recovered properly (each registration
site puts the handle in `rdx` and the mangled name in `r8`; read the strings from `.rdata`, image
base `0x180000000`) — all 29 mappings, so each loop's dispatch list is known rather than guessed.

Two per-block launchers do most of the work:
* **Launcher A `0x180032df0`** — the `k_swin_var<C>` dispatcher (already documented). Used by the
  encoder **and** the decoder loop.
* **Launcher B `0x180033600`** — the attention/FFN block dispatcher. Its per-block recipe is
  `[k_ffwd_inpview | k_ffwd] -> k_ffwd2 -> k_conv_res_views -> [k_qkv_attn2 | k_qkv_attn] ->
  [k_conv_res2 | k_conv_res_views]`. **Every kernel in that recipe is GPU-verified.**

This also dissolved the `ctx+0x230` vs `ctx+0x238` puzzle that ate time in the previous session:
they are distinct stage buffers consumed by different kernels *inside launcher B*, not two halves of
one FFN — which is why no copy between them was ever found. Don't go looking for one again.

Loop bounds, partially done (see the "Stage loop bounds" subsection):
* **ViT loop = blocks 31-38**, confirmed from the counter (`mov r15d, 0x1f` … `cmp r15d, 0x27`).
* **Decoder loop = 4 stages** (`cmp r9d, 0x4` at entry and back edge), each iteration reading a
  **3-dword table entry** (`lea rdx,[rax+2*rax]`; `mov r10d,[r8+4*rdx]`, table base `[rbp+0x498]`).

### The single best next step
**Dump that 3-dword decoder stage table.** It should turn blocks 48-69 into a concrete per-stage
schedule exactly like the encoder's documented 4/4/6/8. I was mid-investigation on this when the
session ended; the addresses above are where to resume.

After that, what still stands between here and an offline frame:
1. Decoder per-stage block counts (the table dump above).
2. Buffer allocation / ping-pong layout across stages.
3. The `k_flag_set` / `k_flag_wait` sync protocol (`k_flag_set` needs `s_sendmsg(MSG_RTN_GET_REALTIME)`
   in the emulator — currently unimplemented).
4. Then write the runner: `k_import` -> 71 blocks -> `k_final_head` -> `k_export`, replaying the
   schedule on the GPU with the already-verified kernels, fed by the captured Cyberpunk frame
   (`INPUT_CONTRACT_CYBERPUNK_FSR3.md`).

**Be honest with the user about this**: the kernels are largely done, but the bookkeeping around
them is not, and no frame has been rendered. Fixing the remaining kernel mismatches (§4) does *not*
by itself produce a frame — the user asked exactly this and deserves the straight answer.

## 3. Verification infrastructure — use it

`rdna2/emu/kernelspec.py` + `rdna2/emu/difftest_spec.py` replaced the ~75-line-per-kernel
hand-written difftests. Adding a kernel is now one registry entry giving only its field layout.

```bash
# one kernel
py sandbox.py run <sandbox> 700 -- py rdna2/emu/difftest_spec.py <name> 1
# whole registry (note: --list emits CRLF, so strip it)
for k in $(py difftest_spec.py --list | tr -d '\r'); do ... ; done
```

**Baseline: 15 PASS / 3 FAIL**, recorded as a table in `VARPARAMS_HOST_CONTRACT.md`. Run it before
and after touching `gfx11emu.py`, `translate_kernels.py` or `translate_final_head.py`. It already
caught a regression I introduced.

Two traps it encodes, both of which produced *false passes* before:
* **Hidden-arg offsets come from the kernel's own `.args` metadata**, never a fixed offset.
* **A kernel that writes nothing is a FAIL.** `k_ffwd2` once "passed" bit-for-bit while writing zero
  bytes because a count field was 0.

The sandbox at `rdna2/build/import-real-20260922/` now also mirrors all `.s` and `.co` files, which
the harness needs. `sandbox.py make` **wipes the target directory** — never run it there.

## 4. Kernel status

**Verified this session (0 mismatches, real weight data):** `k_ffwd2`, `k_final_head`, `k_export`,
`k_dec_upsample`, `k_qkv`, `k_qkv2`, `k_contract2`, `k_conv_splitk`, `k_conv_res2`, `k_expand2`,
`k_ffwd_inpview`, `k_repack`, `k_align_probe`. Coverage went 12/34 → 26/34.

**Open, investigated, not guesses:**
* `k_attention` / `k_attention2` — run correctly, write to the right slot, but mismatch (250/2023
  and 248/2022 bytes). WMMA is **ruled out** (`k_qkv`, `k_contract2` pass with the same lowering).
  Deltas are far too large for rounding (mean 62, cluster near 128 = e4m3 sign flips). Two live
  candidates: a translated transcendental (`k_attention` has 328 exp/log/rcp ops vs 37 in
  `k_contract2`), or fixture realism (random bytes drive softmax to extremes and e4m3's 4-bit
  exponent then rounds near-boundary values to opposite sides). Needs realistic activations +
  `statebisect`.
* `k_mean` — emulator 0.6227 vs hardware 0.1514. **Not** concurrency (reproduces at grid 1×1), not
  the new atomic (exactly one write, value already wrong upstream), not the atomic's translation,
  not cross-lane lowering. It is in the ordinary per-lane arithmetic of its 257 instructions.
* Not yet attempted: `k_reproject` (128 B), `k_post_block_1h_32_fp8` (80 B), `k_flag_set`/`k_flag_wait`.

## 5. Things that cost time — don't repeat them

* **A stranded `git stash`.** The previous session's merge stashed WIP and never popped it, silently
  removing a real `v_pack_b32_f16` fix from the working tree. Four emulator tests were failing for
  that reason, not from anything new. **If tests fail inexplicably, run `git stash list` first.**
  That bug packed `0x0000` instead of `0x3c00` for float inline constants and affects
  `k_swin_var<32,*>`, `k_pre_block`, `k_export`. Restoring it reproduced the documented
  verification to the byte (37,339 bytes changed).
* **A matching `s_load` sequence does NOT imply a matching field shape.** Bitten three times:
  `ConvParams1d`, `FfwdPlParams` and `AttnParams1d` all look like the 5-pointer 40-byte structs but
  are 4 pointers + H/W. I generalized from `k_qkv` and broke a passing `k_contract2`. Confirm each
  struct against hardware individually.
* **Random bytes are not valid float input.** Read as f32 they span ~60 orders of magnitude and
  include NaNs. Reduction/softmax kernels need `fill={slot:'f32'}`; accumulators need
  `fill={slot:'zero'}` or they write nothing at all.
* `/tmp` differs between the Bash tool and Python on Windows — use `cygpath -w` to convert.
  The host disassembly lives at `C:\Users\<user>\AppData\Local\Temp\host_full.txt` (regenerate with
  `llvm-objdump -d` on the PE embedded in `G:\dlss\dlssnr_on_amd_setup.exe` at offset `0x47c00`).

## 6. Standing caveat to keep repeating

All of this is **translation equality against `gfx11emu.py`**, my own interpreter — not a real
gfx1100 GPU — on synthetic fixtures with real weights. It shows the gfx1030 kernels compute what the
reference model computes. It is **not** evidence the network's numerics are correct, and **no frame
has been rendered**. Don't let the passing-test count imply otherwise when reporting to the user.
