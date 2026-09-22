# Handover — 2026-09-22 (third session)

Supersedes `CLAUDE_CONTINUE_2026-09-22b.md` entirely. The goal the user cares about is **rendering a
frame**. This session finished the reverse engineering and got most of the network executing on the
physical GPU. What is left is assembly, not discovery.

Read `rdna2/emu/VARPARAMS_HOST_CONTRACT.md` before doing anything — it is the authoritative record
and it grew by ~870 lines this session.

## 1. Git state — clean, everything pushed

Branch `master`, working tree clean, **0 commits ahead of `origin/main`**. The public repo
(`github.com/philippraschke75-spec/gfx1030-dlss-nr-translation-lab`) is at `9204199`, README and
GitHub About both updated. The push went through without tripping the publication classifier that
blocked an earlier session.

Before any future push: `git diff origin/main master | grep <user>` must be empty, and the diff must
contain no `.co` / `.bin` / `.exe` / disassembly. `.gitignore` covers this but check anyway.

## 2. What is done

**All structural reverse engineering is complete.** The 71-block schedule, the per-block kernel
recipes, the U-net skip wiring, the `ctx` buffer map and the frame-level pipeline are all recovered
and documented. There are no remaining unknown unknowns.

| blocks | stage | status |
|---|---|---|
| 0 | pre-block | kernel verified |
| 1-22 | encoder, 4 stages 4/4/6/8, C=32/64/128/256 | **full chain executed, 0 mismatches** |
| 23-30, 40-47 | C=512 attention | **one complete block executed, 0 mismatches** |
| 31-38 | ViT, C=1024 | recipe decoded, **not executed** |
| 39 | the `k_dec_upsample` transition weight - not a block | resolved |
| 48-69 | decoder, 4 stages 8/6/4/4 (encoder mirrored) | **a stage executed, 0 mismatches** |
| 70 | final head | kernel verified |

Registry sweep: **16 PASS / 2 FAIL** (`difftest_spec.py`). The two failures are `k_attention` /
`k_attention2` at 214 mismatches each, explained below.

Also executed: `k_import` on the real captured Cyberpunk frame at full 1707x960, producing a
viewable PNG whose statistics match the capture's own recorded range.

## 3. The single most important correction this session

**Both outstanding "translation failures" were defects in the reference interpreter, not the port.**

* `stride64` LDS 2addr addressing: the `_ds` matcher captured the variant as a *non-capturing* group
  and discarded it, so addresses used `offset * element_size` instead of `offset * element_size * 64`.
* `v_cvt_f16_f32` wrote the whole destination dword, destroying `D[31:16]`, which must be preserved.

Fixing the second **also fixed `k_mean`**, which had been separately investigated and misdiagnosed
as "ordinary per-lane arithmetic". That is two kernels whose apparent translation failure was really
`gfx11emu.py`.

What remains on `k_attention` is **1 ULP out of `v_wmma_f32_16x16x16_f16`** — a rounding-model
difference. Hardware's internal accumulation order is undocumented and is neither f64-then-round nor
naive sequential f32; both were measured and left the count at 214. The other WMMA kernels pass
because they quantise to e4m3/f16 before storing, which absorbs 1 ULP; `k_attention` has values on
quantisation boundaries where it flips the stored byte.

**Carry this forward:** these difftests judge the *pair* (translation, reference model). Check the
ISA before suspecting the translator.

## 4. Tools

* **`net_run.exe`** (`rdna2/net_run.cpp`) — multi-dispatch runner. Takes a pipe-separated manifest
  (`module|symbol|ka_off|ka_len|gx|gy|threads`), one concatenated kernarg blob, one arena. Rebases
  pointers exactly like `var_gpu_test`. Build:
  `hipcc --offload-host-only -std=c++17 -O2 net_run.cpp -o build/net_run.exe`
* **`kernelspec.py` / `difftest_spec.py`** — registry-driven per-kernel difftests. Adding a kernel is
  one entry giving only its field layout; hidden-arg offsets come from the kernel's own `.args`
  metadata.
* **Chain runners**: `net_encoder_full.py` (22 blocks), `net_encoder_stage1.py` (parameterised by
  block list — works for decoder stages too), `net_block512.py` (one C=512 block), `net_smoke.py`.
* **`statebisect_attn.py`** — adapted bisection. Two latent bugs in the shared `statebisect.py` were
  fixed to get it working: the workgroup-id SGPR base is `user_sgpr_count` (2 here, 4 for
  `k_swin_var`) and was hard-coded; and `coverage.json` only holds the last-built kernel, so
  `source_vgprs` now comes from the disassembly.

Everything runs inside the sandbox:
`py sandbox.py run rdna2/build/import-real-20260922 <timeout> -- py rdna2/emu/<script>.py`
**Never run `sandbox.py make` on that directory — it wipes it.**

## 5. Next step: item 6, the assembly

Build the full 71-block dispatch sequence. Everything needed is decoded:

1. **Chain the ViT** (blocks 31-38) first — it is decoded but never executed, so it is the one piece
   with unverified wiring. Recipe: `k_expand2`(layer 0) -> `k_contract2`(layer 1, count=4) ->
   `k_qkv2`(layer 2) -> `k_attention2`(no weight; ViT layer3 is a 2-byte scalar) ->
   `k_contract2`(layer 4, count=4), ping-ponging two buffers with `ctx+0x260/0x268/0x288` as
   intermediates. Expect `k_attention2`'s 1-ULP residue to show; that is not a wiring error.
2. Assemble encoder -> 23-30 -> ViT -> 39 upsample -> 40-47 -> decoder -> final head, using the
   `ctx` buffer map (§ "The ctx buffer map" in the contract notes).
3. Front with `k_import` (format 0, already working on the real frame) and end with `k_export`.

**`k_export` cannot be validated standalone** — its `+0x00` is the network's own output in its tiled
format, so feeding it anything else produces inf/NaN. Its field layout is decoded and corrected
(`+0x0c` is height, `+0x10` is width — the reverse of the original note). It only becomes testable
once the network actually produces output.

## 6. Traps that cost real time

* **A bit-exact PASS does not mean the fixture asked a meaningful question.** The first full-encoder
  run passed with 0 mismatches while blocks 12-22 ran on weights that scratch had overwritten —
  weight slots overlapped the 18 `VarParams` pointer slots. Chain tests must place buffers beyond
  `len(V.PTR_FIELDS)`, and no weight slot should appear in the written-slot list. Same family as
  `k_ffwd2` "passing" while writing zero bytes.
* **A matching `s_load` sequence does not imply a matching field shape.** `ConvParams1d`,
  `FfwdPlParams` and `AttnParams1d` all look like the 5-pointer 40-byte structs but are 4 pointers
  plus H/W. Generalising from `k_qkv` broke a passing `k_contract2`.
* **Random bytes are not valid float input.** Reduction/softmax kernels need `fill={slot:'f32'}`;
  CAS accumulators need `fill={slot:'zero'}` or they write nothing.
* **Check `git stash list` if tests fail inexplicably** — a previous session stashed a real
  translation fix and never popped it.
* `/tmp` differs between the Bash tool and Python on Windows; use `cygpath -w`. Host disassembly is
  at `C:\Users\<user>\AppData\Local\Temp\host_full.txt` (regenerate with `llvm-objdump -d` on the PE
  embedded in `G:\dlss\dlssnr_on_amd_setup.exe` at offset `0x47c00`).

## 7. Conventions

* Commit as `git -c user.name="Philipp Raschke" -c user.email="philipp.raschke75@gmail.com"`, message
  ending `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
* Never publish vendor binaries, weight bytes, extracted code objects or disassembly. Always redact
  the local Windows username before pushing.
* `py` resolves to the Python 3.10 install and works; the bare Windows Store alias does not.

## 8. The standing caveat

Every result here is **translation equality against `gfx11emu.py`**, an interpreter, not a real
gfx1100 GPU, on synthetic fixtures with real weights. It shows the gfx1030 kernels compute what the
reference computes. It is **not** evidence the network's numerics are correct, and **no frame has
been rendered**. When reporting progress, do not let the passing-test count imply otherwise — the
user has asked for a frame repeatedly and deserves that distinction kept sharp.
