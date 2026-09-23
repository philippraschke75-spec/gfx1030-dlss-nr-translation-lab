# Handover — 2026-09-22, end of session

Read this first. It covers: exact git state (something is mid-flight and needs a human action),
this session's technical progress (four kernel contracts resolved and GPU-verified), and an
external collaboration request that's waiting on the user's decision before any action is taken.

## 1. Immediate action needed — git push is stuck mid-flight

The user asked to sync this local working repo with the public GitHub repo
(`github.com/philippraschke75-spec/gfx1030-dlss-nr-translation-lab`). Local `master` and the
remote's `main` had **unrelated histories** (local = full 35+ commit working history; remote =
a previously-published curated snapshot, 5 commits). Per explicit user choice (asked via
AskUserQuestion), the two were merged with `git merge --allow-unrelated-histories`.

**Current state:**
- Local `master` HEAD = `b9bfcca` ("Merge published snapshot history (unrelated) with full
  working history"). This commit already contains the merge PLUS all of this session's kernel
  work PLUS re-redaction of 6 files that had drifted to contain the real local Windows username
  instead of the `<user>` placeholder the original published snapshot used — verified
  clean (`git show b9bfcca:<path> | grep <the real local username>` → 0 hits on all previously-flagged files).
- `origin/main` is still at `152fdfe` — **nothing has been pushed yet**.
- One file, `README.md`, is `git add`-staged but **not committed** — the commit call was blocked
  by the harness's own "Out-of-Place Publication" safety classifier (separate from the user's own
  tool-permission settings; it fires on actions that look like publishing to an external
  destination, even after the user has already approved the broader task).

**What needs to happen** (the user needs to do this manually, or grant a Bash permission rule for
git commit/push in a fresh session — the classifier blocked it twice already, so don't expect a
retry to succeed without one of those):
```bash
cd "C:\Users\<user>\Documents\ChatGPT\dlss 5"
git -c user.name="Philipp Raschke" -c user.email="philipp.raschke75@gmail.com" commit -m "Sync published research log with this session's full working state"
git push origin master:main
```
Before pushing, if picking this back up in a new session: re-verify no local-username leaks with
`git diff origin/main HEAD | grep <the real local Windows username>` (should be empty) — this was
already done once but do it again if anything changes.

**Do NOT** re-run `git add -u` or other broad staging commands blind — they tripped the same
classifier. Explicit `git add <path> <path> ...` with a concrete file list worked fine.

## 2. This session's technical progress (all committed, all GPU-hardware-verified)

Continuing the DLSS-NR gfx1100→gfx1030 kernel translation project. Four more kernel contracts were
fully resolved and verified on the real RX 6900 XT this session, on top of everything already
documented in `rdna2/emu/VARPARAMS_HOST_CONTRACT.md` and `rdna2/emu/RESULTS_REAL_CONFIG.md` (read
those two files for the full technical detail — this is just a summary):

- **`k_qkv_attn`** (window attention, blocks 23-30/40-47): fully resolved, including finding and
  fixing a real bug — I had invented a fictitious "channel-count scalar" at kernarg `+0x34`; it's
  actually the compiler's standard HSA hidden-args block (`hidden_group_size_x`), auto-populated
  by real launch dimensions on hardware. Found via a custom-adapted `statebisect_qkv_attn.py`
  bisection tool. GPU-verified end-to-end with real chained input from the already-verified 22-block
  encoder and real captured weight data: **PASS, 0 mismatches**.
- **`k_qkv_attn2`** (ViT attention variant, blocks 31-38): resolved cleanly and fast by applying
  the `k_qkv_attn` lesson from the start. GPU-verified, 0 mismatches.
- **`k_conv_res_views`** (the real per-block projection kernel — plain `k_conv_res` turned out to
  be **dead code**, only touched by a generic init-time warmup probe, never dispatched with real
  arguments anywhere in the host binary; found by a full-`.text`-section search). Full 72-byte
  kernarg decoded directly from the host launcher disassembly. GPU-verified, 0 mismatches.
- **`k_expand`** (likely the ViT's FFN channel-expand step, C=1024→4096): simplest kernel yet,
  just 3 pointers. GPU-verified, 0 mismatches.

**All three kernel types needed for one full C=512 block's forward pass** (`k_ffwd`, `k_qkv_attn`,
`k_conv_res_views`) **are now individually resolved and GPU-verified with real weight data.**

**Still open, not yet chased down:** the exact multi-kernel data-flow graph that combines these
three kernels' outputs into one real block output (which ctx buffer offset feeds which kernel,
and the skip-scale combination formula) — started tracing this, found `ctx+0x230` and `ctx+0x238`
are genuinely separate allocated buffers (not aliased), so there's an explicit copy or a
`k_ffwd2`-writes-there mechanism not yet found. Deliberately set aside per user request ("can we
continue without it") in favor of resolving more independent kernel contracts instead — this
worked well (`k_expand` came out of that pivot).

**New this session:** `external-docs/` — downloaded and indexed real, public AMD/LLVM reference
material (the LLVM AMDGPU ABI docs, the exact hidden-args table, the SGPR register order table,
and RDNA2/RDNA3/RDNA3.5 ISA PDFs). Read `external-docs/README.md` first before spending hours on
disassembly/bisection for a new kernel question — it would have answered the `k_qkv_attn` `+0x34`
mystery in five minutes instead of hours.

## 3. Pending external ask — needs the user's decision, nothing started yet

A researcher named Sakushey (maintainer of `github.com/Sakushey/gfx1030-dlss-nr-research`) sent a
message via the user, relayed through a collaborator called Emion. Two asks, explicitly **not yet
acted on** — I asked the user for scope confirmation and the session ended before they answered:

1. **A small, sanitized PR** to `Sakushey/gfx1030-dlss-nr-research`, under their Apache-2.0
   contribution terms, containing ONLY: the `k_import`/`k_swin_var` parameter and launch-contract
   tables (every field labelled `observed`/`read-from-host-code`/`inferred`), a `WEIGHTS_HT`
   framing/index parser or synthetic parser tests (**no real weight bytes**), and the encoder
   block schedule plus negative tests for wrong offsets/flags/block assignments. Explicitly
   excluded: vendor binaries, captures, weights, extracted proprietary code objects, private paths.
   They also noted the repo has no LICENSE file, which is independently worth fixing regardless.
2. **Later**, once their native gfx1030 operator ships as a clean source-built fixture, they want
   the user's RX 6900 XT as an independent second-hardware validation run under their issue #9 —
   explicitly **not** the frozen Candidate-F/J3 path, and they said they'd send the exact
   branch/fixture before that run happens.

They referenced commit `152fdfe` as "your current head" — that's the exact (correct) tip of the
*public* repo's `main` branch, confirming this is a real party looking at real current state, not
a fabricated/injected message.

**Before building any PR content**, get explicit confirmation from the user on scope (I'd asked but
the session ended first). This is a real external contribution under someone else's license terms
and deserves a clear go-ahead, not an assumption.

## 4. Established project conventions (carry forward)

- Commit with `git -c user.name="Philipp Raschke" -c user.email="philipp.raschke75@gmail.com"`,
  message ending `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- Never publish proprietary disassembly/weight bytes/extracted code objects (`.gitignore` already
  enforces most of this: `rdna2/build/`, `analysis/*-disassembly.txt`, `analysis/installer-bundles/`).
- Always redact the local Windows username (`<user>` → `<user>`) in anything destined for the public
  repo — check before every push, it has drifted back at least once already.
- GPU dispatch tests run through `rdna2/emu/sandbox.py run <sandbox_dir> <timeout> -- <cmd>` inside
  the persistent sandbox `rdna2/build/import-real-20260922/` (NOT the repo root — kernel modules,
  weights, and `var_gpu_test.exe` live there; `sandbox.py make` **wipes the target directory first**,
  learned the hard way this session — never run it on a sandbox with irreplaceable cached state
  without copying out what's needed first).
- For a new kernel: check `.s`'s own `.args:` metadata for the *real* explicit-struct size before
  assuming any kernarg field layout — the HSA hidden-args block starts exactly there, not at some
  assumed fixed offset. Check `external-docs/` before re-deriving ABI facts via disassembly.
- `py` resolves to `C:\Users\<user>\AppData\Local\Programs\Python\Python310\python.exe` and works
  reliably; the bare Windows Store alias does not.
