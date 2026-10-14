# New DLSS 5 model: steps for day 1 (2026-10-16)

NVIDIA's stated figures: about 24 % fewer parameters (about 118M) and about half the GPU cost. Our count for 310.8
is 145.76M (143.83M e4m3 matrix weights plus 1.93M table values). NVIDIA probably counts differently, so the
ratio is the better comparison: expect roughly 0.76x.

Tools on branch `new-model-prep`:
* `tools/model_inspect.py` (C1) needs only `py -I`, numpy and `rdna2/emu/weights_ht.py`.
* `E:\dlss 5\dlssnr-pytorch-flex` (C3) is the PyTorch reference. It derives the architecture from the weights it loads.

Never copy the DLL or extracted weights into a repo. Keep the new DLL somewhere like `E:\dlls\<version>\`.

## 0. Get the DLL
Look in the NBA 2K27 update, the driver package or the SDK. Record where it came from and its file version.

## 1. Inspect (2 minutes)
```
py -I tools/model_inspect.py E:\dlls\new\nvngx_dlssnr.dll --json E:\dlls\new\index.json
```
Check these:
* The `version` line (310.8.x → ?) and the sha256.
* `UNKNOWN` records: if there are any, it is scenario C (see step 2).
* The `structure` line: stages, widths and block counts.
* The `totals` line: compare its parameter count with 145.76M.

If `weights_ht.py` fails on an assert, the container framing changed. Fix the parser before anything else. The
DLSS-NR extractor documents the envelope as `u32 type, u32 layout, u64 ndim, u32 dims[ndim]` in the trailer.

## 2. Diff and classify
```
py -I tools/model_inspect.py E:\dlls\new\nvngx_dlssnr.dll --diff "C:\Program Files (x86)\Steam\steamapps\common\Fallout 4\nvngx_dlssnr.dll"
```
The last line reads `SCENARIO A|B|C`:
* **A**: same blocks, layers and families, only the widths changed. Estimate 1-2 days.
* **B**: the same building blocks, rearranged (other block counts, levels or order). Estimate days to a week.
* **C**: a record no known family explains, or a new record kind. Estimate 2-3 weeks.
* **same**: identical weights, nothing to do.

The selftest (`--selftest [310.8 dll]`) checks one synthetic example of each scenario.

## 3. Run the PyTorch reference (CPU; no GPU needed)
```
cd "E:\dlss 5\dlssnr-pytorch-flex"
$env:PYTHONPATH = (Get-Location).Path; $env:DLSSNR_TRITON = "0"
..\dlssnr-pytorch\.venv\Scripts\python.exe -m dlssnr run frame.png -o out.png --dll E:\dlls\new\nvngx_dlssnr.dll --device cpu
..\dlssnr-pytorch\.venv\Scripts\python.exe -m dlssnr plan 1920x1080      # 310.8 plan; the new plan is built from the weights
```
* If extraction fails with "matches no layer layout", the record is a new family or a new layout: scenario C for that layer.
* If it fails with "split feed-forward record layout is unknown for width N", the split-Swin width changed. The
  packed grouped feed-forward record (`layers/split_swin.py::_grouped_ffwd_matrices`) is only traced at 512.
  Trace it at the new width the same way (emulator load trace), then extend the scatter maps.
* The ViT QKV tile order (`extract.unpack_qkv3`) is extrapolated from 1024. If the output is visibly wrong at a new
  ViT width, check that first.
* Check the result for plausibility against the input. A bit-exact check against NVIDIA needs an NVIDIA output
  frame for the same input.

## 4. What to do next, by scenario
* **A**:
  1. **C2**: change the widths, heads and block counts in the generator (`rdna2/emu/net_frame_full.py`) and the
     runtime to the new values. Take them from `model_inspect --json` (`arch.stages`, `vit_hidden`).
  2. **C4**: unpack the kernels from the new DLL, translate them to gfx1030 (`rdna2/translate_kernels.py`) and
     check them on the emulator (`asm_opt/gfx10emu.py`). Check the kernel template widths (`k_swin_var<32/64/...>`)
     against the new widths.
  3. **C5**: make the runtime pick `nr_data\<model>\` by the weights checksum, so that 310.8 and the new DLL both work.
* **B**: do C2 first, deriving the block order and counts. Then C4 and C5. The PyTorch reference (`plan.Arch`) shows
  the new layer table. For other level counts the decoder shift cycle (`plan._rows`, `start = 2 if count % 4 == 2`)
  is a guess; confirm it against a kernel trace.
* **C**: identify the new records from their sizes and the kernels that read them (`trace_reads.py`, as in
  `rdna2/emu/WEIGHTS_HT_FINDINGS.md`). Add the family to `model_inspect.py` (`SHAPES`, the size functions) and to
  `dlssnr-pytorch-flex/dlssnr/extract.py`. Then go back to step 2.

In every case, do **C6** first: until the new model works, the README says "Only works with nvngx_dlssnr.dll 310.8;
a newer DLL turns NR off cleanly."
