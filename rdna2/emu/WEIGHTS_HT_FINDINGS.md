# WEIGHTS_HT resource in nvngx_dlssnr.dll (2026-09-21)

Source: `G:\dlss\Dlss5 2224 1 2026-08-28T18-21Z ZWdvjgs49\dlss5\nvngx_dlssnr.dll` (165,840,496 B), PE resource
`RT_RCDATA(10)/WEIGHTS_HT/1033`, 147,695,410 B at file offset 0x114a160. The 17 MB `.data` section has entropy 8.0 and
was not examined. Tool: `rdna2/emu/weights_ht.py` (read-only; no weights are copied into the repo). Index: `logs/weights_ht_index.json`.

## Proven (parser asserts hold for all 153 records; blob is consumed exactly)
* Magic `0x08cda732`, 153 name-sorted records: 71 `blockN.layer0.layer`, 24 each `layer1/2/3.layer`, 9 `layer4.layer`,
  1 `block70.layer0.blend_scale`.
* Record framing: u64 name_len, name, u64 D+40, u64 D+40, u64 D, u32 1, D data bytes, 20-byte trailer
  (8 zero bytes, u32 1, u32 0, u32 count). `count == D/2` in every record. Total data 147,683,778 B.
* Not encrypted/compressed: per-record byte entropy 1.0..7.13 (median 6.04).
* `blend_scale` is 2 bytes, 0x39eb = 0.7397 as f16.

## NOT established
* Element dtype. Plain f16 is refuted as a general rule: 58/153 records have |value| > 100 or non-finite elements
  (e.g. block2 max 59040), and layer1/layer4 records all sit at |v| <= 1. Content is probably mixed (quantized codes plus
  scales/bias) with a per-layer-type layout. FP8-e4m3 for the whole payload is also unproven (0.13% NaN codes in block0).
* Tensor shapes, strides, which records feed which kernel, ordering, and any dequantization. The container carries no shapes.
* That these weights match the gfx1100 kernels' expected layout. Bridging needs a kernel-side consumer trace
  (which kernarg pointers read which byte ranges) or the caller code in the DLL.

## Next contract-recovery steps
1. Per layer type, correlate record sizes with kernel access patterns (emulator load-address tracing on `k_swin_var`
   given a synthetic weight arena) to find which kernarg pointer expects the weights and their dtype/stride.
2. Locate the DLL host code that walks this table (`.text` 0.7 MB) for shape/dtype metadata.
