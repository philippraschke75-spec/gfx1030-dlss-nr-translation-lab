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

## Kernel-side consumer trace (emulator only, added 2026-09-21)
`trace_reads.py 32_1 <flags>` (synthetic arena, grid 2x2, offsets 0) shows `k_swin_var<32,true>` reads its **kernarg +0x10**
buffer as the weight record:
* flags=63: reads span [0, 0x54b0) = 21680 B; the real `block0.layer0.layer` record is 21696 B (16 B more).
  flags=0: reads span [0, 0x50b0) = 20656 B; `block1.layer0.layer` is 20672 B (again 16 B more).
  The size match in both cases is evidence, not proof (the extent depends on which paths the fixture enables).
* Segment map for flags=63 (access width -> region): `[0,0x1000)` 4 B loads; `[0x1000,0x2000)` 8 B; `[0x2010,0x2450)` 2 B sparse;
  `[0x2460,0x3060)` 8 B; `[0x3060,0x5060)` 2 B; `[0x5070,0x5470)` 8 B; `0x5130..0x54b0` 2 B scattered.
* `segcheck.py <dll> <record>` applies the map to the real block0 bytes. Wide-load segments decode as clean FP8 e4m3
  (no NaN, max |v| 0.56..0.88); 2-byte segments decode as clean f16 (E: median 6.3 / max 8.9; H: gain-like, max 1.0;
  C: max 0.998). This statistical agreement explains why the whole-record f16 fit failed: the record mixes FP8 matrices and f16 tables.
* Other pointers used with flags=63: +0x30 (8 B loads, 4 KiB), +0x40/+0x48/+0x80 (12 B loads, 3 KiB each), +0x78 (4 B loads),
  +0x38/+0x70 written (activations/outputs), +0xa0 read+write (scratch, 32 KiB), +0x00/+0x08 used only with low flags.
Still unproven: which record maps to which kernel instance/block, per-matrix shapes inside each FP8 segment, semantics of the
f16 tables, and that the +0x30..+0x80 inputs are activations. Correspondence rests on size and statistics only.

## Block topology recovered from record sizes + kernel read extents (hypothesis-level: size/extent agreement only)
`trace_reads.py` extents for slot +0x10, each within 16 B of (or exactly equal to) a real record size:
| kernel/flags | extent | matches record size | blocks |
|---|---|---|---|
| 32_1 f=1,2 / 32_0 f=0 | 20656 | 20672 | 1,2,3,67,68,69 |
| 32_1 f=16 (and f=63) | 21680 | 21696 | 0 |
| 32_1 f=4 | 22704 | 22720 | 4 |
| 32_1 f=8 | 22768 | 22784 | 66 |
| 64_0 f=0 | 61744 | 61760 | 5,6,7,63,64,65 |
| 64_0 f=15 | 69936 | 69936 (exact) | 8 (62: 70048 unexplained) |
| 128_0 f=0 | 197168 | 197184 | 9-13,57-61 |
| 128_0 f=15 | 229936 | 229936 (exact) | 14 (56: 230176 unexplained) |
| 256_0 f=0 | 689216 | 689232 | 15-21,49-55 |
| 256_0 f=15 | 820288 | 820288 (exact) | 22 (48: 820784 unexplained) |
Block sequence by record shape: 0-4 (W32), 5-8 (W64), 9-14 (W128), 15-22 (W256) = encoder; 23-30 and 40-47 = 4-layer blocks
(524288/263168/917568/263168 [+524304 at 30]); 31-38 = 5-layer blocks (~4 MiB); 39 (525312) ; 48-70 mirror the encoder.
Unexplained records (21808, 22784 vs flag 8 ok, 70048, 230176, 820784, 525312, block-70 blend_scale) presumably use flag bits
or variants not yet traced. The 4/5-layer blocks belong to kernels other than k_swin_var (not traced yet).
`run_real_weights.py`: with block0.layer0.layer as weights, `<32,true>` flags=63 (2x2 grid, emulator) terminates in 615,735 steps,
all writes stay in slots +0x38/+0x70/+0xa0, output non-degenerate; the synthetic-weight run gave e4m3 |mean| 341 in the +0x38 output vs
2.7 with real weights (e4m3 reading of that slot is an assumption). This is a plausibility check, not correctness.
