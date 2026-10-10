"""CPU proof: packed q8h (u16 min/max variant) == q8ref(f16) for all 2^16 f16 inputs. f16 adds done in f64 then
rounded once to f16 (exact model of v_pk_add_f16 RNE). Exit 1 on mismatch."""
import numpy as np, sys
x = np.arange(65536, dtype=np.uint32).astype(np.uint16)
f = x.view(np.float16).astype(np.float64)
# reference q8 (f32 math, from post_native.hip), applied to the f16 value
a32 = np.minimum(np.abs(f.astype(np.float32)), np.float32(448))
b = a32.view(np.uint32)
rn = ((b + np.uint32(0x7FFFF) + ((b >> 20) & 1)) & np.uint32(0xFFF00000)).view(np.float32)
rs = ((a32 + np.float32(24576)) - np.float32(24576)).astype(np.float32)
nan = np.isnan(f)
ref = np.where(a32 < np.float32(0.015625), rs, rn)
ref = np.copysign(ref, f.astype(np.float32)).astype(np.float16).view(np.uint16)
# candidate
ab = np.minimum(x & 0x7FFF, 0x5F00).astype(np.uint16)
eb = np.maximum(ab & 0x7C00, 0x2400).astype(np.uint16)
A = ab.view(np.float16).astype(np.float64)
K = (eb.view(np.float16).astype(np.float64) * 128).astype(np.float16).astype(np.float64)
s = (A + K).astype(np.float16).astype(np.float64)
r = (s - K).astype(np.float16).view(np.uint16)
got = (r & 0x7FFF) | (x & 0x8000)
bad = (got != ref) & ~nan
# NaN/inf: q8 gives fminf(NaN|inf, 448) = 448 with x's sign; candidate must too
special = ~np.isfinite(f)
bad |= special & (got != (0x5F00 | (x & 0x8000)))
print(f"q8h u16: {bad.sum()} mismatches over 65536 f16 (NaN inputs excluded: {nan.sum()})")
sys.exit(1 if bad.any() else 0)
