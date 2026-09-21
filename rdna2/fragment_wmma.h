#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace rdna2 {
using half2 = _Float16 __attribute__((ext_vector_type(2)));
struct PackedInput { uint32_t word[8]; };
struct Accumulator { float value[8]; };

// GFX11 V_WMMA_F32_16X16X16_F16, wave32 data layout.
// A: lane L owns row L%16, eight packed pairs along K.
// B: lane L owns column L%16, eight packed pairs along K.
// A/B are replicated in lanes L and L+16.
// C/D: lane L, register J owns row 2*J+L/16, column L%16.
// Requires all 32 lanes active; finite FP16 inputs and FP32 accumulators.
// Numerical rounding is NOT claimed bit-identical to native GFX11 WMMA.
// Source for register layout: https://gpuopen.com/learn/wmma_on_rdna3/
__device__ __forceinline__ Accumulator wmma_f32_f16(
    const PackedInput& a, const PackedInput& b, Accumulator c, unsigned lane)
{
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned row = 2 * j + (lane >> 4);
        const unsigned source = row + (lane & 16);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t packed_a = __shfl(a.word[k], source, 32);
            half2 av = __builtin_bit_cast(half2, packed_a);
            half2 bv = __builtin_bit_cast(half2, b.word[k]);
            c.value[j] = __builtin_amdgcn_fdot2(av, bv, c.value[j], false);
        }
    }
    return c;
}
}
