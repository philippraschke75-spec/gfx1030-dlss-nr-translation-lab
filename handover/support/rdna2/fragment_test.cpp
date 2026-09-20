#include "fragment_wmma.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(expr) do { hipError_t e = (expr); if (e != hipSuccess) { \
    std::fprintf(stderr, "%s: %s at line %d\n", #expr, hipGetErrorString(e), __LINE__); \
    std::exit(2); } } while (0)

constexpr int Tiles = 32;
constexpr int Values = Tiles * 256;
constexpr int Guard = 64;
constexpr float Canary = -12345.0f;

// No full matrices are accessible to the replacement. Its inputs/outputs are
// exactly per-lane register fragments, preserving the intended ABI shape.
__global__ void fragment_test(const rdna2::PackedInput* a,
                             const rdna2::PackedInput* b,
                             const rdna2::Accumulator* c,
                             rdna2::Accumulator* d)
{
    const unsigned thread = blockIdx.x * blockDim.x + threadIdx.x;
    d[thread] = rdna2::wmma_f32_f16(a[thread], b[thread], c[thread], threadIdx.x & 31);
}

static uint32_t pack(float lo, float hi) {
    const _Float16 halves[2] = {static_cast<_Float16>(lo), static_cast<_Float16>(hi)};
    uint32_t value;
    std::memcpy(&value, halves, sizeof(value));
    return value;
}

static int errors(const std::vector<float>& got, const std::vector<float>& expected) {
    int n = 0;
    for (int i = 0; i < Values; ++i)
        if (!std::isfinite(got[Guard+i]) || got[Guard+i] != expected[i]) ++n;
    for (int i = 0; i < Guard; ++i)
        if (got[i] != Canary || got[Guard+Values+i] != Canary) ++n;
    return n;
}

int main() {
    hipDeviceProp_t prop{};
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    int runtime = 0, driver = 0;
    HIP_OK(hipRuntimeGetVersion(&runtime));
    HIP_OK(hipDriverGetVersion(&driver));
    std::printf("Device=%s arch=%s wave=%d runtime=%d driver=%d\n",
                prop.name, prop.gcnArchName, prop.warpSize, runtime, driver);
    if ((std::strcmp(prop.gcnArchName, "gfx1030") != 0 &&
         std::strncmp(prop.gcnArchName, "gfx1030:", 8) != 0) || prop.warpSize != 32)
        return 4;

    std::vector<rdna2::PackedInput> af(Tiles*32), bf(Tiles*32);
    std::vector<rdna2::Accumulator> cf(Tiles*32);
    std::vector<float> expected(Values), got(Values+2*Guard, Canary);
    uint32_t seed = 0x69001030;
    auto next = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                       return (static_cast<int>(seed % 33) - 16) / 16.0f; };
    for (int tile = 0; tile < Tiles; ++tile) {
        std::array<float,256> a{}, b{}, c{}, reference{};
        for (int r = 0; r < 16; ++r) for (int col = 0; col < 16; ++col) {
            const int i = 16*r+col;
            a[i] = next(); b[i] = next(); c[i] = next();
            if (tile == 0) a[i] = 0;                     // accumulator preserved
            if (tile == 1) a[i] = r == col ? 1.0f : 0;   // left identity
            if (tile == 2) b[i] = r == col ? 1.0f : 0;   // right identity
            if (tile >= 3 && tile < 19) {                // each K position independently
                a[i] = col == tile-3 ? (r+1)/16.0f : 0;
                b[i] = r == tile-3 ? (col-7)/16.0f : 0;
            }
        }
        // Independent scalar reference; small dyadic operands make every
        // product and sum exact in float, so no loose tolerance can hide errors.
        for (int r = 0; r < 16; ++r) for (int col = 0; col < 16; ++col) {
            double sum = c[16*r+col];
            for (int k = 0; k < 16; ++k) sum += double(a[16*r+k])*b[16*k+col];
            reference[16*r+col] = static_cast<float>(sum);
        }
        for (int lane = 0; lane < 32; ++lane) {
            const int t = tile*32+lane;
            for (int k = 0; k < 8; ++k) {
                af[t].word[k] = pack(a[(lane%16)*16+2*k], a[(lane%16)*16+2*k+1]);
                bf[t].word[k] = pack(b[32*k+lane%16], b[32*k+16+lane%16]);
            }
            for (int j = 0; j < 8; ++j) {
                const int matrix_index = (2*j+lane/16)*16+lane%16;
                cf[t].value[j] = c[matrix_index];
                expected[t*8+j] = reference[matrix_index];
            }
        }
    }
    // Host negative controls prove that skipped stores, a damaged value, and a
    // guard overwrite cannot report success. No extra GPU dispatches involved.
    std::vector<float> check = got;
    std::copy(expected.begin(), expected.end(), check.begin()+Guard);
    if (errors(check, expected)) return 5;
    check[Guard+7] += 1;
    if (!errors(check, expected)) return 5;
    check[Guard+7] = expected[7];
    const uint32_t nan = 0xffffffff;
    std::memcpy(&check[Guard+7], &nan, sizeof(nan));
    if (!errors(check, expected)) return 5;
    check[Guard+7] = expected[7]; check[0] = 0;
    if (!errors(check, expected)) return 5;
    std::puts("Host negative controls: PASS (wrong value, NaN, guard overwrite)");

    rdna2::PackedInput *da=nullptr, *db=nullptr;
    rdna2::Accumulator* dc=nullptr;
    float* dd=nullptr;
    HIP_OK(hipMalloc(&da, af.size()*sizeof(af[0])));
    HIP_OK(hipMalloc(&db, bf.size()*sizeof(bf[0])));
    HIP_OK(hipMalloc(&dc, cf.size()*sizeof(cf[0])));
    HIP_OK(hipMalloc(&dd, got.size()*sizeof(float)));
    HIP_OK(hipMemcpy(da, af.data(), af.size()*sizeof(af[0]), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(db, bf.data(), bf.size()*sizeof(bf[0]), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dc, cf.data(), cf.size()*sizeof(cf[0]), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dd, got.data(), got.size()*sizeof(float), hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dd+Guard, 0xff, Values*sizeof(float)));
    std::puts("One dispatch: 8 blocks x 128 threads, 32 independent 16x16 tiles");
    hipLaunchKernelGGL(fragment_test, dim3(8), dim3(128), 0, 0,
                       da, db, dc, reinterpret_cast<rdna2::Accumulator*>(dd+Guard));
    HIP_OK(hipGetLastError());
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipMemcpy(got.data(), dd, got.size()*sizeof(float), hipMemcpyDeviceToHost));
    const int bad = errors(got, expected);
    std::printf("Compared %d outputs exactly; errors=%d; guards=%s\n", Values, bad,
                std::equal(got.begin(), got.begin()+Guard, got.end()-Guard) &&
                got[0] == Canary ? "checked" : "FAILED");
    if (bad) for (int i=0, shown=0; i<Values && shown<8; ++i)
        if (got[Guard+i] != expected[i]) {
            std::printf("index=%d got=%g expected=%g\n", i, got[Guard+i], expected[i]); ++shown;
        }
    HIP_OK(hipFree(da)); HIP_OK(hipFree(db)); HIP_OK(hipFree(dc)); HIP_OK(hipFree(dd));
    std::puts(bad ? "FRAGMENT TEST FAIL" : "FRAGMENT TEST PASS");
    return bad ? 3 : 0;
}
