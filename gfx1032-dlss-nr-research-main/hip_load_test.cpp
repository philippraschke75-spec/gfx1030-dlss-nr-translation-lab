#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

int main()
{
    const char* file = "embedded_01.elf";

    printf("=== DLSS-NR GFX1032 DIRECT HIP LOAD TEST ===\n");
    printf("File: %s\n", file);

    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);

    printf("hipGetDeviceCount: %d (%s)\n",
           (int)err,
           hipGetErrorString(err));

    if (err != hipSuccess || count == 0)
        return 10;

    hipDeviceProp_t prop{};
    err = hipGetDeviceProperties(&prop, 0);

    printf("hipGetDeviceProperties: %d (%s)\n",
           (int)err,
           hipGetErrorString(err));

    if (err != hipSuccess)
        return 11;

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n",
           prop.major,
           prop.minor);

    err = hipSetDevice(0);

    printf("hipSetDevice(0): %d (%s)\n",
           (int)err,
           hipGetErrorString(err));

    if (err != hipSuccess)
        return 12;

    hipModule_t module = nullptr;

    err = hipModuleLoad(&module, file);

    printf("hipModuleLoad: %d (%s)\n",
           (int)err,
           hipGetErrorString(err));

    if (err == hipSuccess)
    {
        printf("!!! SUCCESS: HIP ACCEPTED THE GFX1032 ELF !!!\n");

        hipError_t unloadErr = hipModuleUnload(module);

        printf("hipModuleUnload: %d (%s)\n",
               (int)unloadErr,
               hipGetErrorString(unloadErr));

        return 0;
    }

    printf("!!! FAILURE: HIP REJECTED THE ELF !!!\n");

    return 20;
}
