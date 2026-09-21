#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

int main()
{
    const char* file = "C:\\DLSSNRResearch\\gfx1032_payload.bin";

    int count = 0;
    hipError_t e = hipGetDeviceCount(&count);
    printf("hipGetDeviceCount: %s (%d), devices=%d\n",
           hipGetErrorString(e), (int)e, count);

    if (e != hipSuccess || count == 0)
        return 1;

    e = hipSetDevice(0);
    printf("hipSetDevice: %s (%d)\n",
           hipGetErrorString(e), (int)e);

    hipModule_t module = nullptr;

    e = hipModuleLoad(&module, file);
    printf("hipModuleLoad: %s (%d)\n",
           hipGetErrorString(e), (int)e);

    if (e == hipSuccess)
    {
        printf("SUCCESS: HIP accepted the gfx1032 ELF.\n");
        hipModuleUnload(module);
        return 0;
    }

    printf("FAILURE: HIP rejected the gfx1032 ELF.\n");
    return 2;
}
