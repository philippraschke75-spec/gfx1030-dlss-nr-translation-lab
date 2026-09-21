#include <windows.h>
#include <stdio.h>

typedef int hipError_t;
typedef void* hipModule_t;

typedef hipError_t (__cdecl *PFN_hipInit)(unsigned int);
typedef hipError_t (__cdecl *PFN_hipGetDeviceCount)(int*);
typedef hipError_t (__cdecl *PFN_hipSetDevice)(int);
typedef hipError_t (__cdecl *PFN_hipModuleLoad)(hipModule_t*, const char*);
typedef hipError_t (__cdecl *PFN_hipModuleUnload)(hipModule_t);
typedef const char* (__cdecl *PFN_hipGetErrorString)(hipError_t);

int main(void)
{
    const char *dll = "amdhip64_7.dll";

    HMODULE h = LoadLibraryA(dll);
    if (!h) {
        printf("LoadLibraryA(%s) FAILED: %lu\n", dll, GetLastError());
        return 1;
    }

    printf("Loaded %s\n", dll);

    PFN_hipInit hipInit =
        (PFN_hipInit)GetProcAddress(h, "hipInit");

    PFN_hipGetDeviceCount hipGetDeviceCount =
        (PFN_hipGetDeviceCount)GetProcAddress(h, "hipGetDeviceCount");

    PFN_hipSetDevice hipSetDevice =
        (PFN_hipSetDevice)GetProcAddress(h, "hipSetDevice");

    PFN_hipModuleLoad hipModuleLoad =
        (PFN_hipModuleLoad)GetProcAddress(h, "hipModuleLoad");

    PFN_hipModuleUnload hipModuleUnload =
        (PFN_hipModuleUnload)GetProcAddress(h, "hipModuleUnload");

    PFN_hipGetErrorString hipGetErrorString =
        (PFN_hipGetErrorString)GetProcAddress(h, "hipGetErrorString");

    printf("hipInit           = %p\n", (void*)hipInit);
    printf("hipGetDeviceCount = %p\n", (void*)hipGetDeviceCount);
    printf("hipSetDevice      = %p\n", (void*)hipSetDevice);
    printf("hipModuleLoad     = %p\n", (void*)hipModuleLoad);
    printf("hipModuleUnload   = %p\n", (void*)hipModuleUnload);

    if (!hipInit || !hipGetDeviceCount || !hipSetDevice ||
        !hipModuleLoad || !hipModuleUnload) {
        printf("Required HIP export missing.\n");
        return 2;
    }

    hipError_t e;

    e = hipInit(0);
    printf("hipInit -> %d", e);
    if (hipGetErrorString) printf(" (%s)", hipGetErrorString(e));
    printf("\n");

    int count = 0;
    e = hipGetDeviceCount(&count);
    printf("hipGetDeviceCount -> %d, count=%d", e, count);
    if (hipGetErrorString) printf(" (%s)", hipGetErrorString(e));
    printf("\n");

    if (e != 0 || count <= 0) {
        return 3;
    }

    e = hipSetDevice(0);
    printf("hipSetDevice(0) -> %d", e);
    if (hipGetErrorString) printf(" (%s)", hipGetErrorString(e));
    printf("\n");

    hipModule_t module = NULL;

    printf("\nAttempting to load embedded_01.elf...\n");

    e = hipModuleLoad(&module, ".\\embedded_01.elf");

    printf("hipModuleLoad -> %d", e);
    if (hipGetErrorString) printf(" (%s)", hipGetErrorString(e));
    printf("\n");

    if (e == 0) {
        printf("========================================\n");
        printf("SUCCESS: HIP accepted embedded_01.elf\n");
        printf("========================================\n");

        hipModuleUnload(module);
        return 0;
    }

    printf("========================================\n");
    printf("FAILED: HIP rejected embedded_01.elf\n");
    printf("========================================\n");

    return 4;
}
