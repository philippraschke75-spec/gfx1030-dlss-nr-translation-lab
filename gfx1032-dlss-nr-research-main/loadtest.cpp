#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    hipModule_t module = nullptr;

    hipError_t e = hipModuleLoad(
        &module,
        "C:\\DLSSNRResearch\\gfx1032_embedded.elf"
    );

    std::cout << "hipModuleLoad result: "
              << hipGetErrorName(e) << " (" << (int)e << ")\n";

    if (e != hipSuccess)
        std::cout << "Description: " << hipGetErrorString(e) << "\n";

    if (module)
        hipModuleUnload(module);

    return 0;
}
