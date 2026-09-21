// Host-only reproduction: register extracted original bundle, then enumerate.
// Does not load the research runtime, hook a game, or launch any kernel.
#include <windows.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include <exception>
#include "capture_exception_trace.h"
struct Wrapper { unsigned magic,version; const void* data; const void* unused; };
int main(int argc,char** argv) {
    if(argc!=3) return 2;
    capture_exception_trace::install(L"registration-probe.exceptions.log");
    std::ifstream file(argv[2],std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(file)),{});
    if(data.empty()) return 3;
    auto lib=LoadLibraryA(argv[1]);
    if(!lib) return 4;
    auto reg=reinterpret_cast<void**(*)(const void*)>(GetProcAddress(lib,"__hipRegisterFatBinary"));
    auto count=reinterpret_cast<int(*)(int*)>(GetProcAddress(lib,"hipGetDeviceCount"));
    auto unreg=reinterpret_cast<void(*)(void**)>(GetProcAddress(lib,"__hipUnregisterFatBinary"));
    if(!reg || !count || !unreg) return 5;
    Wrapper wrapper{0x48495046,1,data.data(),nullptr};
    std::puts("Register original bundle; no launches"); std::fflush(stdout);
    auto handle=reg(&wrapper);
    int n=-1;
    std::puts("ENTER hipGetDeviceCount after registration"); std::fflush(stdout);
    try {
        int rc=count(&n);
        std::printf("RETURN rc=%d count=%d\n",rc,n); std::fflush(stdout);
        unreg(handle);
        return rc ? 6 : 0;
    } catch(const std::exception& error) {
        std::printf("CAUGHT std::exception: %s\n",error.what()); std::fflush(stdout);
    } catch(...) {
        std::puts("CAUGHT unknown C++ exception"); std::fflush(stdout);
    }
    // Skip backend teardown after a failed initialization; process exit cleans up.
    ExitProcess(7);
}
