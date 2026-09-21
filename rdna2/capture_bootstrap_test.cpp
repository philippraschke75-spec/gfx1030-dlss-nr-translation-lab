#include <windows.h>
#include <cassert>
#include <cstdio>
int main(int argc,char** argv) {
    assert(argc==3);
    if(argv[2][0]!='0') {
        SetEnvironmentVariableA("DLSSNR_RESEARCH_CAPTURE","1");
        SetEnvironmentVariableA("DLSSNR_CAPTURE_FILE","mock-only.json");
    } else {
        SetEnvironmentVariableA("DLSSNR_RESEARCH_CAPTURE",nullptr);
        SetEnvironmentVariableA("DLSSNR_CAPTURE_FILE",nullptr);
    }
    assert(LoadLibraryA(argv[1]));
    if(argv[2][0]=='1') {
        for(int i=0;i<100 && !GetModuleHandleA("capture_mock_runtime.dll");++i) Sleep(20);
        assert(GetModuleHandleA("capture_mock_runtime.dll"));
        assert(GetModuleHandleA("amdhip64_7.dll"));
    } else {
        Sleep(100);
        assert(!GetModuleHandleA("amdhip64_7.dll"));
        assert(!GetModuleHandleA("capture_mock_runtime.dll"));
    }
    std::puts("PASS bootstrap environment gate and module loading");
}
