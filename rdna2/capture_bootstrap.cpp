// Opt-in ASI entry point. No hooks or kernel launches in this module.
#include <windows.h>
#include <cwchar>
#include "capture_bootstrap_config.h"
#include "capture_exception_trace.h"

#ifndef CAPTURE_PROCESS_NAME
#define CAPTURE_PROCESS_NAME L"Cyberpunk2077.exe"
#endif

static bool allowed_process() {
    wchar_t path[32768]{};
    DWORD length=GetModuleFileNameW(nullptr,path,32768);
    if(!length || length>=32768) return false;
    const wchar_t* name=wcsrchr(path,L'\\');
    return _wcsicmp(name ? name+1 : path,CAPTURE_PROCESS_NAME)==0;
}

static void report(const char* message) {
    HANDLE f=CreateFileW(CAPTURE_BOOTSTRAP_LOG,FILE_APPEND_DATA,FILE_SHARE_READ,nullptr,
                        OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(f!=INVALID_HANDLE_VALUE) {
        DWORD n; WriteFile(f,message,DWORD(lstrlenA(message)),&n,nullptr); CloseHandle(f);
    }
}
static DWORD WINAPI start(void*) {
    // The worker starts after the loader lock is released.
    wchar_t trace[32768]{};
    DWORD length=GetEnvironmentVariableW(L"DLSSNR_CAPTURE_FILE",trace,32768);
    if(length && length<32740) {
        wcscat_s(trace,L".exceptions.log");
        if(!capture_exception_trace::install(trace))
            report("Exception observer unavailable\r\n");
    }
    if(GetModuleHandleW(L"amdhip64_7.dll")) {
        report("STOP: another HIP7 module is already loaded\r\n"); return 1;
    }
    auto bridge=LoadLibraryExW(CAPTURE_BRIDGE_PATH,nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!bridge) { report("STOP: capture bridge load failed\r\n"); return 2; }
    report("Capture-only bridge loaded\r\n");
    auto runtime=LoadLibraryExW(CAPTURE_RUNTIME_PATH,nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!runtime) { report("STOP: research runtime load failed\r\n"); return 3; }
    report("Research runtime loaded; this does not prove initialization or capture\r\n");
    return 0;
}
BOOL WINAPI DllMain(HINSTANCE instance,DWORD reason,LPVOID) {
    if(reason!=DLL_PROCESS_ATTACH) return TRUE;
    DisableThreadLibraryCalls(instance);
    // Child processes inherit capture variables; never inject the crash reporter.
    if(!allowed_process()) return TRUE;
    wchar_t mode[2]{};
    if(GetEnvironmentVariableW(L"DLSSNR_RESEARCH_CAPTURE",mode,2)!=1 || mode[0]!=L'1') return TRUE;
    wchar_t target[32768]{};
    if(!GetEnvironmentVariableW(L"DLSSNR_CAPTURE_FILE",target,32768)) return TRUE;
    HANDLE t=CreateThread(nullptr,0,start,nullptr,0,nullptr);
    if(t) CloseHandle(t);
    return TRUE;
}
