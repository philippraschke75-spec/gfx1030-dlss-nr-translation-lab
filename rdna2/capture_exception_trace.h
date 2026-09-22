#pragma once
#include <windows.h>
#include <cstdio>

// Observe exceptions without swallowing them or changing their disposition.
// Install from the bootstrap worker, never from DllMain.
namespace capture_exception_trace {
inline HANDLE output=INVALID_HANDLE_VALUE;
inline volatile LONG busy=0;
inline volatile LONG events=0;
inline volatile LONG breakpoints=0;
inline void line(const char* text) {
    DWORD written;
    WriteFile(output,text,DWORD(lstrlenA(text)),&written,nullptr);
}
inline void address(const char* label, void* value) {
    HMODULE module=nullptr;
    char path[32768]{};
    char text[33024]{};
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                         GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(value),&module))
        GetModuleFileNameA(module,path,DWORD(sizeof(path)));
    std::snprintf(text,sizeof(text),"%s address=%p module=%s offset=0x%llx\r\n",
        label,value,path,static_cast<unsigned long long>(
        reinterpret_cast<ULONG_PTR>(value)-reinterpret_cast<ULONG_PTR>(module)));
    line(text);
}
inline LONG CALLBACK observe(EXCEPTION_POINTERS* info) {
    const DWORD code=info->ExceptionRecord->ExceptionCode;
    if(code!=0xe06d7363 && code!=EXCEPTION_ACCESS_VIOLATION &&
       code!=EXCEPTION_ILLEGAL_INSTRUCTION && code!=EXCEPTION_STACK_OVERFLOW &&
       code!=EXCEPTION_BREAKPOINT)
        return EXCEPTION_CONTINUE_SEARCH;
    // Avoid doing stack work on an exhausted stack.
    if(code==EXCEPTION_STACK_OVERFLOW) return EXCEPTION_CONTINUE_SEARCH;
    if(InterlockedCompareExchange(&busy,1,0)) return EXCEPTION_CONTINUE_SEARCH;
    // Ordinary handled C++ exceptions must not exhaust the breakpoint budget.
    LONG ordinal=InterlockedIncrement(code==EXCEPTION_BREAKPOINT ? &breakpoints : &events);
    if(ordinal<=16) {
        char text[256];
        SYSTEMTIME now{}; GetSystemTime(&now);
        std::snprintf(text,sizeof(text),"EXCEPTION observed (may be handled) utc=%04u-%02u-%02uT%02u:%02u:%02u.%03u pid=%lu tid=%lu code=0x%08lx\r\n",
            now.wYear,now.wMonth,now.wDay,now.wHour,now.wMinute,now.wSecond,now.wMilliseconds,
            GetCurrentProcessId(),GetCurrentThreadId(),code);
        line(text);
        address("exception",info->ExceptionRecord->ExceptionAddress);
        void* frames[48]{};
        USHORT count=CaptureStackBackTrace(0,48,frames,nullptr);
        for(USHORT i=0;i<count;++i) address("frame",frames[i]);
        FlushFileBuffers(output);
    }
    InterlockedExchange(&busy,0);
    return EXCEPTION_CONTINUE_SEARCH;
}
inline bool install(const wchar_t* path) {
    output=CreateFileW(path,FILE_APPEND_DATA,FILE_SHARE_READ,nullptr,
                       OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(output==INVALID_HANDLE_VALUE) return false;
    if(!AddVectoredExceptionHandler(1,observe)) {
        CloseHandle(output); output=INVALID_HANDLE_VALUE; return false;
    }
    line("Exception observer installed; first-chance exceptions are not proof of a crash.\r\n");
    return true;
}
}
