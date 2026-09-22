// Host-only first-launch capture. No device reads and no dispatch permission.
#pragma once
#include <windows.h>
#include <cstdio>
#include <cstring>

namespace launch_capture {
inline SRWLOCK lock = SRWLOCK_INIT;
struct Entry { const void* host; void** module; unsigned variant; };
inline Entry entries[128]{};
inline unsigned count = 0;
inline bool attempted = false;
inline const char* symbols[] = {
    "_Z10k_swin_varILi32ELb1EEv9VarParams",
    "_Z10k_swin_varILi32ELb0EEv9VarParams",
    "_Z10k_swin_varILi64ELb0EEv9VarParams",
    "_Z10k_swin_varILi128ELb0EEv9VarParams",
    "_Z10k_swin_varILi256ELb0EEv9VarParams"
};
inline bool read(const void* src, void* dst, size_t size) {
    SIZE_T got = 0;
    return src && ReadProcessMemory(GetCurrentProcess(), src, dst, size, &got) && got == size;
}
inline void registration(void** module, const void* host, const char* name) {
    char copy[96]{};
    // Read one byte at a time: a valid short name can end at a page boundary.
    unsigned i = 0;
    for (; i < sizeof(copy)-1; ++i) {
        if (!read(name ? name+i : nullptr, copy+i, 1)) return;
        if (!copy[i]) break;
    }
    if (i == sizeof(copy)-1) return;
    AcquireSRWLockExclusive(&lock);
    for (unsigned v = 0; v < 5; ++v) {
        if (!std::strcmp(copy, symbols[v]) && count < 128)
            entries[count++] = {host, module, v};
    }
    ReleaseSRWLockExclusive(&lock);
}
inline void unregister_module(void** module) {
    AcquireSRWLockExclusive(&lock);
    for (unsigned i=0; i<count; ) {
        if (entries[i].module == module) entries[i] = entries[--count];
        else ++i;
    }
    ReleaseSRWLockExclusive(&lock);
}
// Returns true whenever capture is requested, including failed/unknown captures.
// Caller MUST block the launch in that case, even if its normal launch gate is on.
inline bool before_launch(const void* host, const unsigned* grid,
                          const unsigned* block, void** args, size_t shared, void* stream) {
    wchar_t path[32768];
    DWORD n = GetEnvironmentVariableW(L"DLSSNR_CAPTURE_FILE", path, 32768);
    if (!n) return false;
    if (n >= 32768) return true;
    AcquireSRWLockExclusive(&lock);
    int variant = -1;
    for (unsigned i=0; i<count; ++i) if(entries[i].host==host) variant=entries[i].variant;
    if (attempted || variant < 0) { ReleaseSRWLockExclusive(&lock); return true; }
    attempted = true;
    void* arg = nullptr;
    unsigned char bytes[168]{}; // Explicit VarParams only, NOT 424 hidden-arg bytes.
    bool ok = read(args, &arg, sizeof(arg)) && read(arg, bytes, sizeof(bytes));
    char hex[337]{};
    const char* digits="0123456789abcdef";
    if (ok) for(unsigned i=0;i<168;++i) { hex[2*i]=digits[bytes[i]>>4]; hex[2*i+1]=digits[bytes[i]&15]; }
    char out[2048];
    int size = std::snprintf(out,sizeof(out),
        "{\"schema\":1,\"kind\":\"blocked_first_swin_launch\",\"pid\":%lu,"
        "\"symbol\":\"%s\",\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
        "\"shared_bytes\":%zu,\"stream\":\"%p\",\"explicit_bytes\":168,"
        "\"args_read_ok\":%s,\"args_hex\":\"%s\",\"payload_identity_verified\":false,"
        "\"device_buffers_captured\":false,\"launch_forwarded\":false}\n",
        GetCurrentProcessId(),symbols[variant],grid[0],grid[1],grid[2],block[0],block[1],block[2],
        shared,stream,ok?"true":"false",hex);
    HANDLE file = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                              CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if(file != INVALID_HANDLE_VALUE) {
        DWORD written=0;
        bool saved = size>0 && size<int(sizeof(out)) && WriteFile(file,out,DWORD(size),&written,nullptr)
                     && written==DWORD(size);
        if (!saved) OutputDebugStringA("DLSSNR capture write failed\n");
        CloseHandle(file);
    } else OutputDebugStringA("DLSSNR capture destination unavailable or already exists\n");
    ReleaseSRWLockExclusive(&lock);
    return true;
}
}
