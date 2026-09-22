// External diagnostic snapshot. Balances each successful thread suspension.
#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>
#include <psapi.h>
#include <cstdio>
#include <cstdlib>
#pragma comment(lib,"dbghelp.lib")
#pragma comment(lib,"psapi.lib")
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    DWORD pid=std::strtoul(argv[1],nullptr,10);
    HANDLE process=OpenProcess(PROCESS_QUERY_INFORMATION|PROCESS_VM_READ,FALSE,pid);
    if(!process) { std::printf("OpenProcess error=%lu\n",GetLastError()); return 3; }
    SymSetOptions(SYMOPT_DEFERRED_LOADS|SYMOPT_FAIL_CRITICAL_ERRORS);
    if(!SymInitialize(process,"",TRUE)) {
        std::printf("SymInitialize error=%lu\n",GetLastError()); CloseHandle(process); return 4;
    }
    HANDLE snapshot=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0);
    THREADENTRY32 entry{}; entry.dwSize=sizeof(entry);
    if(Thread32First(snapshot,&entry)) do {
        if(entry.th32OwnerProcessID!=pid) continue;
        HANDLE thread=OpenThread(THREAD_GET_CONTEXT|THREAD_SUSPEND_RESUME|THREAD_QUERY_INFORMATION,FALSE,entry.th32ThreadID);
        if(!thread) { std::printf("THREAD %lu open error=%lu\n",entry.th32ThreadID,GetLastError()); continue; }
        DWORD previous=SuspendThread(thread);
        if(previous==DWORD(-1)) { CloseHandle(thread); continue; }
        CONTEXT ctx{}; ctx.ContextFlags=CONTEXT_FULL;
        BOOL ok=GetThreadContext(thread,&ctx);
        // Snapshot stack addresses while suspended; print/resolve after resuming.
        DWORD64 addresses[64]{}; unsigned count=0;
        if(ok) {
            addresses[count++]=ctx.Rip;
            STACKFRAME64 frame{};
            frame.AddrPC={ctx.Rip,0,AddrModeFlat};
            frame.AddrStack={ctx.Rsp,0,AddrModeFlat};
            frame.AddrFrame={ctx.Rbp,0,AddrModeFlat};
            while(count<64 && StackWalk64(IMAGE_FILE_MACHINE_AMD64,process,thread,&frame,&ctx,
                  nullptr,SymFunctionTableAccess64,SymGetModuleBase64,nullptr)) {
                if(!frame.AddrPC.Offset) break;
                addresses[count++]=frame.AddrPC.Offset;
            }
        }
        DWORD resumed=ResumeThread(thread);
        CloseHandle(thread);
        std::printf("THREAD %lu previous_suspend=%lu context=%d resume_result=%lu\n",entry.th32ThreadID,previous,int(ok),resumed);
        for(unsigned i=0;i<count;++i) {
            MEMORY_BASIC_INFORMATION memory{};
            char path[2048]{};
            VirtualQueryEx(process,reinterpret_cast<void*>(addresses[i]),&memory,sizeof(memory));
            GetModuleFileNameExA(process,reinterpret_cast<HMODULE>(memory.AllocationBase),path,sizeof(path));
            std::printf("  %02u 0x%llx %s +0x%llx\n",i,addresses[i],path,
                addresses[i]-reinterpret_cast<DWORD64>(memory.AllocationBase));
        }
        std::fflush(stdout);
    } while(Thread32Next(snapshot,&entry));
    CloseHandle(snapshot); SymCleanup(process); CloseHandle(process);
}
