#include <windows.h>
#include <tlhelp32.h>
#include <iostream>
#include <string>

int main()
{
    const wchar_t* processName = L"GTA5_Enhanced.exe";
    const wchar_t* dllPath =
        L"C:\\Program Files (x86)\\Steam\\steamapps\\common\\Grand Theft Auto V Enhanced\\dlssnr_v5.dll";

    DWORD pid = 0;

    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE)
    {
        std::cerr << "CreateToolhelp32Snapshot failed: " << GetLastError() << "\n";
        return 1;
    }

    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);

    if (Process32FirstW(snapshot, &pe))
    {
        do
        {
            if (_wcsicmp(pe.szExeFile, processName) == 0)
            {
                pid = pe.th32ProcessID;
                break;
            }
        } while (Process32NextW(snapshot, &pe));
    }

    CloseHandle(snapshot);

    if (!pid)
    {
        std::cerr << "GTA5_Enhanced.exe not found.\n";
        std::cerr << "Start GTA first, then run this loader.\n";
        return 2;
    }

    std::wcout << L"Found GTA5_Enhanced.exe, PID " << pid << L"\n";

    HANDLE process = OpenProcess(
        PROCESS_CREATE_THREAD |
        PROCESS_QUERY_INFORMATION |
        PROCESS_VM_OPERATION |
        PROCESS_VM_WRITE |
        PROCESS_VM_READ,
        FALSE,
        pid);

    if (!process)
    {
        std::cerr << "OpenProcess failed: " << GetLastError() << "\n";
        return 3;
    }

    SIZE_T pathBytes = (wcslen(dllPath) + 1) * sizeof(wchar_t);

    LPVOID remotePath = VirtualAllocEx(
        process,
        nullptr,
        pathBytes,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE);

    if (!remotePath)
    {
        std::cerr << "VirtualAllocEx failed: " << GetLastError() << "\n";
        CloseHandle(process);
        return 4;
    }

    if (!WriteProcessMemory(
            process,
            remotePath,
            dllPath,
            pathBytes,
            nullptr))
    {
        std::cerr << "WriteProcessMemory failed: " << GetLastError() << "\n";
        VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
        CloseHandle(process);
        return 5;
    }

    HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    FARPROC loadLibraryW = GetProcAddress(kernel32, "LoadLibraryW");

    if (!loadLibraryW)
    {
        std::cerr << "GetProcAddress(LoadLibraryW) failed.\n";
        VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
        CloseHandle(process);
        return 6;
    }

    HANDLE thread = CreateRemoteThread(
        process,
        nullptr,
        0,
        reinterpret_cast<LPTHREAD_START_ROUTINE>(loadLibraryW),
        remotePath,
        0,
        nullptr);

    if (!thread)
    {
        std::cerr << "CreateRemoteThread failed: " << GetLastError() << "\n";
        VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
        CloseHandle(process);
        return 7;
    }

    std::cout << "V5 LoadLibraryW thread created.\n";

    WaitForSingleObject(thread, 10000);

    DWORD exitCode = 0;
    if (GetExitCodeThread(thread, &exitCode))
    {
        std::cout << "Remote LoadLibraryW result: 0x"
                  << std::hex << exitCode << std::dec << "\n";

        if (exitCode == 0)
            std::cout << "V5 DLL load FAILED.\n";
        else
            std::cout << "V5 DLL load reported success.\n";
    }

    CloseHandle(thread);

    VirtualFreeEx(
        process,
        remotePath,
        0,
        MEM_RELEASE);

    CloseHandle(process);

    return 0;
}