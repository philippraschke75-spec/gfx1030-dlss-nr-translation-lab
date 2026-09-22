// DLSS-NR-on-RDNA2 V5 diagnostic companion.
// This DLL is intentionally diagnostic-only: it inspects the already-loaded
// version_v4.dll and tests whether its embedded gfx1032 ELF can be accepted by
// the installed AMD HIP runtime without modifying V4 or altering its state.

#include <windows.h>
#include <psapi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>

#ifndef _WIN64
#error "This V5 diagnostic is intended for 64-bit Windows builds."
#endif

#define V5_LOG_FILE_NAME "dlssnr_v5.log"
#define V5_TARGET_LITERAL "hipv4-amdgcn-amd-amdhsa--gfx1032"
#define V5_OFFLOAD_MARKER "__CLANG_OFFLOAD_BUNDLE__"
static constexpr uint8_t V5_ELF_MAGIC[4] = { 0x7F, 'E', 'L', 'F' };

#ifndef IMAGE_SIZEOF_SHORT_NAME
#define IMAGE_SIZEOF_SHORT_NAME 8
#endif

struct V5SectionRange {
    const char name[IMAGE_SIZEOF_SHORT_NAME + 1];
    uintptr_t va = 0;
    size_t vsize = 0;
    uintptr_t raw = 0;
    size_t rawSize = 0;
    uintptr_t sectionStart = 0;
    uintptr_t sectionEnd = 0;
};

struct V5BundleTarget {
    std::string name;
    uintptr_t start = 0;
    size_t length = 0;
};

struct V5BundleInfo {
    const uint8_t* bundleStart = nullptr;
    size_t bundleSize = 0;
    size_t headerSize = 0;
    size_t targetCount = 0;
    std::vector<V5BundleTarget> targets;
    const uint8_t* elfPointer = nullptr;
    size_t elfSize = 0;
    uintptr_t elfRva = 0;
    uintptr_t elfVa = 0;
};

#include <hip/hip_runtime.h>

typedef int (WINAPI *hipGetDeviceCount_fn)(int*);
typedef int (WINAPI *hipModuleLoadData_fn)(hipModule_t*, const void*);
typedef int (WINAPI *hipModuleGetFunction_fn)(hipFunction_t*, hipModule_t, const char*);
typedef int (WINAPI *hipModuleUnload_fn)(hipModule_t);
typedef int (WINAPI *hipModuleLaunchKernel_fn)(
    hipFunction_t,
    unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int,
    hipStream_t,
    void**,
    void**
);
typedef int (WINAPI *hipDeviceSynchronize_fn)(void);
typedef int (WINAPI *hipGetLastError_fn)(void);
typedef int (WINAPI *hipPeekAtLastError_fn)(void);

#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
using hipGetDevicePropertiesR0600_fn = hipError_t (WINAPI*)(hipDeviceProp_t*, int);
#else
struct hipDeviceProp_t {
    char name[256];
    char gcnArchName[256];
    int major;
    int minor;
    int multiProcessorCount;
    int clockRate;
    int memoryClockRate;
    int memoryBusWidth;
    int totalGlobalMem;
    int sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockInstructionRate;
    int concurrentKernels;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int integrated;
    int canMapHostMemory;
    int maxTexture1D;
    int maxTexture2D[2];
    int maxTexture3D[3];
    char pad[2048];
};
using hipGetDevicePropertiesR0600_fn = int (WINAPI*)(hipDeviceProp_t*, int);
#endif

static FILE* g_v5_log = nullptr;

static uint16_t V5ReadU16LE(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

static uint32_t V5ReadU32LE(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static uint64_t V5ReadU64LE(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

static size_t V5StrnLen(const char* s, size_t maxLen) {
    if (s == nullptr) {
        return 0;
    }

    size_t len = 0;
    while (len < maxLen && s[len] != '\0') {
        ++len;
    }
    return len;
}

static void V5OpenLog(void) {
    if (g_v5_log != nullptr) {
        return;
    }

    fopen_s(&g_v5_log, V5_LOG_FILE_NAME, "w");
    if (g_v5_log == nullptr) {
        g_v5_log = stderr;
    }
}

static void V5CloseLog(void) {
    if (g_v5_log != nullptr && g_v5_log != stderr) {
        fclose(g_v5_log);
    }
    g_v5_log = nullptr;
}

static void V5Log(const char* tag, const char* fmt, ...) {
    V5OpenLog();

    char buffer[2048] = {};
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    fprintf(g_v5_log, "%s %s\n", tag, buffer);
    fflush(g_v5_log);
}

static const char* V5HipStatusString(int rc) {
    switch (rc) {
    case 0:
        return "hipSuccess";
    case 1:
        return "hipErrorInvalidValue";
    case 2:
        return "hipErrorNotInitialized";
    case 3:
        return "hipErrorNoDevice";
    case 11:
        return "hipErrorUnknown";
    default:
        return "hipUnknown";
    }
}

static bool V5FindVersionV4Module(HMODULE* outBase, SIZE_T* outSize) {
    if (outBase == nullptr || outSize == nullptr) {
        return false;
    }

    HMODULE modules[256] = {};
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed)) {
        V5Log("[DLSSNR-V5] ERROR", "EnumProcessModules failed");
        return false;
    }

    const DWORD count = needed / sizeof(HMODULE);
    for (DWORD i = 0; i < count; ++i) {
        wchar_t moduleName[MAX_PATH] = {};
        if (GetModuleBaseNameW(GetCurrentProcess(), modules[i], moduleName, MAX_PATH) == 0) {
            continue;
        }

        std::wstring name(moduleName);
        std::transform(name.begin(), name.end(), name.begin(), [](wchar_t ch) {
            return static_cast<wchar_t>(towlower(ch));
        });

        if (name == L"version.dll") {
            MODULEINFO info = {};
            if (GetModuleInformation(GetCurrentProcess(), modules[i], &info, sizeof(info))) {
                *outBase = modules[i];
                *outSize = info.SizeOfImage;
                V5Log("[DLSSNR-V5] V4", "base=0x%p size=0x%zx", *outBase, *outSize);
                return true;
            }
        }
    }

    V5Log("[DLSSNR-V5] ERROR", "V4_NOT_LOADED");
    return false;
}

static bool V5ValidatePeImage(HMODULE module) {
    if (module == nullptr) {
        return false;
    }

    auto* dos = reinterpret_cast<PIMAGE_DOS_HEADER>(module);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
        V5Log("[DLSSNR-V5] ERROR", "PE DOS signature invalid");
        return false;
    }

    auto* nt = reinterpret_cast<PIMAGE_NT_HEADERS64>(reinterpret_cast<uint8_t*>(module) + dos->e_lfanew);
    if (dos->e_lfanew == 0 || nt->Signature != IMAGE_NT_SIGNATURE) {
        V5Log("[DLSSNR-V5] ERROR", "PE NT headers invalid");
        return false;
    }

    if (nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64) {
        V5Log("[DLSSNR-V5] ERROR", "Image is not AMD64");
        return false;
    }

    V5Log("[DLSSNR-V5] PE", "ImageBase=0x%llx EntryPointRVA=0x%x", nt->OptionalHeader.ImageBase, nt->OptionalHeader.AddressOfEntryPoint);
    return true;
}

static void V5EnumerateSections(HMODULE module) {
    auto* dos = reinterpret_cast<PIMAGE_DOS_HEADER>(module);
    auto* nt = reinterpret_cast<PIMAGE_NT_HEADERS64>(reinterpret_cast<uint8_t*>(module) + dos->e_lfanew);
    auto* sections = IMAGE_FIRST_SECTION(nt);

    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        auto& section = sections[i];
        char name[IMAGE_SIZEOF_SHORT_NAME + 1] = {};
        memcpy(name, section.Name, IMAGE_SIZEOF_SHORT_NAME);

        const uintptr_t va = reinterpret_cast<uintptr_t>(module) + section.VirtualAddress;
        const size_t vsize = section.Misc.VirtualSize ? section.Misc.VirtualSize : section.SizeOfRawData;
        const uintptr_t raw = section.PointerToRawData;
        const size_t rawSize = section.SizeOfRawData;
        const uintptr_t sectionStart = reinterpret_cast<uintptr_t>(module) + section.VirtualAddress;
        const uintptr_t sectionEnd = sectionStart + vsize;

        V5Log("[DLSSNR-V5] SECTION", "name=%s rva=0x%llx vsize=0x%zx raw=0x%llx rawsize=0x%zx", name,
              static_cast<unsigned long long>(section.VirtualAddress), vsize,
              static_cast<unsigned long long>(raw), rawSize);

        if (strcmp(name, ".hip_fat") == 0 || strcmp(name, ".hipFatB") == 0) {
            V5Log("[DLSSNR-V5] V4", "section=%s va=0x%llx end=0x%llx", name,
                  static_cast<unsigned long long>(va), static_cast<unsigned long long>(sectionEnd));
        }
    }
}

static bool V5FindTargetNameInBundle(const uint8_t* start, size_t length, const char* want, std::vector<V5BundleTarget>* targets) {
    if (start == nullptr || targets == nullptr) {
        return false;
    }

    const uint8_t* limit = start + length;
    for (const uint8_t* p = start; p + 1 < limit; ) {
        const auto* s = reinterpret_cast<const char*>(p);
        size_t len = 0;
        while (p + len < limit && len < 256 && s[len] != '\0') {
            ++len;
        }

        if (len > 3) {
            std::string candidate(s, len);
            if (candidate.find("x86_64") != std::string::npos ||
                candidate.find("amdgcn") != std::string::npos ||
                candidate.find("gfx") != std::string::npos ||
                candidate.find("host") != std::string::npos) {
                targets->push_back({ candidate, static_cast<uintptr_t>(p - start), len });
                if (want != nullptr && candidate.find(want) != std::string::npos) {
                    return true;
                }
            }
        }

        p += std::max<size_t>(len + 1, 1);
        if (p >= limit) {
            break;
        }
    }

    return false;
}

static bool V5LocateOffloadBundle(HMODULE module, V5BundleInfo* outBundle)
{
    if (module == nullptr || outBundle == nullptr) {
        return false;
    }

    auto* dos = reinterpret_cast<PIMAGE_DOS_HEADER>(module);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
        V5Log("[DLSSNR-V5] ERROR", "DOS_HEADER_INVALID");
        return false;
    }

    auto* nt = reinterpret_cast<PIMAGE_NT_HEADERS64>(
        reinterpret_cast<uint8_t*>(module) + dos->e_lfanew);

    if (nt->Signature != IMAGE_NT_SIGNATURE) {
        V5Log("[DLSSNR-V5] ERROR", "NT_HEADER_INVALID");
        return false;
    }

    auto* sections = IMAGE_FIRST_SECTION(nt);

    const uint8_t* bundleMarker = nullptr;
    const uint8_t* bundleSectionStart = nullptr;
    size_t bundleSectionLen = 0;

    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        auto& section = sections[i];

        char name[IMAGE_SIZEOF_SHORT_NAME + 1] = {};
        memcpy(name, section.Name, IMAGE_SIZEOF_SHORT_NAME);

        if (strcmp(name, ".hip_fat") != 0 &&
            strcmp(name, ".hipFatB") != 0) {
            continue;
        }

        const uintptr_t secStartVa =
            reinterpret_cast<uintptr_t>(module) + section.VirtualAddress;

        const size_t secLen =
            section.Misc.VirtualSize ?
                section.Misc.VirtualSize :
                section.SizeOfRawData;

        const uint8_t* secStart =
            reinterpret_cast<const uint8_t*>(secStartVa);

        const uint8_t* secEnd = secStart + secLen;

        const size_t markerLen = strlen(V5_OFFLOAD_MARKER);

        for (const uint8_t* p = secStart;
             p + markerLen <= secEnd;
             ++p) {

            if (memcmp(p, V5_OFFLOAD_MARKER, markerLen) == 0) {
                bundleMarker = p;
                bundleSectionStart = secStart;
                bundleSectionLen = secLen;
                break;
            }
        }

        if (bundleMarker != nullptr) {
            break;
        }
    }

    if (bundleMarker == nullptr) {
        V5Log("[DLSSNR-V5] ERROR", "BUNDLE_NOT_FOUND");
        return false;
    }

    outBundle->bundleStart = bundleMarker;
    outBundle->bundleSize =
        bundleSectionLen -
        static_cast<size_t>(bundleMarker - bundleSectionStart);

    const size_t bundleSize = outBundle->bundleSize;

    V5Log("[DLSSNR-V5] BUNDLE",
          "START=0x%p SIZE=0x%zx",
          outBundle->bundleStart,
          bundleSize);

    const size_t magicLen = strlen(V5_OFFLOAD_MARKER);

    // Binary Clang offload-bundle format:
    //
    // 24 bytes  magic
    //  8 bytes  number of entries
    //
    // Per entry:
    //  8 bytes  code-object offset
    //  8 bytes  code-object size
    //  8 bytes  ID length
    //  N bytes  ID
    //
    // All integers are little-endian.
    if (bundleSize < magicLen + 8) {
        V5Log("[DLSSNR-V5] ERROR", "BUNDLE_HEADER_TOO_SMALL");
        return false;
    }

    const uint8_t* cursor = bundleMarker + magicLen;
    const uint8_t* bundleEnd = bundleMarker + bundleSize;

    const uint64_t entryCount = V5ReadU64LE(cursor);
    cursor += 8;

    if (entryCount == 0 || entryCount > 1024) {
        V5Log("[DLSSNR-V5] ERROR",
              "BUNDLE_ENTRY_COUNT_INVALID count=%llu",
              static_cast<unsigned long long>(entryCount));
        return false;
    }

    V5Log("[DLSSNR-V5] BUNDLE",
          "ENTRY_COUNT=%llu",
          static_cast<unsigned long long>(entryCount));

    std::vector<V5BundleTarget> foundTargets;
    foundTargets.reserve(static_cast<size_t>(entryCount));

    const V5BundleTarget* selectedTarget = nullptr;
    uintptr_t selectedOffset = 0;
    size_t selectedSize = 0;

    for (uint64_t i = 0; i < entryCount; ++i) {
        const size_t remaining =
            static_cast<size_t>(bundleEnd - cursor);

        if (remaining < 24) {
            V5Log("[DLSSNR-V5] ERROR",
                  "BUNDLE_ENTRY_HEADER_TRUNCATED index=%llu",
                  static_cast<unsigned long long>(i));
            return false;
        }

        const uint64_t offset = V5ReadU64LE(cursor);
        cursor += 8;

        const uint64_t size = V5ReadU64LE(cursor);
        cursor += 8;

        const uint64_t idLength = V5ReadU64LE(cursor);
        cursor += 8;

        if (idLength > static_cast<uint64_t>(bundleEnd - cursor)) {
            V5Log("[DLSSNR-V5] ERROR",
                  "BUNDLE_ID_TRUNCATED index=%llu idLength=%llu",
                  static_cast<unsigned long long>(i),
                  static_cast<unsigned long long>(idLength));
            return false;
        }

        std::string id(
            reinterpret_cast<const char*>(cursor),
            static_cast<size_t>(idLength));

        cursor += static_cast<size_t>(idLength);

        if (offset > bundleSize ||
            size > bundleSize ||
            offset + size > bundleSize) {

            V5Log("[DLSSNR-V5] ERROR",
                  "BUNDLE_ENTRY_BOUNDS index=%llu offset=0x%llx size=0x%llx",
                  static_cast<unsigned long long>(i),
                  static_cast<unsigned long long>(offset),
                  static_cast<unsigned long long>(size));

            return false;
        }

        foundTargets.push_back({
            id,
            static_cast<uintptr_t>(offset),
            static_cast<size_t>(size)
        });

        V5Log("[DLSSNR-V5] TARGET",
              "INDEX=%llu OFFSET=0x%llx SIZE=0x%llx ID=%s",
              static_cast<unsigned long long>(i),
              static_cast<unsigned long long>(offset),
              static_cast<unsigned long long>(size),
              id.c_str());

        if (id == V5_TARGET_LITERAL) {
            selectedTarget = &foundTargets.back();
            selectedOffset = static_cast<uintptr_t>(offset);
            selectedSize = static_cast<size_t>(size);
        }
    }

    outBundle->targets = foundTargets;
    outBundle->targetCount = foundTargets.size();

    if (selectedTarget == nullptr) {
        V5Log("[DLSSNR-V5] ERROR",
              "GFX1032_TARGET_NOT_FOUND");
        return false;
    }

    const uintptr_t payloadOffset = selectedOffset;
    const size_t payloadSize = selectedSize;

    const uintptr_t payloadStartVa =
        reinterpret_cast<uintptr_t>(bundleMarker) + payloadOffset;

    const uintptr_t payloadEndVa =
        payloadStartVa + payloadSize;

    V5Log("[DLSSNR-V5] ELF",
          "BUNDLE_REL_START=0x%llx SIZE=0x%zx",
          static_cast<unsigned long long>(payloadOffset),
          payloadSize);

    V5Log("[DLSSNR-V5] ELF",
          "START=0x%llx SIZE=0x%zx END=0x%llx",
          static_cast<unsigned long long>(payloadStartVa),
          payloadSize,
          static_cast<unsigned long long>(payloadEndVa));

    // The bundle lives inside the loaded PE section. Verify the
    // selected code object stays inside that section.
    bool isInRange = false;

    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        auto& section = sections[i];

        const uintptr_t secStart =
            reinterpret_cast<uintptr_t>(module) +
            section.VirtualAddress;

        const uintptr_t secEnd =
            secStart +
            (section.Misc.VirtualSize ?
                section.Misc.VirtualSize :
                section.SizeOfRawData);

        if (payloadStartVa >= secStart &&
            payloadEndVa <= secEnd) {

            isInRange = true;
            break;
        }
    }

    if (!isInRange) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_BOUNDS");
        return false;
    }

    outBundle->elfPointer =
        reinterpret_cast<const uint8_t*>(payloadStartVa);

    outBundle->elfSize = payloadSize;

    outBundle->elfVa = payloadStartVa;

    // This is intentionally the RVA of the code object from the
    // PE image, not merely the offset relative to the bundle.
    outBundle->elfRva =
        payloadStartVa - reinterpret_cast<uintptr_t>(module);

    outBundle->headerSize =
        static_cast<size_t>(cursor - bundleMarker);

    V5Log("[DLSSNR-V5] BUNDLE",
          "PARSED_HEADER_SIZE=0x%zx",
          outBundle->headerSize);

    return true;
}
static bool V5ValidateElf64Amdgpu(const void* ptr, size_t size) {
    if (ptr == nullptr || size < 64) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_INVALID size=%zu", size);
        return false;
    }

    const uint8_t* p = reinterpret_cast<const uint8_t*>(ptr);
    if (memcmp(p, V5_ELF_MAGIC, 4) != 0) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_MAGIC=FAIL");
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "MAGIC=PASS");

    if (p[4] != 2) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_CLASS=FAIL");
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "CLASS=ELF64");

    if (p[5] != 1) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_ENDIAN=FAIL");
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "ENDIAN=LITTLE");

    const uint16_t e_type = V5ReadU16LE(p + 16);
    const uint16_t e_machine = V5ReadU16LE(p + 18);
    const uint64_t e_phoff = V5ReadU64LE(p + 32);
    const uint64_t e_shoff = V5ReadU64LE(p + 40);
    const uint16_t e_phentsize = V5ReadU16LE(p + 54);
    const uint16_t e_phnum = V5ReadU16LE(p + 56);
    const uint16_t e_shentsize = V5ReadU16LE(p + 58);
    const uint16_t e_shnum = V5ReadU16LE(p + 60);
    const uint16_t e_shstrndx = V5ReadU16LE(p + 62);

    if (e_machine != 0x00E0) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_MACHINE=FAIL value=0x%x", e_machine);
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "MACHINE=AMDGPU");

    if (e_type != 3) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_TYPE=FAIL value=0x%x", e_type);
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "TYPE=ET_DYN");

    if (e_phoff == 0 || e_phnum == 0 || e_phoff + static_cast<uint64_t>(e_phentsize) * e_phnum > size) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_PHDRS=FAIL phoff=0x%llx phnum=%u phentsize=%u size=%zu", static_cast<unsigned long long>(e_phoff), e_phnum, e_phentsize, size);
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "PHDRS=%u", e_phnum);

    if (e_shoff == 0 || e_shnum == 0 || e_shoff + static_cast<uint64_t>(e_shentsize) * e_shnum > size) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_SHDRS=FAIL shoff=0x%llx shnum=%u shentsize=%u size=%zu", static_cast<unsigned long long>(e_shoff), e_shnum, e_shentsize, size);
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "SHDRS=%u", e_shnum);

    if (e_shstrndx >= e_shnum) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_SHSTRIDX_INVALID idx=%u shnum=%u", e_shstrndx, e_shnum);
        return false;
    }

    const uint8_t* shstrHeader = p + e_shoff + e_shstrndx * e_shentsize;
    const uint64_t shstrOff = V5ReadU64LE(shstrHeader + 24);
    const uint64_t shstrSize = V5ReadU64LE(shstrHeader + 32);
    if (shstrOff + shstrSize > size) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_SHSTR_BOUNDS");
        return false;
    }

    const uint8_t* shstrData = p + shstrOff;
    bool foundText = false;
    uint64_t textOff = 0;
    uint64_t textSize = 0;

    for (uint16_t i = 0; i < e_shnum; ++i) {
        const uint8_t* sh = p + e_shoff + i * e_shentsize;
        const uint32_t nameOff = V5ReadU32LE(sh);
        const uint64_t shOffset = V5ReadU64LE(sh + 24);
        const uint64_t shSize = V5ReadU64LE(sh + 32);
        const char* name = reinterpret_cast<const char*>(shstrData + nameOff);
        if (strcmp(name, ".text") == 0) {
            textOff = shOffset;
            textSize = shSize;
            foundText = true;
            break;
        }
    }

    if (!foundText || textOff + textSize > size) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_TEXT_BOUNDS");
        return false;
    }
    V5Log("[DLSSNR-V5] ELF", "TEXT offset=0x%llx size=0x%llx", static_cast<unsigned long long>(textOff), static_cast<unsigned long long>(textSize));

    return true;
}

static bool V5ResolveHipRuntime(HMODULE* outRuntime, FARPROC* outGetDeviceCount, FARPROC* outGetDeviceProperties,
                               FARPROC* outModuleLoadData, FARPROC* outModuleGetFunction,
                               FARPROC* outGetLastError, FARPROC* outPeekAtLastError) {
    if (outRuntime == nullptr || outGetDeviceCount == nullptr || outGetDeviceProperties == nullptr ||
        outModuleLoadData == nullptr || outModuleGetFunction == nullptr) {
        return false;
    }

    HMODULE hipRuntime = GetModuleHandleA("amdhip64_7.dll");
    if (hipRuntime == nullptr) {
        hipRuntime = LoadLibraryA("amdhip64_7.dll");
    }

    if (hipRuntime == nullptr) {
        V5Log("[DLSSNR-V5] ERROR", "HIP runtime not found: amdhip64_7.dll");
        return false;
    }

    *outRuntime = hipRuntime;
    V5Log("[DLSSNR-V5] HIP", "DLL=%s", "amdhip64_7.dll");

    *outGetDeviceCount = GetProcAddress(hipRuntime, "hipGetDeviceCount");
    *outGetDeviceProperties = GetProcAddress(hipRuntime, "hipGetDevicePropertiesR0600");
    *outModuleLoadData = GetProcAddress(hipRuntime, "hipModuleLoadData");
    *outModuleGetFunction = GetProcAddress(hipRuntime, "hipModuleGetFunction");
    *outGetLastError = GetProcAddress(hipRuntime, "hipGetLastError");
    *outPeekAtLastError = GetProcAddress(hipRuntime, "hipPeekAtLastError");

    V5Log("[DLSSNR-V5] HIP", "hipGetDeviceCount=%s", *outGetDeviceCount ? "OK" : "FAIL");
    V5Log("[DLSSNR-V5] HIP", "hipGetDeviceProperties=%s", *outGetDeviceProperties ? "OK" : "FAIL");
    V5Log("[DLSSNR-V5] HIP", "hipModuleLoadData=%s", *outModuleLoadData ? "OK" : "FAIL");
    V5Log("[DLSSNR-V5] HIP", "hipModuleGetFunction=%s", *outModuleGetFunction ? "OK" : "FAIL");
    if (*outGetDeviceCount == nullptr || *outGetDeviceProperties == nullptr ||
        *outModuleLoadData == nullptr || *outModuleGetFunction == nullptr) {
        V5Log("[DLSSNR-V5] ERROR", "HIP_API_MISSING");
        return false;
    }

    return true;
}

static void V5QueryDevices(hipGetDeviceCount_fn hipGetDeviceCount, hipGetDevicePropertiesR0600_fn hipGetDevicePropertiesR0600) {
    if (hipGetDeviceCount == nullptr || hipGetDevicePropertiesR0600 == nullptr) {
        return;
    }

    int deviceCount = 0;
    const int rc = hipGetDeviceCount(&deviceCount);
    V5Log("[DLSSNR-V5] DEVICE", "COUNT=%d rc=%d (%s)", deviceCount, rc, V5HipStatusString(rc));
    if (rc != 0 || deviceCount <= 0) {
        V5Log("[DLSSNR-V5] ERROR", "DEVICE_QUERY_FAILED");
        return;
    }

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t props = {};
        const int deviceRc = hipGetDevicePropertiesR0600(&props, i);
        if (deviceRc != 0) {
            V5Log("[DLSSNR-V5] DEVICE", "index=%d rc=%d (%s)", i, deviceRc, V5HipStatusString(deviceRc));
            continue;
        }

        V5Log("[DLSSNR-V5] DEVICE", "index=%d name=%s gcnArchName=%s major=%d minor=%d mpCount=%d wavefrontSize=%d",
              i, props.name, props.gcnArchName, props.major, props.minor,
              props.multiProcessorCount, props.warpSize);
    }
}

static void V5DoModuleLoadExperiment(HMODULE hipRuntime, FARPROC moduleLoadProc, FARPROC moduleGetFunctionProc,
                                    FARPROC getLastErrorProc, FARPROC peekAtLastErrorProc,
                                    const void* elfPointer, size_t elfSize) {
    if (hipRuntime == nullptr || moduleLoadProc == nullptr || moduleGetFunctionProc == nullptr) {
        V5Log("[DLSSNR-V5] ERROR", "MODULE_LOAD_SKIPPED");
        return;
    }

    auto* hipModuleLoadData = reinterpret_cast<hipModuleLoadData_fn>(moduleLoadProc);
    auto* hipModuleGetFunction = reinterpret_cast<hipModuleGetFunction_fn>(moduleGetFunctionProc);
    auto* hipModuleLaunchKernel = reinterpret_cast<hipModuleLaunchKernel_fn>(
        GetProcAddress(hipRuntime, "hipModuleLaunchKernel"));
    auto* hipDeviceSynchronize = reinterpret_cast<hipDeviceSynchronize_fn>(
        GetProcAddress(hipRuntime, "hipDeviceSynchronize"));
    auto* hipGetLastError = reinterpret_cast<hipGetLastError_fn>(getLastErrorProc);
    auto* hipPeekAtLastError = reinterpret_cast<hipPeekAtLastError_fn>(peekAtLastErrorProc);

    V5Log("[DLSSNR-V5] MODULE", "LOAD BEGIN PTR=0x%p SIZE=0x%zx", elfPointer, elfSize);

    hipModule_t module = nullptr;
    const int loadRc = hipModuleLoadData(&module, elfPointer);
    V5Log("[DLSSNR-V5] MODULE", "LOAD RESULT=%d STATUS=%s", loadRc, V5HipStatusString(loadRc));

    if (hipGetLastError != nullptr) {
        const int lastErrRc = hipGetLastError();
        V5Log("[DLSSNR-V5] MODULE", "LAST_ERROR=%d (%s)", lastErrRc, V5HipStatusString(lastErrRc));
    }

    if (hipPeekAtLastError != nullptr) {
        const int peekRc = hipPeekAtLastError();
        V5Log("[DLSSNR-V5] MODULE", "PEEK_LAST_ERROR=%d (%s)", peekRc, V5HipStatusString(peekRc));
    }

    if (loadRc == 0 && module != nullptr) {
        V5Log("[DLSSNR-V5] RESULT", "MODULE_LOAD_SUCCESS");

        // Diagnostic only: resolve several known gfx1032 kernels, but do not launch them.
        const char* testKernels[] = {
            "_Z6k_ffwd10FfwdParams",
            "_Z7k_ffwd211Ffwd2Params",
            "_Z5k_qkv9QkvParams",
            "_Z6k_qkv29QkvParams",
            "_Z6k_mean10MeanParams"
        };

        bool allKernelLookupsSucceeded = true;

        for (const char* testKernel : testKernels) {
            hipFunction_t testFunction = nullptr;
            const int funcRc = hipModuleGetFunction(&testFunction, module, testKernel);

            V5Log("[DLSSNR-V5] KERNEL",
                  "LOOKUP NAME=%s RESULT=%d STATUS=%s PTR=0x%p",
                  testKernel, funcRc, V5HipStatusString(funcRc), testFunction);

            if (funcRc != 0 || testFunction == nullptr) {
                allKernelLookupsSucceeded = false;
            }
        // First execution probe: k_align_probe is a 4-byte s_endpgm kernel.
        hipFunction_t probeFunction = nullptr;
        const int probeLookupRc = hipModuleGetFunction(
            &probeFunction, module, "_Z13k_align_probePh");

        V5Log("[DLSSNR-V5] PROBE",
              "LOOKUP RESULT=%d STATUS=%s PTR=0x%p",
              probeLookupRc, V5HipStatusString(probeLookupRc), probeFunction);

        if (probeLookupRc == 0 && probeFunction != nullptr &&
            hipModuleLaunchKernel != nullptr && hipDeviceSynchronize != nullptr) {

            void* nullArg = nullptr;
            void* kernelArgs[] = { &nullArg };

            V5Log("[DLSSNR-V5] PROBE", "LAUNCH BEGIN");

            const int launchRc = hipModuleLaunchKernel(
                probeFunction,
                1, 1, 1,
                1, 1, 1,
                0,
                nullptr,
                kernelArgs,
                nullptr);

            V5Log("[DLSSNR-V5] PROBE",
                  "LAUNCH RESULT=%d STATUS=%s",
                  launchRc, V5HipStatusString(launchRc));

            const int syncRc = hipDeviceSynchronize();

            V5Log("[DLSSNR-V5] PROBE",
                  "SYNC RESULT=%d STATUS=%s",
                  syncRc, V5HipStatusString(syncRc));

            if (launchRc == 0 && syncRc == 0) {
                V5Log("[DLSSNR-V5] RESULT", "KERNEL_EXECUTION_SUCCESS");
            } else {
                V5Log("[DLSSNR-V5] RESULT", "KERNEL_EXECUTION_FAILED");
            }
        } else {
            V5Log("[DLSSNR-V5] RESULT", "KERNEL_EXECUTION_SKIPPED");
        }
        }

        if (allKernelLookupsSucceeded) {
            V5Log("[DLSSNR-V5] RESULT", "ALL_KERNEL_LOOKUPS_SUCCESS");
        } else {
            V5Log("[DLSSNR-V5] RESULT", "ONE_OR_MORE_KERNEL_LOOKUPS_FAILED");
        }
        // Safe cleanup only when a module was successfully loaded.
        auto* hipModuleUnload = reinterpret_cast<hipModuleUnload_fn>(GetProcAddress(hipRuntime, "hipModuleUnload"));
        if (hipModuleUnload != nullptr) {
            const int unloadRc = hipModuleUnload(module);
            V5Log("[DLSSNR-V5] MODULE", "UNLOAD RESULT=%d STATUS=%s", unloadRc, V5HipStatusString(unloadRc));
        }
        return;
    }

    V5Log("[DLSSNR-V5] RESULT", "MODULE_LOAD_FAILED code=%d", loadRc);
}

static DWORD WINAPI V5WorkerThread(LPVOID) {
    V5Log("[DLSSNR-V5] INIT", "worker thread started");

    HMODULE v4Module = nullptr;
    SIZE_T v4Size = 0;
    if (!V5FindVersionV4Module(&v4Module, &v4Size)) {
        return 0;
    }

    if (!V5ValidatePeImage(v4Module)) {
        V5Log("[DLSSNR-V5] ERROR", "V4_PE_INVALID");
        return 0;
    }

    V5EnumerateSections(v4Module);

    V5BundleInfo bundle = {};
    if (!V5LocateOffloadBundle(v4Module, &bundle)) {
        V5Log("[DLSSNR-V5] ERROR", "OFFLOAD_BUNDLE_NOT_FOUND");
        return 0;
    }

    if (bundle.elfPointer == nullptr || bundle.elfSize == 0) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_BOUNDS");
        return 0;
    }

    if (!V5ValidateElf64Amdgpu(bundle.elfPointer, bundle.elfSize)) {
        V5Log("[DLSSNR-V5] ERROR", "ELF_INVALID");
        return 0;
    }

    HMODULE hipRuntime = nullptr;
    FARPROC getDeviceCountProc = nullptr;
    FARPROC getDevicePropsProc = nullptr;
    FARPROC moduleLoadDataProc = nullptr;
    FARPROC moduleGetFunctionProc = nullptr;
    FARPROC getLastErrorProc = nullptr;
    FARPROC peekLastErrorProc = nullptr;

    if (!V5ResolveHipRuntime(&hipRuntime, &getDeviceCountProc, &getDevicePropsProc,
                             &moduleLoadDataProc, &moduleGetFunctionProc,
                             &getLastErrorProc, &peekLastErrorProc)) {
        return 0;
    }

    auto* hipGetDeviceCount = reinterpret_cast<hipGetDeviceCount_fn>(getDeviceCountProc);
    auto* hipGetDevicePropertiesR0600 = reinterpret_cast<hipGetDevicePropertiesR0600_fn>(getDevicePropsProc);
    V5QueryDevices(hipGetDeviceCount, hipGetDevicePropertiesR0600);

    V5DoModuleLoadExperiment(hipRuntime, moduleLoadDataProc, moduleGetFunctionProc,
                             getLastErrorProc, peekLastErrorProc,
                             bundle.elfPointer, bundle.elfSize);

    V5Log("[DLSSNR-V5] RESULT", "DIAGNOSTIC_COMPLETE");
    return 0;
}

extern "C" BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    UNREFERENCED_PARAMETER(hinstDLL);
    UNREFERENCED_PARAMETER(lpvReserved);

    if (fdwReason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinstDLL);
        V5OpenLog();
        V5Log("[DLSSNR-V5] INIT", "module loaded");

        HANDLE worker = CreateThread(nullptr, 0, V5WorkerThread, nullptr, 0, nullptr);
        if (worker != nullptr) {
            CloseHandle(worker);
        }
    }

    return TRUE;
}

extern "C" __declspec(dllexport) void V5DiagnosticInitialize(void) {
    V5OpenLog();
    V5Log("[DLSSNR-V5] INIT", "manual initialization requested");

    HMODULE v4Module = nullptr;
    SIZE_T v4Size = 0;
    if (!V5FindVersionV4Module(&v4Module, &v4Size)) {
        return;
    }

    if (!V5ValidatePeImage(v4Module)) {
        return;
    }

    V5EnumerateSections(v4Module);

    V5BundleInfo bundle = {};
    if (!V5LocateOffloadBundle(v4Module, &bundle)) {
        return;
    }

    if (!V5ValidateElf64Amdgpu(bundle.elfPointer, bundle.elfSize)) {
        return;
    }

    HMODULE hipRuntime = nullptr;
    FARPROC getDeviceCountProc = nullptr;
    FARPROC getDevicePropsProc = nullptr;
    FARPROC moduleLoadDataProc = nullptr;
    FARPROC moduleGetFunctionProc = nullptr;
    FARPROC getLastErrorProc = nullptr;
    FARPROC peekLastErrorProc = nullptr;
    if (!V5ResolveHipRuntime(&hipRuntime, &getDeviceCountProc, &getDevicePropsProc,
                             &moduleLoadDataProc, &moduleGetFunctionProc,
                             &getLastErrorProc, &peekLastErrorProc)) {
        return;
    }

    auto* hipGetDeviceCount = reinterpret_cast<hipGetDeviceCount_fn>(getDeviceCountProc);
    auto* hipGetDevicePropertiesR0600 = reinterpret_cast<hipGetDevicePropertiesR0600_fn>(getDevicePropsProc);
    V5QueryDevices(hipGetDeviceCount, hipGetDevicePropertiesR0600);
    V5DoModuleLoadExperiment(hipRuntime, moduleLoadDataProc, moduleGetFunctionProc,
                             getLastErrorProc, peekLastErrorProc,
                             bundle.elfPointer, bundle.elfSize);
}
