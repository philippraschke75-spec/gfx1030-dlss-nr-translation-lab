// Offline benchmark for the vendor-neutral DLSS-NR port (Optional\nvngx.dll_dlssnr.dll).
// usage: nrbench <port.dll> <nvngx_dlssnr.dll> [width height frames out.bin]
// The port reads nr_port.ini from its own folder and writes nr_port.log there.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using PFN_Create = void*(__cdecl*)(const wchar_t*, const wchar_t*, ID3D12Device*, ID3D12GraphicsCommandList*,
                                   void*, unsigned, unsigned, int, float, int, float, float, float, int, int);
using PFN_Eval = int(__cdecl*)(ID3D12GraphicsCommandList*, void*, void*, ID3D12Resource*, ID3D12Resource*,
                               ID3D12Resource*, ID3D12Resource*, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, unsigned, unsigned, int, int, float, int, float, float,
                               float, int, float, float);
using PFN_Err = const char*(__cdecl*)();
using PFN_Rel = void(__cdecl*)(void*);

#define CK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { printf("FAIL %s = 0x%08lx (line %d)\n", #x, hr_, __LINE__); return 1; } } while (0)

static ID3D12Device* dev;

// --- per-shader profiling (NR_BENCH_PROFILE=1): patch the device/command-list vtables so every Dispatch
// is wrapped in timestamps and attributed to the compute shader its pipeline was built from.
#include <map>
#include <string>
static std::map<void*, std::string> g_psoName;      // pipeline -> shader label
static void* g_curPso;
static ID3D12QueryHeap* g_profHeap;
static UINT g_profN, g_profCap = 8192;
static std::vector<std::string> g_profLabel;        // label per timestamp pair
static bool g_profOn;

using PFN_CreateCPSO = HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, REFIID, void**);
using PFN_Dispatch = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
using PFN_SetPSO = void(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);
static PFN_CreateCPSO o_CreateCPSO; static PFN_Dispatch o_Dispatch; static PFN_SetPSO o_SetPSO;

// Shader label = its entry name found in the bytecode (Linear, Attn, ...) plus size, which tells the Linear variants apart.
static std::string labelFor(const void* code, size_t size) {
    static const char* names[] = {"AttnGlobal", "AttnH", "Attn", "Compose", "CopyRaw", "Copy", "Grade", "InColor", "InMv",
                                  "LinEpi", "Linear", "OutFin", "Pool", "PostIn", "PreFeat", "UpAdd"};
    std::string s((const char*)code, size);
    for (const char* n : names) {
        size_t at = s.find(std::string(n) + '\0');
        if (at != std::string::npos) return std::string(n) + "#" + std::to_string(size);
    }
    return "other#" + std::to_string(size);  // DirectML's own shaders land here
}
static HRESULT STDMETHODCALLTYPE h_CreateCPSO(ID3D12Device* d, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID riid, void** out) {
    HRESULT hr = o_CreateCPSO(d, desc, riid, out);
    if (SUCCEEDED(hr) && out && *out) g_psoName[*out] = labelFor(desc->CS.pShaderBytecode, desc->CS.BytecodeLength);
    return hr;
}
static void STDMETHODCALLTYPE h_SetPSO(ID3D12GraphicsCommandList* c, ID3D12PipelineState* p) { g_curPso = p; o_SetPSO(c, p); }
static void STDMETHODCALLTYPE h_Dispatch(ID3D12GraphicsCommandList* c, UINT x, UINT y, UINT z) {
    if (!g_profOn || g_profN + 2 > g_profCap) { o_Dispatch(c, x, y, z); return; }
    c->EndQuery(g_profHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_profN);
    o_Dispatch(c, x, y, z);
    c->EndQuery(g_profHeap, D3D12_QUERY_TYPE_TIMESTAMP, g_profN + 1);
    auto it = g_psoName.find(g_curPso);
    g_profLabel.push_back((it != g_psoName.end() ? it->second : "unknown-pso") + "," + std::to_string(x) + "," +
                          std::to_string(y) + "," + std::to_string(z));
    g_profN += 2;
}
static void patch(void* obj, int idx, void* hook, void** orig) {
    void** vt = *(void***)obj; DWORD old;
    VirtualProtect(&vt[idx], sizeof(void*), PAGE_READWRITE, &old);
    *orig = vt[idx]; vt[idx] = hook;
    VirtualProtect(&vt[idx], sizeof(void*), old, &old);
}
static ID3D12Resource* tex(DXGI_FORMAT f, UINT w, UINT h) {
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC d{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, w, h, 1, 1, f, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r));
    return r;
}
static ID3D12Resource* buf(D3D12_HEAP_TYPE t, UINT64 size) {
    D3D12_HEAP_PROPERTIES hp{t};
    D3D12_RESOURCE_DESC d{D3D12_RESOURCE_DIMENSION_BUFFER, 0, size, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                          D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                 t == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                                 nullptr, IID_PPV_ARGS(&r));
    return r;
}

// float -> fp16, truncating; good enough for synthetic test content.
static unsigned short h(float f) {
    unsigned u; memcpy(&u, &f, 4);
    int e = ((u >> 23) & 0xff) - 127 + 15; unsigned m = (u >> 13) & 0x3ff;
    return e <= 0 ? 0 : (unsigned short)(((u >> 16) & 0x8000) | (e << 10) | m);
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: nrbench <port.dll> <nvngx_dlssnr.dll> [w h frames out.bin]\n"); return 2; }
    UINT W = argc > 4 ? atoi(argv[3]) : 1920, H = argc > 4 ? atoi(argv[4]) : 1080;
    int frames = argc > 5 ? atoi(argv[5]) : 60;
    const char* outPath = argc > 6 ? argv[6] : nullptr;
    wchar_t snippet[MAX_PATH]; MultiByteToWideChar(CP_ACP, 0, argv[2], -1, snippet, MAX_PATH);

    IDXGIFactory6* fac; CK(CreateDXGIFactory2(0, IID_PPV_ARGS(&fac)));
    IDXGIAdapter1* ad; CK(fac->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&ad)));
    DXGI_ADAPTER_DESC1 ades; ad->GetDesc1(&ades); printf("GPU: %ls\n", ades.Description);
    CK(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));

    HMODULE port = LoadLibraryA(argv[1]);
    if (!port) { printf("LoadLibrary port failed %lu\n", GetLastError()); return 1; }
    auto create = (PFN_Create)GetProcAddress(port, "dlssnr_call_create");
    auto eval = (PFN_Eval)GetProcAddress(port, "dlssnr_call_evaluate_v2");
    auto err = (PFN_Err)GetProcAddress(port, "dlssnr_call_error");
    auto rel = (PFN_Rel)GetProcAddress(port, "dlssnr_call_release");
    if (!create || !eval || !GetProcAddress(port, "dlssnr_backend_id")) { printf("not the port DLL\n"); return 1; }

    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    ID3D12CommandQueue* q; CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ID3D12CommandAllocator* ca; CK(dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&ca)));
    ID3D12GraphicsCommandList* cl; CK(dev->CreateCommandList(0, qd.Type, ca, nullptr, IID_PPV_ARGS(&cl)));
    ID3D12Fence* fence; CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    UINT64 fv = 0; HANDLE ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    const bool profile = getenv("NR_BENCH_PROFILE") != nullptr;
    if (profile) {  // device slot 11 = CreateComputePipelineState; list slots 14 = Dispatch, 25 = SetPipelineState
        patch(dev, 11, (void*)h_CreateCPSO, (void**)&o_CreateCPSO);
        patch(cl, 14, (void*)h_Dispatch, (void**)&o_Dispatch);
        patch(cl, 25, (void*)h_SetPSO, (void**)&o_SetPSO);
        D3D12_QUERY_HEAP_DESC ph{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, g_profCap};
        CK(dev->CreateQueryHeap(&ph, IID_PPV_ARGS(&g_profHeap)));
    }
    auto submit = [&] {
        cl->Close(); ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l);
        q->Signal(fence, ++fv); fence->SetEventOnCompletion(fv, ev); WaitForSingleObject(ev, INFINITE);
        ca->Reset(); cl->Reset(ca, nullptr);
    };

    ID3D12Resource* color = tex(DXGI_FORMAT_R16G16B16A16_FLOAT, W, H);
    ID3D12Resource* depth = tex(DXGI_FORMAT_R32_FLOAT, W, H);
    ID3D12Resource* mv = tex(DXGI_FORMAT_R16G16_FLOAT, W, H);
    ID3D12Resource* out = tex(DXGI_FORMAT_R16G16B16A16_FLOAT, W, H);

    // Colour: gradient plus a 16px checker so the output is deterministic and comparable across runs.
    UINT64 pitch = ((W * 8 + 255) & ~255ull);
    ID3D12Resource* up = buf(D3D12_HEAP_TYPE_UPLOAD, pitch * H);
    unsigned char* p; up->Map(0, nullptr, (void**)&p);
    for (UINT y = 0; y < H; ++y) for (UINT x = 0; x < W; ++x) {
        unsigned short* px = (unsigned short*)(p + y * pitch + x * 8);
        float c = ((x / 16 + y / 16) & 1) ? 0.15f : 0.0f;
        px[0] = h(float(x) / W * 0.8f + c); px[1] = h(float(y) / H * 0.8f + c); px[2] = h(0.3f + c); px[3] = h(1.f);
    }
    up->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION src{up, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}, dst{color, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    src.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, W, H, 1, (UINT)pitch};
    cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER b{D3D12_RESOURCE_BARRIER_TYPE_TRANSITION};
    b.Transition = {color, 0, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON};
    cl->ResourceBarrier(1, &b);
    submit();

    // The port never reads the capability block, but the pointer must be non-null.
    static void* dummyParams[64] = {};
    LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
    void* feat = create(snippet, L".", dev, cl, dummyParams, W, H, 0, 1.0f, 0, 1.0f, 1.0f, 1.0f, 0, 0);
    submit(); QueryPerformanceCounter(&t1);
    if (!feat) { printf("create failed: %s\n", err ? err() : "?"); return 1; }
    printf("create: %.1f ms\n", (t1.QuadPart - t0.QuadPart) * 1000.0 / f.QuadPart);
    // jb132: our port loads asynchronously (state 5 = sync path ready); wait for it so no pass-through frame is timed
    if (auto st = (int (*)(void*))GetProcAddress(port, "dlssnr_port_state")) {
        for (int i = 0; i < 1800 && st(feat) != 5 && st(feat) != 4; ++i) Sleep(50);
        printf("port state %d\n", st(feat));
        for (int i = 0; i < 10; ++i) { eval(cl, feat, dummyParams, color, depth, mv, out, W, H, W, H, W, H, 0, 0, 0, 0, 0, i == 0, 1.0f, 0, 1.0f, 1.0f, 1.0f, 0, 1.0f, 1.0f); submit(); }   // warm-up
    }

    D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2};
    ID3D12QueryHeap* qheap; CK(dev->CreateQueryHeap(&qh, IID_PPV_ARGS(&qheap)));
    ID3D12Resource* qbuf = buf(D3D12_HEAP_TYPE_READBACK, 16);
    UINT64 tsFreq; q->GetTimestampFrequency(&tsFreq);

    std::vector<double> ms;
    for (int i = 0; i < frames; ++i) {
        cl->EndQuery(qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        int r = eval(cl, feat, dummyParams, color, depth, mv, out, W, H, W, H, W, H, 0, 0, 0, 0,
                     0, i == 0, 1.0f, 0, 1.0f, 1.0f, 1.0f, 0, 1.0f, 1.0f);
        if (r != 1) { printf("evaluate failed (%d): %s\n", r, err ? err() : "?"); return 1; }
        cl->EndQuery(qheap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        cl->ResolveQueryData(qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, qbuf, 0);
        submit();
        UINT64* ts; qbuf->Map(0, nullptr, (void**)&ts);
        ms.push_back((ts[1] - ts[0]) * 1000.0 / tsFreq);
        qbuf->Unmap(0, nullptr);
    }
    std::vector<double> warm(ms.begin() + std::min<size_t>(5, ms.size() - 1), ms.end());
    std::sort(warm.begin(), warm.end());
    double sum = 0; for (double v : warm) sum += v;
    printf("evaluate GPU ms @%ux%u over %zu frames: median %.2f  mean %.2f  min %.2f  max %.2f  (first %.2f)\n",
           W, H, warm.size(), warm[warm.size() / 2], sum / warm.size(), warm.front(), warm.back(), ms[0]);

    if (profile) {  // two profiled frames (vit_every=2 alternates), per-dispatch timestamps, summed per shader
        ID3D12Resource* pbuf = buf(D3D12_HEAP_TYPE_READBACK, g_profCap * 8ull);
        std::map<std::string, std::pair<double, int>> agg;
        double total = 0;
        const int profFrames = 2;
        for (int fr = 0; fr < profFrames; ++fr) {
            g_profN = 0; g_profLabel.clear(); g_profOn = true;
            eval(cl, feat, dummyParams, color, depth, mv, out, W, H, W, H, W, H, 0, 0, 0, 0,
                 0, 0, 1.0f, 0, 1.0f, 1.0f, 1.0f, 0, 1.0f, 1.0f);
            g_profOn = false;
            cl->ResolveQueryData(g_profHeap, D3D12_QUERY_TYPE_TIMESTAMP, 0, g_profN, pbuf, 0);
            submit();
            UINT64* ts; pbuf->Map(0, nullptr, (void**)&ts);
            FILE* csv = fr == 0 ? fopen("profile.csv", "w") : nullptr;
            if (csv) fprintf(csv, "idx,shader,gx,gy,gz,ms\n");
            for (size_t i = 0; i < g_profLabel.size(); ++i) {
                double d = (ts[2 * i + 1] - ts[2 * i]) * 1000.0 / tsFreq;
                if (csv) fprintf(csv, "%zu,%s,%.4f\n", i, g_profLabel[i].c_str(), d);
                std::string key = g_profLabel[i].substr(0, g_profLabel[i].find(','));
                agg[key].first += d / profFrames; agg[key].second++;
                total += d / profFrames;
            }
            if (csv) fclose(csv);
            pbuf->Unmap(0, nullptr);
        }
        std::vector<std::pair<double, std::string>> rows;
        for (auto& [k, v] : agg) rows.push_back({v.first, k + "  x" + std::to_string(v.second / profFrames)});
        std::sort(rows.rbegin(), rows.rend());
        printf("profile (serialised dispatches, avg of %d frames): total %.2f ms\n", profFrames, total);
        for (auto& [ms_, name] : rows) printf("  %8.2f ms  %5.1f%%  %s\n", ms_, 100 * ms_ / total, name.c_str());
    }

    if (outPath) {  // raw fp16 RGBA rows, W*8 bytes each
        ID3D12Resource* rb = buf(D3D12_HEAP_TYPE_READBACK, pitch * H);
        D3D12_TEXTURE_COPY_LOCATION s2{out, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}, d2{rb, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        d2.PlacedFootprint.Footprint = src.PlacedFootprint.Footprint;
        b.Transition = {out, 0, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE};
        cl->ResourceBarrier(1, &b);
        cl->CopyTextureRegion(&d2, 0, 0, 0, &s2, nullptr);
        submit();
        unsigned char* r; rb->Map(0, nullptr, (void**)&r);
        FILE* fo = fopen(outPath, "wb");
        for (UINT y = 0; y < H; ++y) fwrite(r + y * pitch, 1, W * 8, fo);
        fclose(fo); rb->Unmap(0, nullptr);
        printf("wrote %s\n", outPath);
    }
    rel(feat);
    return 0;
}
