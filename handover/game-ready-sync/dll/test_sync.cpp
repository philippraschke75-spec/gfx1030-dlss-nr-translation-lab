// test_sync <color.bin RGBA16F WxH> <dll dir> [frames]
// Plays the host for the sync path: one D3D12 direct queue; per frame ONE command list = upload this frame's colour ->
// evaluate_v2 -> copy `output` to readback (the consumer sits in the same list, after evaluate, as in the game).
// Every frame gets a different input (the base image shifted by 13 px per frame), so a stale or skipped result cannot match.
// Reference: nr_run from the same nr_runtime.dll in a second context fed the same frame sequence.
// Pace mode also passes motion vectors (RG16F, uv units, mvScale = W,H; previous = current + mv, as janblade) and depth (R32F, reversed Z):
// NR_TEST_DX/DY = image shift per frame (default 13, 0); NR_TEST_OBJ=1: a W/3 x H/3 object moves by DX over a static background (disocclusion).
// NR_TEST_GPU_MS (pace mode): a dummy compute load of that many ms per frame on the same queue (game rendering).
// Every GPU wait has a timeout (deadlock -> FAIL, never a hang). Env: NR_TEST_W/H (default 1280x720), NR_TEST_FMT=r11 (R11G11B10 textures) | e5 (R9G9B9E5 colour input, RGBA16F output).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <numeric>
#include <atomic>
#include <thread>
#include <d3dcompiler.h>
#include "dlssnr_iface.h"
#include "r11g10.h"
#define CK(x) do { HRESULT _h = (x); if (FAILED(_h)) { printf("FAIL %s -> 0x%08lx (line %d)\n", #x, (unsigned long)_h, __LINE__); return 10; } } while (0)
static UINT W = 1280, H = 720, MW, MH, UP = 1; static DXGI_FORMAT FMT = DXGI_FORMAT_R16G16B16A16_FLOAT; static UINT BPP = 8;   // W,H frame; MW,MH model (NR_TEST_UP=2: frame = model upsampled 2x, tests the resampling path)
static ID3D12Device* dev; static ID3D12CommandQueue* q; static ID3D12CommandAllocator* al; static ID3D12GraphicsCommandList* cl; static ID3D12Fence* fence; static UINT64 fv; static HANDLE ev;
static double now() { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return c.QuadPart * 1000.0 / f.QuadPart; }
static bool wait(DWORD ms) { q->Signal(fence, ++fv); if (fence->GetCompletedValue() >= fv) return true; fence->SetEventOnCompletion(fv, ev); return WaitForSingleObject(ev, ms) == WAIT_OBJECT_0; }
static ID3D12Resource* buf(D3D12_HEAP_TYPE t, UINT64 sz, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = t; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; ID3D12Resource* r = nullptr; dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r)); return r; }
static ID3D12Resource* tex(D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st, DXGI_FORMAT fmt) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = W; d.Height = H; d.DepthOrArraySize = 1;
    d.MipLevels = 1; d.Format = fmt; d.SampleDesc.Count = 1; d.Flags = fl; ID3D12Resource* r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r)))) {   // a non-UAV format may reject a UAV initial state
        r = nullptr; dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r)); }
    if (!r) printf("texture create failed (format %d)\n", (int)FMT);
    return r; }
static void bar(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x = {}; x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource = r; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a; x.Transition.StateAfter = b; cl->ResourceBarrier(1, &x); }
static D3D12_TEXTURE_COPY_LOCATION fp(ID3D12Resource* b, UINT pitch, DXGI_FORMAT FMT) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = b; l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; l.PlacedFootprint.Footprint = {FMT, W, H, 1, pitch}; return l; }
static D3D12_TEXTURE_COPY_LOCATION ts(ID3D12Resource* t) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = t; l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; return l; }
static uint16_t f2h(float f) {                              // round to nearest half, clamp to +-65504
    if (!(f == f)) return 0; if (f > 65504.f) f = 65504.f; if (f < -65504.f) f = -65504.f;
    uint32_t b; memcpy(&b, &f, 4); uint16_t sgn = (uint16_t)((b >> 16) & 0x8000); b &= 0x7FFFFFFF; float a; memcpy(&a, &b, 4);
    if (a < 6.1035156e-5f) return sgn | (uint16_t)lrintf(a * 16777216.f);                  // subnormal half: units of 2^-24
    int e; float m = frexpf(a, &e); uint32_t mant = (uint32_t)lrintf((m * 2.f - 1.f) * 1024.f); int he = e - 1 + 15;   // a = m*2^e, m in [0.5,1)
    if (mant == 1024) { mant = 0; he++; } if (he >= 31) return sgn | 0x7BFF; return sgn | (uint16_t)((he << 10) | mant);
}
static double h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; double v = e == 0 ? ldexp((double)m, -24) : e == 31 ? 1e9 : ldexp((double)(1024 + m), e - 25); return (h & 0x8000) ? -v : v; }
typedef void* (*P_create)(const char*, int, int, int, char*, int); typedef int (*P_run)(void*, const void*, void*, float*);

// NR_TEST_V1=1: every evaluate goes through the v1 export (AMDNR's forwarder path); depth/motion must then share one extent at (0,0)
static PFN_NrEvaluateV1 g_v1;
static int __cdecl v1shim(ID3D12GraphicsCommandList* cmd, void* f, void* p, ID3D12Resource* c, ID3D12Resource* d, ID3D12Resource* m, ID3D12Resource* o, unsigned w, unsigned h,
                          unsigned dW, unsigned dH, unsigned mW, unsigned mH, unsigned dBx, unsigned dBy, unsigned mBx, unsigned mBy, int di, int rs, float in, int st, float ls, float lt, float ss, int am, float sx, float sy) {
    if (dW != mW || dH != mH || dBx || dBy || mBx || mBy) { puts("FAIL NR_TEST_V1: this test path has separate depth / motion extents"); exit(1); }
    return g_v1(cmd, f, p, c, d, m, o, w, h, dW, dH, di, rs, in, st, ls, lt, ss, am, sx, sy); }
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (getenv("NR_TEST_W")) { W = atoi(getenv("NR_TEST_W")); H = atoi(getenv("NR_TEST_H")); }
    MW = W; MH = H; const bool fl = getenv("NR_TEST_FW") != nullptr; if (fl) { W = atoi(getenv("NR_TEST_FW")); H = atoi(getenv("NR_TEST_FH")); }   // flicker test: any frame size
    if (getenv("NR_TEST_UP")) { UP = atoi(getenv("NR_TEST_UP")); W = MW * UP; H = MH * UP; }
    const bool r11 = getenv("NR_TEST_FMT") && !strcmp(getenv("NR_TEST_FMT"), "r11"); if (r11) { FMT = DXGI_FORMAT_R11G11B10_FLOAT; BPP = 4; }
    const bool e5 = getenv("NR_TEST_FMT") && !strcmp(getenv("NR_TEST_FMT"), "e5"); if (e5) { FMT = DXGI_FORMAT_R9G9B9E5_SHAREDEXP; BPP = 4; }
    auto pk5 = [](const uint16_t* h) {   // half4 -> R9G9B9E5 (reference encoder, as D3D)
        float c[3]; for (int i = 0; i < 3; i++) { float v = h2f(h[i]); c[i] = v != v || v < 0 ? 0.f : std::min(v, 65408.f); }
        float m = std::max(c[0], std::max(c[1], c[2])); int e = std::max(-16, (int)floorf(log2f(std::max(m, 1e-30f)))) + 16;
        float d = exp2f((float)(e - 24)); if (floorf(m / d + .5f) >= 512.f) { d *= 2.f; e++; }
        uint32_t v = (uint32_t)(e) << 27; for (int i = 0; i < 3; i++) v |= (uint32_t)floorf(c[i] / d + .5f) << (9 * i); return v; };
    auto up5 = [](uint32_t v, uint16_t* o) { float d = exp2f((float)((int)(v >> 27) - 24)); for (int i = 0; i < 3; i++) o[i] = f2h((float)((v >> (9 * i)) & 511) * d); o[3] = f2h(1.f); };
    if (argc < 3) { puts("usage: test_sync color.bin dll_dir [frames]"); return 1; }
    const int frames = argc > 3 ? atoi(argv[3]) : 300; std::string dir = argv[2]; SetDllDirectoryA(dir.c_str());
    const UINT SW = getenv("NR_TEST_SW") ? atoi(getenv("NR_TEST_SW")) : MW, SH = getenv("NR_TEST_SH") ? atoi(getenv("NR_TEST_SH")) : MH;   // flicker mode: source texture size (default model size)
    std::vector<uint16_t> base((size_t)SW * SH * 4); { FILE* f = fopen(argv[1], "rb"); if (!f || fread(base.data(), 8, (size_t)SW * SH, f) != (size_t)SW * SH) { puts("cannot read color.bin"); return 1; } fclose(f); }
    char dd[1024]; GetPrivateProfileStringA("nr", "data_dir", "", dd, sizeof dd, (dir + "\\nr_port.ini").c_str());
    if (GetFileAttributesA((std::string(dd) + "\\meta.txt").c_str()) == INVALID_FILE_ATTRIBUTES) snprintf(dd + strlen(dd), sizeof dd - strlen(dd), "\\%ux%u", MW, MH);   // step dirs: nr_data\<W>x<H>

    IDXGIFactory4* fac; CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac))); IDXGIAdapter1* ad = nullptr;
    for (UINT i = 0;; i++) { IDXGIAdapter1* a; if (fac->EnumAdapters1(i, &a) != S_OK) break; DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d); if (d.VendorId == 0x1002) { ad = a; break; } a->Release(); }
    if (!ad) { puts("no AMD adapter"); return 1; }
    CK(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd = {}; CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al)));
    CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, IID_PPV_ARGS(&cl))); cl->Close();
    CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))); ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    const UINT pitch = (W * BPP + 255) & ~255u; const UINT64 sz = (UINT64)pitch * H;
    const bool o16 = e5 || getenv("NR_TEST_OFMT");   // NR_TEST_OFMT=f16: RGBA16F output whatever the colour format (format-converting pass-through)
    const DXGI_FORMAT OFMT = o16 ? DXGI_FORMAT_R16G16B16A16_FLOAT : FMT; const UINT opitch = o16 ? (W * 8 + 255) & ~255u : pitch;   // R9G9B9E5 cannot be a UAV on AMD
    auto color = tex(e5 ? D3D12_RESOURCE_FLAG_NONE : D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, FMT),
         out = tex(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, OFMT);
    auto up = buf(D3D12_HEAP_TYPE_UPLOAD, sz, D3D12_RESOURCE_STATE_GENERIC_READ), rb = buf(D3D12_HEAP_TYPE_READBACK, (UINT64)opitch * H, D3D12_RESOURCE_STATE_COPY_DEST);
    uint8_t *upp, *rbp; up->Map(0, nullptr, (void**)&upp); rb->Map(0, nullptr, (void**)&rbp);

    HMODULE m = LoadLibraryA((dir + "\\nvngx.dll_dlssnr.dll").c_str()); if (!m) { printf("load failed %lu\n", GetLastError()); return 1; }
    auto create = (PFN_NrCreate)GetProcAddress(m, "dlssnr_call_create"); auto eval = (PFN_NrEvaluate)GetProcAddress(m, "dlssnr_call_evaluate_v2");
    if (getenv("NR_TEST_V1")) { g_v1 = (PFN_NrEvaluateV1)GetProcAddress(m, "dlssnr_call_evaluate"); auto qr = (PFN_NrQueryScalingRatio)GetProcAddress(m, "dlssnr_query_scaling_ratio"); float ratio = -1.f;
        if (!g_v1 || !qr || qr(L"nvngx_dlssnr.dll", nullptr, 2, &ratio) != 1 || ratio != 1.f) { printf("FAIL v1 exports (evaluate %p, query_scaling_ratio %p, ratio %.2f)\n", (void*)g_v1, (void*)qr, ratio); return 1; }
        eval = v1shim; puts("v1: evaluate through dlssnr_call_evaluate, scaling ratio 1.00"); }
    auto release = (PFN_NrRelease)GetProcAddress(m, "dlssnr_call_release"); auto err = (PFN_NrError)GetProcAddress(m, "dlssnr_call_error");
    auto state = (int (*)(void*))GetProcAddress(m, "dlssnr_port_state");
    HMODULE rt = GetModuleHandleA("nr_runtime.dll"); if (!rt) rt = LoadLibraryA((dir + "\\nr_runtime.dll").c_str());
    auto rcreate = (P_create)GetProcAddress(rt, "nr_create"); auto rrun = (P_run)GetProcAddress(rt, "nr_run");
    if (!create || !eval || !release || !state || !rcreate || !rrun) { puts("missing exports"); return 1; }

    al->Reset(); cl->Reset(al, nullptr); void* f = create(L"nvngx_dlssnr.dll", L"data", dev, cl, nullptr, W, H, getenv("NR_TEST_PRESET") ? atoi(getenv("NR_TEST_PRESET")) : 0, 1.f, 0, 1.f, 1.f, -1.f, 1, 1); cl->Close();   // NR_TEST_PRESET: the host's DLSSNR.Hint.Render.Preset (310.9: 1 = large model)
    { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } if (!wait(5000)) { puts("FAIL timeout"); return 2; }
    if (!f) { printf("FAIL create: %s\n", err()); return 3; }
    if (auto pdir = (const char* (*)(void*))GetProcAddress(m, "dlssnr_port_dir")) if (*pdir(f)) { snprintf(dd, sizeof dd, "%s", pdir(f)); printf("data dir (picked by the DLL): %s\n", dd); }   // model folder + step
    if (getenv("NR_TEST_GUARD")) { double t = now(); while (state(f) == 0 && now() - t < 90000) Sleep(50); }   // also catch a runtime-side refusal (weights.map)
    const bool guard = state(f) == 4;   // model guard: unknown nvngx_dlssnr.dll, NR off -> pass-through test below
    if (guard) printf("NR off at create: %s\n", err());
    if (getenv("NR_TEST_JB")) {   // janblade v0.1.32 host pattern (DlssNr_Dx12.cpp): NR_TEST_JB=cycles. A changed look/preset (epoch) creates the new main
        // feature while the old chain is parked (released NR_TEST_JBPARK evaluates later, default 32; 0 = right after recording, GPU still busy);
        // pass count walks 1 -> 2 -> 3 -> 1 (extra features created / parked); the game's history reset (reset=1) every NR_TEST_JBRESET frames;
        // a failed create is retried every frame (as jb132 does). 3 lists in flight. FAIL: evaluate != 1, GPU timeout (hang), device removed,
        // a failed create, wrong pass labels (dlssnr_port_pass), free VRAM spread > 200 MB at the cycle ends after warm-up.
        typedef int (*P_mem)(size_t*, size_t*); HMODULE hip = GetModuleHandleA("amdhip64_6.dll"); auto mem = hip ? (P_mem)GetProcAddress(hip, "hipMemGetInfo") : nullptr;
        auto ppass = (int (*)(void*))GetProcAddress(m, "dlssnr_port_pass");
        if (!mem || guard) { puts("FAIL JB: no hipMemGetInfo / NR off"); return 1; }
        const int N = atoi(getenv("NR_TEST_JB")), park = getenv("NR_TEST_JBPARK") ? atoi(getenv("NR_TEST_JBPARK")) : 32, rsEvery = getenv("NR_TEST_JBRESET") ? atoi(getenv("NR_TEST_JBRESET")) : 7;
        { for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) memcpy(upp + (size_t)y * pitch + x * 8, &base[((size_t)(y * SH / H) * SW + x * SW / W) * 4], 8); }   // colour: the source scaled to the frame
        auto gtx = [&](DXGI_FORMAT fm) { return tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, fm); };
        ID3D12Resource *mvt = gtx(DXGI_FORMAT_R16G16_FLOAT), *dpt = gtx(DXGI_FORMAT_R32_FLOAT), *outB = tex(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, FMT);
        const UINT gp = (W * 4 + 255) & ~255u; ID3D12Resource* gu = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gp * H * 2, D3D12_RESOURCE_STATE_GENERIC_READ); uint8_t* gup; gu->Map(0, nullptr, (void**)&gup);
        for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { uint16_t mv[2] = { f2h(4.f / W), 0 }; float d = 0.1f; memcpy(gup + (size_t)y * gp + x * 4, mv, 4); memcpy(gup + (size_t)(H + y) * gp + x * 4, &d, 4); }
        const int NQ = 3; ID3D12CommandAllocator* A[NQ]; ID3D12GraphicsCommandList* L[NQ]; UINT64 fq[NQ] = {};
        for (int i = 0; i < NQ; i++) { CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&A[i]))); CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, A[i], nullptr, IID_PPV_ARGS(&L[i]))); L[i]->Close(); }
        auto waitV = [&](UINT64 v) { if (fence->GetCompletedValue() >= v) return true; fence->SetEventOnCompletion(v, ev); return WaitForSingleObject(ev, 5000) == WAIT_OBJECT_0; };
        std::vector<void*> chain = {f}; std::vector<std::pair<void*, int>> parked; std::vector<bool> fresh = {true}; int want = 1, k = 0, bad = 0, fails = 0; size_t lo = SIZE_MAX, hi = 0; bool first = true;
        auto mk = [&](ID3D12GraphicsCommandList* c) { return create(L"nvngx_dlssnr.dll", L"data", dev, c, nullptr, W, H, 0, 1.f, 0, 1.f, 1.f, -1.f, 1, 1); };
        auto tick = [&] { for (size_t i = 0; i < parked.size();) if (--parked[i].second <= 0) { release(parked[i].first); parked.erase(parked.begin() + i); } else i++; };
        auto frameJB = [&](int reset) -> int {   // one game frame: (create a missing pass) + evaluate the chain colour -> A -> B -> A ...
            const int s = k % NQ; if (!waitV(fq[s])) return -100; A[s]->Reset(); ID3D12GraphicsCommandList* c = L[s]; c->Reset(A[s], nullptr); ID3D12GraphicsCommandList* sv = cl; cl = c;
            if (first) { for (int g = 0; g < 2; g++) { ID3D12Resource* t = g ? dpt : mvt; bar(t, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST); D3D12_TEXTURE_COPY_LOCATION s0 = {}; s0.pResource = gu; s0.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                s0.PlacedFootprint.Offset = g ? (UINT64)gp * H : 0; s0.PlacedFootprint.Footprint = {g ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16_FLOAT, W, H, 1, gp}; auto d0 = ts(t); c->CopyTextureRegion(&d0, 0, 0, 0, &s0, nullptr); bar(t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); }
                bar(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST); { auto d = ts(color), s1 = fp(up, pitch, FMT); c->CopyTextureRegion(&d, 0, 0, 0, &s1, nullptr); } bar(color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); first = false; }
            int r = 1;
            if ((int)chain.size() < want) {   // jb132: build at most one missing feature on this invocation and evaluate nothing afterwards
                void* g = mk(c); if (g) { chain.push_back(g); fresh.push_back(true); } else { fails++; if (fails <= 3) printf("frame %d: create of pass %zu failed: %s\n", k, chain.size(), err()); } }
            else for (size_t p = 0; p < chain.size(); p++) {
                ID3D12Resource* in = p == 0 ? color : (p & 1) ? out : outB; ID3D12Resource* o = (p & 1) ? outB : out;
                if (p) bar(in, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                const int e = eval(c, chain[p], nullptr, in, dpt, mvt, o, W, H, W, H, W, H, 0, 0, 0, 0, 1, reset || fresh[p] ? 1 : 0, 1.f, 0, 1.f, 1.f, -1.f, 1, (float)W, (float)H); fresh[p] = false;
                if (p) bar(in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                if (e != 1) r = e; }
            tick(); c->Close(); cl = sv; ID3D12CommandList* l[] = {c}; q->ExecuteCommandLists(1, l); q->Signal(fence, ++fv); fq[s] = fv; k++; return r; };
        auto loaded = [&] { for (void* g : chain) if (state(g) == 0) return false; return (int)chain.size() >= want; };
        auto labels = [&]() -> bool { if (!ppass) return true; for (size_t p = 0; p < chain.size(); p++) if (ppass(chain[p]) != (int)p) { printf("cycle label: chain[%zu] has pass %d (expected %zu)\n", p, ppass(chain[p]), p); return false; } return true; };
        if (const int rapid = getenv("NR_TEST_JBRAPID") ? atoi(getenv("NR_TEST_JBRAPID")) : 0) {   // a slider dragged: a new epoch every 3 frames, the features never finish loading
            for (int i = 0; i < rapid && !bad; i++) { for (void* g : chain) parked.push_back({g, park}); chain.clear(); fresh.clear(); want = 1;
                for (int j = 0; j < 3 && !bad; j++) { const int r = frameJB(j == 2); if (r != 1 || dev->GetDeviceRemovedReason() != S_OK || fails) { printf("FAIL JB rapid %d: evaluate %d, device 0x%08lx, %d failed creates\n", i, r, (unsigned long)dev->GetDeviceRemovedReason(), fails); bad = 9; } } }
            size_t fr2 = 0, tot = 0; mem(&fr2, &tot); printf("rapid: %d epochs, %zu parked, free VRAM %zu MB\n", rapid, parked.size(), fr2 >> 20); }
        for (int cy = 0; cy < N && !bad; cy++) {
            const int kind = cy % 4;   // 0 new epoch (look preset), 1 -> 2 passes, 2 -> 3 passes, 3 -> 1 pass
            if (kind == 0 && cy) { for (void* g : chain) parked.push_back({g, park}); chain.clear(); fresh.clear(); if (!park) tick(); want = 1; }   // park = 0: released after this frame's list is recorded
            else if (kind == 1) want = 2; else if (kind == 2) want = 3;
            else if (kind == 3) { while (chain.size() > 1) { parked.push_back({chain.back(), park}); chain.pop_back(); fresh.pop_back(); } want = 1; }
            const double t0 = now(); int fr = 0;
            while ((!loaded() || fr < 12 || !parked.empty()) && now() - t0 < 30000 && !bad) {
                const int r = frameJB(k % rsEvery == 0); fr++;
                if (r == -100) { printf("FAIL JB cycle %d frame %d: GPU wait timed out (hang)\n", cy, k); bad = 7; }
                else if (r != 1) { printf("FAIL JB cycle %d frame %d: evaluate returned %d\n", cy, k, r); bad = 9; }
                if (dev->GetDeviceRemovedReason() != S_OK) { printf("FAIL JB cycle %d frame %d: device removed 0x%08lx\n", cy, k, (unsigned long)dev->GetDeviceRemovedReason()); bad = 9; }
                if (fails) bad = bad ? bad : 3; }
            if (!bad && !loaded()) { printf("FAIL JB cycle %d: features not loaded after 30 s\n", cy); bad = 4; }
            if (!bad && !labels()) bad = 5;
            for (void* g : chain) if (!bad && state(g) != 5) { printf("FAIL JB cycle %d: a feature is in state %d (%s)\n", cy, state(g), err()); bad = 4; }
            if (!waitV(fv)) { printf("FAIL JB cycle %d: GPU idle wait timed out\n", cy); bad = 7; }
            size_t fr2 = 0, tot = 0; mem(&fr2, &tot); if (cy >= 4 && kind == 3) { lo = std::min(lo, fr2); hi = std::max(hi, fr2); }
            printf("cycle %2d kind %d passes %zu frames %d free VRAM %zu MB device 0x%08lx\n", cy, kind, chain.size(), fr, fr2 >> 20, (unsigned long)dev->GetDeviceRemovedReason());
        }
        waitV(fv); for (auto& pg : parked) release(pg.first); for (void* g : chain) release(g);
        const size_t spread = hi > lo ? (hi - lo) >> 20 : 0; printf("JB: %d frames, free VRAM spread %zu MB over the 1-pass cycle ends\n", k, spread);
        if (!bad && spread > 200) { puts("FAIL JB: VRAM grows"); bad = 9; }
        puts(bad ? "FAIL JB" : "PASS JB"); return bad;
    }
    char e2[512] = ""; void* ref = guard ? nullptr : rcreate(dd, (int)MW, (int)MH, 1, e2, sizeof e2); if (!ref && !guard) { printf("FAIL reference nr_create: %s\n", e2); return 3; }
    if (ref && getenv("NR_TEST_COND")) { int s = 0, am = 1; float t = 1, st = 1, sk = -1; sscanf(getenv("NR_TEST_COND"), "%d,%f,%f,%f,%d", &s, &t, &st, &sk, &am);   // the reference's conditioning (style,tone,structure,skin,auto_mask), as the ini sliders give the DLL
        auto sc = (int (*)(void*, int, float, float, float, int))GetProcAddress(rt, "nr_set_conditioning"); if (!sc || sc(ref, s, t, st, sk, am)) { puts("FAIL NR_TEST_COND"); return 3; } }
    if (getenv("NR_TEST_RECREATE")) {   // VRAM leak test: N cycles of create -> load -> evaluate -> release, alternating frame sizes. Every 3rd cycle releases
        // while the runtime is still loading, every 3rd abandons the feature without release (MSFS 2024 host) -> the next create must evict it.
        // Free VRAM (hipMemGetInfo) is sampled once each fully loaded cycle has its feature live; PASS = spread <= 100 MB (+-50) after warm-up.
        typedef int (*P_mem)(size_t*, size_t*); HMODULE hip = GetModuleHandleA("amdhip64_6.dll"); auto mem = hip ? (P_mem)GetProcAddress(hip, "hipMemGetInfo") : nullptr;
        if (!mem) { puts("FAIL no hipMemGetInfo"); return 1; }
        release(f); const int N = atoi(getenv("NR_TEST_RECREATE")); std::vector<void*> abandoned; size_t lo = SIZE_MAX, hi = 0;
        for (int i = 0; i < N; i++) {
            const UINT fw = i % 2 ? W * 3 / 4 : W, fh = i % 2 ? H * 3 / 4 : H; const int kind = i % 3;   // 0 normal, 1 abandon, 2 release while loading
            if (i && (i - 1) % 3 == 1) Sleep(2100);   // the abandoned one must be idle >= evict_idle_ms (2000) when the next create loads
            // NR_TEST_EVICTRACE=1: the host evaluates the abandoned feature while this create's worker evicts it. dlssnr_port_test(ms) holds that
            // evaluate between its first state check and its lock until the eviction is done, so it lands exactly there (v0.4.3: null heap -> crash).
            std::thread race; int raceR = -1; ID3D12CommandAllocator* ra = nullptr; ID3D12GraphicsCommandList* rl = nullptr; void* rg = nullptr; auto ptest = (void (*)(int))GetProcAddress(m, "dlssnr_port_test");
            if (getenv("NR_TEST_EVICTRACE") && kind == 2 && !abandoned.empty()) {
                if (!ptest) { puts("FAIL no dlssnr_port_test"); return 1; }
                CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ra))); CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ra, nullptr, IID_PPV_ARGS(&rl)));
                rg = abandoned.back(); const UINT pw = (i - 1) % 2 ? W * 3 / 4 : W, ph = (i - 1) % 2 ? H * 3 / 4 : H; ptest(60000);
                race = std::thread([&, pw, ph] { raceR = eval(rl, rg, nullptr, color, nullptr, nullptr, out, pw, ph, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.f, 0, 1.f, 1.f, 1.f, 0, 1.f, 1.f); });
                Sleep(200); }
            al->Reset(); cl->Reset(al, nullptr); void* g = create(L"nvngx_dlssnr.dll", L"data", dev, cl, nullptr, fw, fh, 0, 1.f, 0, 1.f, 1.f, 1.f, 0, 1); cl->Close();
            if (race.joinable()) { const double t2 = now(); while (state(rg) != 4 && now() - t2 < 60000) Sleep(20); const bool ev = state(rg) == 4; ptest(0); race.join();   // the new worker evicts it, then the held evaluate goes on
                rl->Close(); ID3D12CommandList* l[] = {rl}; q->ExecuteCommandLists(1, l); const bool ok = wait(5000);
                printf("evict race: feature %s, the evaluate held across it returned %d, list %s, device 0x%08lx\n", ev ? "evicted" : "NOT evicted (race not exercised)", raceR, ok ? "executed" : "TIMEOUT", (unsigned long)dev->GetDeviceRemovedReason());
                if (!ev || raceR != 1 || !ok || dev->GetDeviceRemovedReason() != S_OK) { puts("FAIL evict race"); return 9; } rl->Release(); ra->Release(); }
            { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } if (!wait(5000) || !g) { printf("FAIL recreate %d: %s\n", i, err()); return 3; }
            if (kind == 2) { release(g); printf("cycle %2d %ux%u released while loading, device 0x%08lx\n", i, fw, fh, (unsigned long)dev->GetDeviceRemovedReason()); continue; }
            const double t1 = now(); while (state(g) == 0 && now() - t1 < 30000) Sleep(20);
            if (state(g) != 5) { printf("FAIL recreate %d: state %d (%s)\n", i, state(g), err()); return 4; }
            for (int k = 0; k < 3; k++) { al->Reset(); cl->Reset(al, nullptr); eval(cl, g, nullptr, color, nullptr, nullptr, out, fw, fh, 0, 0, 0, 0, 0, 0, 0, 0, 0, k == 0, 1.f, 0, 1.f, 1.f, 1.f, 0, 1.f, 1.f); cl->Close();
                ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); if (!wait(5000)) { puts("FAIL recreate: GPU timeout"); return 2; } }
            size_t fr = 0, tot = 0; mem(&fr, &tot); if (i >= 3) { lo = std::min(lo, fr); hi = std::max(hi, fr); }
            printf("cycle %2d %ux%u loaded in %.0f ms, free VRAM %zu MB%s, device 0x%08lx\n", i, fw, fh, now() - t1, fr >> 20, kind == 1 ? " (abandoned, no release)" : "", (unsigned long)dev->GetDeviceRemovedReason());
            if (dev->GetDeviceRemovedReason() != S_OK) { puts("FAIL recreate: device removed"); return 9; }   // a removed device completes every fence: the waits above cannot see it
            if (kind == 1) abandoned.push_back(g); else release(g);
        }
        for (void* g : abandoned) release(g);
        Sleep(3000); size_t fr = 0, tot = 0; mem(&fr, &tot); printf("after all releases: free VRAM %zu MB\n", fr >> 20);
        const size_t spread = hi > lo ? (hi - lo) >> 20 : 0; printf("%s recreate: %d cycles, free VRAM spread %zu MB over the loaded cycles\n", spread <= 100 ? "PASS" : "FAIL", N, spread);
        return spread <= 100 ? 0 : 9;
    }
    // NR_TEST_PASSES=2 (flicker mode): OptiScaler "2 passes" -- a second feature, evaluated in the same list on pass 1's output (janblade DlssNr_Dx12.cpp)
    const bool two = getenv("NR_TEST_PASSES") && atoi(getenv("NR_TEST_PASSES")) > 1; void* f2 = nullptr; ID3D12Resource* out2 = nullptr;
    if (two) { al->Reset(); cl->Reset(al, nullptr); f2 = create(L"nvngx_dlssnr.dll", L"data", dev, cl, nullptr, W, H, 0, 1.f, 0, 1.f, 1.f, -1.f, 1, 1); cl->Close();
        { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } if (!wait(5000) || !f2) { printf("FAIL create pass 2: %s\n", err()); return 3; }
        out2 = tex(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, OFMT); }
    double t0 = now(); while ((state(f) == 0 || (f2 && state(f2) == 0)) && now() - t0 < 90000) Sleep(50);
    printf("dll state %d after %.0f ms (5 = sync)\n", state(f), now() - t0); if (state(f) != 5 && !guard) { printf("FAIL sync path not active: %s\n", err()); return 4; }
    printf("device after create: 0x%08lx\n", (unsigned long)dev->GetDeviceRemovedReason());

    std::vector<uint16_t> in((size_t)MW * MH * 4), refo((size_t)MW * MH * 4), raw((size_t)MW * MH * 4);   // raw: the frame before the DLL's [0,1] input clamp
    const int DX = getenv("NR_TEST_DX") ? atoi(getenv("NR_TEST_DX")) : getenv("NR_TEST_SHIFT") ? atoi(getenv("NR_TEST_SHIFT")) : 13, DY = getenv("NR_TEST_DY") ? atoi(getenv("NR_TEST_DY")) : 0; const bool OBJ = getenv("NR_TEST_OBJ") != nullptr;
    const UINT gpitch = (W * 4 + 255) & ~255u; std::vector<uint8_t> mvScratch((size_t)gpitch * H), dScratch((size_t)gpitch * H); uint8_t *mvp = mvScratch.data(), *dpp = dScratch.data();   // this frame's motion (RG16F) / depth (R32F) upload rows
    const UINT ow = MW / 3, oh = MH / 3, oy = MH / 3; auto objX = [&](int k) { return (UINT)(((long long)k * DX % (MW - ow) + (MW - ow)) % (MW - ow)); };
    auto inObj = [&](int k, UINT x, UINT y) { UINT ox = objX(k); return OBJ && k >= 0 && x >= ox && x < ox + ow && y >= oy && y < oy + oh; };
    const double ZOOM = getenv("NR_TEST_ZOOM") ? atof(getenv("NR_TEST_ZOOM")) : 0.0; const float MVK = getenv("NR_TEST_MVK") ? (float)atof(getenv("NR_TEST_MVK")) : 1.f;
    const double NOISE =getenv("NR_TEST_NOISE") ? atof(getenv("NR_TEST_NOISE")) : 0.0;
    auto makeFrame = [&](int k) {   // shifted base image -> texture format bytes in upload, and the exact half4 the network sees in `in`
        const int sx = (int)(((long long)k * DX % MW + MW) % MW), sy = (int)(((long long)k * DY % MH + MH) % MH);
        for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) {
            const UINT bx = x * MW / W, by = y * MH / H; const bool ob = inObj(k, bx, by);   // frame -> model-size source (any frame size)
            const double zk = ZOOM > 0 ? pow(ZOOM, -(double)k) : 1.0;   // NR_TEST_ZOOM=s: walking forward, the image grows by s per frame about the centre (per-pixel vectors, as FO4)
            const uint16_t* s = OBJ ? (ob ? &base[((size_t)(by - oy + MH / 2) * MW + (bx - objX(k) + MW / 2)) * 4] : &base[((size_t)by * MW + bx) * 4])
                              : ZOOM > 0 ? &base[((size_t)std::min<double>(MH - 1, MH / 2 + (by + 0.5 - MH / 2.0) * zk) * MW + (size_t)std::min<double>(MW - 1, MW / 2 + (bx + 0.5 - MW / 2.0) * zk)) * 4]
                                    : &base[((size_t)((by + sy) % MH) * MW + (bx + sx) % MW) * 4];
            { float mv[2] = { OBJ ? (ob ? (float)-DX : 0.f) : (float)DX, OBJ ? 0.f : (float)DY };   // object moves +DX: previous = current - DX
              if (ZOOM > 0) { mv[0] = (float)((x + 0.5 - W / 2.0) * (1.0 / ZOOM - 1.0)) * MW / W; mv[1] = (float)((y + 0.5 - H / 2.0) * (1.0 / ZOOM - 1.0)) * MH / H; }   // previous = centre + (cur - centre) / s
              mv[0] *= MVK; mv[1] *= MVK;   // NR_TEST_MVK: scale the vectors (-1 = wrong sign, 0 = none) to show the ghost metric catches a bad warp
              uint16_t hv[2] = { f2h(mv[0] / MW), f2h(mv[1] / MH) }; memcpy(mvp + (size_t)y * gpitch + x * 4, hv, 4);
              float dv = OBJ && ob ? 0.5f : 0.1f; memcpy(dpp + (size_t)y * gpitch + x * 4, &dv, 4); } uint16_t* d = &in[((size_t)by * MW + bx) * 4];
            if (r11) { uint32_t v; pack_r11g11b10(s, &v); memcpy(upp + (size_t)y * pitch + x * 4, &v, 4); unpack_r11g11b10(v, d); }
            else if (e5) { uint32_t v = pk5(s); memcpy(upp + (size_t)y * pitch + x * 4, &v, 4); up5(v, d); }
            else if (NOISE > 0) { uint16_t t[4]; for (int c = 0; c < 4; c++) { uint32_t hsh = (uint32_t)(k * 73856093) ^ (x * 19349663u) ^ (y * 83492791u) ^ (c * 2654435761u); hsh ^= hsh >> 13; hsh *= 0x5bd1e995u; hsh ^= hsh >> 15;
                    t[c] = c == 3 ? s[c] : f2h(std::max(0.0, h2f(s[c]) * (1.0 + NOISE * ((hsh & 0xFFFF) / 32767.5 - 1.0)))); }   // NR_TEST_NOISE: relative per-frame noise
                  memcpy(d, t, 8); memcpy(upp + (size_t)y * pitch + x * 8, t, 8); }
            else { memcpy(d, s, 8); memcpy(upp + (size_t)y * pitch + x * 8, s, 8); }
            memcpy(&raw[d - in.data()], d, 8); for (int c = 0; c < 3; c++) d[c] = (d[c] & 0x8000) || (d[c] & 0x7C00) == 0x7C00 && (d[c] & 0x3FF) ? 0 : d[c] > 0x3C00 ? 0x3C00 : d[c]; } };   // the DLL (ldr_input=1, LDR) clamps the network input to [0,1]; NaN -> 0
    ID3D12Resource *fmv = nullptr, *fdp = nullptr, *fmu = nullptr, *fdu = nullptr; bool ffirst = true;   // flicker mode: motion/depth textures, their upload buffers (mvp/dpp)
    ID3D12QueryHeap* tsq = nullptr; ID3D12Resource* tsb = nullptr; uint64_t* tsp = nullptr; UINT64 tsf = 1; std::vector<double> tsv;   // NR_TEST_TS: GPU time of the evaluate call per frame (timestamps around it)
    if (getenv("NR_TEST_TS")) { D3D12_QUERY_HEAP_DESC qd = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0}; CK(dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&tsq))); tsb = buf(D3D12_HEAP_TYPE_READBACK, 256, D3D12_RESOURCE_STATE_COPY_DEST); tsb->Map(0, nullptr, (void**)&tsp); q->GetTimestampFrequency(&tsf); }   // flicker mode: motion/depth textures, their upload buffers (mvp/dpp)
    int dInvT = 1;   // depthInv passed with the guides (flicker mode: 1; NR_TEST_SEQ: NR_TEST_DINV, default 0 = standard Z as FO4)
    auto frame = [&](int k, bool exec, double* ms) -> int {   // one game frame; returns evaluate's result, -100 on GPU timeout
        al->Reset(); cl->Reset(al, nullptr);
        if (fmv) { for (int g = 0; g < 2; g++) { ID3D12Resource* t = g ? fdp : fmv; DXGI_FORMAT fm = g ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16_FLOAT;
            bar(t, ffirst ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION s0 = {}; s0.pResource = g ? fdu : fmu; s0.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; s0.PlacedFootprint.Footprint = {fm, W, H, 1, gpitch};
            auto d0 = ts(t); cl->CopyTextureRegion(&d0, 0, 0, 0, &s0, nullptr); bar(t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); } ffirst = false; }
        bar(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST); { auto d = ts(color), s = fp(up, pitch, FMT); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
        bar(color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (tsq) cl->EndQuery(tsq, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        int r = k < 0 ? 1 : eval(cl, f, nullptr, color, fdp, fmv, out, W, H, W, H, W, H, 0, 0, 0, 0, fmv ? dInvT : 0, k == 0, 1.f, 0, 1.f, 1.f, -1.f, 1, fmv ? (float)W : 1.f, fmv ? (float)H : 1.f);
        if (tsq) { cl->EndQuery(tsq, D3D12_QUERY_TYPE_TIMESTAMP, 1); cl->ResolveQueryData(tsq, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, tsb, 0); }
        ID3D12Resource* fin_ = out;
        if (f2 && k >= 0) { bar(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);   // pass 2: pass 1's output is the input
            int r2 = eval(cl, f2, nullptr, out, fdp, fmv, out2, W, H, W, H, W, H, 0, 0, 0, 0, fmv ? dInvT : 0, k == 0, 1.f, 0, 1.f, 1.f, -1.f, 1, fmv ? (float)W : 1.f, fmv ? (float)H : 1.f);
            bar(out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); if (r2 != 1) r = r2; fin_ = out2; }
        bar(fin_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); { auto d = fp(rb, opitch, OFMT), s = ts(fin_); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
        bar(fin_, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->Close();
        if (!exec) return r;                                  // the host dropped this list
        double a = now(); ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); if (!wait(5000)) return -100; *ms = now() - a; if (tsq && k >= 8) tsv.push_back((tsp[1] - tsp[0]) * 1000.0 / tsf); return r; };
    double se_o = 0, se_i = 0; auto psnr = [&](double se) { return 10 * log10(3.0 * MW * MH / std::max(se, 1e-12)); };
    UINT bx0 = 0, by0 = 0, bx1 = 0, by1 = 0;   // bounding box of the values beyond tolerance (printed on MISMATCH)
    auto compare = [&](size_t* beyond, double* maxerr) {
        *beyond = 0; *maxerr = 0; se_o = se_i = 0; bx0 = by0 = ~0u; bx1 = by1 = 0;
        for (UINT y = 0; y < MH; y++) for (UINT x = 0; x < MW; x++) {
            double ov[3] = {0, 0, 0};   // UP > 1: mean of the UPxUP frame block (bilinear up + box down ~ identity)
            for (UINT j = 0; j < UP * UP; j++) { size_t fy = y * UP + j / UP, fx = x * UP + j % UP; uint16_t o[4];
                if (r11) unpack_r11g11b10(*(uint32_t*)(rbp + fy * pitch + fx * 4), o); else memcpy(o, rbp + fy * opitch + fx * 8, 8);
                for (int c = 0; c < 3; c++) ov[c] += h2f(o[c]) / (UP * UP); }
            for (int c = 0; c < 3; c++) {
                const size_t m = ((size_t)y * MW + x) * 4 + c; uint16_t b = refo[m];   // expected: frame + (NR - clamped input), so values above 1 keep their excess
                double bv = std::max(0.0, ((b & 0x7C00) == 0x7C00 ? 0 : (double)h2f(b)) + ((raw[m] & 0x7C00) == 0x7C00 ? 0 : (double)h2f(raw[m])) - h2f(in[m]));
                double e = fabs(ov[c] - bv);
                { double iv = h2f(raw[m]); se_o += std::min(e * e, 1.0); se_i += std::min((iv - bv) * (iv - bv), 1.0); }   // UP > 1 metric: PSNR (peak 1, clamped) of NR result vs reference, against doing nothing
                if (e > *maxerr) *maxerr = e; if (e > 0.004 + (r11 ? 0.035 : 0.004) * fabs(bv)) { (*beyond)++; bx0 = std::min(bx0, x); by0 = std::min(by0, y); bx1 = std::max(bx1, x); by1 = std::max(by1, y); } } } };   // r11: 5-6 bit mantissa
    if (guard) {   // every frame must come back as the native input, evaluate returning 1
        int bad = 0; double ms = 0;
        for (int k = 0; k < 5; k++) { makeFrame(k); int r = frame(k, true, &ms); if (r == -100) { puts("FAIL GPU wait timed out"); return 7; }
            size_t diff = 0;
            if (OFMT == FMT) for (UINT y = 0; y < H; y++) diff += memcmp(rbp + (size_t)y * opitch, upp + (size_t)y * pitch, (size_t)W * BPP) != 0;
            else for (UINT y = 0; y < H; y++) { bool rd = false;   // converted: the decoded colour (r11 / e5 -> half, exact) vs the RGBA16F output, rgb
                for (UINT x = 0; x < W && !rd; x++) { uint16_t a[4], o[4]; uint32_t v; memcpy(&v, upp + (size_t)y * pitch + x * 4, 4); if (r11) unpack_r11g11b10(v, a); else up5(v, a);
                    memcpy(o, rbp + (size_t)y * opitch + x * 8, 8); for (int c = 0; c < 3; c++) rd |= fabs(h2f(a[c]) - h2f(o[c])) > 1e-3 * fabs(h2f(a[c])) + 1e-6; }
                diff += rd; }
            printf("frame %d: evaluate %d, %zu rows differ from the input\n", k, r, diff); bad += r != 1 || diff; }
        release(f); printf("device 0x%08lx\n", (unsigned long)dev->GetDeviceRemovedReason());
        puts(bad || dev->GetDeviceRemovedReason() != S_OK ? "FAIL passthrough" : "PASSTHROUGH PASS (NR off, native frames)"); return bad ? 9 : 0; }

    if (getenv("NR_TEST_PACE")) {   // pacing: a game submitting every NR_TEST_CPU_MS with up to 2 frames queued; frame time = gap between GPU completions
        const int n = atoi(getenv("NR_TEST_PACE")); const int QD = getenv("NR_TEST_QUEUE") ? std::max(1, std::min(7, atoi(getenv("NR_TEST_QUEUE")))) : 2, NQ = QD + 1; const double cpu = getenv("NR_TEST_CPU_MS") ? atof(getenv("NR_TEST_CPU_MS")) : 8.0;
        auto stats = (void (*)(void*, unsigned*, unsigned*, unsigned*))GetProcAddress(m, "dlssnr_port_stats"); if (!stats) { puts("missing dlssnr_port_stats"); return 1; }
        ID3D12CommandAllocator* A[8]; ID3D12GraphicsCommandList* L[8]; for (int i = 0; i < NQ; i++) { CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&A[i]))); CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, A[i], nullptr, IID_PPV_ARGS(&L[i]))); L[i]->Close(); }
        const double gms = getenv("NR_TEST_GPU_MS") ? atof(getenv("NR_TEST_GPU_MS")) : 0; ID3D12RootSignature* lrs = nullptr; ID3D12PipelineState* lps = nullptr; ID3D12Resource* lb = nullptr; UINT lit = 20000;
        if (gms > 0) {   // dummy "game rendering": an ALU loop over 4096x64 threads, iteration count calibrated to gms (from branch pipeline)
            const char* hl = "RWByteAddressBuffer u : register(u0); cbuffer C : register(b0) { uint n; };\n"
                             "[numthreads(64,1,1)] void main(uint3 id : SV_DispatchThreadID) { float x = id.x; for (uint i = 0; i < n; i++) x = x * 0.9999 + 0.5; if (x == 12345.0) u.Store(0, asuint(x)); }\n";
            ID3DBlob *cs = nullptr, *er = nullptr; if (FAILED(D3DCompile(hl, strlen(hl), "load", nullptr, nullptr, "main", "cs_5_0", 0, 0, &cs, &er))) { printf("FAIL load shader: %s\n", er ? (char*)er->GetBufferPointer() : "?"); return 1; }
            D3D12_ROOT_PARAMETER rp[2] = {}; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[0].Constants.Num32BitValues = 1; rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 2; rsd.pParameters = rp; ID3DBlob* sb = nullptr; CK(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sb, &er));
            CK(dev->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&lrs)));
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = lrs; pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()}; CK(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&lps)));
            { D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = 256; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
              d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; CK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&lb))); }
            for (int it = 0; it < 4; it++) {   // calibrate (submit -> done wall time, list holds only the load)
                al->Reset(); cl->Reset(al, nullptr); cl->SetComputeRootSignature(lrs); cl->SetPipelineState(lps); cl->SetComputeRoot32BitConstants(0, 1, &lit, 0); cl->SetComputeRootUnorderedAccessView(1, lb->GetGPUVirtualAddress()); cl->Dispatch(4096, 1, 1); cl->Close();
                double a = now(); ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); if (!wait(5000)) { puts("FAIL load calibration timeout"); return 7; } double t = now() - a;
                printf("gpu load calibration: %u iterations -> %.2f ms\n", lit, t); if (it < 3) lit = (UINT)std::min(5e7, std::max(1000.0, lit * gms / std::max(t, 0.05))); }
        }
        auto gtex = [&](DXGI_FORMAT fm) { auto r = tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, fm); return r; };
        ID3D12Resource *mvt = gtex(DXGI_FORMAT_R16G16_FLOAT), *dpt = gtex(DXGI_FORMAT_R32_FLOAT), *mu[8], *du[8]; uint8_t *mup[8], *dup[8];
        for (int i = 0; i < NQ; i++) { mu[i] = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ); du[i] = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ); mu[i]->Map(0, nullptr, (void**)&mup[i]); du[i]->Map(0, nullptr, (void**)&dup[i]); }
        mvp = mup[0]; dpp = dup[0];
        makeFrame(0); for (int i = 1; i < NQ; i++) { memcpy(mup[i], mup[0], (size_t)gpitch * H); memcpy(dup[i], dup[0], (size_t)gpitch * H); }
        bool gfirst = true; std::vector<double> done; std::vector<UINT64> fvs; unsigned n0, b0, r0; stats(f, &n0, &b0, &r0);
        // NR_TEST_GHOST=1: the image moves (makeFrame(k) shift) every frame; reused frames are compared with the reference NR of the SAME frame
        const bool ghost = getenv("NR_TEST_GHOST") && !r11;   /* UP > 1: each frame pixel vs the model pixel it was made from (the game's resampled path) */ ID3D12Resource *gu[8] = {}, *gr[8] = {}; uint8_t *gup[8] = {}, *grp[8] = {};
        std::vector<int> greu(n, 0); std::vector<std::vector<uint16_t>> gout(n); ID3D12Resource* up0 = up; uint8_t* upp0 = upp;
        if (ghost) for (int i = 0; i < NQ; i++) { gu[i] = buf(D3D12_HEAP_TYPE_UPLOAD, sz, D3D12_RESOURCE_STATE_GENERIC_READ); gr[i] = buf(D3D12_HEAP_TYPE_READBACK, (UINT64)opitch * H, D3D12_RESOURCE_STATE_COPY_DEST); gu[i]->Map(0, nullptr, (void**)&gup[i]); gr[i]->Map(0, nullptr, (void**)&grp[i]); }
        auto grab = [&](int j) { if (!ghost || j < 0 || !greu[j]) return; auto& o = gout[j]; o.resize((size_t)W * H * 4); for (UINT y = 0; y < H; y++) memcpy(&o[(size_t)y * W * 4], grp[j % NQ] + (size_t)y * opitch, (size_t)W * 8); };
        auto waitFv = [&](UINT64 v) { if (fence->GetCompletedValue() >= v) return true; fence->SetEventOnCompletion(v, ev); return WaitForSingleObject(ev, 5000) == WAIT_OBJECT_0; };
        auto poll = [&]() { while (done.size() < fvs.size() && fence->GetCompletedValue() >= fvs[done.size()]) done.push_back(now()); };
        // completion times from a poller thread: the DLL may block the submit thread (busy_mode=wait), which must not skew the GPU frame times
        const UINT64 fv0 = fv; std::vector<double> tv(n, 0); std::atomic<bool> prun{true};
        std::thread pt([&] { UINT64 seen = fv0; while (prun) { UINT64 c = fence->GetCompletedValue(); const double t = now(); while (seen < c && seen - fv0 < (UINT64)n) tv[(++seen) - fv0 - 1] = t; Sleep(0); } });
        double next = now();
        for (int k = 0; k < n; k++) {
            while (now() < next) { poll(); Sleep(0); } next = now() + cpu;   // game CPU time per frame
            if (k >= QD) { if (!waitFv(fvs[k - QD])) { prun = false; pt.join(); puts("FAIL pace: GPU wait timed out"); return 7; } poll(); grab(k - QD); }   // swapchain: at most QD frames queued (NR_TEST_QUEUE, default 2)
            if (ghost) { up = gu[k % NQ]; upp = gup[k % NQ]; mvp = mup[k % NQ]; dpp = dup[k % NQ]; makeFrame(k); }
            auto* c = L[k % NQ]; A[k % NQ]->Reset(); c->Reset(A[k % NQ], nullptr); std::swap(cl, c);
            bar(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST); { auto d = ts(color), s = fp(up, pitch, FMT); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
            bar(color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            for (int g = 0; g < 2; g++) { ID3D12Resource* t = g ? dpt : mvt; DXGI_FORMAT fm = g ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16_FLOAT;   // guides: upload -> texture -> shader resource
                bar(t, gfirst ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_TEXTURE_COPY_LOCATION s0 = {}; s0.pResource = g ? du[k % NQ] : mu[k % NQ]; s0.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; s0.PlacedFootprint.Footprint = {fm, W, H, 1, gpitch};
                auto d0 = ts(t); cl->CopyTextureRegion(&d0, 0, 0, 0, &s0, nullptr); bar(t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); } gfirst = false;
            if (lps) { cl->SetComputeRootSignature(lrs); cl->SetPipelineState(lps); cl->SetComputeRoot32BitConstants(0, 1, &lit, 0); cl->SetComputeRootUnorderedAccessView(1, lb->GetGPUVirtualAddress()); cl->Dispatch(4096, 1, 1);
                       D3D12_RESOURCE_BARRIER ub = {}; ub.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; cl->ResourceBarrier(1, &ub); }
            unsigned x1, x2, ru0, ru1; stats(f, &x1, &x2, &ru0);
            eval(cl, f, nullptr, color, dpt, mvt, out, W, H, W, H, W, H, 0, 0, 0, 0, 1, k == 0, 1.f, 0, 1.f, 1.f, -1.f, 1, (float)W, (float)H);
            stats(f, &x1, &x2, &ru1); greu[k] = ru1 != ru0;
            if (ghost) { bar(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); { auto d = fp(gr[k % NQ], opitch, OFMT), s = ts(out); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
                         bar(out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); }
            cl->Close();
            ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); q->Signal(fence, ++fv); fvs.push_back(fv); std::swap(cl, c); poll();
        }
        if (!waitFv(fvs.back())) { prun = false; pt.join(); puts("FAIL pace: GPU wait timed out"); return 7; } prun = false; pt.join(); done = tv;
        for (int j = std::max(0, n - QD); j < n; j++) grab(j);
        if (ghost) {   // ghosting: mean |out - refNR(same frame)| over reused frames; "live" = the plain live frame for scale
            double e = 0, el = 0, er = 0, erl = 0; size_t cnt = 0, cr = 0; int nf = 0; up = up0; upp = upp0; mvp = mvScratch.data(); dpp = dScratch.data();
            for (int j = 0; j < n; j++) { if (gout[j].empty()) continue; makeFrame(j); float gms; if (rrun(ref, in.data(), refo.data(), &gms)) { puts("FAIL reference nr_run"); return 6; } nf++;
                for (size_t i = 0; i < (size_t)W * H * 4; i++) { if ((i & 3) == 3) continue; const UINT px = (UINT)(i / 4 % W), py = (UINT)(i / 4 / W); const size_t mi = ((size_t)(py / UP) * MW + px / UP) * 4 + (i & 3);
                    uint16_t b = refo[mi]; double bv = (b & 0x7C00) == 0x7C00 ? 0 : std::max(0.0, (double)h2f(b));
                    double eo = fabs(h2f(gout[j][i]) - bv), ei = fabs(h2f(in[mi]) - bv); e += eo; el += ei; cnt++; if (inObj(j, px / UP, py / UP) || inObj(j - 1, px / UP, py / UP)) { er += eo; erl += ei; cr++; } } }
            printf("ghost: %d reused frames, mean |out - NR(same frame)| %.5f (live frame without NR: %.5f)\n", nf, cnt ? e / cnt : 0, cnt ? el / cnt : 0);
            if (OBJ) printf("ghost object region (object now or one frame ago): %.5f (live %.5f)\n", cr ? er / cr : 0, cr ? erl / cr : 0);
            if (getenv("NR_TEST_DUMP") && UP == 1) for (int j = n - 1; j >= 0; j--) if (!gout[j].empty()) {   // last reused frame: input, output, NR of the same frame (f32 RGB, W x H)
                makeFrame(j); float gms; rrun(ref, in.data(), refo.data(), &gms); const uint16_t* src[3] = { in.data(), gout[j].data(), refo.data() }; const char* nm[3] = { "_in.bin", "_out.bin", "_ref.bin" };
                for (int q = 0; q < 3; q++) { FILE* df = fopen((std::string(getenv("NR_TEST_DUMP")) + nm[q]).c_str(), "wb"); if (!df) continue;
                    for (size_t i = 0; i < (size_t)W * H; i++) for (int c = 0; c < 3; c++) { float v = h2f(src[q][i * 4 + c]); v = v == v ? v : 0.f; fwrite(&v, 4, 1, df); } fclose(df); }
                printf("ghost dump: frame %d\n", j); break; }
        }
        std::vector<double> d; for (size_t i = 10; i < done.size(); i++) d.push_back(done[i] - done[i - 1]);   // skip warm-up
        double s = 0, s2 = 0, oe[2] = {}; for (size_t i = 0; i < d.size(); i++) { s += d[i]; s2 += d[i] * d[i]; oe[(i + 11) & 1] += d[i]; } double av = s / d.size(), sd = sqrt(std::max(0.0, s2 / d.size() - av * av));
        printf("pace odd/even: frame time odd %.2f even %.2f |diff| %.2f ms\n", oe[1] / (d.size() / 2), oe[0] / ((d.size() + 1) / 2), fabs(oe[1] / (d.size() / 2) - oe[0] / ((d.size() + 1) / 2))); std::sort(d.begin(), d.end());
        unsigned n1, b1, r1; stats(f, &n1, &b1, &r1);
        printf("pace (gpu load %.1f ms) ", gms);
        printf("pace (cpu %.1f ms/frame, %d frames): frame time avg %.2f ms (%.1f fps) sd %.2f p50 %.2f p95 %.2f p99 %.2f max %.2f | NR %u, reused %u, native %u\n",
               cpu, n, av, 1000 / av, sd, d[d.size() / 2], d[d.size() * 95 / 100], d[d.size() * 99 / 100], d.back(), n1 - n0, r1 - r0, b1 - b0);
        release(f); return 0;
    }
    if (getenv("NR_TEST_HIST")) {   // temporal test (UP=1, RGBA16F): N frames shifted by DX,DY per frame (+NR_TEST_NOISE) with exact MVs and depth, one list per frame.
        // Flicker F = mean |P(O_t) - P(O_{t-1}) warped| - the same for the inputs (P = sRGB(x/(1+x)); exact integer warp, wraps like the image).
        // Also: mean brightness of P(O) - P(I) per frame (drift), and mean |O - NR_ref(same frame, no history)| over background / object / trailing (ghost) region.
        const int n = atoi(getenv("NR_TEST_HIST")); auto P = [](double x) { x = std::max(x, 0.0); x = x / (1 + x); return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055; };
        auto gtex = [&](DXGI_FORMAT fm) { return tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, fm); };
        ID3D12Resource *mvt = gtex(DXGI_FORMAT_R16G16_FLOAT), *dpt = gtex(DXGI_FORMAT_R32_FLOAT), *mu = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ), *du = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ);
        mu->Map(0, nullptr, (void**)&mvp); du->Map(0, nullptr, (void**)&dpp); bool gfirst = true;
        std::vector<double> po((size_t)MW * MH * 3), pi(po.size()), qo(po.size()), qi(po.size()); std::vector<double> tl; double sf = 0, sfi = 0, sg[3] = {}, sgl[3] = {}; size_t ng[3] = {}; int nf = 0; double bmin = 1e9, bmax = -1e9;
        for (int k = 0; k < n; k++) {
            makeFrame(k); al->Reset(); cl->Reset(al, nullptr);
            bar(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST); { auto d = ts(color), s = fp(up, pitch, FMT); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
            bar(color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            for (int g = 0; g < 2; g++) { ID3D12Resource* t = g ? dpt : mvt; DXGI_FORMAT fm = g ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16_FLOAT;
                bar(t, gfirst ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_TEXTURE_COPY_LOCATION s0 = {}; s0.pResource = g ? du : mu; s0.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; s0.PlacedFootprint.Footprint = {fm, W, H, 1, gpitch};
                auto d0 = ts(t); cl->CopyTextureRegion(&d0, 0, 0, 0, &s0, nullptr); bar(t, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); } gfirst = false;
            int r = eval(cl, f, nullptr, color, dpt, mvt, out, W, H, W, H, W, H, 0, 0, 0, 0, 1, k == 0, 1.f, 0, 1.f, 1.f, -1.f, 1, (float)W, (float)H);
            bar(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); { auto d = fp(rb, opitch, OFMT), s = ts(out); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
            bar(out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->Close();
            double a = now(); ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); if (!wait(5000)) { printf("FAIL frame %d: GPU wait timed out\n", k); return 7; } tl.push_back(now() - a);
            if (r != 1) { printf("FAIL frame %d: evaluate returned %d\n", k, r); return 9; }
            float gms; if (rrun(ref, in.data(), refo.data(), &gms)) { puts("FAIL reference nr_run"); return 6; }
            double bo = 0, bi = 0, fo = 0, fi = 0; size_t cnt = (size_t)MW * MH * 3;
            for (UINT y = 0; y < MH; y++) for (UINT x = 0; x < MW; x++) { const uint16_t* o = (const uint16_t*)(rbp + (size_t)y * opitch + x * 8); size_t i = ((size_t)y * MW + x) * 3;
                const int reg = !OBJ ? 0 : inObj(k, x, y) ? 1 : inObj(k - 1, x, y) ? 2 : 0;   // 0 background, 1 object, 2 just uncovered (ghost trail)
                for (int c = 0; c < 3; c++) { qo[i + c] = P(h2f(o[c])); qi[i + c] = P(h2f(in[((size_t)y * MW + x) * 4 + c])); bo += qo[i + c]; bi += qi[i + c];
                    uint16_t b = refo[((size_t)y * MW + x) * 4 + c]; double bv = (b & 0x7C00) == 0x7C00 ? 0 : std::max(0.0, h2f(b)); sg[reg] += fabs(qo[i + c] - P(bv)); sgl[reg] += fabs(qi[i + c] - P(bv)); ng[reg]++;
                    if (k > 0 && !OBJ) { size_t j = ((size_t)((y + DY) % MH + MH) % MH * MW + ((x + DX) % MW + MW) % MW) * 3 + c; fo += fabs(qo[i + c] - po[j]); fi += fabs(qi[i + c] - pi[j]); } } }
            const double db = (bo - bi) / cnt; bmin = std::min(bmin, db); bmax = std::max(bmax, db);
            if (k > 1 && !OBJ) { sf += fo / cnt; sfi += fi / cnt; nf++; }
            if (k < 3 || k % 4 == 0) printf("frame %d: list %.1f ms | mean P(O)-P(I) %.5f | |dO| %.5f |dI| %.5f\n", k, tl.back(), db, k ? fo / cnt : 0, k ? fi / cnt : 0);
            std::swap(po, qo); std::swap(pi, qi); }
        std::sort(tl.begin() + 1, tl.end());
        if (nf) printf("HIST flicker: mean |O_t - warp(O_t-1)| %.5f, inputs %.5f, F = %.5f (frames 2..%d)\n", sf / nf, sfi / nf, (sf - sfi) / nf, n - 1);
        printf("HIST brightness: mean P(O)-P(I) range %.5f .. %.5f (drift %.5f)\n", bmin, bmax, bmax - bmin);
        printf("HIST vs no-history NR (same frame), mean |P(O) - P(ref)|: background %.5f, object %.5f, uncovered %.5f (input vs ref: %.5f %.5f %.5f)\n", ng[0] ? sg[0] / ng[0] : 0, ng[1] ? sg[1] / ng[1] : 0, ng[2] ? sg[2] / ng[2] : 0,
               ng[0] ? sgl[0] / ng[0] : 0, ng[1] ? sgl[1] / ng[1] : 0, ng[2] ? sgl[2] / ng[2] : 0);
        printf("HIST list ms p50 %.2f\n", tl[tl.size() / 2]);
        release(f); return 0;
    }
    if (getenv("NR_TEST_HDR")) {   // HDR white-out test (RE Engine-like scene-linear colour): NR_TEST_HDR=frames, frame W x H (NR_TEST_FW/FH), source NR_TEST_SW x SH.
        // Scene: the source (display sRGB) decoded to linear x NR_TEST_EXP (default 3), a 5..50 light strip at the top, four skin-like patches (~1.5..4).
        // Static sequence; NR_TEST_PASSES=2 evaluates twice per frame (output -> input, as OptiScaler Passes=2). Last frame: white-out = share of
        // pixels whose luma rises above 1.5 x input + 0.05 (linear), and in display space P = sRGB(x/(1+x)): share with P(out) - P(in) > 0.1.
        // NR_TEST_HDRDUMP=prefix: writes prefix_in.ppm / prefix_out.ppm (P-tonemapped, 8 bit) for viewing.
        const int n = atoi(getenv("NR_TEST_HDR")), passes = getenv("NR_TEST_PASSES") ? atoi(getenv("NR_TEST_PASSES")) : 1; const float ex = getenv("NR_TEST_EXP") ? (float)atof(getenv("NR_TEST_EXP")) : 3.f;
        const double ws = getenv("NR_TEST_WHITE") ? atof(getenv("NR_TEST_WHITE")) / ex : 1.0;   // NR_TEST_WHITE=w: the whole scene x w/ex (scene white at w); P and the dumps are normalised back by 1/ws, so an exposure-invariant DLL gives the same numbers at every w
        auto P = [ws](double x) { x = std::max(x / ws, 0.0); x = x / (1 + x); return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055; };
        auto lin = [](double v) { v = std::min(std::max(v, 0.0), 1.0); return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4); };
        std::vector<float> hin((size_t)W * H * 3); std::vector<char> skin((size_t)W * H);
        const float sk[4][3] = {{0.30f, 0.30f, 0.12f}, {0.62f, 0.45f, 0.10f}, {0.45f, 0.70f, 0.14f}, {0.80f, 0.75f, 0.09f}};   // centre u, v, radius (fraction of H)
        for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { const size_t i = (size_t)y * W + x; const UINT sx = x * SW / W, sy = y * SH / H; float l = 0;
            for (int c = 0; c < 3; c++) { hin[i * 3 + c] = ex * (float)lin(h2f(base[((size_t)sy * SW + sx) * 4 + c])); l += hin[i * 3 + c] / 3; }
            if (y < H / 8) { const float v = 5.f + 45.f * x / W; for (int c = 0; c < 3; c++) hin[i * 3 + c] = v * (c == 2 ? 0.8f : 1.f) * (0.8f + 0.4f * l / ex); }
            for (auto& s : sk) { const float du = (x + .5f) / H - s[0] * W / H, dv = (y + .5f) / H - s[1]; if (du * du + dv * dv < s[2] * s[2]) { const float k = 1.5f + 2.5f * (1 - sqrtf(du * du + dv * dv) / s[2]) * (0.7f + 0.6f * std::min(l / ex, 1.f));
                hin[i * 3] = 1.0f * k; hin[i * 3 + 1] = 0.66f * k; hin[i * 3 + 2] = 0.5f * k; skin[i] = 1; } }  for (int c = 0; c < 3; c++) hin[i * 3 + c] *= (float)ws; }
        for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { uint16_t h4[4]; float* p = &hin[((size_t)y * W + x) * 3]; for (int c = 0; c < 3; c++) h4[c] = f2h(p[c]); h4[3] = f2h(1.f);
            if (e5) { uint32_t v = pk5(h4); memcpy(upp + (size_t)y * pitch + x * 4, &v, 4); up5(v, h4); } else if (r11) { uint32_t v; pack_r11g11b10(h4, &v); memcpy(upp + (size_t)y * pitch + x * 4, &v, 4); unpack_r11g11b10(v, h4); }
            else memcpy(upp + (size_t)y * pitch + x * 8, h4, 8);
            for (int c = 0; c < 3; c++) p[c] = (float)h2f(h4[c]); }   // what the DLL really gets
        // passes 2: the second pass is a second feature (f2), as janblade's passFeature[pass]; NR_TEST_P2CLAMP=1 feeds it through the host's ClampProxy (CPU copy of it)
        if (passes > 1 && !f2) { puts("FAIL passes 2 needs NR_TEST_PASSES=2 at create"); return 3; }
        const bool clampP2 = getenv("NR_TEST_P2CLAMP") && atoi(getenv("NR_TEST_P2CLAMP"));
        ID3D12Resource* out2 = passes > 1 ? tex(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, OFMT) : nullptr;
        ID3D12Resource *p2in = clampP2 ? tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, OFMT) : nullptr, *p2up = clampP2 ? buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)opitch * H, D3D12_RESOURCE_STATE_GENERIC_READ) : nullptr;
        uint8_t* p2u = nullptr; if (p2up) p2up->Map(0, nullptr, (void**)&p2u);
        for (int k = 0; k < n; k++) {
            al->Reset(); cl->Reset(al, nullptr);
            bar(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST); { auto d = ts(color), s = fp(up, pitch, FMT); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
            bar(color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            int r = eval(cl, f, nullptr, color, nullptr, nullptr, out, W, H, W, H, W, H, 0, 0, 0, 0, 0, k == 0, 1.f, 0, 1.f, 1.f, -1.f, 1, 1.f, 1.f); ID3D12Resource* last = out;
            if (passes > 1 && clampP2) {   // host ClampProxy (janblade dlssnr.hlsl mode 12, PassFeedback 1): next = saturate(P + saturate(alpha) * (raw - P)), alpha fits raw - P into the unit cube, P = pass 1 input
                bar(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); { auto d = fp(rb, opitch, OFMT), s = ts(out); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
                bar(out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->Close();
                { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } if (!wait(5000)) { printf("FAIL frame %d: GPU wait timed out\n", k); return 7; }
                for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { uint16_t o[4]; memcpy(o, rbp + (size_t)y * opitch + x * 8, 8); const float* pp = &hin[((size_t)y * W + x) * 3]; float d[3], a = 1.f;
                    for (int c = 0; c < 3; c++) { float v = h2f(o[c]); d[c] = (v == v && fabsf(v) < 1e30f ? v : 0.5f) - pp[c]; if (d[c] > 1e-6f) a = std::min(a, (1.f - pp[c]) / d[c]); else if (d[c] < -1e-6f) a = std::min(a, -pp[c] / d[c]); }
                    a = std::min(std::max(a, 0.f), 1.f); for (int c = 0; c < 3; c++) o[c] = f2h(std::min(std::max(pp[c] + a * d[c], 0.f), 1.f)); memcpy(p2u + (size_t)y * opitch + x * 8, o, 8); }
                al->Reset(); cl->Reset(al, nullptr);
                bar(p2in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST); { auto d = ts(p2in), s = fp(p2up, opitch, OFMT); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
                bar(p2in, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                r &= eval(cl, f2, nullptr, p2in, nullptr, nullptr, out2, W, H, W, H, W, H, 0, 0, 0, 0, 0, k == 0, 1.f, 0, 1.f, 1.f, -1.f, 1, 1.f, 1.f); last = out2; }
            else if (passes > 1) { bar(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);   // raw: pass 1's output (RGBA16F) straight into the second feature
                r &= eval(cl, f2, nullptr, out, nullptr, nullptr, out2, W, H, W, H, W, H, 0, 0, 0, 0, 0, k == 0, 1.f, 0, 1.f, 1.f, -1.f, 1, 1.f, 1.f);
                bar(out, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); last = out2; }
            bar(last, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); { auto d = fp(rb, opitch, OFMT), s = ts(last); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr); }
            bar(last, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); cl->Close();
            ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); if (!wait(5000)) { printf("FAIL frame %d: GPU wait timed out\n", k); return 7; }
            if (r != 1) { printf("FAIL frame %d: evaluate returned %d\n", k, r); return 9; }
            size_t wo = 0, wp = 0, sw = 0, ns = 0; double dp = 0, mx = 0, so = 0, si = 0;
            for (size_t i = 0; i < (size_t)W * H; i++) { const size_t y = i / W, x = i % W; uint16_t o[4]; if (r11) unpack_r11g11b10(*(uint32_t*)(rbp + y * pitch + x * 4), o); else memcpy(o, rbp + y * opitch + x * 8, 8);
                double lo = 0, li = 0; for (int c = 0; c < 3; c++) { const double wl = c == 0 ? .2126 : c == 1 ? .7152 : .0722, v = h2f(o[c]); lo += wl * (v == v ? v : 1e9); li += wl * hin[i * 3 + c]; }
                const bool w1 = lo > 1.5 * li + 0.05 * ws, w2 = P(lo) - P(li) > 0.1; wo += w1; wp += w2; dp += fabs(P(lo) - P(li)); mx = std::max(mx, lo);
                if (skin[i]) { ns++; sw += w2; so += lo; si += li; } }
            const double N = (double)W * H;
            if (k < 2 || k == n - 1) printf("HDR frame %d (passes %d): white-out (lin) %.3f%%, white-out (display) %.3f%%, mean |dP| %.5f, max luma %.1f | skin: white-out %.3f%%, mean luma out/in %.3f\n",
                                            k, passes, 100 * wo / N, 100 * wp / N, dp / N, mx, 100.0 * sw / std::max<size_t>(ns, 1), so / std::max(si, 1e-9));
            if (k == n - 1 && getenv("NR_TEST_HDRDUMP")) for (int g = 0; g < 2; g++) { std::string fn = std::string(getenv("NR_TEST_HDRDUMP")) + (g ? "_out.ppm" : "_in.ppm"); FILE* pf = fopen(fn.c_str(), "wb"); if (!pf) continue;
                fprintf(pf, "P6\n%u %u\n255\n", W, H); std::vector<uint8_t> row((size_t)W * 3);
                for (UINT y = 0; y < H; y++) { for (UINT x = 0; x < W; x++) { uint16_t o[4]; if (r11) unpack_r11g11b10(*(uint32_t*)(rbp + (size_t)y * pitch + x * 4), o); else memcpy(o, rbp + (size_t)y * opitch + x * 8, 8);
                    for (int c = 0; c < 3; c++) { double v = g ? h2f(o[c]) : hin[((size_t)y * W + x) * 3 + c]; row[x * 3 + c] = (uint8_t)std::min(255.0, P(v == v ? v : 1e9) * 255 + .5); } } fwrite(row.data(), 1, row.size(), pf); }
                fclose(pf); }
        }
        release(f); return 0;
    }
    if (fl) {   // NR_TEST_FW/FH: temporal flicker test at any frame size (frame W x H, network MW x MH). See HARNESS.md (handover/temporal-2026-10-11).
        if (const char* seq = getenv("NR_TEST_SEQ")) {   // replay a dump_frame sequence (frame_dump.h: dump_{colour,motion,depth}_fN_WxH_*.f32 at the frame size W x H, NR_TEST_FW/FH)
            // NR_TEST_SEQN frames in it (default 8), NR_TEST_WARM static copies of f0 first (default 12), then f1.. with the game's vectors and depth (NR_TEST_DINV, default 0).
            // Per frame: the shown edit (out - in) box-downsampled to the model size vs the reference NR edit of the SAME frame (nr_run on the box-downsampled frame, clamped
            // [0,1] as the DLL). ALIGN: per quadrant, the shift (model px) where the shown edit correlates best with the reference edit; it must be 0,0 (a stretched or
            // offset edit, e.g. padded vs valid size, moves the peak by up to ~5% of the position). NR_TEST_SEQOUT=prefix: f32 RGB output per frame (prefix_fN.bin).
            const int sn = getenv("NR_TEST_SEQN") ? atoi(getenv("NR_TEST_SEQN")) : 8, warm = getenv("NR_TEST_WARM") ? atoi(getenv("NR_TEST_WARM")) : 12; dInvT = getenv("NR_TEST_DINV") ? atoi(getenv("NR_TEST_DINV")) : 0;
            fmv = tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, DXGI_FORMAT_R16G16_FLOAT); fdp = tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, DXGI_FORMAT_R32_FLOAT);
            fmu = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ); fdu = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ);
            if (!fmv || !fdp || !fmu || !fdu) { puts("FAIL guide resources"); return 1; } fmu->Map(0, nullptr, (void**)&mvp); fdu->Map(0, nullptr, (void**)&dpp);
            auto rd = [&](const char* n, int fr, int ch, std::vector<float>& v) { char fn[1024]; snprintf(fn, sizeof fn, "%s\\dump_%s_f%d_%ux%u_%dch.f32", seq, n, fr, W, H, ch);
                v.assign((size_t)W * H * ch, 0.f); FILE* fp = fopen(fn, "rb"); const bool ok = fp && fread(v.data(), 4, v.size(), fp) == v.size(); if (fp) fclose(fp); return ok; };
            auto down = [&](auto px, std::vector<float>& o) {   // box: each frame pixel into the model pixel its centre falls in (frame -> model, RGB)
                o.assign((size_t)MW * MH * 3, 0.f); std::vector<int> cnt((size_t)MW * MH, 0);
                for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { const size_t m = (size_t)(y * MH / H) * MW + x * MW / W; cnt[m]++; for (int c = 0; c < 3; c++) o[m * 3 + c] += px(x, y, c); }
                for (size_t m = 0; m < cnt.size(); m++) for (int c = 0; c < 3; c++) o[m * 3 + c] /= std::max(cnt[m], 1); };
            std::vector<float> esP, col, mot, dep, fo((size_t)W * H * 3), ci, co, em((size_t)MW * MH * 3), es((size_t)MW * MH * 3); int bad = 0; double ms = 0, hole0 = 0, holeMax = 0;
            for (int k = 0; k < warm + sn - 1; k++) {
                const int fr = k < warm ? 0 : k - warm + 1;
                if (!rd("colour", fr, 4, col) || !rd("depth", fr, 1, dep)) { printf("FAIL cannot read %s frame %d\n", seq, fr); return 1; }
                if (fr == 0 || !rd("motion", fr, 2, mot)) mot.assign((size_t)W * H * 2, 0.f);
                for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { const size_t i = (size_t)y * W + x; uint16_t h4[4] = { f2h(col[i * 4]), f2h(col[i * 4 + 1]), f2h(col[i * 4 + 2]), f2h(1.f) };
                    memcpy(upp + (size_t)y * pitch + x * 8, h4, 8); const uint16_t hv[2] = { f2h(mot[i * 2]), f2h(mot[i * 2 + 1]) }; memcpy(mvp + (size_t)y * gpitch + x * 4, hv, 4); memcpy(dpp + (size_t)y * gpitch + x * 4, &dep[i], 4); }
                const int r = frame(k, true, &ms); if (r != 1) { printf("FAIL frame %d: evaluate returned %d\n", k, r); return 7; }
                for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { uint16_t o[4]; memcpy(o, rbp + (size_t)y * opitch + x * 8, 8); for (int c = 0; c < 3; c++) { const double v = h2f(o[c]); fo[((size_t)y * W + x) * 3 + c] = v == v && v < 1e8 ? (float)v : 0.f; } }
                if (getenv("NR_TEST_SEQOUT") && *getenv("NR_TEST_SEQOUT")) { char fn[1024]; snprintf(fn, sizeof fn, "%s_f%d.bin", getenv("NR_TEST_SEQOUT"), k); FILE* df = fopen(fn, "wb"); if (df) { fwrite(fo.data(), 4, fo.size(), df); fclose(df); } }
                if (k < warm - 1) continue;   // measured: the last static frame and every moving one
                auto cl01 = [](float v) { return v == v ? std::min(std::max(v, 0.f), 1.f) : 0.f; };
                down([&](UINT x, UINT y, int c) { return cl01(col[((size_t)y * W + x) * 4 + c]); }, ci);
                down([&](UINT x, UINT y, int c) { const size_t i = (size_t)y * W + x; return fo[i * 3 + c] - std::max(col[i * 4 + c] == col[i * 4 + c] ? col[i * 4 + c] : 0.f, 0.f); }, es);   // shown edit (compose add: out = frame + edit)
                for (size_t i = 0; i < (size_t)MW * MH; i++) { for (int c = 0; c < 3; c++) in[i * 4 + c] = f2h(ci[i * 3 + c]); in[i * 4 + 3] = f2h(1.f); }
                float gms; if (rrun(ref, in.data(), refo.data(), &gms)) { puts("FAIL reference nr_run"); return 6; }
                for (size_t i = 0; i < (size_t)MW * MH; i++) for (int c = 0; c < 3; c++) { const uint16_t v = refo[i * 4 + c]; em[i * 3 + c] = ((v & 0x7C00) == 0x7C00 ? 0.f : (float)h2f(v)) - (float)h2f(in[i * 4 + c]); }
                double pe = 0, re = 0; for (size_t i = 0; i < em.size(); i++) { pe += fabs(es[i] - em[i]); re += fabs(em[i]); }
                size_t nh = 0, ns = 0;   // HOLE: on flat frame areas (3x3 luma range < 0.03, where a halo shows: sky, walls) the reference edits the px (|luma| > 0.02) but the shown edit has < 30% of it
                auto Y = [&](int x, int y) { const float* q = &ci[((size_t)y * MW + x) * 3]; return .2126f * q[0] + .7152f * q[1] + .0722f * q[2]; };
                for (UINT y = 1; y + 1 < MH; y++) for (UINT x = 1; x + 1 < MW; x++) { const size_t i = (size_t)y * MW + x; const float* u = &em[i * 3]; const float* v = &es[i * 3];
                    float lo = 9, hi = -9; for (int j = 0; j < 9; j++) { const float t = Y(x + j % 3 - 1, y + j / 3 - 1); lo = std::min(lo, t); hi = std::max(hi, t); } if (hi - lo >= 0.03f) continue;
                    const float lr = .2126f * u[0] + .7152f * u[1] + .0722f * u[2], ls = .2126f * v[0] + .7152f * v[1] + .0722f * v[2]; if (fabsf(lr) > 0.02f) { ns++; nh += ls * lr < 0.3f * lr * lr; } }
                const double hole = 100.0 * nh / std::max<size_t>(ns, 1); if (k == warm - 1) hole0 = hole; else holeMax = std::max(holeMax, hole);
                const int R = std::max(4, (int)(MW / 80)); std::string sh; bool fr_bad = false;   // search +-R model px (>= 5% of a quadrant's centre position)
                for (int qd = 0; qd < 4; qd++) {
                    const int x0 = qd & 1 ? MW / 2 : R + 8, x1 = qd & 1 ? MW - R - 8 : MW / 2, y0 = qd & 2 ? MH / 2 : R + 8, y1 = qd & 2 ? MH - R - 8 : MH / 2;
                    auto L = [&](const std::vector<float>& v, int x, int y) { const float* p = &v[((size_t)y * MW + x) * 3]; return .2126f * p[0] + .7152f * p[1] + .0722f * p[2]; };
                    double best = -2, r0 = 0; int bx = 0, by = 0;
                    for (int dy = -R; dy <= R; dy++) for (int dx = -R; dx <= R; dx++) { double sab = 0, saa = 0, sbb = 0;
                        for (int y = y0; y < y1; y += 2) for (int x = x0; x < x1; x += 2) { const double a = L(es, x, y), b = L(em, x + dx, y + dy); sab += a * b; saa += a * a; sbb += b * b; }
                        const double rr = sab / sqrt(std::max(saa * sbb, 1e-30)); if (!dx && !dy) r0 = rr; if (rr > best + 1e-9) { best = rr; bx = dx; by = dy; } }
                    char b[96]; snprintf(b, sizeof b, " q%d %+d,%+d (r %.3f, at 0,0 %.3f)", qd, bx, by, best, r0); sh += b; fr_bad |= (bx || by) && best > r0 + 0.01; }
                bad += fr_bad;
                double cg = 0, cm = 0;   // ECHG: shown edit vs the previous frame's shown edit warped by the game's vectors (model px, bilinear), relative to |shown edit|
                if (!esP.empty()) for (UINT y = 2; y + 2 < MH; y++) for (UINT x = 2; x + 2 < MW; x++) { const size_t fi = (size_t)((y + .5f) * H / MH) * W + (size_t)((x + .5f) * W / MW);
                    const float u = std::min(std::max(x + mot[fi * 2] * MW, 0.f), MW - 1.001f), v = std::min(std::max(y + mot[fi * 2 + 1] * MH, 0.f), MH - 1.001f); const int x0 = (int)u, y0 = (int)v; const float fa = u - x0, fb = v - y0;
                    for (int c = 0; c < 3; c++) { auto g = [&](int xx, int yy) { return esP[((size_t)yy * MW + xx) * 3 + c]; };
                        const float w = (g(x0, y0) * (1 - fa) + g(x0 + 1, y0) * fa) * (1 - fb) + (g(x0, y0 + 1) * (1 - fa) + g(x0 + 1, y0 + 1) * fa) * fb, e = es[((size_t)y * MW + x) * 3 + c]; cg += fabs(e - w); cm += fabs(e); } }
                esP = es;
                printf("SEQ k %d (f%d): shown edit vs reference edit of the same frame: mean |diff| %.5f, reference |edit| %.5f, HOLE %.2f%%, ECHG %.1f%% | ALIGN%s%s\n", k, fr, pe / em.size(), re / em.size(), hole, 100.0 * cg / std::max(cm, 1e-9), sh.c_str(), fr_bad ? " MISALIGNED" : ""); }
            const double holeLim = getenv("NR_TEST_HOLEMAX") ? atof(getenv("NR_TEST_HOLEMAX")) : 6.0; const bool hb = holeMax > hole0 + holeLim;   // HOLE: moving frames may lose the edit on at most 6 points more of the edited flat px than the static frame (FO4 dump, v0.4.3/4 nr_split presets: +6.5..+18.6 without halo_fix, <= +2.3 with it; the static share is history weakening the edit, not a halo)
            printf("HOLE static %.2f%%, worst moving %.2f%% (limit static + %.1f) %s\n", hole0, holeMax, holeLim, hb ? "FAIL (edit missing in disoccluded bands: halos)" : "ok");
            release(f); puts(bad ? "ALIGN FAIL (the shown edit is offset from the reference edit)" : hb ? "SEQ FAIL" : "SEQ PASS"); return bad || hb ? 9 : 0;
        }
        // Sequence (a game-like camera): the source texture (argv[1], NR_TEST_SW x NR_TEST_SH, default the model size) bilinear-resampled to the
        // frame with a pan of NR_TEST_VX/VY frame px per frame (default 0.37, 0.13), TAA-like sub-pixel jitter (Halton 2,3, amplitude NR_TEST_JIT px,
        // default 0.25), per-pixel per-frame noise (relative sigma NR_TEST_NOISE, default 0.01) and optional bright specks (NR_TEST_SPECK value, 0 = none).
        // Motion vectors (RG16F uv, previous = current + mv, mvScale W,H) and depth (R32F 0.1, inverted) match the motion and go to evaluate.
        // Metric, frames k >= 1, interior (8 px + motion margin), values clamped to [0,1] (display): inst = mean over RGB of |x_k - warp(x_{k-1})|
        // (warp = bilinear fetch at the exact known motion) for output and input; excess = inst_out - inst_in (0 = as stable as the input).
        // "white jumps": pixels whose luma rises against the warped previous output by > 3/255 more than the input's own rise; "visible": |that excess change| > 2/255 (either sign).
        // NR_TEST_REF=1: the same metric for the network alone (nr_run, model-size sequence clamped to [0,1]); NR_TEST_DUMPDIR=dir: writes
        // that sequence's model-size input/output (float32 RGB, f%03d_in.bin / f%03d_out.bin) for the PyTorch reference (temporal/flicker_torch.py).
        auto envf = [](const char* n, float d) { return getenv(n) ? (float)atof(getenv(n)) : d; };
        const float vx = envf("NR_TEST_VX", 0.37f), vy = envf("NR_TEST_VY", 0.13f), jit = envf("NR_TEST_JIT", 0.25f), nsg = envf("NR_TEST_NOISE", 0.01f), sv = envf("NR_TEST_SPECK", 0.f);
        const bool doRef = getenv("NR_TEST_REF") != nullptr; const char* dumpDir = getenv("NR_TEST_DUMPDIR"); double ms = 0;
        auto halton = [](int i, int b) { float f = 1, r = 0; for (i++; i > 0; i /= b) { f /= b; r += f * (i % b); } return r; };
        auto shift = [&](int k, float* s) { s[0] = k * vx + jit * 2 * (halton(k % 8, 2) - .5f); s[1] = k * vy + jit * 2 * (halton(k % 8, 3) - .5f); };   // frame px
        auto src = [&](float u, float v, int c) { u = fmodf(fmodf(u, (float)SW) + SW, (float)SW); v = std::min(std::max(v, 0.f), SH - 1.001f); int x0 = (int)u, y0 = (int)v; float a = u - x0, b = v - y0; int x1 = (x0 + 1) % SW;
            auto g = [&](int x, int y) { return std::max(0.f, (float)h2f(base[((size_t)y * SW + x) * 4 + c])); }; return (g(x0, y0) * (1 - a) + g(x1, y0) * a) * (1 - b) + (g(x0, y0 + 1) * (1 - a) + g(x1, y0 + 1) * a) * b; };
        auto gn = [](uint32_t x, uint32_t y, uint32_t k) { float s = 0; for (uint32_t i = 0; i < 4; i++) { uint32_t h = x * 0x8DA6B343u ^ y * 0xD8163841u ^ (k * 4 + i + 1) * 0xCB1AB31Fu;
            h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16; s += (h >> 8) * (1.f / 16777216.f); } return (s - 2.f) * 1.7320508f; };   // ~N(0,1)
        auto gen = [&](std::vector<float>& o, UINT w, UINT h, int k) { float s[2]; shift(k, s); const float sc = (float)w / W;   // frame k at w x h (same aspect as the frame)
            for (UINT y = 0; y < h; y++) for (UINT x = 0; x < w; x++) { const float n = 1 + nsg * gn(x, y, k);
                for (int c = 0; c < 3; c++) o[((size_t)y * w + x) * 3 + c] = n * src((x + .5f + s[0] * sc) * SW / w - .5f, (y + .5f + s[1] * sc) * SH / h - .5f, c); }
            if (sv > 0) for (int i = 0; i < 400; i++) { int x = (int)fmodf((i * 7919u) % w - s[0] * sc + 4.f * w, (float)w), y = (int)((i * 104729u) % h); for (int c = 0; c < 3; c++) o[((size_t)y * w + x) * 3 + c] = sv; } };
        struct Fr { double o = 0, i = 0; size_t j = 0, v = 0, n = 0; };
        auto measure = [&](const std::vector<float>& ci, const std::vector<float>& co, const std::vector<float>& pi, const std::vector<float>& po, UINT w, UINT h, int k) {
            float a[2], b[2]; shift(k, a); shift(k - 1, b); const float sc = (float)w / W, dx = (a[0] - b[0]) * sc, dy = (a[1] - b[1]) * sc;   // warp(prev)(x) = prev(x + d)
            auto cl1 = [](float v) { return std::min(std::max(v, 0.f), 1.f); };
            auto bil = [&](const std::vector<float>& p, float u, float v, int c) { int x0 = (int)floorf(u), y0 = (int)floorf(v); float fa = u - x0, fb = v - y0;
                auto g = [&](int x, int y) { return cl1(p[((size_t)y * w + x) * 3 + c]); }; return (g(x0, y0) * (1 - fa) + g(x0 + 1, y0) * fa) * (1 - fb) + (g(x0, y0 + 1) * (1 - fa) + g(x0 + 1, y0 + 1) * fa) * fb; };
            const int m = 8 + (int)ceilf(std::max(fabsf(dx), fabsf(dy))); Fr r;
            for (int y = m; y < (int)h - m; y++) for (int x = m; x < (int)w - m; x++) { const size_t i = ((size_t)y * w + x) * 3; float lo = 0, lpo = 0, li = 0, lpi = 0;
                for (int c = 0; c < 3; c++) { const float wl = c == 0 ? .2126f : c == 1 ? .7152f : .0722f, wo = bil(po, x + dx, y + dy, c), wi = bil(pi, x + dx, y + dy, c), o = cl1(co[i + c]), in_ = cl1(ci[i + c]);
                    r.o += fabsf(o - wo); r.i += fabsf(in_ - wi); lo += wl * o; lpo += wl * wo; li += wl * in_; lpi += wl * wi; }
                const float d = (lo - lpo) - (li - lpi); r.j += d > 3 / 255.f; r.v += fabsf(d) > 2 / 255.f; r.n++; }
            r.o /= 3.0 * r.n; r.i /= 3.0 * r.n; return r; };
        auto report = [&](const char* tag, const Fr& t, int nf, UINT w, UINT h) {
            printf("%s %ux%u (model %ux%u): inst out %.5f in %.5f excess %+.5f (x%.2f) | white jumps %.1f/frame (%.3f%% of px), visible flicker %.3f%% of px, over %d frames\n", tag, w, h, MW, MH, t.o / nf, t.i / nf, (t.o - t.i) / nf,
                   t.i > 0 ? t.o / t.i : 0, (double)t.j / nf, 100.0 * t.j / std::max<size_t>(t.n, 1), 100.0 * t.v / std::max<size_t>(t.n, 1), nf); };
        fmv = tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, DXGI_FORMAT_R16G16_FLOAT); fdp = tex(D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON, DXGI_FORMAT_R32_FLOAT);
        fmu = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ); fdu = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)gpitch * H, D3D12_RESOURCE_STATE_GENERIC_READ);
        if (!fmv || !fdp || !fmu || !fdu) { puts("FAIL guide resources"); return 1; } fmu->Map(0, nullptr, (void**)&mvp); fdu->Map(0, nullptr, (void**)&dpp);
        std::vector<float> fin((size_t)W * H * 3), fout(fin.size()), pin(fin.size()), pout(fin.size()), mi((size_t)MW * MH * 3), mo(mi.size()), pmi(mi.size()), pmo(mi.size());
        double hfo = 0, hfi = 0, lmax = 0; int nhf = 0, lmaxK = -1; Fr tot, rtot, sst; int nf = 0, nss = 0; double pulse[2] = {}; int npulse[2] = {}; double eabs = 0, pipeE = 0, pipeR = 0; int neabs = 0, nPipe = 0; std::vector<std::pair<double, double>> ech; uint64_t hsh = 1469598103934665603ull;
        for (int k = 0; k < frames; k++) {
            gen(fin, W, H, k); float a[2], b[2]; shift(k, a); shift(k - 1, b); const uint16_t hv[2] = { f2h((a[0] - b[0]) / W), f2h((a[1] - b[1]) / H) }; const float dv = 0.1f;
            for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { memcpy(mvp + (size_t)y * gpitch + x * 4, hv, 4); memcpy(dpp + (size_t)y * gpitch + x * 4, &dv, 4);
                uint16_t h4[4]; float* p = &fin[((size_t)y * W + x) * 3]; for (int c = 0; c < 3; c++) h4[c] = f2h(p[c]); h4[3] = f2h(1.f);
                if (r11) { uint32_t v; pack_r11g11b10(h4, &v); memcpy(upp + (size_t)y * pitch + x * 4, &v, 4); unpack_r11g11b10(v, h4); }
                else memcpy(upp + (size_t)y * pitch + x * 8, h4, 8);
                for (int c = 0; c < 3; c++) p[c] = (float)h2f(h4[c]); }   // what the network really gets
            int r = frame(k, true, &ms); if (r == -100) { printf("FAIL frame %d: GPU wait timed out\n", k); return 7; } if (r != 1) { printf("FAIL frame %d: evaluate returned %d\n", k, r); return 7; }
            if (k >= 2 && ms > lmax) { lmax = ms; lmaxK = k; }   // longest list (submit -> done) after warm-up: a GPU wait running into its bound shows here
            for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) { uint16_t o[4]; if (r11) unpack_r11g11b10(*(uint32_t*)(rbp + (size_t)y * pitch + x * 4), o); else memcpy(o, rbp + (size_t)y * opitch + x * 8, 8);
                for (int c = 0; c < 3; c++) { double v = h2f(o[c]); fout[((size_t)y * W + x) * 3 + c] = v == v ? (float)v : 0.f; } }
            { double e = 0, ea = 0; for (size_t i = 0; i < fin.size(); i += 3) for (int c = 0; c < 3; c++) { const double d = std::min(std::max(fout[i + c], 0.f), 1.f) - std::min(std::max(fin[i + c], 0.f), 1.f); e += (c == 0 ? .2126 : c == 1 ? .7152 : .0722) * d; ea += fabs(d); }
              if (k >= 4) { eabs += ea / fin.size(); neabs++; }   // effect strength: mean |out - in| (display, RGB)
              if (getenv("NR_TEST_PERFRAME")) printf("EDITF %d %.5f\n", k, ea / fin.size());   // per-frame effect strength (temporal audit: does history weaken/strengthen it over time)
              uint64_t fh = 1469598103934665603ull; for (float v : fout) { uint32_t b; memcpy(&b, &v, 4); hsh = (hsh ^ b) * 1099511628211ull; fh = (fh ^ b) * 1099511628211ull; }   // output hash over all frames (bit-exactness between builds)
              if (k >= frames - 4) printf("FHASH %d %016llx\n", k, (unsigned long long)fh);   // per frame (static sequences: split vs non-split once settled)
              if (k > 0) pulse[k & 1] += e / (fin.size() / 3), npulse[k & 1]++; }   // odd/even pulsing: mean luma edit (out - in, display) per frame parity
            if (doRef) { gen(mi, MW, MH, k); for (size_t i = 0; i < (size_t)MW * MH; i++) { for (int c = 0; c < 3; c++) { in[i * 4 + c] = f2h(std::min(std::max(mi[i * 3 + c], 0.f), 1.f)); mi[i * 3 + c] = (float)h2f(in[i * 4 + c]); } in[i * 4 + 3] = f2h(1.f); }
                float gms; if (rrun(ref, in.data(), refo.data(), &gms)) { puts("FAIL reference nr_run"); return 6; }
                for (size_t i = 0; i < (size_t)MW * MH; i++) for (int c = 0; c < 3; c++) { uint16_t v = refo[i * 4 + c]; mo[i * 3 + c] = (v & 0x7C00) == 0x7C00 ? 0.f : (float)h2f(v); }
                if (dumpDir) { char fn[1024]; for (int q = 0; q < 2; q++) { snprintf(fn, sizeof fn, "%s\\f%03d_%s.bin", dumpDir, k, q ? "out" : "in"); FILE* df = fopen(fn, "wb"); if (df) { fwrite(q ? mo.data() : mi.data(), 4, mi.size(), df); fclose(df); } } } }
            { auto hf = [&](const std::vector<float>& v) { double s = 0; for (UINT y = 8; y < H - 8; y++) for (UINT x = 8; x < W - 8; x++) { auto L = [&](UINT xx, UINT yy) { const float* p = &v[((size_t)yy * W + xx) * 3];
                    return .2126f * std::min(std::max(p[0], 0.f), 1.f) + .7152f * std::min(std::max(p[1], 0.f), 1.f) + .0722f * std::min(std::max(p[2], 0.f), 1.f); };
                    s += fabs(4 * L(x, y) - L(x - 1, y) - L(x + 1, y) - L(x, y - 1) - L(x, y + 1)); } return s / ((double)(W - 16) * (H - 16)); };   // high-frequency energy: mean |Laplacian| of display luma
              if (k >= 4) { hfo += hf(fout); hfi += hf(fin); nhf++; }
              if (k == frames - 1 && getenv("NR_TEST_DUMP")) { for (int q = 0; q < 2; q++) { std::string fn = std::string(getenv("NR_TEST_DUMP")) + (q ? "_out.bin" : "_in.bin"); FILE* df = fopen(fn.c_str(), "wb"); if (df) { fwrite(q ? fout.data() : fin.data(), 4, fin.size(), df); fclose(df); } } } }
            if (doRef && k >= 8) {   // PIPE: the shown frame vs the live frame + the reference NR edit of the SAME frame (bilinear up, add, boost past intensity 1 as boost()):
                const double I = getenv("NR_TEST_INT") ? atof(getenv("NR_TEST_INT")) : 1.0; double pe = 0, re = 0;   // what the split / reuse / history / edit_temporal path adds on top of the network
                for (UINT y = 0; y < H; y++) for (UINT x = 0; x < W; x++) {
                    float u = std::min(std::max((x + .5f) * MW / W - .5f, 0.f), MW - 1.001f), v = std::min(std::max((y + .5f) * MH / H - .5f, 0.f), MH - 1.001f); int x0 = (int)u, y0 = (int)v; float fa = u - x0, fb = v - y0;
                    float o[3], a[3]; const float* fi = &fin[((size_t)y * W + x) * 3];
                    for (int c = 0; c < 3; c++) { auto d = [&](int xx, int yy) { const size_t i = ((size_t)yy * MW + xx) * 3 + c; return mo[i] - mi[i]; };
                        a[c] = std::max(fi[c], 0.f); o[c] = std::max(a[c] + (d(x0, y0) * (1 - fa) + d(x0 + 1, y0) * fa) * (1 - fb) + (d(x0, y0 + 1) * (1 - fa) + d(x0 + 1, y0 + 1) * fa) * fb, 0.f); }
                    if (I > 1) { const double la = .2126 * a[0] + .7152 * a[1] + .0722 * a[2], lo = .2126 * o[0] + .7152 * o[1] + .0722 * o[2], r = std::max((lo + 1 / 512.) / (la + 1 / 512.), 1e-6);
                        const double s = std::min(std::max(pow(r, I), 0.5), 2.0) / r; float m = 0, lim = std::max(1.f, std::max(o[0], std::max(o[1], o[2]))); float b[3];
                        for (int c = 0; c < 3; c++) { b[c] = (float)(o[c] * s); m = std::max(m, b[c]); } for (int c = 0; c < 3; c++) o[c] = m > lim ? b[c] * lim / m : b[c]; }
                    for (int c = 0; c < 3; c++) { const float sh = std::min(std::max(fout[((size_t)y * W + x) * 3 + c], 0.f), 1.f), rf = std::min(std::max(o[c], 0.f), 1.f); pe += fabs(sh - rf); re += fabs(rf - std::min(std::max(fi[c], 0.f), 1.f)); } }
                pipeE += pe / (3.0 * W * H); pipeR += re / (3.0 * W * H); nPipe++; }
            if (k >= 8) {   // ECHG: the shown edit (display out - in) against the previous frame's shown edit warped by the exact motion; the top third of the frames = where a new split job lands
                float a[2], b[2]; shift(k, a); shift(k - 1, b); const float dx = a[0] - b[0], dy = a[1] - b[1]; const int m = 8 + (int)ceilf(std::max(fabsf(dx), fabsf(dy)));
                auto ed = [&](const std::vector<float>& o, const std::vector<float>& i, size_t j) { return std::min(std::max(o[j], 0.f), 1.f) - std::min(std::max(i[j], 0.f), 1.f); };
                double dc = 0, em = 0; size_t nn = 0;
                for (int y = m; y < (int)H - m; y += 2) for (int x = m; x < (int)W - m; x += 2) { const float u = x + dx, v = y + dy; const int x0 = (int)floorf(u), y0 = (int)floorf(v); const float fa = u - x0, fb = v - y0;
                    for (int c = 0; c < 3; c++) { auto g = [&](int xx, int yy) { const size_t j = ((size_t)yy * W + xx) * 3 + c; return ed(pout, pin, j); };
                        const float w = (g(x0, y0) * (1 - fa) + g(x0 + 1, y0) * fa) * (1 - fb) + (g(x0, y0 + 1) * (1 - fa) + g(x0 + 1, y0 + 1) * fa) * fb, e = ed(fout, fin, ((size_t)y * W + x) * 3 + c);
                        dc += fabs(e - w); em += fabs(e); } nn += 3; }
                ech.push_back({dc / nn, em / nn}); if (getenv("NR_TEST_PERFRAME")) printf("ECHGF %d %.5f %.5f\n", k, dc / nn, em / nn); }
            if (k > 0) { Fr e = measure(fin, fout, pin, pout, W, H, k); tot.o += e.o; tot.i += e.i; tot.j += e.j; tot.v += e.v; tot.n += e.n; nf++;
                if (k >= 8) { sst.o += e.o; sst.i += e.i; sst.j += e.j; sst.v += e.v; sst.n += e.n; nss++; } Fr er;   // steady state (k >= 8): without the first edit's fade-in
                if (doRef) { er = measure(mi, mo, pmi, pmo, MW, MH, k); rtot.o += er.o; rtot.i += er.i; rtot.j += er.j; rtot.v += er.v; rtot.n += er.n; }
                if (k < 4 || k % 10 == 0 || getenv("NR_TEST_PERFRAME")) printf("frame %d: inst out %.5f in %.5f, white jumps %zu, visible %.3f%%%s\n", k, e.o, e.i, e.j, 100.0 * e.v / std::max<size_t>(e.n, 1), doRef ? (std::string(" | network alone: inst out ") + std::to_string(er.o) + " in " + std::to_string(er.i) + ", jumps " + std::to_string(er.j)).c_str() : ""); }
            std::swap(fin, pin); std::swap(fout, pout); std::swap(mi, pmi); std::swap(mo, pmo); }
        printf("OUTHASH %016llx\n", (unsigned long long)hsh);
        if (!ech.empty()) { double sc = 0, se = 0, tc = 0; std::vector<double> r; for (auto& p : ech) { sc += p.first; se += p.second; r.push_back(p.first / std::max(p.second, 1e-9)); }
          std::sort(r.rbegin(), r.rend()); const size_t nt = std::max<size_t>(r.size() / 3, 1); for (size_t i = 0; i < nt; i++) tc += r[i];
          printf("ECHG shown-edit change vs warped previous (k >= 8): mean %.5f of |edit| %.5f (%.1f%%), top third of frames %.1f%%, max %.1f%%\n", sc / ech.size(), se / ech.size(), 100.0 * sc / std::max(se, 1e-9), 100.0 * tc / nt, 100.0 * r[0]); }
        printf("PULSE mean luma edit odd %+.5f even %+.5f |diff| %.5f\n", pulse[1] / std::max(npulse[1], 1), pulse[0] / std::max(npulse[0], 1), fabs(pulse[1] / std::max(npulse[1], 1) - pulse[0] / std::max(npulse[0], 1)));
        if (!tsv.empty()) { std::sort(tsv.begin(), tsv.end()); printf("TS evaluate GPU ms (frames >= 8): p10 %.3f median %.3f mean-of-lowest-half %.3f\n", tsv[tsv.size() / 10], tsv[tsv.size() / 2],
            std::accumulate(tsv.begin(), tsv.begin() + tsv.size() / 2, 0.0) / (tsv.size() / 2)); }
        printf("EDIT mean |out - in| (display RGB) %.5f over %d frames\n", eabs / std::max(neabs, 1), neabs);
        printf("LISTMAX %.1f ms (frame %d)\n", lmax, lmaxK);
        if (nPipe) printf("PIPE mean |shown - (live + reference NR edit of the same frame)| %.5f; reference edit mean |.| %.5f; over %d frames (k >= 8)\n", pipeE / nPipe, pipeR / nPipe, nPipe);
        printf("HF mean |Laplacian| luma: out %.5f in %.5f (x%.3f) over %d frames\n", hfo / std::max(nhf, 1), hfi / std::max(nhf, 1), hfi > 0 ? hfo / hfi : 0, nhf);
        report("FLICKER frame", tot, nf, W, H); report("FLICKER steady(k>=8)", sst, nss, W, H); if (doRef) report("FLICKER network-alone", rtot, nf, MW, MH);
        if (f2) release(f2); release(f); return 0;
    }
    // baseline: the same list without evaluate
    std::vector<double> tb, ts_, tr; double ms = 0;
    makeFrame(0); for (int i = 0; i < 20; i++) { if (frame(-1, true, &ms) == -100) { puts("FAIL timeout baseline"); return 5; } tb.push_back(ms); }
    if (dev->GetDeviceRemovedReason() != S_OK) { printf("FAIL device removed after baseline: 0x%08lx\n", (unsigned long)dev->GetDeviceRemovedReason()); return 5; }
    int failed = 0, notRun = 0;
    for (int k = 0; k < frames; k++) {
        makeFrame(k);
        double a = now(); float gms = 0; if (rrun(ref, in.data(), refo.data(), &gms)) { puts("FAIL reference nr_run"); return 6; } tr.push_back(now() - a);
        int r = frame(k, true, &ms);
        if (r == -100) { printf("FAIL frame %d: GPU wait timed out (deadlock?)\n", k); return 7; }
        if (r != 1) { notRun++; printf("frame %d: evaluate returned %d\n", k, r); continue; }
        ts_.push_back(ms); size_t bad; double mx; compare(&bad, &mx);
        bool ok = UP > 1 ? psnr(se_o) >= 30 : bad <= (size_t)MW * MH * 3 / 1000; failed += !ok;   // UP > 1 (resampled): compare quality, not values
        if (k < 3 || k % 50 == 0 || !ok) printf("frame %d: %.1f ms list (ref nr_run %.1f ms) | %zu rgb values beyond tol, max err %.4g %s\n", k, ms, tr.back(), bad, mx, ok ? "ok" : "MISMATCH");
        if (!ok) printf("  beyond-tol box x %u..%u y %u..%u\n", bx0, bx1, by0, by1);
        static const bool psall = getenv("NR_TEST_PSNR") != nullptr;   // NR_TEST_PSNR=1: PSNR vs the vit_every=1 reference every frame (lossy ini settings)
        if (psall) printf("psnr %d %.2f %.2f\n", k, psnr(se_o), ms);
        if (UP > 1 && (k < 3 || !ok)) printf("  psnr vs reference: NR %.1f dB, no NR %.1f dB\n", psnr(se_o), psnr(se_i));
    }
    // robustness: the host records a frame but never executes it, then carries on
    makeFrame(frames); frame(frames, false, &ms); int after = 0, afterBad = 0; double afterMax = 0;
    for (int k = frames + 1; k < frames + 11; k++) { makeFrame(k); int r = frame(k, true, &ms); if (r == -100) { puts("FAIL deadlock after a dropped list"); return 8; } if (r == 1) after++; afterMax = std::max(afterMax, ms); afterBad += ms > 1000; }
    double tr0 = now(); release(f); double trel = now() - tr0;
    auto stat = [](std::vector<double> v, const char* n) { if (v.empty()) return; std::sort(v.begin(), v.end()); double s = 0; for (double x : v) s += x; printf("%-34s avg %6.2f ms  p50 %6.2f  p95 %6.2f  max %6.2f  (n=%zu)\n", n, s / v.size(), v[v.size() / 2], v[v.size() * 95 / 100], v.back(), v.size()); };
    stat(tb, "list without NR (baseline)"); stat(ts_, "list with sync NR (submit->done)"); stat(tr, "reference nr_run (CPU in/out)");
    printf("frames %d: %d evaluated+matched, %d mismatched, %d not run | after dropped list: %d/10 ran, worst %.1f ms | release %.0f ms\n", frames, (int)ts_.size() - failed, failed, notRun, after, afterMax, trel);
    bool pass = failed == 0 && notRun == 0 && after == 10 && afterBad == 0 && trel < 3000;
    puts(pass ? "PASS" : "FAIL"); fflush(stdout); return pass ? 0 : 9;
}
