// test_amdnr <dll dir> <mode> [W H]  - plays AMDNR's lmxxf host (LmxxfBackend.cpp) against Rdna2NrRuntime.dll.
// Per frame ONE command list: RecordOutputs(previous job) + copy of private_output to a readback buffer, then this frame's colour upload,
// PrepareFrame + RecordInputs; ExecuteCommandLists; EnqueueHipAsync (AMDNR calls it from its submission hook). The answer is consumed one
// frame late. Odd frames first poll OutputReady (AMDNR's no-motion mode), even frames record the consumer at once (GPU spin path).
// Reference: nr_run2 from the same nr_runtime.dll in its own contexts, fed the same frames (inputs pre-clamped to [0,1], LDR encode).
// Modes: exact | hist | passes2 | passes3 | abandon | resample | leak | refuse | refuse_dll | bench
// The test writes <dll dir>\Rdna2NrRuntime.ini first. Env: NR_TEST_DATA (nr_data), NR_TEST_WEIGHTS (nvngx_dlssnr.dll), NR_TEST_COLOR (RGBA16F WxH).
// Every GPU wait has a timeout (FAIL, never a hang).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "LmxxfNrApi.h"
#define CK(x) do { HRESULT _h = (x); if (FAILED(_h)) { printf("FAIL %s -> 0x%08lx (line %d)\n", #x, (unsigned long)_h, __LINE__); exit(10); } } while (0)
#define REQ(c, ...) do { if (!(c)) { printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); exit(11); } } while (0)

static ID3D12Device* dev; static ID3D12CommandQueue* q; static ID3D12CommandAllocator* al; static ID3D12GraphicsCommandList* cl; static ID3D12Fence* fence; static UINT64 fv; static HANDLE ev;
static LmxxfNrApi api; static std::string dir; static unsigned W = 960, H = 540;
static double now() { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return c.QuadPart * 1000.0 / f.QuadPart; }
static bool gpuWait(DWORD ms) { q->Signal(fence, ++fv); if (fence->GetCompletedValue() >= fv) return true; fence->SetEventOnCompletion(fv, ev); return WaitForSingleObject(ev, ms) == WAIT_OBJECT_0; }
static std::string lastErr() { char b[512] = ""; api.GetLastError(b, sizeof b); return b; }
static ID3D12Resource* buf(D3D12_HEAP_TYPE t, UINT64 sz, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = t; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; ID3D12Resource* r = nullptr; CK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r))); return r; }
static ID3D12Resource* tex(unsigned w, unsigned h, DXGI_FORMAT f) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT; D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = w; d.Height = h; d.DepthOrArraySize = 1;
    d.MipLevels = 1; d.Format = f; d.SampleDesc.Count = 1; ID3D12Resource* r = nullptr; CK(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&r))); return r; }
static void bar(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x = {}; x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource = r; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a; x.Transition.StateAfter = b; cl->ResourceBarrier(1, &x); }
static D3D12_TEXTURE_COPY_LOCATION fp(ID3D12Resource* b, unsigned w, unsigned h, UINT pitch, DXGI_FORMAT f) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = b; l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; l.PlacedFootprint.Footprint = {f, w, h, 1, pitch}; return l; }
static D3D12_TEXTURE_COPY_LOCATION sub(ID3D12Resource* t) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = t; l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; return l; }
static float h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(1024 + m), e - 25); return (h & 0x8000) ? -v : v; }
static uint16_t f2h(float f) {
    if (!(f == f)) return 0; if (f > 65504.f) f = 65504.f; if (f < -65504.f) f = -65504.f;
    uint32_t b; memcpy(&b, &f, 4); uint16_t sgn = (uint16_t)((b >> 16) & 0x8000); b &= 0x7FFFFFFF; float a; memcpy(&a, &b, 4);
    if (a < 6.1035156e-5f) return sgn | (uint16_t)lrintf(a * 16777216.f);
    int e; float m = frexpf(a, &e); uint32_t mant = (uint32_t)lrintf((m * 2.f - 1.f) * 1024.f); int he = e - 1 + 15;
    if (mant == 1024) { mant = 0; he++; } if (he >= 31) return sgn | 0x7BFF; return sgn | (uint16_t)((he << 10) | mant);
}

// ---- nr_runtime reference
struct nr_ctx;
static nr_ctx* (*r_create)(const char*, int, int, int, char*, int); static int (*r_run2)(nr_ctx*, const void*, void*, float*, int); static void (*r_destroy)(nr_ctx*);
static int (*r_cond)(nr_ctx*, int, float, float, float, int); static void (*r_select)(const void*, int); static void (*r_weights)(const char*);
static void loadRef() {
    HMODULE m = GetModuleHandleA("nr_runtime.dll"); if (!m) m = LoadLibraryA((dir + "\\nr_runtime.dll").c_str()); REQ(m, "no nr_runtime.dll");
    r_create = (decltype(r_create))GetProcAddress(m, "nr_create"); r_run2 = (decltype(r_run2))GetProcAddress(m, "nr_run2"); r_destroy = (decltype(r_destroy))GetProcAddress(m, "nr_destroy");
    r_cond = (decltype(r_cond))GetProcAddress(m, "nr_set_conditioning"); r_select = (decltype(r_select))GetProcAddress(m, "nr_select_device"); r_weights = (decltype(r_weights))GetProcAddress(m, "nr_set_weights_dll");
    REQ(r_create && r_run2 && r_destroy && r_cond && r_select && r_weights, "nr_runtime exports");
}

// ---- host
struct Host {
    ID3D12Resource *color, *motion, *up, *rb; uint8_t *upp, *rbp; unsigned cw, ch; UINT pitch, opitch; void* ctx = nullptr;
    LmxxfNrJob prev{}; bool havePrev = false; ID3D12Resource* prevOut = nullptr; unsigned frame = 0, readyFirst = 0, polled = 0; double readyMs = 0;
};
static void hostInit(Host& h, unsigned cw, unsigned ch) {
    h.cw = cw; h.ch = ch; h.pitch = (cw * 8 + 255) & ~255u; h.opitch = h.pitch;
    h.color = tex(cw, ch, DXGI_FORMAT_R16G16B16A16_FLOAT); h.motion = tex(cw, ch, DXGI_FORMAT_R16G16_FLOAT);
    h.up = buf(D3D12_HEAP_TYPE_UPLOAD, (UINT64)h.pitch * ch + 65536, D3D12_RESOURCE_STATE_GENERIC_READ); h.rb = buf(D3D12_HEAP_TYPE_READBACK, (UINT64)h.opitch * ch, D3D12_RESOURCE_STATE_COPY_DEST);
    CK(h.up->Map(0, nullptr, (void**)&h.upp)); CK(h.rb->Map(0, nullptr, (void**)&h.rbp));
    // zero motion: upload once
    memset(h.upp, 0, (size_t)h.pitch * ch); CK(al->Reset()); CK(cl->Reset(al, nullptr)); bar(h.motion, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    auto d = sub(h.motion); auto s = fp(h.up, cw, ch, (cw * 4 + 255) & ~255u, DXGI_FORMAT_R16G16_FLOAT); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
    bar(h.motion, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE); CK(cl->Close()); { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); } REQ(gpuWait(5000), "init timeout");
}
static void hostFree(Host& h) { h.color->Release(); h.motion->Release(); h.up->Release(); h.rb->Release(); }
struct FrameOpt { unsigned passes = 1; bool temporal = false, submit = true, feed = true, abandonNow = false; uint32_t flags = 0; };
// One host frame. Returns true when the previous job's answer was read back into `answer` (cw*ch half4, packed).
static bool hostFrame(Host& h, const uint16_t* img, const FrameOpt& o, std::vector<uint16_t>* answer, LmxxfNrJob* jobOut = nullptr) {
    if (h.havePrev && (h.frame & 1)) {   // AMDNR no-motion mode: consume only when OutputReady says so (bounded here)
        uint32_t r = 0; const double t = now(); h.polled++; REQ(api.OutputReady(h.ctx, h.prev.handle, &r) == 0, "OutputReady: %s", lastErr().c_str()); if (r) h.readyFirst++;
        while (!r && now() - t < 3000) { Sleep(0); api.OutputReady(h.ctx, h.prev.handle, &r); } REQ(r, "OutputReady never 1"); h.readyMs += now() - t;
    }
    CK(al->Reset()); CK(cl->Reset(al, nullptr));
    const bool consume = h.havePrev && o.submit;   // a list that is never executed consumes nothing (the previous job stays pending)
    if (consume) {
        REQ(api.RecordOutputs(h.ctx, h.prev.handle, cl) == 0, "RecordOutputs: %s", lastErr().c_str());
        ID3D12Resource* po = (ID3D12Resource*)h.prev.private_output; bar(po, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto d = fp(h.rb, h.cw, h.ch, h.opitch, DXGI_FORMAT_R16G16B16A16_FLOAT); auto s = sub(po); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
        bar(po, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    LmxxfNrJob job{}; job.struct_size = sizeof job; bool fed = false;
    if (o.feed) {
        for (unsigned y = 0; y < h.ch; y++) memcpy(h.upp + (size_t)y * h.pitch, img + (size_t)y * h.cw * 4, (size_t)h.cw * 8);
        bar(h.color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        auto d = sub(h.color); auto s = fp(h.up, h.cw, h.ch, h.pitch, DXGI_FORMAT_R16G16B16A16_FLOAT); cl->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
        bar(h.color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        LmxxfNrFrameInfo fi{}; fi.struct_size = sizeof fi; fi.session_id = 1; fi.frame_id = fi.list_generation = h.frame + 1; fi.command_list = cl;
        fi.color_width = h.cw; fi.color_height = h.ch; fi.color = h.color; fi.color_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        fi.flags = LMXXF_NR_FRAME_FLAG_STRENGTH | (o.temporal ? LMXXF_NR_FRAME_FLAG_TEMPORAL : 0) | o.flags; fi.transfer_strength = fi.color_strength = 1.f; fi.model_scale = 1.f;
        fi.motion = h.motion; fi.motion_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; fi.motion_width = h.cw; fi.motion_height = h.ch; fi.motion_scale_x = fi.motion_scale_y = 1.f;
        fi.history_reset = h.frame == 0; fi.passes = o.passes; fi.output_smooth_threshold = 6.f / 255.f;
        REQ(api.PrepareFrame(h.ctx, &fi, &job) == 0, "PrepareFrame: %s", lastErr().c_str());
        REQ(api.RecordInputs(h.ctx, job.handle, cl) == 0, "RecordInputs: %s", lastErr().c_str()); fed = true;
    }
    CK(cl->Close());
    if (o.submit) { ID3D12CommandList* l[] = {cl}; q->ExecuteCommandLists(1, l); }
    if (fed) REQ(api.EnqueueHipAsync(h.ctx, job.handle, q) == 0, "EnqueueHipAsync: %s", lastErr().c_str());
    if (fed && o.abandonNow) { const double t = now(); REQ(api.AbandonJob(h.ctx, job.handle) == 0, "AbandonJob: %s", lastErr().c_str()); printf("AbandonJob returned after %.1f ms (job %s)\n", now() - t, o.submit ? "in flight" : "never submitted"); fed = false; }
    if (o.submit) REQ(gpuWait(5000), "GPU timeout");
    if (consume) { api.Retire(h.ctx, h.prev.handle); if (answer) { answer->resize((size_t)h.cw * h.ch * 4); for (unsigned y = 0; y < h.ch; y++) memcpy(answer->data() + (size_t)y * h.cw * 4, h.rbp + (size_t)y * h.opitch, (size_t)h.cw * 8); } }
    h.havePrev = fed || (h.havePrev && !consume); if (fed) h.prev = job; if (jobOut) *jobOut = job; h.frame++;
    return consume;
}
static bool waitReady(Host& h, std::vector<uint16_t>& img, FrameOpt o, double ms) {   // passthrough frames until the network is loaded (GetStatus has no "loading")
    const double t = now(); char st[512];
    for (;;) { api.GetStatus(h.ctx, st, sizeof st); if (!strstr(st, "loading") && strstr(st, "net=") && !strstr(st, "net=none")) { if (strstr(st, "failed")) { printf("network failed: %s\n", st); return false; } break; }
               if (now() - t > ms) { printf("network not ready after %.0f ms: %s\n", ms, st); return false; }
               hostFrame(h, img.data(), o, nullptr); Sleep(20); }
    return true;
}
static void drain(Host& h, std::vector<uint16_t>* a) { FrameOpt o; o.feed = false; hostFrame(h, nullptr, o, a); }

static std::vector<uint16_t> loadColor(unsigned w, unsigned h) {
    char p[512]; if (const char* e = getenv("NR_TEST_COLOR")) snprintf(p, sizeof p, "%s", e); else snprintf(p, sizeof p, "E:\\dlss-runtime\\color_%ux%u.bin", w, h);
    std::vector<uint16_t> v((size_t)w * h * 4); FILE* f = fopen(p, "rb"); REQ(f && fread(v.data(), 8, (size_t)w * h, f) == (size_t)w * h, "cannot read %s", p); fclose(f); return v;
}
static std::vector<uint16_t> shifted(const std::vector<uint16_t>& b, unsigned w, unsigned h, int dx) {   // frame i: base moved by dx px (wrap), rgb clamped to [0,1], alpha 1
    std::vector<uint16_t> o(b.size());
    for (unsigned y = 0; y < h; y++) for (unsigned x = 0; x < w; x++) { const uint16_t* s = &b[((size_t)y * w + (x + dx) % w) * 4]; uint16_t* d = &o[((size_t)y * w + x) * 4];
        for (int c = 0; c < 3; c++) { float v = h2f(s[c]); d[c] = f2h(v == v ? std::min(std::max(v, 0.f), 1.f) : 0.f); } d[3] = 0x3C00; }
    return o;
}
struct Cmp { double psnr; size_t exact, n; };
static Cmp compare(const std::vector<uint16_t>& out, const std::vector<uint16_t>& ref) {   // rgb; ref = network output (half), expected max(san(ref), 0)
    double se = 0; size_t ex = 0, n = 0;
    for (size_t i = 0; i < ref.size(); i++) { if ((i & 3) == 3) continue; float r = h2f(ref[i]); r = r == r ? std::min(std::max(r, 0.f), 65504.f) : 0.f;
        float o = h2f(out[i]); ex += f2h(r) == out[i] || (r == 0.f && o == 0.f); se += (double)(o - r) * (o - r); n++; }
    return {se ? 10 * log10(n / se) : 999.0, ex, n};
}
static const char* writeIni(const char* extra) {
    const char* data = getenv("NR_TEST_DATA") ? getenv("NR_TEST_DATA") : "E:\\dlss5-v0.4\\nr_data";
    const char* wts = getenv("NR_TEST_WEIGHTS") ? getenv("NR_TEST_WEIGHTS") : "E:\\dlss-runtime\\v042\\nvngx_dlssnr.dll";
    FILE* f = fopen((dir + "\\Rdna2NrRuntime.ini").c_str(), "w"); REQ(f, "cannot write ini");
    fprintf(f, "[nr]\ndata_dir=%s\nweights_dll=%s\nlog=test_amdnr.log\n%s\n", data, wts, extra); fclose(f); return data;
}
static void* create(ID3D12Device* d, ID3D12CommandQueue* qq, int* rc) {
    LmxxfNrCreateInfo ci{}; ci.struct_size = sizeof ci; ci.device = d; ci.queue = qq; ci.assets_directory = L""; void* c = nullptr; *rc = api.Create(&ci, &c);
    if (!*rc) REQ(api.PrepareSession(c) == 0, "PrepareSession"); return c;
}
static std::string stepDir(void* ctx, const char* data, unsigned w, unsigned h) {
    char st[512]; api.GetStatus(ctx, st, sizeof st); const char* m = strstr(st, "model="); REQ(m, "status: %s", st); char id[17] = {}; memcpy(id, m + 6, 16);
    char p[600]; snprintf(p, sizeof p, "%s\\%s\\%ux%u", data, id, w, h); return p;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 3) { puts("usage: test_amdnr <dll dir> <mode> [W H]"); return 1; }
    dir = argv[1]; const std::string mode = argv[2]; if (argc > 4) { W = atoi(argv[3]); H = atoi(argv[4]); }
    const bool strict = mode == "exact" || mode == "abandon" || mode.rfind("passes", 0) == 0;
    const char* data = writeIni(mode == "hist" ? "vit_every=1\nencode=ldr\nhistory=1\nhistory_gate_lo=1\nhistory_gate_hi=2" : strict ? "vit_every=1\nencode=ldr\nhistory=0"
                              : mode == "refuse_dll" ? "weights_dll=C:\\Windows\\System32\\d3d12.dll" : mode == "resample" ? "vit_every=1\nmodel_res=960x540" : mode == "leak" ? "vit_every=1" : "");
    if (mode == "refuse_dll") { FILE* f = fopen((dir + "\\Rdna2NrRuntime.ini").c_str(), "w"); fprintf(f, "[nr]\ndata_dir=%s\nweights_dll=C:\\Windows\\System32\\d3d12.dll\nlog=test_amdnr.log\n", data); fclose(f); }
    HMODULE m = LoadLibraryA((dir + "\\Rdna2NrRuntime.dll").c_str()); REQ(m, "load Rdna2NrRuntime.dll: %lu", GetLastError());
    auto getApi = (int32_t(*)(uint32_t, LmxxfNrApi*))GetProcAddress(m, "LmxxfNrGetApi"); REQ(getApi, "no LmxxfNrGetApi");
    api.struct_size = sizeof api; REQ(getApi(LMXXF_NR_ABI_VERSION, &api) == 0 && api.EnqueueHipAsync && api.AbandonJob && api.OutputReady, "LmxxfNrGetApi");   // AMDNR's own acceptance check
    { void** f = (void**)&api.QueryCapabilities; for (int i = 0; i < 19; i++) REQ(f[i], "table entry %d null", i); }
    REQ(GetProcAddress(m, "LmxxfNrGetImportPoolStats") && GetProcAddress(m, "LmxxfNrSetTierPolicy"), "optional exports");
    { LmxxfNrApi a2{}; a2.struct_size = sizeof a2; REQ(getApi(2, &a2) == LMXXF_NR_UNSUPPORTED_ABI, "ABI 2 accepted"); printf("ABI 2 refused: %s\n", lastErr().c_str()); }
    LmxxfNrCapabilities caps{}; caps.struct_size = sizeof caps; REQ(api.QueryCapabilities(&caps) == 0, "caps");

    IDXGIFactory4* fac; CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac))); IDXGIAdapter1* ad = nullptr;
    for (UINT i = 0;; i++) { IDXGIAdapter1* a; if (fac->EnumAdapters1(i, &a) != S_OK) break; DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d); if (d.VendorId == 0x1002 && d.DedicatedVideoMemory > (2ull << 30)) { ad = a; break; } a->Release(); }
    REQ(ad, "no AMD dGPU adapter");
    CK(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd = {}; CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al)));
    CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, al, nullptr, IID_PPV_ARGS(&cl))); cl->Close();
    CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))); ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    if (mode == "refuse" || mode == "refuse_dll") {
        int rc; void* c;
        if (mode == "refuse_dll") { c = create(dev, q, &rc); REQ(rc != 0 && !c, "unknown nvngx_dlssnr.dll accepted"); printf("PASS unknown DLL refused (%d): %s\n", rc, lastErr().c_str()); return 0; }
        const bool sim = getenv("NR_HIP_DEVICE") && atoi(getenv("NR_HIP_DEVICE")) < 0;
        c = create(dev, q, &rc);
        if (sim) { REQ(rc != 0 && !c, "LUID mismatch accepted"); printf("PASS LUID mismatch refused (%d): %s\n", rc, lastErr().c_str()); return 0; }
        REQ(rc == 0 && c, "real RDNA2 device refused: %s", lastErr().c_str()); api.Destroy(c); printf("real device accepted\n");
        int refused = 0;
        for (UINT i = 0;; i++) {   // every other adapter (iGPU, WARP, other vendors) must be refused cleanly
            IDXGIAdapter1* a; if (fac->EnumAdapters1(i, &a) != S_OK) break; DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d); if (a == ad || (d.VendorId == 0x1002 && d.DedicatedVideoMemory > (2ull << 30))) { a->Release(); continue; }
            ID3D12Device* d2 = nullptr; if (FAILED(D3D12CreateDevice(a, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d2)))) { a->Release(); continue; }
            ID3D12CommandQueue* q2; CK(d2->CreateCommandQueue(&qd, IID_PPV_ARGS(&q2))); void* c2 = create(d2, q2, &rc);
            printf("adapter %ls (%04x:%04x): rc %d %s\n", d.Description, d.VendorId, d.DeviceId, rc, lastErr().c_str()); REQ(rc != 0 && !c2, "non-RDNA2 adapter accepted"); refused++;
            q2->Release(); d2->Release(); a->Release();
        }
        { IDXGIAdapter* wa; CK(fac->EnumWarpAdapter(IID_PPV_ARGS(&wa))); ID3D12Device* d2; CK(D3D12CreateDevice(wa, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d2))); ID3D12CommandQueue* q2; CK(d2->CreateCommandQueue(&qd, IID_PPV_ARGS(&q2)));
          void* c2 = create(d2, q2, &rc); printf("WARP: rc %d %s\n", rc, lastErr().c_str()); REQ(rc != 0 && !c2, "WARP accepted"); refused++;
          ID3D12CommandQueue* qx; CK(d2->CreateCommandQueue(&qd, IID_PPV_ARGS(&qx))); void* c3 = create(dev, qx, &rc); REQ(rc != 0 && !c3, "queue of another device accepted"); printf("queue/device mismatch: rc %d %s\n", rc, lastErr().c_str());
          qx->Release(); q2->Release(); d2->Release(); wa->Release(); }
        printf("PASS refuse: %d non-RDNA2 adapters refused, LUID check: run again with NR_HIP_DEVICE=-1\n", refused); return 0;
    }

    loadRef(); r_weights(getenv("NR_TEST_WEIGHTS") ? getenv("NR_TEST_WEIGHTS") : "E:\\dlss-runtime\\v042\\nvngx_dlssnr.dll"); { LUID l = dev->GetAdapterLuid(); r_select(&l, 0); }
    std::vector<uint16_t> base = loadColor(W, H);
    if (mode == "leak") {
        typedef int (*P_mem)(size_t*, size_t*); HMODULE hip = GetModuleHandleA("amdhip64_6.dll"); auto mem = hip ? (P_mem)GetProcAddress(hip, "hipMemGetInfo") : nullptr; REQ(mem, "hipMemGetInfo");
        std::vector<uint16_t> img = shifted(base, W, H, 0); size_t v0 = 0, p0 = 0; DWORD h0 = 0; const double t0 = now(); const int N = argc > 5 ? atoi(argv[5]) : 50;
        for (int i = 0; i < N; i++) {
            int rc; Host h; h.ctx = create(dev, q, &rc); REQ(rc == 0, "Create %d: %s", i, lastErr().c_str()); hostInit(h, W, H);
            REQ(waitReady(h, img, FrameOpt(), 30000), "cycle %d", i); for (int k = 0; k < 2; k++) hostFrame(h, img.data(), FrameOpt(), nullptr);
            if (i % 3 == 2) { FrameOpt o; o.abandonNow = true; hostFrame(h, img.data(), o, nullptr); }   // every 3rd cycle ends on an abandoned job
            drain(h, nullptr); REQ(api.Destroy(h.ctx) == 0, "Destroy"); hostFree(h);
            size_t fr = 0, tot = 0; mem(&fr, &tot); PROCESS_MEMORY_COUNTERS_EX pm{}; GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pm, sizeof pm); DWORD hc = 0; GetProcessHandleCount(GetCurrentProcess(), &hc);
            if (i == 2) { v0 = fr; p0 = pm.PrivateUsage; h0 = hc; }
            if (i < 3 || i % 10 == 9 || i == N - 1) printf("cycle %2d: free VRAM %zu MB, private %zu MB, handles %lu (%.0f ms)\n", i, fr >> 20, pm.PrivateUsage >> 20, hc, now() - t0);
            if (i == N - 1) { const double dv = ((double)v0 - fr) / 1048576, dp = ((double)pm.PrivateUsage - p0) / 1048576; const long dh = (long)hc - (long)h0;
                const bool ok = dv < 64 && dp < 64 && dh < 64; printf("%s leak: %d cycles, after cycle 2 -> %d: VRAM %+.0f MB used, private %+.0f MB, handles %+ld\n", ok ? "PASS" : "FAIL", N, N - 1, dv, dp, dh); return ok ? 0 : 9; }
        }
    }
    int rc; Host h; h.ctx = create(dev, q, &rc); REQ(rc == 0, "Create: %s", lastErr().c_str()); hostInit(h, W, H);
    printf("caps: max %ux%u history %u overlap %u hip_ready %u\n", caps.max_input_width, caps.max_input_height, caps.history_supported, caps.overlap_supported, caps.hip_ready);
    FrameOpt o; o.passes = mode == "passes2" ? 2 : mode == "passes3" ? 3 : 1; o.temporal = mode == "hist" || mode == "bench";
    std::vector<uint16_t> img0 = shifted(base, W, H, 0);
    if (mode == "bench") {   // timings: default ini (data dir's vit_every + bottleneck reuse, history on), EnqueueHipAsync -> OutputReady
        if (argc > 5) o.passes = atoi(argv[5]);
        { const double t = now(); REQ(waitReady(h, img0, o, 60000), "ready"); printf("network load %.0f ms\n", now() - t); }
        drain(h, nullptr); std::vector<double> lat, fr; const int N = 120;
        for (int i = 0; i < N; i++) {
            const double t = now(); LmxxfNrJob j; FrameOpt b = o; hostFrame(h, img0.data(), b, nullptr, &j); uint32_t r = 0;
            while (now() - t < 3000 && api.OutputReady(h.ctx, j.handle, &r) == 0 && !r) Sleep(0);
            if (i >= 10) { lat.push_back(now() - t); }
        }
        drain(h, nullptr); std::sort(lat.begin(), lat.end()); char st[512]; api.GetStatus(h.ctx, st, sizeof st);
        printf("BENCH %ux%u passes %u: frame (record+submit+GPU copy) -> OutputReady median %.2f ms, p90 %.2f, min %.2f (%zu jobs) | %s\n", W, H, o.passes, lat[lat.size() / 2], lat[lat.size() * 9 / 10], lat[0], lat.size(), st);
        api.Destroy(h.ctx); return 0;
    }
    REQ(waitReady(h, img0, o, 60000), "network ready");
    drain(h, nullptr);   // passthrough jobs from the loading phase are consumed
    const std::string sd = stepDir(h.ctx, data, mode == "resample" ? 960 : W, mode == "resample" ? 540 : H);
    if (mode == "resample") {   // colour WxH, network 960x540: output finite, near the input, and != input
        std::vector<uint16_t> a; double d = 0, mx = 0; size_t bad = 0, ch = 0;
        for (int i = 0; i < 6; i++) hostFrame(h, img0.data(), o, i ? &a : nullptr);
        drain(h, &a);
        for (size_t i = 0; i < a.size(); i++) { if ((i & 3) == 3) continue; float v = h2f(a[i]); if (!(v == v) || fabsf(v) > 1e4f) { bad++; continue; } float e = fabsf(v - h2f(img0[i])); d += e; mx = std::max(mx, (double)e); ch += e > 1e-3f; }
        printf("%s resample %ux%u -> 960x540 network: non-finite %zu, mean |out-in| %.4f, max %.3f, changed %.1f%%\n", bad == 0 && d > 0 ? "PASS" : "FAIL", W, H, bad, d / (a.size() * 0.75), mx, 100.0 * ch / (a.size() * 0.75));
        api.Destroy(h.ctx); return bad == 0 && d > 0 ? 0 : 9;
    }
    // reference contexts (one per pass, as the runtime chains them)
    nr_ctx* ref[3] = {}; char e[512] = "";
    for (unsigned p = 0; p < o.passes; p++) { ref[p] = r_create(sd.c_str(), (int)W, (int)H, 1, e, sizeof e); REQ(ref[p], "reference nr_create %s: %s", sd.c_str(), e); r_cond(ref[p], 0, 1.f, 1.f, -1.f, 1); }
    const size_t nb = (size_t)W * H * 4; std::vector<uint16_t> t1(nb), t2(nb);
    auto refRun = [&](const std::vector<uint16_t>& in, int hv, std::vector<uint16_t>& out) {
        out.resize(nb); const uint16_t* src = in.data();
        for (unsigned p = 0; p < o.passes; p++) { std::vector<uint16_t>& dst = p + 1 == o.passes ? out : (p & 1 ? t2 : t1); float ms; REQ(r_run2(ref[p], src, dst.data(), &ms, p == 0 ? hv : 0) == 0, "ref run"); src = dst.data(); } };
    const int N = mode == "hist" ? 12 : 10; double worst = 999; size_t exact = 0, total = 0; int compared = 0, refFrame = 0;
    std::vector<uint16_t> a, r;
    std::vector<std::vector<uint16_t>> pending;   // inputs whose answer is still to be read back, in order
    for (int i = 0; i <= N; i++) {
        const bool last = i == N; std::vector<uint16_t> img = mode == "hist" ? img0 : shifted(base, W, H, 13 * i);
        FrameOpt fo = o; if (last) fo.feed = false;
        if (mode == "abandon" && i == 4) fo.abandonNow = true;                      // in flight on the GPU
        if (mode == "abandon" && i == 6) { fo.abandonNow = true; fo.submit = false; }   // list never executed
        const bool got = hostFrame(h, img.data(), fo, &a);
        if (got) {
            const std::vector<uint16_t> in = pending.front(); pending.erase(pending.begin());
            refRun(in, mode == "hist" && refFrame > 0 ? 1 : 0, r); refFrame++;
            Cmp c = compare(a, r); worst = std::min(worst, c.psnr); exact += c.exact; total += c.n; compared++;
            printf("job %d: %.2f dB, %zu/%zu values bit-exact\n", compared, c.psnr, c.exact, c.n);
        }
        if (fo.feed && !fo.abandonNow) pending.push_back(img);
        else if (fo.feed && fo.submit) { refRun(img, 0, r); refFrame++; }   // abandoned in flight: it ran in the runtime (the reference runs it too, keeps the frame count)
    }
    char st[512]; api.GetStatus(h.ctx, st, sizeof st);
    const bool ok = compared >= N - 3 && (exact == total || worst >= 50.0);
    printf("%s %s: %d jobs compared, worst %.2f dB, %zu/%zu values bit-exact (%.4f%%), OutputReady at first poll %u/%u (mean wait %.1f ms) | %s\n", ok ? "PASS" : "FAIL", mode.c_str(), compared, worst, exact, total,
           100.0 * exact / std::max<size_t>(total, 1), h.readyFirst, h.polled, h.polled ? h.readyMs / h.polled : 0.0, st);
    for (auto x : ref) if (x) r_destroy(x);
    REQ(api.Destroy(h.ctx) == 0, "Destroy"); return ok ? 0 : 9;
}
