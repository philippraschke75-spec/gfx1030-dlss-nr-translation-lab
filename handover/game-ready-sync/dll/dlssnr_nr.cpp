// Drop-in nvngx.dll_dlssnr.dll that runs the RDNA2 DLSS-NR port through nr_runtime.dll (HIP).
// Same 12 exports as the shipped vendor-neutral port (see ../INTERFACE.md). Fail-safe: anything unexpected -> evaluate
// returns 0 ("not run") and the host keeps the native frame.
//
// Sync design (default, sync=1; DESIGN.md): evaluate N records into the host's list: colour -> shared io buffer (compute),
// WriteBufferImmediate(flag_in = N), a one-thread bounded spin on flag_out >= N, then io output -> `output` (compute).
// The HIP stream (queued at evaluate time) waits on flag_in, runs the network on io, writes flag_out. No CPU readback, no lag.
// If interop is unavailable (old runtime, import fails, no ID3D12GraphicsCommandList2) the async path below is used.
//
// Async design (fallback; forced by the host: evaluate only RECORDS into `cmd`, the host executes it later, so we cannot read the
// frame back synchronously and the network takes ~143 ms anyway):
//   evaluate N : if the worker is idle, record color -> readback copy (job). If a finished result exists, record
//                result -> output copy and return 1 (the result is stale by the network latency, ~10 frames at 60 fps).
//   worker     : create the HIP runtime once (seconds), then per job: wait until the readback copy landed (sentinel
//                polling, no queue access needed), run nr_run, publish the answer into an upload buffer.
// ponytail: single feature/pass; model size = one step dir (nr_data\\WxH) picked at create, fixed until the host recreates.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include "dlssnr_iface.h"
#include "r11g10.h"
#include "../runtime/weights_ht.h"
#include <cmath>
#include <algorithm>
#include <d3dcommon.h>

// ---------------------------------------------------------------- logging / settings
static CRITICAL_SECTION g_cs; static FILE* g_log; static char g_err[512] = "";
static std::wstring g_dir;                         // folder of this DLL
static std::string g_dataDir, g_modelRes = "auto", g_model = "auto", g_logName = "dlssnr_nr.log"; static int g_reuseBn = 0, g_vitEvery = 1, g_settleMs = 20, g_jobTimeoutMs = 3000;
static std::atomic<int> g_evalCalls{0};
static float g_intensity = -1.0f;                       // ini override of the host's intensity (output blend towards the NR result, 0..2); <0 = the host's value
static float g_iniCond[5] = {NAN, NAN, NAN, NAN, NAN};   // ini overrides: style, local_tone, local_structure, skin_structure, auto_mask; NaN = the host's value
static std::atomic<bool> g_exiting{false};              // set by atexit: at process exit never join/destroy anything (leak it, the OS reclaims)
static struct ExitHook { ExitHook() { atexit([] { g_exiting = true; }); } } g_exitHook;
static int g_apply = 1, g_gate = 1, g_ldrIn = 1, g_hdr = 2; static float g_hdrFrac = 0.125f, g_gain = 1.0f, g_gateLo = 0.03f, g_gateHi = 0.15f;   // apply=1: out = current frame + gain * (NR(old frame) - old frame) (no lag); apply=0: copy the finished (stale) NR image
static int g_allowAny = 0, g_evictMs = 2000; static int g_sync = 1, g_spinMs = 250, g_maxInflight = 3, g_fg = 0, g_fgWaitMs = 4, g_fgMaxAge = 3, g_mv = 0, g_nrEvery = 1; static float g_mvLo = 0.05f, g_mvHi = 0.25f;
static int g_busy = 0, g_busyWaitMs = 100;   // busy_mode: 0 wait (CPU waits, bounded, for a free HIP slot), 1 reuse (last NR edit, warped) - never a native frame once NR ran
static int g_compose = 0; static float g_detail = 1.f;   // compose= add (0: the edit added as is) | nv (1: NVIDIA's intensity) | ratio (2: OkLab matched residual); detail= live-frame detail above the model raster (1 = as is). handover/community-look/README.md
static float g_pass2Gain = 0.5f; static std::atomic<int> g_live{0}; static UINT g_hdrP0 = 0; static ID3D12Resource* g_hl = nullptr; static float g_hdrKey = 0.18f, g_hdrAdapt = 0.05f, g_hdrBias = 1.f, g_hdrExp = 0.f;   // OptiScaler "2 passes": a later pass (a second feature on pass 1's output) adds its edit at gain * pass2_gain
static float g_eT = 0.f, g_eL = 0.f; static int g_eFl = 0;   // edit_temporal (blend of the shown edit with the previous one, warped), edit_limit (cap vs local contrast), edit_warp bicubic (flag bit 2)
static int g_split = 0, g_splitK[2] = {0, 0}; static int g_hist = 0; static float g_hLo = 0.08f, g_hHi = 0.25f;   // history=1: temporal history (previous NR output warped by the game's MVs) into the pre-block's +0x40, see handover/temporal-2026-10-11/DLL.md   // mv_reuse/nr_every: NR every Nth frame, in between the last edit warped by the game's motion vectors   // sync=1: same-frame NR through D3D12<->HIP shared memory (see DESIGN.md); falls back to the async path when unavailable

static void logf(const char* fmt, ...) {
    EnterCriticalSection(&g_cs);
    if (g_log) { va_list a; va_start(a, fmt); vfprintf(g_log, fmt, a); va_end(a); fputc('\n', g_log); fflush(g_log); }
    LeaveCriticalSection(&g_cs);
}
static void seterr(const char* fmt, ...) { va_list a; va_start(a, fmt); vsnprintf(g_err, sizeof g_err, fmt, a); va_end(a); logf("ERROR: %s", g_err); }
static std::string narrow(const std::wstring& w) { int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, 0, 0, 0, 0); std::string s(n ? n - 1 : 0, 0); if (n) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, &s[0], n, 0, 0); return s; }

static void loadSettings() {
    std::wstring ini = g_dir + L"nr_port.ini"; char b[1024];
    GetPrivateProfileStringA("nr", "data_dir", "nr_data", b, sizeof b, narrow(ini).c_str()); g_dataDir = b;
    if (g_dataDir.size() < 2 || (g_dataDir[1] != ':' && g_dataDir[0] != '\\' && g_dataDir[0] != '/')) g_dataDir = narrow(g_dir) + g_dataDir;   // relative -> next to the DLL
    GetPrivateProfileStringA("nr", "model_res", "auto", b, sizeof b, narrow(ini).c_str()); g_modelRes = b;   // WxH (a subfolder of data_dir) or auto
    GetPrivateProfileStringA("nr", "model", "auto", b, sizeof b, narrow(ini).c_str()); g_model = b;   // small (310.9 WEIGHTS_CC_SMALL_FP8) | large (WEIGHTS_HT) | auto = small when the DLL has it
    GetPrivateProfileStringA("nr", "log", "dlssnr_nr.log", b, sizeof b, narrow(ini).c_str()); g_logName = b;
    g_vitEvery = (int)GetPrivateProfileIntA("nr", "vit_every", 0, narrow(ini).c_str());   // 0/absent = the data dir's meta vit_every (build_plan: new model 3, old 2), else 1
    { char r[32]; GetPrivateProfileStringA("nr", "reuse", "vit", r, sizeof r, narrow(ini).c_str()); g_reuseBn = !_stricmp(r, "bottleneck"); }   // what vit_every>1 reuses
    g_settleMs = (int)GetPrivateProfileIntA("nr", "settle_ms", 20, narrow(ini).c_str());
    g_jobTimeoutMs = (int)GetPrivateProfileIntA("nr", "job_timeout_ms", 3000, narrow(ini).c_str());
    g_apply = (int)GetPrivateProfileIntA("nr", "apply", 1, narrow(ini).c_str());
    GetPrivateProfileStringA("nr", "gain", "1.0", b, sizeof b, narrow(ini).c_str()); g_gain = (float)atof(b);
    g_gate = (int)GetPrivateProfileIntA("nr", "gate", 1, narrow(ini).c_str());
    g_ldrIn = (int)GetPrivateProfileIntA("nr", "ldr_input", 1, narrow(ini).c_str());   // resampled path: clamp the network input to [0,1] (0 = old HDR input, white flicker)
    { char r[16]; GetPrivateProfileStringA("nr", "hdr_mode", "auto", r, sizeof r, narrow(ini).c_str()); g_hdr = !_stricmp(r, "off") ? 0 : !_stricmp(r, "on") ? 1 : 2; }
    GetPrivateProfileStringA("nr", "hdr_auto_frac", "0.125", b, sizeof b, narrow(ini).c_str()); g_hdrFrac = (float)atof(b); if (!(g_hdrFrac > 0.f && g_hdrFrac <= 1.f)) g_hdrFrac = 0.125f;   // auto: share of model pixels above 1.0 in one frame that switches HDR on (sticky)   // scene-linear HDR colour: tonemap into the network space and back (auto: RGBA16F/R9G9B9E5 with values > 4)
    GetPrivateProfileStringA("nr", "hdr_key", "0.18", b, sizeof b, narrow(ini).c_str()); g_hdrKey = (float)atof(b); if (!(g_hdrKey > 0.f && g_hdrKey < 10.f)) g_hdrKey = 0.18f;   // HDR auto-exposure: E = key / exp(mean log L), smoothed by hdr_adapt per frame
    GetPrivateProfileStringA("nr", "hdr_adapt", "0.05", b, sizeof b, narrow(ini).c_str()); g_hdrAdapt = (float)atof(b); if (!(g_hdrAdapt > 0.f && g_hdrAdapt <= 1.f)) g_hdrAdapt = 0.05f;
    GetPrivateProfileStringA("nr", "hdr_exposure_bias", "1.0", b, sizeof b, narrow(ini).c_str()); g_hdrBias = (float)atof(b); if (!(g_hdrBias > 0.f && g_hdrBias < 100.f)) g_hdrBias = 1.f;
    GetPrivateProfileStringA("nr", "hdr_exposure", "0", b, sizeof b, narrow(ini).c_str()); g_hdrExp = (float)atof(b); if (!(g_hdrExp > 0.f && g_hdrExp < 1000.f)) g_hdrExp = 0.f;   // fixed E (0 = auto)
    GetPrivateProfileStringA("nr", "gate_lo", "0.03", b, sizeof b, narrow(ini).c_str()); g_gateLo = (float)atof(b);
    GetPrivateProfileStringA("nr", "gate_hi", "0.15", b, sizeof b, narrow(ini).c_str()); g_gateHi = (float)atof(b);
    GetPrivateProfileStringA("nr", "intensity", "-1", b, sizeof b, narrow(ini).c_str()); g_intensity = (float)atof(b); if (!(g_intensity >= 0.f)) g_intensity = -1.f; if (g_intensity > 2.f) g_intensity = 2.f;
    { const char* k[5] = {"style", "local_tone", "local_structure", "skin_structure", "auto_mask"};   // empty = follow the host (OptiScaler menu)
      for (int i = 0; i < 5; i++) { GetPrivateProfileStringA("nr", k[i], "", b, sizeof b, narrow(ini).c_str()); g_iniCond[i] = b[0] ? (float)atof(b) : NAN; } }
    g_sync = (int)GetPrivateProfileIntA("nr", "sync", 1, narrow(ini).c_str());
    g_allowAny = (int)GetPrivateProfileIntA("nr", "allow_any_gpu", 0, narrow(ini).c_str()); g_evictMs = (int)GetPrivateProfileIntA("nr", "evict_idle_ms", 2000, narrow(ini).c_str());
    g_spinMs = (int)GetPrivateProfileIntA("nr", "sync_timeout_ms", 250, narrow(ini).c_str()); g_spinMs = g_spinMs < 10 ? 10 : g_spinMs > 1000 ? 1000 : g_spinMs;   // GPU wait bound, far below the 2 s TDR
    g_fg = (int)GetPrivateProfileIntA("nr", "fg_mode", 0, narrow(ini).c_str());   // 1: frame-generation friendly sync (never native once an NR result exists; short bounded queue wait)
    g_fgWaitMs = (int)GetPrivateProfileIntA("nr", "fg_wait_ms", 4, narrow(ini).c_str()); g_fgWaitMs = g_fgWaitMs < 0 ? 0 : g_fgWaitMs > 1000 ? 1000 : g_fgWaitMs;
    g_fgMaxAge = (int)GetPrivateProfileIntA("nr", "fg_max_age", 3, narrow(ini).c_str()); g_fgMaxAge = g_fgMaxAge < 0 ? 0 : g_fgMaxAge > 60 ? 60 : g_fgMaxAge;   // fg: reuse an NR edit at most this many frames, fading out
    g_mv = (int)GetPrivateProfileIntA("nr", "mv_reuse", 0, narrow(ini).c_str());
    g_nrEvery = (int)GetPrivateProfileIntA("nr", "nr_every", 1, narrow(ini).c_str()); g_nrEvery = g_nrEvery < 1 ? 1 : g_nrEvery > 8 ? 8 : g_nrEvery;
    GetPrivateProfileStringA("nr", "mv_gate_lo", "0.05", b, sizeof b, narrow(ini).c_str()); g_mvLo = (float)atof(b);
    GetPrivateProfileStringA("nr", "mv_gate_hi", "0.25", b, sizeof b, narrow(ini).c_str()); g_mvHi = (float)atof(b);
    g_hist = (int)GetPrivateProfileIntA("nr", "history", 0, narrow(ini).c_str()); g_hist = g_hist < 0 ? 0 : g_hist > 2 ? 2 : g_hist;   // 1 = history into pre- and post-block, 2 = post-block lerp only (runtime hist_valid)
    GetPrivateProfileStringA("nr", "history_gate_lo", "0.08", b, sizeof b, narrow(ini).c_str()); g_hLo = (float)atof(b);
    GetPrivateProfileStringA("nr", "history_gate_hi", "0.25", b, sizeof b, narrow(ini).c_str()); g_hHi = (float)atof(b);
    { char m[16]; GetPrivateProfileStringA("nr", "compose", "add", m, sizeof m, narrow(ini).c_str()); g_compose = !_stricmp(m, "nv") ? 1 : !_stricmp(m, "ratio") ? 2 : 0; }
    GetPrivateProfileStringA("nr", "detail", "1", b, sizeof b, narrow(ini).c_str()); g_detail = (float)atof(b); g_detail = g_detail == g_detail ? (std::min)((std::max)(g_detail, 0.f), 3.f) : 1.f;
    GetPrivateProfileStringA("nr", "pass2_gain", "0.5", b, sizeof b, narrow(ini).c_str()); g_pass2Gain = (float)atof(b); g_pass2Gain = g_pass2Gain == g_pass2Gain ? (std::min)((std::max)(g_pass2Gain, 0.f), 1.f) : 0.5f;
    { const int i8 = (int)GetPrivateProfileIntA("nr", "int8_swin", -1, narrow(ini).c_str()); if (i8 >= 0) SetEnvironmentVariableA("NR_INT8_SWIN", i8 ? "1" : "0"); }   // int8 swin GEMMs; absent = the data dir's meta (build_plan: 1). nr_runtime reads it at nr_create
    GetPrivateProfileStringA("nr", "edit_temporal", "0", b, sizeof b, narrow(ini).c_str()); g_eT = (float)atof(b); g_eT = g_eT > 0.f ? (std::min)(g_eT, 0.95f) : 0.f;   // 0 = off (bit-exact); blend weight of the previous shown edit (warped, depth-gated, clamped to the 3x3 range of the current edit)
    GetPrivateProfileStringA("nr", "edit_limit", "0", b, sizeof b, narrow(ini).c_str()); g_eL = (float)atof(b); g_eL = g_eL > 0.f ? (std::min)(g_eL, 100.f) : 0.f;   // 0 = off; |edit| <= edit_limit x (3x3 luma range + 0.02)
    { char m[16]; GetPrivateProfileStringA("nr", "edit_warp", "bilinear", m, sizeof m, narrow(ini).c_str()); g_eFl = !_stricmp(m, "bicubic") ? 4 : 0; }   // with edit_temporal/edit_limit: Catmull-Rom fetches (sharper, more flicker)
    { const int i8 = (int)GetPrivateProfileIntA("nr", "int8_prepost", -1, narrow(ini).c_str()); if (i8 >= 0) SetEnvironmentVariableA("NR_INT8_PREPOST", i8 ? "1" : "0"); }   // opt-in int8 post-block MLP; absent = meta (default off)
    g_split = (int)GetPrivateProfileIntA("nr", "nr_split", 0, narrow(ini).c_str()); g_split = g_split <= 0 ? 0 : g_split == 1 ? 2 : (std::min)(g_split, 8);   // N >= 2: the network in N parts over N frames (1 = 2, as before), output = last finished NR, warped
    if (g_split) g_nrEvery = g_split;
    GetPrivateProfileStringA("nr", "nr_split_k", "0", b, sizeof b, narrow(ini).c_str()); g_splitK[0] = g_splitK[1] = 0; sscanf_s(b, "%d,%d", &g_splitK[0], &g_splitK[1]);   // first dispatch of part 2 (full[,skipping] frames); 0 = measure
    { char m[16]; GetPrivateProfileStringA("nr", "busy_mode", "wait", m, sizeof m, narrow(ini).c_str()); g_busy = !_stricmp(m, "reuse"); }
    g_busyWaitMs = (int)GetPrivateProfileIntA("nr", "busy_wait_ms", 100, narrow(ini).c_str()); g_busyWaitMs = g_busyWaitMs < 0 ? 0 : g_busyWaitMs > 1000 ? 1000 : g_busyWaitMs;
    g_maxInflight = (int)GetPrivateProfileIntA("nr", "max_inflight", g_fg ? 1 : 3, narrow(ini).c_str()); g_maxInflight = g_maxInflight < 1 ? 1 : g_maxInflight > 8 ? 8 : g_maxInflight;
    auto cl = [](float v, float lo, float hi) { return v != v ? lo : v < lo ? lo : v > hi ? hi : v; };   // ini garbage (NaN, huge, negative) never reaches the GPU constants
    g_gain = cl(g_gain, 0.f, 2.f); g_gateLo = cl(g_gateLo, 0.f, 1.f); g_gateHi = cl(g_gateHi, g_gateLo, 1.f); g_mvLo = cl(g_mvLo, 0.f, 4.f); g_mvHi = cl(g_mvHi, g_mvLo + 1e-4f, 8.f); g_hLo = cl(g_hLo, 0.f, 1.f); g_hHi = cl(g_hHi, g_hLo + 1e-4f, 2.f);
    g_settleMs = g_settleMs < 0 ? 0 : g_settleMs > 500 ? 500 : g_settleMs; g_jobTimeoutMs = g_jobTimeoutMs < 200 ? 200 : g_jobTimeoutMs > 60000 ? 60000 : g_jobTimeoutMs; if (g_vitEvery < 0) g_vitEvery = 0;
    std::string lp = (g_logName.size() > 1 && g_logName[1] == ':') ? g_logName : narrow(g_dir) + g_logName;
    fopen_s(&g_log, lp.c_str(), "a");
}
BOOL WINAPI DllMain(HINSTANCE h, DWORD why, LPVOID) {
    if (why == DLL_PROCESS_ATTACH) {
        InitializeCriticalSection(&g_cs);
        wchar_t p[MAX_PATH]; GetModuleFileNameW(h, p, MAX_PATH); g_dir = p; g_dir = g_dir.substr(0, g_dir.find_last_of(L'\\') + 1);
    }
    return TRUE;
}

// ---------------------------------------------------------------- nr_runtime (loaded dynamically)
struct nr_ctx;
typedef nr_ctx* (*P_create)(const char*, int, int, int, char*, int);
typedef int (*P_run)(nr_ctx*, const void*, void*, float*);
typedef void (*P_destroy)(nr_ctx*);
typedef int (*P_cond)(nr_ctx*, int, float, float, float, int);
static P_create p_create; static P_run p_run; static P_destroy p_destroy; static P_cond p_cond;
typedef int (*P_extOpen)(nr_ctx*, void*, unsigned long long, unsigned long long, unsigned long long, unsigned long long, void*, unsigned long long, unsigned long long, char*, int);
typedef int (*P_extEnqueue)(nr_ctx*, unsigned, int);
typedef int (*P_extHist)(nr_ctx*, unsigned long long, unsigned long long); typedef int (*P_histLayout)(nr_ctx*, unsigned*, unsigned*); typedef int (*P_extEnqueue2)(nr_ctx*, unsigned, int, int);
typedef int (*P_extEnqueue3)(nr_ctx*, unsigned, int, int, int); typedef int (*P_extSplit)(nr_ctx*, int*, int*, float*); static P_extEnqueue3 p_extEnqueue3; static P_extSplit p_extSplit;
typedef int (*P_extSplitN)(nr_ctx*, int, int*, int*, float*); static P_extSplitN p_extSplitN;
static void (*p_select)(const void*, int); static P_extOpen p_extOpen; static P_extEnqueue p_extEnqueue; static P_extHist p_extHist; static P_histLayout p_histLayout; static P_extEnqueue2 p_extEnqueue2;   // p_extHist: absent in an old runtime -> history off   // absent in an old nr_runtime.dll -> async path
static bool loadRuntime() {
    if (p_create) return true;
    std::wstring p = g_dir + L"nr_runtime.dll";
    HMODULE m = LoadLibraryExW(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);   // its HIP dependencies are searched beside it too
    if (!m) { seterr("cannot load nr_runtime.dll (error %lu) - is the HIP runtime (amdhip64_*.dll) on PATH or beside it?", GetLastError()); return false; }
    p_create = (P_create)GetProcAddress(m, "nr_create"); p_run = (P_run)GetProcAddress(m, "nr_run"); p_destroy = (P_destroy)GetProcAddress(m, "nr_destroy");
    if (!p_create || !p_run || !p_destroy) { p_create = nullptr; seterr("nr_runtime.dll lacks nr_create/nr_run/nr_destroy"); return false; }
    p_extOpen = (P_extOpen)GetProcAddress(m, "nr_ext_open"); p_extEnqueue = (P_extEnqueue)GetProcAddress(m, "nr_ext_enqueue"); p_extHist = (P_extHist)GetProcAddress(m, "nr_ext_history"); p_histLayout = (P_histLayout)GetProcAddress(m, "nr_hist_layout"); p_extEnqueue2 = (P_extEnqueue2)GetProcAddress(m, "nr_ext_enqueue2");
    p_extEnqueue3 = (P_extEnqueue3)GetProcAddress(m, "nr_ext_enqueue3"); p_extSplit = (P_extSplit)GetProcAddress(m, "nr_ext_split"); p_extSplitN = (P_extSplitN)GetProcAddress(m, "nr_ext_splitn");
    p_select = (void (*)(const void*, int))GetProcAddress(m, "nr_select_device");   // absent in an old runtime: it uses HIP device 0
    p_cond = (P_cond)GetProcAddress(m, "nr_set_conditioning");                   // absent in an old nr_runtime.dll: the conditioning baked into the data dir stays
    if (auto sl = (void (*)(void (*)(const char*)))GetProcAddress(m, "nr_set_log")) sl([](const char* s) { logf("%s", s); });   // hardware line + output self-check
    if (!p_cond) logf("nr_runtime.dll has no nr_set_conditioning (old runtime): style / tone / structure / skin stay at the data dir's values");
    return true;
}
// SEH wrappers: an access violation inside the HIP runtime must fail the feature, never the game (worker thread has no other guard)
static nr_ctx* safeCreate(const char* d, int w, int h, int v, char* e, int n) { __try { return p_create(d, w, h, v, e, n); } __except (EXCEPTION_EXECUTE_HANDLER) { snprintf(e, n, "access violation 0x%08lx in nr_create", (unsigned long)GetExceptionCode()); return nullptr; } }
static int safeRun(nr_ctx* c, const void* i, void* o, float* ms) { __try { return p_run(c, i, o, ms); } __except (EXCEPTION_EXECUTE_HANDLER) { return -1000; } }
static void sanitizeHalf4(uint16_t* p, size_t n) {   // NaN -> 0, +Inf -> 65504, -Inf -> 0 (k_import turns +Inf into NaN; unlanded readback rows are 0xFF.. = NaN)
    for (size_t i = 0; i < n; i++) { uint16_t h = p[i]; if ((h & 0x7C00) == 0x7C00) p[i] = ((h & 0x3FF) || (h & 0x8000)) ? 0 : 0x7BFF; }
}

// ---------------------------------------------------------------- half helpers
static float h2f(uint16_t h) { int e = (h >> 10) & 31, m = h & 1023; float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? INFINITY : std::ldexp((float)(1024 + m), e - 25); return (h & 0x8000) ? -v : v; }
static uint16_t f2h(float f) {                              // round to nearest half, clamp to +-65504
    if (!(f == f)) return 0; if (f > 65504.f) f = 65504.f; if (f < -65504.f) f = -65504.f;
    uint32_t b; memcpy(&b, &f, 4); uint16_t sgn = (uint16_t)((b >> 16) & 0x8000); b &= 0x7FFFFFFF; float a; memcpy(&a, &b, 4);
    if (a < 6.1035156e-5f) return sgn | (uint16_t)lrintf(a * 16777216.f);                  // subnormal half: units of 2^-24
    int e; float m = frexpf(a, &e); uint32_t mant = (uint32_t)lrintf((m * 2.f - 1.f) * 1024.f); int he = e - 1 + 15;   // a = m*2^e, m in [0.5,1)
    if (mant == 1024) { mant = 0; he++; } if (he >= 31) return sgn | 0x7BFF; return sgn | (uint16_t)((he << 10) | mant);
}

// ---------------------------------------------------------------- delta compute pass
static const char* kHlsl =
"Texture2D<float4> col : register(t0);\n"
"ByteAddressBuffer edit : register(t1);\n"
"ByteAddressBuffer oldc : register(t2);\n"
"RWTexture2D<float4> outp : register(u0);\n"
"cbuffer C : register(b0) { uint W; uint H; uint pitch; uint gateOn; float g0; float g1; };\n"
"[numthreads(8,8,1)] void main(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return;\n"
"  float4 c = col[id.xy]; uint2 r = edit.Load2(id.y * pitch + id.x * 8);\n"
"  float3 e = float3(f16tof32(r.x & 0xFFFF), f16tof32(r.x >> 16), f16tof32(r.y & 0xFFFF));\n"
"  float w = 1.0;\n"
"  if (gateOn != 0) { uint2 q = oldc.Load2(id.y * pitch + id.x * 8); float3 o = float3(f16tof32(q.x & 0xFFFF), f16tof32(q.x >> 16), f16tof32(q.y & 0xFFFF));\n"
"    float d = max(abs(c.r - o.r), max(abs(c.g - o.g), abs(c.b - o.b))); w = (d < 1e30) ? 1.0 - smoothstep(g0, g1, d) : 1.0; }\n"
"  outp[id.xy] = float4(max(c.rgb + e * w, 0.0), c.a);\n"
"}\n";
static const int kRing = 8, kDesc = 4;                                 // descriptor sets in flight

// ---------------------------------------------------------------- sync path shaders (one source, three entry points)
// imp : game colour (any of the 3 formats, the SRV converts) -> sanitized half4 network input in the shared io buffer
// spin: one thread polls the flag the HIP stream writes when frame `seq` is done; bounded (TDR-safe), result -> st[0]
// exp : NR result (or the live frame if spin timed out) -> output texture (the UAV converts to the game's format)
static const char* kSyncHlsl =
"Texture2D<float4> col : register(t0);\n"
"RWTexture2D<float4> outp : register(u0);\n"
"#ifdef NC\n#define GC\n#else\n#define GC globallycoherent\n#endif\n"   // NC (edit stabilisation on: exp, tea, teb): plain cached views; their reads are all behind UAV barriers (only spin polls)
"GC RWByteAddressBuffer io : register(u1);\n"
"GC RWByteAddressBuffer st : register(u2);\n"
"cbuffer C : register(b0) { uint W; uint H; uint seq; uint maxIt; uint outOff; uint flagOff; float gain; uint FWH; uint minSeq; float gLo; float gHi; uint gate; uint frame; uint maxAge;\n"
"  uint mvOn; float mvSx; float mvSy; uint mWH; uint mBase; uint dWH; uint dBase; uint dFl; float mLo; float mHi;\n"
"  uint histOff; uint hPWH; uint histOk; uint hMv; float hLo; float hHi; float tmap; uint histOut; uint split; uint hdrM; float hk; float hx; float hkb; float hA; float amp; uint cm; float dt; float eT; float eL; uint tFl; };\n"   // history: io offset of the PAD_W x PAD_H RGB32F slot, padded size, CPU says the previous frame was NR, motion usable, colour gate   // mv reuse: motion (t1) / depth (t2) size and subrect base (x | y << 16); dFl bit0 depth present, bit1 depth inverted
"Texture2D<float4> mot : register(t1);\n"
"Texture2D<float> dep : register(t2);\n"   // minSeq != 0: fg mode, reuse the edit captured from a frame >= minSeq;   // W,H model; FWH = frame W | H << 16
"RWByteAddressBuffer ob : register(u3);\n"
"globallycoherent RWByteAddressBuffer hl : register(u4);\n"   // process-global HDR latch: pass 0 publishes its st[16], extra passes (hdrM 3) follow it   // PACKED: frame-size R9G9B9E5 staging, copied into the output texture afterwards
"uint e5(float3 c) {\n"   // float3 -> R9G9B9E5_SHAREDEXP (N=9, B=15)
"  c = clamp(c, 0.0, 65408.0); float m = max(c.r, max(c.g, c.b)); int e = max(-16, (int)floor(log2(max(m, 1e-30)))) + 16;\n"
"  float d = exp2(e - 24); if (floor(m / d + 0.5) >= 512.0) { d *= 2.0; e++; } uint3 q = (uint3)floor(c / d + 0.5);\n"
"  return q.r | (q.g << 9) | (q.b << 18) | ((uint)e << 27); }\n"
"float3 boost(float4 c, float3 v);\n"
"void put(uint2 p, float4 v) {\n"
"  if (amp > 1.0 && any(v.rgb != col[p].rgb)) v.rgb = boost(col[p], v.rgb);\n"
"#ifdef PACKED\n"
"  ob.Store(p.y * ((((FWH & 0xFFFF) * 4) + 255) & ~255) + p.x * 4, e5(v.rgb));\n"
"#else\n"
"  outp[p] = v;\n"
"#endif\n"
"}\n"   // W,H model; FWH = frame W | H << 16
"float san(float v) { return (v != v || v < -65504.0) ? 0.0 : min(v, 65504.0); }\n"   // NaN/-Inf -> 0, +Inf -> 65504 (as the CPU path)
"uint pk(float a, float b) { return f32tof16(san(a)) | (f32tof16(san(b)) << 16); }\n"
"float4 bil(float2 p, uint off) {\n"   // bilinear fetch of a half4 model-size image in io at model coords p (pixel centres at .5)
"  p = clamp(p - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a = (uint2)p; uint2 b = min(a + 1, uint2(W - 1, H - 1)); float2 t = p - a;\n"
"  uint2 q[4] = { io.Load2(off + (a.y * W + a.x) * 8), io.Load2(off + (a.y * W + b.x) * 8), io.Load2(off + (b.y * W + a.x) * 8), io.Load2(off + (b.y * W + b.x) * 8) };\n"
"  float4 v[4]; for (int i = 0; i < 4; i++) v[i] = float4(f16tof32(q[i].x & 0xFFFF), f16tof32(q[i].x >> 16), f16tof32(q[i].y & 0xFFFF), f16tof32(q[i].y >> 16));\n"
"  return lerp(lerp(v[0], v[1], t.x), lerp(v[2], v[3], t.x), t.y); }\n"
"float4 sn(float4 v) { return float4(san(v.x), san(v.y), san(v.z), san(v.w)); }\n"
// compose= (cm != 0, handover/community-look/README.md). OkLab with Ottosson's constants, for the ratio mode. The ratio mode adapts
// NvidiaResidualModel from janblade's OptiScaler-F5-DLSSNR (GPL-3, shaders/dlssnr/precompile/dlssnr.hlsl); the nv mode follows
// dlssnr-pytorch layers/pre_post.py compose(); its intensity > 1 rule is boost() below (one implementation, cb[38] amp).
"float3 okl(float3 c) { c = saturate(c); float3 l = float3(dot(c, float3(0.4122214708, 0.5363325363, 0.0514459929)), dot(c, float3(0.2119034982, 0.6806995451, 0.1073969566)), dot(c, float3(0.0883024619, 0.2817188376, 0.6299787005)));\n"
"  l = pow(max(l, 1e-12), 1.0 / 3.0); return float3(dot(l, float3(0.2104542553, 0.7936177850, -0.0040720468)), dot(l, float3(1.9779984951, -2.4285922050, 0.4505937099)), dot(l, float3(0.0259040371, 0.7827717662, -0.8086757660))); }\n"
"float3 okr(float3 b) { float3 l = float3(dot(b, float3(1.0, 0.3963377774, 0.2158037573)), dot(b, float3(1.0, -0.1055613458, -0.0638541728)), dot(b, float3(1.0, -0.0894841775, -1.2914855480))); l = l * l * l;\n"
"  return float3(dot(l, float3(4.0767416621, -3.3077115913, 0.2309699292)), dot(l, float3(-1.2684380046, 2.6097574011, -0.3413193965)), dot(l, float3(-0.0041960863, -0.7034186147, 1.7076147010))); }\n"
// The network-space edit for the live colour y (network space), given the model input a and the network's answer b at this pixel
// (bilinear from the model raster) and the intensity g. ponytail: a and b are interpolated before the non-linear step, not per texel.
"float3 ced(float3 y, float3 a, float3 b, float g) {\n"
"  float3 d;\n"
"  if (cm == 1) d = saturate(a + g * (b - a)) - a;\n"   // nv: NVIDIA's post block (dlssnr-pytorch compose()): blend up to 1 (g <= 1, the CPU caps it); past 1 put()'s boost (amp) applies the luminance ratio ^ intensity, for every compose mode
"  else { float3 la = okl(a), lb = okl(b), ly = okl(y);\n"   // ratio: OkLab matched residual (janblade's Transfer 2, NvidiaResidualModel): log-luminance ratio x g (bounded 2x), chroma difference x g, on the live frame's own OkLab (g <= 1; intensity > 1 = boost)
"    float r = clamp(log(max(lb.x, 0.1)) - log(max(la.x, 0.1)), -2.0, 2.0) * saturate(10.0 * la.x);\n"
"    ly.x *= exp(clamp(g * r, -0.6931472, 0.6931472)); ly.yz += g * (lb.yz - la.yz); d = okr(ly) - saturate(y); }\n"
"  return d + (dt - 1.0) * (y - a); }\n"   // detail: (dt - 1) x the live frame's detail above what the network saw (janblade's ReplaceDetail idea)
// hdr_mode: scene-linear HDR colour (RE Engine) goes to the network as T(x) = sRGB(x / (1 + x)) (the network's own mode-1 proxy at scale 1)
// and comes back as Ti(T(live) + gain * edit). hdrM 0 off, 1 on, 2 auto: on once lat has latched st[16] (imp counts pixels > 1 into st[20]; lat latches when one frame had >= hdrM >> 2 of them, then resets the count).
// Off, every path below is the old LDR code (bit-exact).
"bool hd() { return hdrM == 1 || ((hdrM & 3) == 2 && st.Load(16) != 0) || ((hdrM & 3) == 3 && hl.Load(0) != 0); }\n"
"float Ex() { if (hx > 0.0) return hx; float e = asfloat(hl.Load(4)); return e > 0.0 ? e : 1.0; }\n"   // HDR exposure: hx fixed (extra pass 1, hdr_exposure), else the metered E lat keeps in hl[4]
"float3 T(float3 x) { x = max(x * Ex(), 0.0); x = x / (1.0 + x); return lerp(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, step(x, 0.0031308)); }\n"
"float3 Ti(float3 y) { y = saturate(y); float3 z = lerp(pow((y + 0.055) / 1.055, 2.4), y / 12.92, step(y, 0.04045)); z = min(z, 0.999); return z / (1.0 - z) / Ex(); }\n"   // <= 999: finite for e5 / half
"float3 L(float4 c) { return hd() ? T(sn(c).rgb) : sn(c).rgb; }\n"   // live colour in network space (gates)
"float3 cmp(float4 c, float3 d) { return hd() ? min(Ti(T(sn(c).rgb) + d), (1.0 + hk) * max(sn(c).rgb, 0.0) + 0.1 * hk / Ex()) : max(c.rgb + d, 0.0); }\n"   // live colour + network-space edit; HDR: at most (1+hk)x + 0.1hk brighter (pass 0 hk 1 = 2x + 0.1; extra pass hk pass2_gain, so passes cannot compound to 4x) (Ti is steep near 1, a small edit must not blow a highlight out)
// intensity > 1 (amp): dlssnr-pytorch PostBlock.compose / NVIDIA's library: the luminance ratio r = out/frame (network space, at intensity 1)
// is raised to the intensity (bounded to 0.5..2x) and the colour rescaled by it, which keeps the hue (the old linear extrapolation pushed the
// channels apart: chroma shift x2 at intensity 2). Over-range: the peak channel is brought back by one scalar (hue kept), not clamped per channel.
"float3 boost(float4 c, float3 v) {\n"
"  float3 a = L(c), b = hd() ? T(max(v, 0.0)) : max(v, 0.0); const float3 w = float3(0.2126, 0.7152, 0.0722);\n"
"  float r = max((dot(b, w) + 1.0 / 512.0) / (dot(a, w) + 1.0 / 512.0), 1e-6); float3 o = b * (clamp(pow(r, amp), 0.5, 2.0) / r);\n"
"  float m = max(o.r, max(o.g, o.b)), lim = max(1.0, max(b.r, max(b.g, b.b))); if (m > lim) o *= lim / m;\n"
"  return hd() ? min(Ti(o), (1.0 + hk) * max(sn(c).rgb, 0.0) + 0.1 * hk / Ex()) : o; }\n"
"float4 ld(float4 v) { v = sn(v); if (hd()) return float4(T(v.rgb), v.a);\n"
"#ifdef HDR_IN\n"
"  return v; }\n"
"#else\n"
"  return float4(saturate(v.rgb), v.a); }\n"
"#endif\n"   // resampled path: the network only knows [0,1] (its output clips at 1). HDR texels
                                                                             // (specular, emissive) light up whole dark patches around them -> white flicker as they move.
                                                                             // exp adds out - in on top of the live frame, so the excess above 1 survives.
"float4 colAt(float2 p) {\n"           // frame colour averaged over the model pixel footprint (4 bilinear taps), p = frame coords of the centre
"  uint FW = FWH & 0xFFFF, FH = FWH >> 16; float2 s = float2(FW, FH) / float2(W, H) * 0.25; float4 r = 0;\n"
"  for (int j = 0; j < 4; j++) { float2 q = clamp(p + s * float2((j & 1) ? 1 : -1, (j & 2) ? 1 : -1) - 0.5, 0.0, float2(FW - 1, FH - 1));\n"
"    uint2 a = (uint2)q; uint2 b = min(a + 1, uint2(FW - 1, FH - 1)); float2 t = q - a;\n"
"    r += lerp(lerp(ld(col[a]), ld(col[uint2(b.x, a.y)]), t.x), lerp(ld(col[uint2(a.x, b.y)]), ld(col[b]), t.x), t.y); }\n"
"  return r * 0.25; }\n"
"[numthreads(8,8,1)] void imp(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return; uint FW = FWH & 0xFFFF, FH = FWH >> 16;\n"
"  float4 c = (FW == W && FH == H) ? ld(col[id.xy]) : colAt((id.xy + 0.5) * float2(FW, FH) / float2(W, H));\n"   // frame == model size: clamped as the resampled path (raw > 1 input made block artifacts)
"  if ((hdrM & 3) == 1 || (hdrM & 3) == 2) { float3 r = sn(col[min((uint2)((id.xy + 0.5) * float2(FW, FH) / float2(W, H)), uint2(FW - 1, FH - 1))]).rgb; if ((hdrM & 3) == 2 && max(r.r, max(r.g, r.b)) > 1.0) st.InterlockedAdd(20, 1);\n"
"    if (((id.x | id.y) & 7) == 0) { st.InterlockedAdd(24, (uint)(int)round(clamp(log2(max(dot(r, float3(0.2126, 0.7152, 0.0722)), 1e-6)), -12.0, 12.0) * 256.0)); st.InterlockedAdd(28, 1); } }\n"   // auto: scene-linear HDR evidence (an LDR frame has a few % of specks above 1, HDR much more)
"  io.Store2((id.y * W + id.x) * 8, uint2(pk(c.r, c.g), pk(c.b, c.a))); }\n"
"[numthreads(1,1,1)] void lat() { if ((hdrM & 3) == 2) { if (st.Load(20) >= (hdrM >> 2)) st.Store(16, 1u); st.Store(20, 0u); hl.Store(0, st.Load(16)); }\n"
"  uint n = st.Load(28); float e = asfloat(hl.Load(4)); if (n) { float t = clamp(hkb * exp2(-(float)(int)st.Load(24) / (256.0 * n)), 1.0 / 1024.0, 16.0); e = e > 0.0 ? exp2(lerp(log2(e), log2(t), hA)) : t; }\n"
"  st.Store(24, 0u); st.Store(28, 0u); hl.Store(4, asuint(e)); }\n"   // hdr_mode auto: latch HDR at the start of a frame (sticky for the feature)
"[numthreads(1,1,1)] void spin() {\n"
"  uint v = 0; for (uint i = 0; i < maxIt; i++) { io.InterlockedOr(flagOff, 0, v); if ((int)(v - seq) >= 0) break; }\n"
"  st.Store(0, (int)(v - seq) >= 0 ? 1u : 0u); }\n"
"float depAt(float2 uv) {\n"   // far-is-zero depth at image uv (0 = no depth): reversed Z as is, standard Z as 1 - d (as janblade's detail reuse)
"  if ((dFl & 1) == 0) return 0; uint2 n = uint2(dWH & 0xFFFF, dWH >> 16); float d = dep[uint2(dBase & 0xFFFF, dBase >> 16) + min((uint2)(uv * n), n - 1)];\n"
"  return (dFl & 2) ? d : 1.0 - d; }\n"
"float4 stAt(float2 p, uint off) {\n"   // bilinear fetch of a half4 model-size image in st (as bil for io)
"  p = clamp(p - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a = (uint2)p; uint2 b = min(a + 1, uint2(W - 1, H - 1)); float2 t = p - a;\n"
"  uint2 q[4] = { st.Load2(off + (a.y * W + a.x) * 8), st.Load2(off + (a.y * W + b.x) * 8), st.Load2(off + (b.y * W + a.x) * 8), st.Load2(off + (b.y * W + b.x) * 8) };\n"
"  float4 v[4]; for (int i = 0; i < 4; i++) v[i] = float4(f16tof32(q[i].x & 0xFFFF), f16tof32(q[i].x >> 16), f16tof32(q[i].y & 0xFFFF), f16tof32(q[i].y >> 16));\n"
"  return lerp(lerp(v[0], v[1], t.x), lerp(v[2], v[3], t.x), t.y); }\n"
"[numthreads(8,8,1)] void cap(uint3 id : SV_DispatchThreadID) {\n"   // fg: keep the NR edit (out - in, model size) of a finished frame at st+256
"  if (id.x >= W || id.y >= H || st.Load(0) == 0) return; uint o = (id.y * W + id.x) * 8;\n"
"  float4 i = bil(id.xy + 0.5, 0), d = sn(bil(id.xy + 0.5, outOff) - i); st.Store2(256 + o, uint2(pk(d.r, d.g), pk(d.b, 0)));\n"
"  st.Store2(256 + W * H * 8 + o, uint2(pk(i.r, i.g), pk(i.b, depAt((id.xy + 0.5) / float2(W, H)))));\n"   // + the model input the edit belongs to (motion gate) and its depth
"  if (id.x == 0 && id.y == 0) { st.Store(4, seq); st.Store(8, frame); } }\n"
"float2 accAt(float2 q, uint f); float2 slotAt(float2 q, uint k);\n"
// ---- edit stabilisation (edit_temporal / edit_limit / edit_warp, tFl bit0 on). tea: the edit exp would show, at model texel centres
// (split: last job's edit warped + gated as exp does, live colour = the frame pixel at the texel centre; full NR: out - in) + live luma -> st slot 6. teb: cap vs the 3x3 luma range,
// blend with the previous S (slot 7/8 by frame parity, warped by this frame's vectors, depth-gated) clamped to the 3x3 range of slot 6 -> S.
// exp shows S (bilinear, model size) instead of its own per-pixel warp and colour gate. Bit2: Catmull-Rom fetches (job edit, history, S).
"float3 stCR(float2 p, uint off) {\n"   // Catmull-Rom fetch of a half4 model-size image in st
"  p -= 0.5; float2 f = floor(p), t = p - f; float2 w[4] = { t * (-0.5 + t * (1.0 - 0.5 * t)), 1.0 + t * t * (-2.5 + 1.5 * t), t * (0.5 + t * (2.0 - 1.5 * t)), t * t * (-0.5 + 0.5 * t) }; float3 r = 0;\n"
"  for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) { int2 q = clamp(int2(f) + int2(i - 1, j - 1), 0, int2(W - 1, H - 1)); uint2 u = st.Load2(off + (q.y * W + q.x) * 8);\n"
"    r += w[i].x * w[j].y * float3(f16tof32(u.x & 0xFFFF), f16tof32(u.x >> 16), f16tof32(u.y & 0xFFFF)); }\n"
"  return r; }\n"
"float3 jobE(float2 p) { return (tFl & 4) ? stCR(p, 256) : stAt(p, 256).rgb; }\n"
"float4 ld4(uint k, int2 q) { q = clamp(q, 0, int2(W - 1, H - 1)); uint2 u = st.Load2(256 + k * W * H * 8 + (q.y * W + q.x) * 8); return float4(f16tof32(u.x & 0xFFFF), f16tof32(u.x >> 16), f16tof32(u.y & 0xFFFF), f16tof32(u.y >> 16)); }\n"
"[numthreads(8,8,1)] void tea(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return; uint FW = FWH & 0xFFFF, FH = FWH >> 16; float2 uv = (id.xy + 0.5) / float2(W, H), pf = uv * float2(FW, FH); float3 e = 0, l;\n"
"  if (split != 0) { l = L(col[min((uint2)pf, uint2(FW - 1, FH - 1))]); uint age = frame - st.Load(8);\n"
"    if ((int)(st.Load(4) - minSeq) >= 0 && age != 0 && ((split & 8) != 0 || age == (split >> 4))) {\n"   // as exp's split branch, at this texel centre
"      if (mvOn) { uint2 mn = uint2(mWH & 0xFFFF, mWH >> 16); pf += mot[uint2(mBase & 0xFFFF, mBase >> 16) + min((uint2)(uv * mn), mn - 1)].xy * float2(mvSx, mvSy); }\n"
"      if (split & 2) pf += accAt(pf / float2(FW, FH), frame - 1) * float2(FW, FH);\n"
"      if (split & 4) pf += slotAt(pf / float2(FW, FH), 5) * float2(FW, FH);\n"
"      if (all(pf == pf) && all(pf >= 0.0) && all(pf < float2(FW, FH))) { float2 p = pf * float2(W, H) / float2(FW, FH); float4 o = stAt(p, 256 + W * H * 8); float3 a = abs(l - o.rgb);\n"
"        float wt = 1.0 - smoothstep(mLo, mHi, max(a.r, max(a.g, a.b)));\n"
"        if (dFl & 1) { float dn = depAt(uv); wt *= 1.0 - smoothstep(0.05, 0.2, abs(dn - o.a) / max(max(dn, o.a), 1e-6)); }\n"
"        e = wt * jobE(p); } } }\n"
"  else { float4 i = bil(id.xy + 0.5, 0); l = i.rgb; if (st.Load(0) != 0) e = sn(bil(id.xy + 0.5, outOff) - i).rgb; }\n"   // full NR frame: out - in
"  st.Store2(256 + 6 * W * H * 8 + (id.y * W + id.x) * 8, uint2(pk(e.r, e.g), pk(e.b, dot(l, float3(0.2126, 0.7152, 0.0722))))); }\n"
"[numthreads(8,8,1)] void teb(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return; uint FW = FWH & 0xFFFF, FH = FWH >> 16; float2 uv = (id.xy + 0.5) / float2(W, H);\n"
"  float4 c = ld4(6, id.xy); float3 e = c.rgb, mn = e, mx = e; float lo = c.a, hi = c.a;\n"
"  for (int j = 0; j < 9; j++) { float4 v = ld4(6, int2(id.xy) + int2(j % 3 - 1, j / 3 - 1)); mn = min(mn, v.rgb); mx = max(mx, v.rgb); lo = min(lo, v.a); hi = max(hi, v.a); }\n"
"  if (eL > 0.0) { float cap = eL * (hi - lo + 0.02); e = clamp(e, -cap, cap); mn = clamp(mn, -cap, cap); mx = clamp(mx, -cap, cap); }\n"
"  float dn = depAt(uv);\n"
"  if (eT > 0.0 && (tFl & 8) != 0 && (mvOn | hMv) != 0) {\n"   // bit3: the CPU ran teb on the frame before (no reset since)
"    uint2 m2 = uint2(mWH & 0xFFFF, mWH >> 16); float2 pf = uv * float2(FW, FH) + mot[uint2(mBase & 0xFFFF, mBase >> 16) + min((uint2)(uv * m2), m2 - 1)].xy * float2(mvSx, mvSy);\n"
"    if (all(pf == pf) && all(pf >= 0.0) && all(pf < float2(FW, FH))) { float2 ph = pf * float2(W, H) / float2(FW, FH); uint so = 256 + (7 + ((frame - 1) & 1)) * W * H * 8; float4 h = stAt(ph, so); if (tFl & 4) h.rgb = stCR(ph, so); float w = eT;\n"
"      if (dFl & 1) w *= 1.0 - smoothstep(0.05, 0.2, abs(dn - h.a) / max(max(dn, h.a), 1e-6));\n"   // disocclusion: current edit only
"      e = lerp(e, clamp(h.rgb, mn, mx), w); } }\n"
"  st.Store2(256 + (7 + (frame & 1)) * W * H * 8 + (id.y * W + id.x) * 8, uint2(pk(e.r, e.g), pk(e.b, dn))); }\n"
"float3 tE(uint2 q) { uint FW = FWH & 0xFFFF, FH = FWH >> 16; float2 p = (q + 0.5) * float2(W, H) / float2(FW, FH); uint so = 256 + (7 + (frame & 1)) * W * H * 8; return (tFl & 4) ? stCR(p, so) : stAt(p, so).rgb; }\n"
"[numthreads(8,8,1)] void exp(uint3 id : SV_DispatchThreadID) {\n"
"  uint FW = FWH & 0xFFFF, FH = FWH >> 16; if (id.x >= FW || id.y >= FH) return; float4 c = col[id.xy];\n"
"  if (split != 0) {\n"   // nr_split: the last finished NR edit (st+256, job frame st[8]) moved to this frame: by this frame's vectors, + (bit 2) the acc chain of the frame before, + (bit 4) the anchor (job frame -> frame of the edit before); age expected by the CPU in bits 4+
"    uint age = frame - st.Load(8); if ((int)(st.Load(4) - minSeq) < 0 || age == 0 || ((split & 8) == 0 && age != (split >> 4))) { put(id.xy, c); return; }\n"   // none since the reset yet / not the edit the CPU expects (spin timeout): live frame; bit 8: chain broken, vectors only, any age
"    if (tFl & 1) { float3 y = L(c); put(id.xy, float4(cmp(c, cm ? ced(y, y, y + tE(id.xy), gain) : gain * tE(id.xy)), c.a)); return; }\n"   // edit stabilisation: the stabilised edit (warped and gated per model texel by tea)
"    float2 uv = (id.xy + 0.5) / float2(FW, FH), pf = id.xy + 0.5;\n"   // frame px, same arithmetic as the mv reuse branch (age 1 is bit-identical to it)
"    if (mvOn) { uint2 mn = uint2(mWH & 0xFFFF, mWH >> 16); pf = id.xy + 0.5 + mot[uint2(mBase & 0xFFFF, mBase >> 16) + min((uint2)(uv * mn), mn - 1)].xy * float2(mvSx, mvSy); }\n"
"    if (split & 2) pf += accAt(pf / float2(FW, FH), frame - 1) * float2(FW, FH);\n"
"    if (split & 4) pf += slotAt(pf / float2(FW, FH), 5) * float2(FW, FH);\n"
"    if (!all(pf == pf) || any(pf < 0.0) || any(pf >= float2(FW, FH))) { put(id.xy, c); return; }\n"   // from off screen: live frame
"    float2 p = pf * float2(W, H) / float2(FW, FH); float3 e = stAt(p, 256).rgb; float4 o = stAt(p, 256 + W * H * 8); float3 a = abs(L(c) - o.rgb);\n"
"    float wt = 1.0 - smoothstep(mLo, mHi, max(a.r, max(a.g, a.b)));\n"   // colour gate as mv reuse
"    if (dFl & 1) { float dn = depAt(uv); wt *= 1.0 - smoothstep(0.05, 0.2, abs(dn - o.a) / max(max(dn, o.a), 1e-6)); }\n"
"    put(id.xy, float4(cmp(c, cm ? ced(L(c), o.rgb, o.rgb + wt * e, gain) : gain * wt * e), c.a)); return; }\n"
"  if (st.Load(0) == 0) {\n"
"    uint age = frame - st.Load(8);\n"   // fg: frames since the edit's own frame; older than maxAge -> live frame only
"    if (mvOn != 0 && minSeq != 0 && (int)(st.Load(4) - minSeq) >= 0 && age <= maxAge) {\n"   // mv reuse: the edit moved by the game's vectors (previous = current + mv, frame pixels)
"      uint2 mn = uint2(mWH & 0xFFFF, mWH >> 16); float2 uv = (id.xy + 0.5) / float2(FW, FH);\n"
"      float2 pf = id.xy + 0.5 + mot[uint2(mBase & 0xFFFF, mBase >> 16) + min((uint2)(uv * mn), mn - 1)].xy * float2(mvSx, mvSy);\n"
"      if (any(pf < 0.0) || any(pf >= float2(FW, FH))) { put(id.xy, c); return; }\n"   // came from off screen: live frame
"      float2 p = pf * float2(W, H) / float2(FW, FH); float3 e = stAt(p, 256).rgb; float4 o = stAt(p, 256 + W * H * 8); float3 a = abs(L(c) - o.rgb);\n"
"      float wt = 1.0 - smoothstep(mLo, mHi, max(a.r, max(a.g, a.b)));\n"   // colour gate: live vs the edit's own input at the source
"      if (dFl & 1) { float dn = depAt(uv); wt *= 1.0 - smoothstep(0.05, 0.2, abs(dn - o.a) / max(max(dn, o.a), 1e-6)); }\n"   // depth gate (relative); ponytail: fixed thresholds, add ini keys if a game needs them
"      put(id.xy, float4(cmp(c, cm ? ced(L(c), o.rgb, o.rgb + wt * e, gain) : gain * wt * e), c.a)); return; }\n"
"    if (minSeq != 0 && (int)(st.Load(4) - minSeq) >= 0 && age <= maxAge) { float2 p = (id.xy + 0.5) * float2(W, H) / float2(FW, FH);\n"   // fg: live frame + last NR edit
"      p = clamp(p - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a = (uint2)p; uint2 b = min(a + 1, uint2(W - 1, H - 1)); float2 t = p - a; float3 e[4]; uint2 ix[4] = { a, uint2(b.x, a.y), uint2(a.x, b.y), b };\n"
"      for (int i = 0; i < 4; i++) { uint2 q = st.Load2(256 + (ix[i].y * W + ix[i].x) * 8); e[i] = float3(f16tof32(q.x & 0xFFFF), f16tof32(q.x >> 16), f16tof32(q.y & 0xFFFF));\n"
"        if (gate) { uint2 u = st.Load2(256 + W * H * 8 + (ix[i].y * W + ix[i].x) * 8); float3 o = float3(f16tof32(u.x & 0xFFFF), f16tof32(u.x >> 16), f16tof32(u.y & 0xFFFF));\n"   // motion gate per model pixel: live colour (sampled as imp does) vs the edit's input
"          float3 l = ((FW == W && FH == H) ? float4(L(col[ix[i]]), 0) : colAt((ix[i] + 0.5) * float2(FW, FH) / float2(W, H))).rgb; float3 a = abs(l - o);\n"
"          e[i] *= 1.0 - smoothstep(gLo, gHi, max(a.r, max(a.g, a.b))); } }\n"
"      float3 d = lerp(lerp(e[0], e[1], t.x), lerp(e[2], e[3], t.x), t.y) * (1.0 - (float)age / (maxAge + 1)); put(id.xy, float4(cmp(c, gain * d), c.a)); return; }\n"
"    put(id.xy, c); return; }\n"
"  if (FW != W || FH != H) {\n"     // resampled: carry the model's change (out - in, model size) up to the frame
"    float2 p = (id.xy + 0.5) * float2(W, H) / float2(FW, FH); float4 d = sn(bil(p, outOff) - bil(p, 0));\n"
"    if (tFl & 1) { float3 y = L(c); put(id.xy, float4(cmp(c, cm ? ced(y, y, y + tE(id.xy), gain) : gain * tE(id.xy)), c.a)); return; }\n"   // edit stabilisation: the stabilised edit (tea/teb); compose= applies to it as to any edit
"    put(id.xy, float4(cmp(c, cm ? ced(L(c), sn(bil(p, 0)).rgb, sn(bil(p, outOff)).rgb, gain) : gain * d.rgb), c.a)); return; }\n"                       // timeout: native frame
"  uint2 r = io.Load2(outOff + (id.y * W + id.x) * 8);\n"
"  float3 e = float3(san(f16tof32(r.x & 0xFFFF)), san(f16tof32(r.x >> 16)), san(f16tof32(r.y & 0xFFFF)));\n"
"  uint2 q = io.Load2((id.y * W + id.x) * 8); float3 n = float3(f16tof32(q.x & 0xFFFF), f16tof32(q.x >> 16), f16tof32(q.y & 0xFFFF));\n"   // the network's (clamped) input
"  float3 o = cm ? cmp(c, ced(L(c), n, e, gain)) : hd() ? cmp(c, gain * (e - T(sn(c).rgb))) : c.rgb + gain * (e - n);\n"   // LDR: frame + edit, so pixels above 1 keep their excess
"  put(id.xy, float4(max(o, 0.0), c.a)); }\n"
// ---- temporal history (history=1). hst: after the spin, keep this frame's NR output in proxy space P (+ depth) at st+256+2n.
// hwp: before the HIP release, write the network's history input into io+histOff (PAD_W x PAD_H x RGB32F, pre-block +0x40): the kept P warped
// by the game's vectors to this frame (previous = current + mv*mvScale, frame pixels), with colour/depth gates; current P where invalid or reset.
"float3 Pf(float3 x) { x = max(x, 0.0); if (tmap == 0.0) return x; x = max(tmap * x, 0.0); x = min(x / (1.0 + x), 1.0); return lerp(1.055 * pow(x, 1.0 / 2.4) - 0.055, x * 12.92, step(x, 0.0031308)); }\n"   // as k_import: mode 0 raw; else max(0, scale*x), x/(1+x) clamped, sRGB OETF (tmap = scale, 0 = mode 0)
"[numthreads(8,8,1)] void hst(uint3 id : SV_DispatchThreadID) {\n"
"  if (id.x >= W || id.y >= H) return; uint ok = st.Load(0);\n"
"  if (id.x == 0 && id.y == 0) st.Store(12, ok ? frame : 0u);\n"   // frame whose output is kept (0 = none)
"  if (ok == 0) return; uint2 r = io.Load2(outOff + (id.y * W + id.x) * 8);\n"
"  float3 p = Pf(float3(san(f16tof32(r.x & 0xFFFF)), san(f16tof32(r.x >> 16)), san(f16tof32(r.y & 0xFFFF))));\n"
"  st.Store2(256 + 2 * W * H * 8 + (id.y * W + id.x) * 8, uint2(pk(p.r, p.g), pk(p.b, depAt((id.xy + 0.5) / float2(W, H))))); }\n"
"float2 slotAt(float2 q, uint k) {\n"   // nearest fetch of a model-size uv offset field in st slot k (3, 4: acc ping-pong by frame parity, 5: split anchor)
"  uint2 p = (uint2)clamp(q * float2(W, H), 0.0, float2(W - 1, H - 1)); return asfloat(st.Load2(256 + k * W * H * 8 + (p.y * W + p.x) * 8)); }\n"
"float2 accAt(float2 q, uint f) { return slotAt(q, 3 + (f & 1)); }\n"   // frame f's accumulated uv offset to the last NR frame
"void accB(uint3 id, uint k) {\n"   // in-between frame: acc_f(x) = m_f(x) + acc_{f-1}(x + m_f(x)) (acc of the NR frame itself = 0); histOk = 1 when f-1 was the NR frame
"  if (id.x >= W || id.y >= H) return; uint FW = FWH & 0xFFFF, FH = FWH >> 16; float2 uv = (id.xy + 0.5) / float2(W, H), m = 0;\n"
"  if (hMv) { uint2 mn = uint2(mWH & 0xFFFF, mWH >> 16); m = mot[uint2(mBase & 0xFFFF, mBase >> 16) + min((uint2)(uv * mn), mn - 1)].xy * float2(mvSx, mvSy) / float2(FW, FH); }\n"
"  if (histOk > 1) m += accAt(uv + m, frame - 1);\n"
"  st.Store2(256 + k * W * H * 8 + (id.y * W + id.x) * 8, asuint(m)); }\n"
"[numthreads(8,8,1)] void acc(uint3 id : SV_DispatchThreadID) { accB(id, 3 + (frame & 1)); }\n"
"[numthreads(8,8,1)] void anc(uint3 id : SV_DispatchThreadID) { accB(id, 5); }\n"
"void hwpB(uint3 id, uint age) {\n"
"  uint PW = hPWH & 0xFFFF, PH = hPWH >> 16; if (id.x >= PW || id.y >= PH) return; uint o = histOff + (id.y * PW + id.x) * 12;\n"
"  if (id.x >= W || id.y >= H) { io.Store3(o, uint3(0, 0, 0)); return; }\n"   // padding
"  uint2 r = io.Load2((id.y * W + id.x) * 8); float3 cur = Pf(float3(f16tof32(r.x & 0xFFFF), f16tof32(r.x >> 16), f16tof32(r.y & 0xFFFF))), res = cur;\n"
"  if (histOk != 0 && st.Load(12) == frame - age) {\n"
"    uint FW = FWH & 0xFFFF, FH = FWH >> 16; float2 uv = (id.xy + 0.5) / float2(W, H), m = 0;\n"
"    if (hMv) { uint2 mn = uint2(mWH & 0xFFFF, mWH >> 16); m = mot[uint2(mBase & 0xFFFF, mBase >> 16) + min((uint2)(uv * mn), mn - 1)].xy * float2(mvSx, mvSy) / float2(FW, FH); }\n"   // UV towards the previous frame
"    if (age > 1) m += accAt(uv + m, frame - 1);\n"   // older NR output (nr_every > 1): + the offsets accumulated over the in-between frames
"    float2 q = uv + m;\n"
"    if (all(m == m) && all(abs(m) < 0.25) && all(q >= 0.0) && all(q <= 1.0)) {\n"   // off screen / NaN / absurd vector: current
"      float2 p = clamp(q * float2(W, H) - 0.5, 0.0, float2(W - 1, H - 1)); uint2 a0 = (uint2)p, b0 = min(a0 + 1, uint2(W - 1, H - 1)); float2 t = p - a0;\n"   // bilinear in the runtime's P_{t-1} (io+histOut, PW stride)
"      float3 v00 = asfloat(io.Load3(histOut + (a0.y * PW + a0.x) * 12)), v10 = asfloat(io.Load3(histOut + (a0.y * PW + b0.x) * 12)), v01 = asfloat(io.Load3(histOut + (b0.y * PW + a0.x) * 12)), v11 = asfloat(io.Load3(histOut + (b0.y * PW + b0.x) * 12));\n"
"      float4 h = float4(lerp(lerp(v00, v10, t.x), lerp(v01, v11, t.x), t.y), stAt(q * float2(W, H), 256 + 2 * W * H * 8).a); float3 a = abs(h.rgb - cur);\n"
"      float w = 1.0 - smoothstep(hLo, hHi, max(a.r, max(a.g, a.b)));\n"   // colour disocclusion / wrong vector
"      if (dFl & 1) { float dn = depAt(uv); w *= 1.0 - smoothstep(0.05, 0.2, abs(dn - h.a) / max(max(dn, h.a), 1e-6)); }\n"   // depth disocclusion (as mv reuse)
"      res = lerp(cur, h.rgb, w); } }\n"
"  io.Store3(o, asuint(res)); }\n"
"[numthreads(8,8,1)] void hwp(uint3 id : SV_DispatchThreadID) { hwpB(id, 1); }\n"   // previous frame was NR (bit-exact to the one-frame history)
"[numthreads(8,8,1)] void hwp2(uint3 id : SV_DispatchThreadID) { hwpB(id, histOk); }\n";   // older NR output (nr_every > 1): histOk = its age in frames
static const int kSyncRing = 16, kCb = 44;                                     // descriptor pairs (colour SRV, output UAV) in flight

// ---------------------------------------------------------------- feature
enum { S_LOADING = 0, S_IDLE = 1, S_WAIT_COPY = 2, S_RUNNING = 3, S_FAILED = 4, S_SYNC = 5 };
struct Sync {                                       // D3D12 side of the sync path; HIP imports io + sig
    ID3D12Resource *io = nullptr, *sig = nullptr, *st = nullptr; HANDLE hIo = nullptr, hSig = nullptr;
    UINT64 ioAlloc = 0, ioSize = 0, sigAlloc = 0, sigSize = 0, outOff = 0, flagOff = 0;
    ID3D12RootSignature* rs = nullptr; ID3D12PipelineState *imp = nullptr, *spin = nullptr, *exp = nullptr, *cap = nullptr, *expPk = nullptr; ID3D12Resource* ob = nullptr; unsigned obW = 0, obH = 0; ID3D12DescriptorHeap* heap = nullptr; UINT incr = 0;
    unsigned seq = 0; unsigned busy = 0; unsigned ran = 0, rec = 0, reused = 0, minSeq = 1; unsigned frm = 0, frmOf[16] = {}, sinceNr = 0; bool warnedMv = false;
    ID3D12PipelineState *hwp = nullptr, *hst = nullptr, *acc = nullptr, *hwp2 = nullptr, *lat = nullptr; UINT64 histOff = 0, histOut = 0; float impScale = 0;   // k_import of the data dir: 0 = mode 0 (raw), else its +0x2c scale (mode != 0 tonemaps)
    unsigned waitFails = 0; unsigned padW = 0, padH = 0, lastNrFrm = 0xFFFFFFF0u, accFrm = 0xFFFFFFF0u; bool histOn = false;   // history=1   // fg: frame counter, frame of each in-flight seq (edit age)
    ID3D12PipelineState* anc = nullptr; unsigned np = 0, ancFrm = 0xFFFFFFF0u, ancTo = 0xFFFFFFF0u, ageSum = 0, ageCnt = 0, ageMax = 0, loose = 0;   // nr_split N: parts per job; anchor (st slot 5) = offsets from frame ancFrm to ancTo; edit age stats
    ID3D12PipelineState *tea = nullptr, *teb = nullptr; unsigned teFrm = 0xFFFFFFF0u;   // edit stabilisation passes, frame teb last ran
    bool split = false; unsigned part = 0, jobFrm = 0, doneFrm = 0xFFFFFFF0u, accTo = 0xFFFFFFF0u;   // nr_split: next part (0 = start a job), frame of the job in flight / last finished one, NR frame the acc chain points to
};
struct Feature;
static void applyCond(Feature* F); static float curIntensity(Feature* F);
struct Feature {
    unsigned W = 0, H = 0, pitch = 0;
    std::atomic<float> hostI{1.f}, hostTone{1.f}, hostStruct{1.f}, hostSkin{-1.f}; std::atomic<int> hostStyle{0}, hostMask{1};   // the host's controls (NVIDIA defaults)
    float appliedCond[5] = {NAN, NAN, NAN, NAN, NAN};
    ID3D12Resource* rb = nullptr; ID3D12Resource* up[2] = {}; uint8_t* rbp = nullptr; uint8_t* upp[2] = {};
    std::atomic<int> state{S_LOADING}; std::atomic<int> published{-1}; std::atomic<bool> quit{false};
    std::thread th; std::string dir; nr_ctx* ctx = nullptr; DWORD jobStart = 0; bool warnedFmt = false, warnedSize = false;
    std::atomic<unsigned> resetGen{0}, slotEval[2] = {{0}, {0}}; unsigned jobGen = 0; float appliedIntensity = -2.f;   // resetGen: bumped by a real reset; slotEval: last evaluate that recorded a read of a slot
    std::vector<uint8_t> tin, tout;
    ID3D12Device* dev = nullptr; ID3D12RootSignature* rs = nullptr; ID3D12PipelineState* pso = nullptr; ID3D12DescriptorHeap* heap = nullptr; UINT incr = 0; bool delta = false;
    ID3D12Resource* upOld[2] = {}; uint8_t* upOldp[2] = {}; ID3D12Resource* zero = nullptr; uint8_t* zerop = nullptr;   // old model input per slot (motion gate); all-zero buffer for "no result yet"
    std::atomic<int> inFmt{(int)DXGI_FORMAT_R16G16B16A16_FLOAT}, outFmt{(int)DXGI_FORMAT_R16G16B16A16_FLOAT};   // game texture formats (set by evaluate)
    int pass = 0;                                                       // 0 = the host's main feature; >0 = an extra pass (created while another feature lives)
    Sync* sy = nullptr;                                                 // non-null: sync resources exist (worker tries nr_ext_open)
    std::mutex mu; std::atomic<DWORD> lastUse{GetTickCount()}; std::atomic<bool> done{false}; std::atomic<bool> evaluated{false};   // mu: evaluate vs eviction; done: worker finished (release-while-loading handoff)
};
// Hosts (OptiScaler in MSFS 2024) create features again and again without releasing the old ones. Every live feature
// holds a runtime context (~1 GB VRAM), so before loading a new one the contexts of features idle for evict_idle_ms are freed.
static std::mutex g_featMu; static std::vector<Feature*> g_feats;
static const DXGI_FORMAT kFmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
static bool fmtOk(DXGI_FORMAT f) { return f == DXGI_FORMAT_R16G16B16A16_FLOAT || f == DXGI_FORMAT_R11G11B10_FLOAT || f == DXGI_FORMAT_R8G8B8A8_UNORM; }
static unsigned fmtBpp(int f) { return f == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4; }
static unsigned align256(unsigned v) { return (v + 255) & ~255u; }

static ID3D12Resource* makeBuffer(ID3D12Device* d, D3D12_HEAP_TYPE type, UINT64 size, uint8_t** map) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = type;
    D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    void* p = nullptr; D3D12_RANGE none = {0, 0};
    if (FAILED(r->Map(0, type == D3D12_HEAP_TYPE_READBACK ? nullptr : &none, &p))) { r->Release(); return nullptr; }
    *map = (uint8_t*)p; return r;
}

static bool sampledDone(Feature* F) {             // every sampled texel no longer the 0xFF.. sentinel => the copy landed
    const unsigned bpp = fmtBpp(F->inFmt);
    auto sentinel = [&](size_t off) { uint64_t v = 0; memcpy(&v, F->rbp + off, bpp); return bpp == 8 ? v == ~0ull : (uint32_t)v == ~0u; };
    if (F->inFmt == (int)DXGI_FORMAT_R8G8B8A8_UNORM) {   // 0xFFFFFFFF is a valid white pixel in RGBA8: any changed sample means the copy ran (settle_ms covers the rest)
        for (unsigned y = 0; y < F->H; y += 16) for (int side = 0; side < 2; side++) if (!sentinel((size_t)y * F->pitch + (side ? (F->W - 1) * bpp : 0))) return true;
        return !sentinel((size_t)(F->H - 1) * F->pitch + (F->W - 1) * bpp); }
    for (unsigned y = 0; y < F->H; y += 16)
        for (int side = 0; side < 2; side++) if (sentinel((size_t)y * F->pitch + (side ? (F->W - 1) * bpp : 0))) return false;
    return !sentinel((size_t)(F->H - 1) * F->pitch + (F->W - 1) * bpp);
}

// Input/output statistics of one job for the log: tells the first game session whether the value range the host hands the network is sane.
static void logStats(Feature* F) {
    std::vector<float> v; double d = 0; size_t nd = 0; int bad = 0;
    for (unsigned y = 0; y < F->H; y += 5) for (unsigned x = 0; x < F->W; x += 5) {
        const uint16_t* a = (const uint16_t*)(F->tin.data() + ((size_t)y * F->W + x) * 8); const uint16_t* b = (const uint16_t*)(F->tout.data() + ((size_t)y * F->W + x) * 8);
        if ((b[0] & 0x7C00) == 0x7C00 || (b[1] & 0x7C00) == 0x7C00 || (b[2] & 0x7C00) == 0x7C00) { bad++; continue; }
        v.push_back((h2f(a[0]) + h2f(a[1]) + h2f(a[2])) * (1.0f / 3.0f)); for (int c = 0; c < 3; c++) { d += fabs((double)h2f(b[c]) - (double)h2f(a[c])); nd++; } }
    if (v.empty()) { logf("stats: no finite samples (%d non-finite outputs)", bad); return; }
    std::sort(v.begin(), v.end()); double m = 0; for (float f : v) m += f; m /= v.size();
    logf("stats: model input mean %.4g p50 %.4g p99 %.4g max %.4g | network changes it by mean %.4f per channel (%zu samples, %d non-finite outputs)", m, v[v.size() / 2], v[v.size() * 99 / 100], v.back(), d / nd, v.size(), bad);
}

static float importScale(const std::string& dir);
static int metaInt(const std::string& dir, const char* key, int def) {   // meta.txt "<key>=<int>" (at least 1), else def
    FILE* f = nullptr; fopen_s(&f, (dir + "\\meta.txt").c_str(), "r"); if (!f) return def; char l[256]; const size_t n = strlen(key);
    while (fgets(l, sizeof l, f)) if (!strncmp(l, key, n) && l[n] == '=') { def = atoi(l + n + 1); break; }
    fclose(f); return def < 1 ? 1 : def;
}
static void freeFeature(Feature* F);
static void evictIdle(Feature* self);
static void workerBody(Feature* F);
static void worker(Feature* F) { workerBody(F); if (F->done.exchange(true)) freeFeature(F); }   // released while loading: the last one out frees F
static void workerBody(Feature* F) {
    try {
        char err[512] = {0};
        if (!loadRuntime()) { F->state = S_FAILED; return; }
        const int ve = g_vitEvery ? g_vitEvery : metaInt(F->dir, "vit_every", 1);
        logf("worker: nr_create(%s, %ux%u, vit_every %d%s%s)...", F->dir.c_str(), F->W, F->H, ve, g_reuseBn ? " bottleneck" : "", g_vitEvery ? "" : " (data dir default)");
        evictIdle(F);
        if (p_select) { LUID l = F->dev->GetAdapterLuid(); p_select(&l, g_allowAny); }   // HIP on the GPU the game renders on (not an iGPU / HIP device 0)
        DWORD t0 = GetTickCount();
        F->ctx = safeCreate(F->dir.c_str(), (int)F->W, (int)F->H, g_reuseBn && ve > 1 ? -ve : ve, err, sizeof err);
        if (!F->ctx) { seterr("nr_create failed: %s", err); F->state = S_FAILED; return; }
        if (F->quit) return;   // released while the runtime was loading: worker() frees it
        logf("worker: runtime ready after %lu ms", GetTickCount() - t0);
        if (F->sy) {
            if (!p_extOpen || !p_extEnqueue) logf("sync: nr_runtime.dll has no nr_ext_* (old runtime) - async path");
            else if (p_extOpen(F->ctx, F->sy->hIo, F->sy->ioAlloc, F->sy->ioSize, F->sy->outOff, F->sy->flagOff, F->sy->hSig, F->sy->sigAlloc, F->sy->sigSize, err, sizeof err)) logf("sync: nr_ext_open failed (%s) - async path", err);
            else { applyCond(F);
                   if (g_hist) { unsigned pw = 0, ph = 0; int r = p_extHist && p_histLayout && p_extEnqueue2 ? p_histLayout(F->ctx, &pw, &ph) : -1; if (!r) r = p_extHist(F->ctx, F->sy->histOff, F->sy->histOut);
                       if (r || (UINT64)pw * ph * 12 > F->sy->histOut - F->sy->histOff || (unsigned)pw < F->W || (unsigned)ph < F->H) logf("history: unavailable (%s, code %d) - history off", p_extHist ? "this data dir / runtime cannot take it" : "nr_runtime.dll has no nr_ext_history", r);
                       else { F->sy->padW = pw; F->sy->padH = ph; F->sy->histOn = true; F->sy->impScale = importScale(F->dir); logf("history: k_import scale %.3g (0 = mode 0, raw colour)", F->sy->impScale); logf("history: mode %d on (warped history io+%llu, runtime output io+%llu, %ux%u RGB32F, gate %.2f..%.2f)", g_hist, F->sy->histOff, F->sy->histOut, pw, ph, g_hLo, g_hHi); } }
                   if (g_split) { const int np = g_split > 2 && !p_extSplitN ? 2 : g_split; int kf[8] = {}, ks[8] = {}; float ms[16] = {};
                       if (np == 2) { kf[0] = g_splitK[0]; ks[0] = g_splitK[1] > 0 ? g_splitK[1] : g_splitK[0]; }   // nr_split_k fixes the 2-part point; N > 2: always measured
                       const int r = !p_extEnqueue3 ? -1 : p_extSplitN ? p_extSplitN(F->ctx, np, kf, ks, ms) : p_extSplit ? p_extSplit(F->ctx, kf, ks, ms) : -1;
                       if (np < g_split) logf("nr_split: nr_runtime.dll has no nr_ext_splitn (old runtime) - 2 parts instead of %d", g_split);
                       if (r) logf("nr_split: unavailable (%s, code %d) - whole network per NR frame", p_extSplit ? "split setup failed" : "nr_runtime.dll has no nr_ext_split", r);
                       else { F->sy->split = true; F->sy->np = np; char t[256] = ""; size_t o = 0;
                              for (int f = 0; f < 2; f++) { o += snprintf(t + o, sizeof t - o, f ? " | skipping frames: at" : "full frames: at"); for (int j = 0; j < np - 1; j++) o += snprintf(t + o, sizeof t - o, " %d", (f ? ks : kf)[j]); o += snprintf(t + o, sizeof t - o, ", ms"); for (int j = 0; j < np; j++) o += snprintf(t + o, sizeof t - o, " %.1f", ms[f * np + j]); }
                              logf("nr_split: on, %d parts; %s GPU", np, t); } }
                   logf("sync: D3D12<->HIP interop ready, same-frame NR (GPU wait bound %d ms, max %d frames in flight)", g_spinMs, g_maxInflight); F->state = S_SYNC; return; }   // the worker has nothing more to do
        }
        F->state = S_IDLE;
        while (!F->quit) {
            if (F->state == S_WAIT_COPY) {
                if (GetTickCount() - F->jobStart > (DWORD)g_jobTimeoutMs) { logf("worker: copy never landed (frame not executed?), dropping job"); F->state = S_IDLE; continue; }
                if (!sampledDone(F)) { Sleep(1); continue; }
                Sleep(g_settleMs);                          // let the rest of the copy finish
                F->state = S_RUNNING;
                for (unsigned y = 0; y < F->H; y++) {
                    uint8_t* d = F->tin.data() + (size_t)y * F->W * 8; const uint8_t* sr = F->rbp + (size_t)y * F->pitch;
                    if (F->inFmt == (int)DXGI_FORMAT_R11G11B10_FLOAT) for (unsigned x = 0; x < F->W; x++) unpack_r11g11b10(((const uint32_t*)sr)[x], (uint16_t*)d + 4 * x);
                    else if (F->inFmt == (int)DXGI_FORMAT_R8G8B8A8_UNORM) for (unsigned x = 0; x < F->W * 4; x++) ((uint16_t*)d)[x] = f2h(sr[x] / 255.f);
                    else memcpy(d, sr, (size_t)F->W * 8); }
                sanitizeHalf4((uint16_t*)F->tin.data(), (size_t)F->W * F->H * 4);
                if (g_ldrIn) { uint16_t* t = (uint16_t*)F->tin.data(); for (size_t i = 0; i < (size_t)F->W * F->H * 4; i++) if ((i & 3) != 3) t[i] = (t[i] & 0x8000) ? 0 : t[i] > 0x3C00 ? 0x3C00 : t[i]; }   // rgb to [0,1] as the sync path (edit = NR - clamped input)
                applyCond(F);
                float ms = 0; int r = safeRun(F->ctx, F->tin.data(), F->tout.data(), &ms);
                if (r) { seterr("nr_run failed (%d)", r); F->state = S_FAILED; return; }
                { static unsigned jobs = 0; if (jobs < 3 || jobs % 8 == 0) logStats(F); jobs++; }
                if (F->jobGen != F->resetGen) { logf("worker: result dropped (a reset/camera cut happened while it was computed)"); F->state = S_IDLE; continue; }
                int slot = F->published.load() < 0 ? 0 : 1 - F->published.load();
                { const DWORD tw = GetTickCount(); while (!F->quit && (unsigned)(g_evalCalls.load() - F->slotEval[slot].load()) < 6u && GetTickCount() - tw < 250) Sleep(2); }   // the GPU may still read this slot from a list recorded a few frames ago (deadline in ms: Sleep(1) is ~15 ms on Windows)
                for (unsigned y = 0; y < F->H; y++) {
                    uint8_t* d = F->upp[slot] + (size_t)y * F->pitch; const uint8_t* sr = F->tout.data() + (size_t)y * F->W * 8;
                    if (F->delta) {                                             // edit = NR(old) - old, half4 (rgb used)
                        const uint16_t* a = (const uint16_t*)sr; const uint16_t* b0 = (const uint16_t*)(F->tin.data() + (size_t)y * F->W * 8); uint16_t* e = (uint16_t*)d;
                        for (unsigned x = 0; x < F->W * 4; x++) e[x] = f2h(g_gain * curIntensity(F) * (h2f(a[x]) - h2f(b0[x])));
                        memcpy(F->upOldp[slot] + (size_t)y * F->pitch, b0, (size_t)F->W * 8); }
                    else if (F->outFmt == (int)DXGI_FORMAT_R11G11B10_FLOAT) for (unsigned x = 0; x < F->W; x++) pack_r11g11b10((const uint16_t*)sr + 4 * x, (uint32_t*)d + x);
                    else if (F->outFmt == (int)DXGI_FORMAT_R8G8B8A8_UNORM) for (unsigned x = 0; x < F->W * 4; x++) { float v = h2f(((const uint16_t*)sr)[x]); d[x] = (uint8_t)(v <= 0.f ? 0 : v >= 1.f ? 255 : v * 255.f + .5f); }
                    else memcpy(d, sr, (size_t)F->W * 8); }
                F->published = slot; logf("worker: result published in slot %d (network %.1f ms GPU)", slot, ms);
                F->state = S_IDLE;
            } else Sleep(2);
        }
    } catch (...) { seterr("worker exception"); F->state = S_FAILED; }
}

static bool makeDelta(Feature* F) {
    typedef HRESULT (WINAPI *PFN_Compile)(const void*, SIZE_T, const char*, const void*, void*, const char*, const char*, UINT, UINT, ID3DBlob**, ID3DBlob**);
    HMODULE c = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); /* System32 only: never a copy lying next to the game exe */ auto comp = c ? (PFN_Compile)GetProcAddress(c, "D3DCompile") : nullptr;
    if (!comp) { logf("delta: d3dcompiler_47.dll missing, falling back to copy"); return false; }
    ID3DBlob *cs = nullptr, *er = nullptr;
    if (FAILED(comp(kHlsl, strlen(kHlsl), "nr_delta", nullptr, nullptr, "main", "cs_5_0", 1 << 15, 0, &cs, &er))) { logf("delta: shader compile failed: %s", er ? (char*)er->GetBufferPointer() : "?"); return false; }
    D3D12_DESCRIPTOR_RANGE rg[4] = {}; for (int i = 0; i < 4; i++) { rg[i].RangeType = i == 3 ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV : D3D12_DESCRIPTOR_RANGE_TYPE_SRV; rg[i].NumDescriptors = 1; rg[i].BaseShaderRegister = i == 3 ? 0 : i; rg[i].OffsetInDescriptorsFromTableStart = i; }
    D3D12_ROOT_PARAMETER rp[2] = {}; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[0].DescriptorTable.NumDescriptorRanges = 4; rp[0].DescriptorTable.pDescriptorRanges = rg;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[1].Constants.Num32BitValues = 6;
    D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 2; rsd.pParameters = rp;
    ID3DBlob *sb = nullptr, *se = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sb, &se)) || FAILED(F->dev->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&F->rs)))) { logf("delta: root signature failed"); return false; }
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = F->rs; pd.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
    if (FAILED(F->dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&F->pso)))) { logf("delta: pipeline state failed"); return false; }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = kDesc * kRing; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(F->dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&F->heap)))) { logf("delta: descriptor heap failed"); return false; }
    F->incr = F->dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

static void freeSync(Sync* s) {
    if (!s) return;
    for (IUnknown* u : std::initializer_list<IUnknown*>{s->io, s->sig, s->st, s->rs, s->imp, s->spin, s->exp, s->cap, s->expPk, s->ob, s->heap, s->hwp, s->hst, s->acc, s->hwp2, s->lat, s->anc, s->tea, s->teb}) if (u) u->Release();
    if (s->hIo) CloseHandle(s->hIo); if (s->hSig) CloseHandle(s->hSig); delete s;
}
static ID3D12Resource* sharedBuffer(ID3D12Device* d, UINT64 size, D3D12_HEAP_FLAGS hf, UINT64* alloc) {
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* r = nullptr; if (FAILED(d->CreateCommittedResource(&hp, hf, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    if (alloc) *alloc = d->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes; return r;
}
// D3D12 half of the sync path: shared io (input | output | flag) and sig (flag_in) buffers + NT handles, a private status word, 3 compute PSOs.
static Sync* makeSync(Feature* F) {
    typedef HRESULT (WINAPI *PFN_Compile)(const void*, SIZE_T, const char*, const void*, void*, const char*, const char*, UINT, UINT, ID3DBlob**, ID3DBlob**);
    HMODULE c = LoadLibraryExW(L"d3dcompiler_47.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); auto comp = c ? (PFN_Compile)GetProcAddress(c, "D3DCompile") : nullptr;
    if (!comp) { logf("sync: d3dcompiler_47.dll missing"); return nullptr; }
    auto* s = new Sync; ID3D12Device* d = F->dev;
    const UINT64 n = (UINT64)F->W * F->H * 8, a = (n + 65535) & ~65535ull;
    s->outOff = a; s->flagOff = 2 * a; s->ioSize = 2 * a + 65536; s->sigSize = 65536;
    if (g_hist) { const UINT64 hb = ((UINT64)((F->W + 127) & ~127u) * ((F->H + 127) & ~127u) * 12 + 65535) & ~65535ull; s->histOff = s->ioSize; s->histOut = s->ioSize + hb; s->ioSize += 2 * hb; }   // + history slot (PAD_W x PAD_H RGB32F; PAD = 128-aligned, checked by nr_ext_history)
    s->io = sharedBuffer(d, s->ioSize, D3D12_HEAP_FLAG_SHARED, &s->ioAlloc); s->sig = sharedBuffer(d, s->sigSize, D3D12_HEAP_FLAG_SHARED, &s->sigAlloc); s->st = sharedBuffer(d, 256 + (g_eT > 0.f || g_eL > 0.f ? 9 : g_split ? 6 : g_hist ? 5 : 2) * n, D3D12_HEAP_FLAG_NONE, nullptr);   // status words | fg edit | its model input (half4, model size each)
    if (!s->io || !s->sig || !s->st) { logf("sync: cannot create shared buffers"); freeSync(s); return nullptr; }
    if (FAILED(d->CreateSharedHandle(s->io, nullptr, GENERIC_ALL, nullptr, &s->hIo)) || FAILED(d->CreateSharedHandle(s->sig, nullptr, GENERIC_ALL, nullptr, &s->hSig))) { logf("sync: CreateSharedHandle failed"); freeSync(s); return nullptr; }
    D3D12_DESCRIPTOR_RANGE rg[2] = {}; rg[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; rg[0].NumDescriptors = 3; rg[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; rg[1].NumDescriptors = 1; rg[1].OffsetInDescriptorsFromTableStart = 3;   // colour, motion, depth | output
    D3D12_ROOT_PARAMETER rp[6] = {}; rp[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[5].Descriptor.ShaderRegister = 4; rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[0].DescriptorTable.NumDescriptorRanges = 2; rp[0].DescriptorTable.pDescriptorRanges = rg;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor.ShaderRegister = 1; rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[2].Descriptor.ShaderRegister = 2;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[3].Constants.Num32BitValues = kCb; rp[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[4].Descriptor.ShaderRegister = 3;
    D3D12_ROOT_SIGNATURE_DESC rsd = {}; rsd.NumParameters = 6; rsd.pParameters = rp; ID3DBlob *sb = nullptr, *er = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sb, &er)) || FAILED(d->CreateRootSignature(0, sb->GetBufferPointer(), sb->GetBufferSize(), IID_PPV_ARGS(&s->rs)))) { logf("sync: root signature failed"); freeSync(s); return nullptr; }
    sb->Release();
    ID3D12PipelineState** pso[13] = {&s->imp, &s->spin, &s->exp, &s->cap, &s->expPk, &s->lat, &s->hwp, &s->hst, &s->acc, &s->hwp2, &s->anc, &s->tea, &s->teb}; const char* ep[13] = {"imp", "spin", "exp", "cap", "exp", "lat", "hwp", "hst", "acc", "hwp2", "anc", "tea", "teb"};
    for (int i = 0; i < (g_eT > 0.f || g_eL > 0.f ? 13 : g_split ? 11 : g_hist ? 10 : 6); i++) {
        ID3DBlob* cs = nullptr; er = nullptr; const char* mac[8] = {}; int nm = 0;   // D3D_SHADER_MACRO[], null-terminated
        if (i == 4) { mac[nm++] = "PACKED"; mac[nm++] = "1"; }                     // expPk
        if (!g_ldrIn) { mac[nm++] = "HDR_IN"; mac[nm++] = "1"; }
        if ((g_eT > 0.f || g_eL > 0.f) && (i == 2 || i == 4 || i >= 11)) { mac[nm++] = "NC"; mac[nm++] = "1"; }   // exp, expPk, tea, teb
        if (FAILED(comp(kSyncHlsl, strlen(kSyncHlsl), "nr_sync", nm ? mac : nullptr, nullptr, ep[i], "cs_5_0", 1 << 15, 0, &cs, &er))) { logf("sync: shader %s: %s", ep[i], er ? (char*)er->GetBufferPointer() : "?"); freeSync(s); return nullptr; }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {}; pd.pRootSignature = s->rs; pd.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
        HRESULT h = d->CreateComputePipelineState(&pd, IID_PPV_ARGS(pso[i])); cs->Release(); if (FAILED(h)) { logf("sync: PSO %s failed", ep[i]); freeSync(s); return nullptr; }
    }
    D3D12_DESCRIPTOR_HEAP_DESC hd = {}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 4 * kSyncRing; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&s->heap)))) { logf("sync: descriptor heap failed"); freeSync(s); return nullptr; }
    s->incr = d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return s;
}

// Model resolution step: data_dir holds one subfolder per step (640x360, 960x540, 1280x720, ...), or is itself one
// data dir (legacy). nr_port.ini model_res=WxH picks a step; auto (default) takes the step nearest in height to the size
// the host creates the feature with (OptiScaler's Model resolution slider sets that size; a change recreates the feature).
static bool readMeta(const std::string& dir, unsigned* w, unsigned* h);
static float importScale(const std::string& dir) {   // k_import kernarg +0x28 mode, +0x2c scale, from manifest.txt + kernargs.bin. 0 = mode 0 / unknown
    FILE* f = nullptr; fopen_s(&f, (dir + "\\manifest.txt").c_str(), "r"); if (!f) return 0; char l[1024]; size_t off = 0, len = 0; bool hit = false;
    while (!hit && fgets(l, sizeof l, f)) { std::string t(l); size_t a = t.find('|'); if (a == std::string::npos || t.substr(a + 1, t.find('|', a + 1) - a - 1).find("k_import") == std::string::npos) continue;   // field 2 = symbol
        size_t b = t.find('|', a + 1), c = t.find('|', b + 1); off = strtoull(t.c_str() + b + 1, nullptr, 10); len = strtoull(t.c_str() + c + 1, nullptr, 10); hit = len >= 0x30; }
    fclose(f); if (!hit) return 0;
    fopen_s(&f, (dir + "\\kernargs.bin").c_str(), "rb"); if (!f) return 0; int mode = 0; float sc = 0; _fseeki64(f, (long long)off + 0x28, SEEK_SET);
    bool ok = fread(&mode, 4, 1, f) == 1 && fread(&sc, 4, 1, f) == 1; fclose(f); return ok && mode != 0 && sc > 0 && sc < 1e6f ? sc : 0;
}
// Model guard. The model id of an nvngx_dlssnr.dll = FNV-1a 64 over its WEIGHTS_HT records' names + sizes (nr_model_id,
// runtime/model_id.py); 310.8 (original and ShortFuse) = 1ff9997855c80dd5. A data dir names the model it was built for in
// meta.json "model_id" (no field: v0.3 data = 310.8). Weight values are checked by the runtime (weights.map fnv).
// Layout: nr_data\<model id>\<W>x<H>\ when that folder exists (several models side by side), else the flat v0.3 nr_data\<W>x<H>\.
// Mismatch / missing / unreadable DLL: NR stays off for the session and evaluate passes the frame through.
#pragma comment(lib, "version.lib")
#pragma comment(lib, "bcrypt.lib")
#include <bcrypt.h>
static const char* kModel3108 = "1ff9997855c80dd5";
static std::string fileVersion(const std::wstring& p) {
    DWORD z = 0, n = GetFileVersionInfoSizeW(p.c_str(), &z); std::vector<uint8_t> vi(n); VS_FIXEDFILEINFO* ff = nullptr; UINT fl = 0; char v[64] = "?";
    if (n && GetFileVersionInfoW(p.c_str(), 0, n, vi.data()) && VerQueryValueW(vi.data(), L"\\", (void**)&ff, &fl) && ff)
        snprintf(v, sizeof v, "%u.%u.%u.%u", HIWORD(ff->dwFileVersionMS), LOWORD(ff->dwFileVersionMS), HIWORD(ff->dwFileVersionLS), LOWORD(ff->dwFileVersionLS));
    return v;
}
static std::string sha256Prefix(const std::string& p) {   // first 12 hex digits; "-" if unreadable
    FILE* f = nullptr; fopen_s(&f, p.c_str(), "rb"); if (!f) return "-";
    BCRYPT_ALG_HANDLE a = nullptr; BCRYPT_HASH_HANDLE h = nullptr; UCHAR d[32] = {}; std::vector<UCHAR> b(1 << 20); size_t n;
    bool ok = !BCryptOpenAlgorithmProvider(&a, BCRYPT_SHA256_ALGORITHM, nullptr, 0) && !BCryptCreateHash(a, &h, nullptr, 0, nullptr, 0, 0);
    while (ok && (n = fread(b.data(), 1, b.size(), f)) > 0) ok = !BCryptHashData(h, b.data(), (ULONG)n, 0);
    ok = ok && !BCryptFinishHash(h, d, 32, 0); if (h) BCryptDestroyHash(h); if (a) BCryptCloseAlgorithmProvider(a, 0); fclose(f);
    char s[13] = "-"; if (ok) for (int i = 0; i < 6; i++) snprintf(s + 2 * i, 3, "%02x", d[i]); return s;
}
// 310.9 carries two models: WEIGHTS_HT (= 310.8) and WEIGHTS_CC_SMALL_FP8 (small, 113M params). nr_port.ini [nr] model=small|large|auto
// (auto = small when present). How NVIDIA's own DLL chooses is not known; the id names the chosen resource's model.
static const char* g_res = "WEIGHTS_HT";
static uint64_t modelIdRaw(const char* p) {
    if (_stricmp(g_model.c_str(), "large")) if (uint64_t id = nr_model_id(p, "WEIGHTS_CC_SMALL_FP8")) { g_res = "WEIGHTS_CC_SMALL_FP8"; return id; }
    g_res = "WEIGHTS_HT"; return nr_model_id(p);
}
static uint64_t modelIdSafe(const char* p) { __try { return modelIdRaw(p); } __except (EXCEPTION_EXECUTE_HANDLER) { return 0; } }   // truncated / broken PE
static std::string metaModelId(const std::string& dir) {   // meta.json "model_id": "<16 hex>"; absent = v0.3 data = 310.8
    FILE* f = nullptr; fopen_s(&f, (dir + "\\meta.json").c_str(), "rb"); if (!f) return kModel3108;
    std::string s; char b[4096]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n); fclose(f);
    size_t k = s.find("\"model_id\""); if (k == std::string::npos) return kModel3108;
    size_t q = s.find('"', s.find(':', k) + 1); return q == std::string::npos ? std::string(kModel3108) : s.substr(q + 1, 16);
}
static std::string g_dllId, g_dllDesc;   // the user's DLL: model id ("" = not checked yet, 0...0 = none) and "version x sha256 y"
static bool pickModelBase(std::string* base) {
    if (g_dllId.empty()) {
        std::string from; uint64_t id = 0;
        for (const std::string& d : {narrow(g_dir), g_dataDir + "\\..\\"}) if (GetFileAttributesA((d + "nvngx_dlssnr.dll").c_str()) != INVALID_FILE_ATTRIBUTES) { from = d + "nvngx_dlssnr.dll"; break; }
        if (!from.empty()) { id = modelIdSafe(from.c_str()); wchar_t w[MAX_PATH] = L""; MultiByteToWideChar(CP_UTF8, 0, from.c_str(), -1, w, MAX_PATH);
                             g_dllDesc = "version " + fileVersion(w) + " sha256 " + sha256Prefix(from); }
        else g_dllDesc = "(not found)";
        char hx[17]; snprintf(hx, sizeof hx, "%016llx", (unsigned long long)id); g_dllId = hx;
        logf("model: %s %s, model id %s (%s, model=%s)", from.empty() ? "nvngx_dlssnr.dll" : from.c_str(), g_dllDesc.c_str(), id ? hx : "none (no WEIGHTS_HT / unreadable)", g_res, g_model.c_str());
    }
    const std::string m = g_dataDir + "\\" + g_dllId; const DWORD at = GetFileAttributesA(m.c_str());
    *base = at != INVALID_FILE_ATTRIBUTES && (at & FILE_ATTRIBUTE_DIRECTORY) ? m : g_dataDir;
    return true;
}
static bool modelMatches(const std::string& dir) {   // the step dir create picked was built for the user's DLL
    const std::string want = metaModelId(dir); if (!_stricmp(want.c_str(), g_dllId.c_str())) return true;
    seterr("nvngx_dlssnr.dll %s has a different model than this port supports (%s). NR stays off; update the port. (DLL model id %s, data %s)",
           g_dllDesc.c_str(), want == kModel3108 ? "310.8 / 310.9.16" : want.c_str(), g_dllId.c_str(), dir.c_str());
    return false;
}
static std::string pickModelDir(unsigned w, unsigned h) {
    std::string base; if (!pickModelBase(&base)) return "";
    unsigned mw, mh; if (readMeta(base, &mw, &mh)) return base;
    std::vector<std::pair<unsigned, unsigned>> st; WIN32_FIND_DATAA fd; HANDLE fh = FindFirstFileA((base + "\\*x*").c_str(), &fd);
    if (fh != INVALID_HANDLE_VALUE) { do { unsigned a, b; if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && sscanf_s(fd.cFileName, "%ux%u", &a, &b) == 2) st.push_back({a, b}); } while (FindNextFileA(fh, &fd)); FindClose(fh); }
    auto name = [](unsigned a, unsigned b) { char n[32]; snprintf(n, sizeof n, "%ux%u", a, b); return std::string(n); };
    if (st.empty()) return base;   // create reports the missing meta.txt
    std::string want = g_modelRes;
    if (want != "auto" && std::none_of(st.begin(), st.end(), [&](const std::pair<unsigned, unsigned>& p) { return name(p.first, p.second) == want; })) { logf("model_res %s: no such step in %s, using auto", want.c_str(), base.c_str()); want = "auto"; }
    if (want == "auto" && !st.empty()) {
        auto best = std::make_pair(1280u, 720u); bool def = std::find(st.begin(), st.end(), best) != st.end(); if (!def) best = st[0];
        if (h) for (auto& p : st) if (p.second <= 720 && abs((int)p.second - (int)h) < abs((int)best.second - (int)h)) best = p;   // ties keep 1280x720; 1080p / 1440p steps only by explicit model_res (hosts often pass the output size)
        want = name(best.first, best.second);
    }
    std::string steps; for (auto& p : st) steps += " " + name(p.first, p.second);
    logf("model resolution: %s (model_res=%s, host size %ux%u, steps:%s)", want.c_str(), g_modelRes.c_str(), w, h, steps.c_str());
    return base + "\\" + want;
}

// ---------------------------------------------------------------- exports
static bool readMeta(const std::string& dir, unsigned* w, unsigned* h) {
    std::string p = dir + "\\meta.txt"; FILE* f = nullptr; fopen_s(&f, p.c_str(), "r"); if (!f) return false;
    char line[256]; *w = *h = 0;
    while (fgets(line, sizeof line, f)) { unsigned v; if (sscanf_s(line, "width=%u", &v) == 1) *w = v; if (sscanf_s(line, "height=%u", &v) == 1) *h = v; }
    fclose(f); return *w && *h;
}

extern "C" {
__declspec(dllexport) int dlssnr_call_last_init = 1, dlssnr_call_last_create = 1;
__declspec(dllexport) int dlssnr_last_ratio_result = 0, dlssnr_last_ratio_stage = 0;
__declspec(dllexport) const char* dlssnr_backend_id() { return "rdna2-hip"; }   // presence marks the "port" flavour
__declspec(dllexport) const char* dlssnr_call_error() { return g_err; }
__declspec(dllexport) int dlssnr_port_state(void* f) { return f ? ((Feature*)f)->state.load() : -1; }   // diagnostics/tests: 0 loading, 1-3 async, 4 failed, 5 sync
__declspec(dllexport) void dlssnr_port_stats(void* f, unsigned* nr, unsigned* native, unsigned* reused) { Sync* s = f ? ((Feature*)f)->sy : nullptr; *nr = s ? s->ran : 0; *native = s ? s->busy : 0; *reused = s ? s->reused : 0; }   // tests: sync-path frame counts
__declspec(dllexport) const char* dlssnr_port_dir(void* f) { return f ? ((Feature*)f)->dir.c_str() : ""; }   // tests: the data dir create picked (model + step)
__declspec(dllexport) void dlssnr_call_set_float_slot(int) {}
__declspec(dllexport) void dlssnr_call_probe_float(void*, const char*, float, int) {}
__declspec(dllexport) void dlssnr_call_set_extras(void*, float, ID3D12Resource*, ID3D12Resource*, ID3D12Resource*, unsigned, unsigned, unsigned, unsigned) {}

// Once per process: our build, the game exe and the calling host module (OptiScaler under its proxy name) with its file version.
#include <intrin.h>
static void logHost(void* caller) {
    static bool done = false; if (done) return; done = true;
    wchar_t exe[MAX_PATH] = L"", host[MAX_PATH] = L"?"; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    HMODULE hm = nullptr; if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)caller, &hm)) GetModuleFileNameW(hm, host, MAX_PATH);
    logf("host: dlssnr_nr build %s %s | game %ls | host module %ls (version %s)", __DATE__, __TIME__, exe, host, fileVersion(host).c_str());
}

}  // extern "C" (the helpers below are C++)
// Host controls -> runtime (ini keys override). Values outside NVIDIA's ranges are ignored (the previous ones stay).
static void setHost(Feature* F, float i, int style, float structure, float tone, float skin, int mask) {
    if (i >= 0.f && i <= 2.f) F->hostI = i; if (style >= 0 && style <= 2) F->hostStyle = style; if (structure >= 0.f && structure <= 2.f) F->hostStruct = structure;
    if (tone >= 0.f && tone <= 2.f) F->hostTone = tone; if (skin >= -1.f && skin <= 2.f) F->hostSkin = skin; F->hostMask = mask ? 1 : 0;
}
static float curIntensity(Feature* F) { return g_intensity >= 0.f ? g_intensity : F->hostI.load(); }
static void applyCond(Feature* F) {
    if (!p_cond || !F->ctx) return;
    const float host[5] = {(float)F->hostStyle.load(), F->hostTone.load(), F->hostStruct.load(), F->hostSkin.load(), (float)F->hostMask.load()};
    float v[5]; for (int i = 0; i < 5; i++) v[i] = g_iniCond[i] == g_iniCond[i] ? g_iniCond[i] : host[i];
    if (!memcmp(v, F->appliedCond, sizeof v)) return;
    const int r = p_cond(F->ctx, (int)v[0], v[1], v[2], v[3], v[4] != 0.f);
    logf("conditioning: style %d tone %.2f structure %.2f skin %.2f auto_mask %d, intensity %.2f%s", (int)v[0], v[1], v[2], v[3], (int)(v[4] != 0.f), curIntensity(F), r ? " - rejected by the runtime" : "");
    memcpy(F->appliedCond, v, sizeof v);
}

extern "C" {
__declspec(dllexport) void* dlssnr_call_create(const wchar_t* snippet, const wchar_t*, ID3D12Device* dev, ID3D12GraphicsCommandList*, void*,
        unsigned w, unsigned h, int, float intensity, int style, float localStructure, float localTone, float skinStructure, int useAutoMask, int) {
    try {
        if (!g_log) loadSettings();
        logf("create: snippet=%ls dev=%p %ux%u", snippet ? snippet : L"", (void*)dev, w, h);
        logHost(_ReturnAddress());
        dlssnr_call_last_create = 0;
        if (!dev) { seterr("create: no D3D12 device"); return nullptr; }
        unsigned mw = 0, mh = 0; const std::string dir = pickModelDir(w, h);
        if (dir.empty() || !modelMatches(dir)) {   // unknown / missing nvngx_dlssnr.dll: a feature that only passes frames through (returns 1; OptiScaler ends NR on 0)
            auto* F = new Feature; F->state = S_FAILED; F->W = w; F->H = h; F->pass = g_live++; dlssnr_call_last_create = 1; return F; }
        if (!readMeta(dir, &mw, &mh)) { seterr("create: no meta.txt in '%s' (run gen_runtime_data.py, see INSTALL.md)", dir.c_str()); return nullptr; }
        if ((mw != w || mh != h) && g_sync) { logf("create: frame %ux%u, model %ux%u - sync path resamples (universal for any resolution / upscaler mode)", w, h, mw, mh); w = mw; h = mh; }
        if (mw != w || mh != h) { seterr("create: model size %ux%u but data dir '%s' is for %ux%u - generate data for this size", w, h, dir.c_str(), mw, mh); return nullptr; }
        auto* F = new Feature; F->dir = dir; F->pass = g_live++; setHost(F, intensity, style, localStructure, localTone, skinStructure, useAutoMask);   // ponytail: pass = live features at create (janblade creates the main one first); recreating only the main feature while extras live would mislabel it
        if (F->pass) logf("create: extra pass %d (pass2_gain %.2f)", F->pass, g_pass2Gain); F->W = w; F->H = h; F->pitch = align256(w * 8);
        F->tin.resize((size_t)w * h * 8); F->tout.resize((size_t)w * h * 8);
        UINT64 sz = (UINT64)F->pitch * h;
        F->rb = makeBuffer(dev, D3D12_HEAP_TYPE_READBACK, sz, &F->rbp);
        F->up[0] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upp[0]); F->up[1] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upp[1]);
        if (!F->rb || !F->up[0] || !F->up[1]) { seterr("create: cannot allocate staging buffers"); if (F->rb) F->rb->Release(); for (auto u : F->up) if (u) u->Release(); delete F; return nullptr; }
        F->dev = dev;
        if (g_apply) {
            F->upOld[0] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upOldp[0]); F->upOld[1] = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->upOldp[1]);
            F->zero = makeBuffer(dev, D3D12_HEAP_TYPE_UPLOAD, sz, &F->zerop);
            if (F->upOld[0] && F->upOld[1] && F->zero) { memset(F->zerop, 0, (size_t)sz); F->delta = makeDelta(F); } else logf("delta: cannot allocate extra buffers, falling back to copy");
        }
        if (g_sync) F->sy = makeSync(F);
        logf("create: sync %s; async fallback apply mode %s gain %.2f; compose %s detail %.2f", F->sy ? "resources ready" : "off", F->delta ? "delta (live frame + NR edit)" : "copy (stale NR image)", g_gain, g_compose == 1 ? "nv" : g_compose == 2 ? "ratio" : "add", g_detail);
        { std::lock_guard<std::mutex> g(g_featMu); g_feats.push_back(F); }
        F->th = std::thread(worker, F);
        dlssnr_call_last_create = 1; return F;
    } catch (...) { seterr("create: exception"); return nullptr; }
}

// Pass-through: copy the live colour into the output (host contract: colour NON_PIXEL_SHADER_RESOURCE, output UNORDERED_ACCESS)
// and report success. Returns 0 only when the formats differ (no plain copy possible).
static int passThrough(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color, ID3D12Resource* out, unsigned w, unsigned h) {
    D3D12_RESOURCE_DESC cd = color->GetDesc(), od = out->GetDesc();
    if (cd.Format != od.Format || cd.Width < w || cd.Height < h || od.Width < w || od.Height < h) return 0;
    D3D12_RESOURCE_BARRIER b[2] = {}; for (auto& x : b) { x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; }
    b[0].Transition.pResource = color; b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    b[1].Transition.pResource = out; b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    cmd->ResourceBarrier(2, b);
    D3D12_TEXTURE_COPY_LOCATION dl = {}, sl = {}; dl.pResource = out; dl.Type = sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; sl.pResource = color;
    D3D12_BOX bx = {0, 0, 0, w, h, 1}; cmd->CopyTextureRegion(&dl, 0, 0, 0, &sl, &bx);
    for (auto& x : b) std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
    cmd->ResourceBarrier(2, b);
    return 1;
}

__declspec(dllexport) int dlssnr_call_evaluate_v2(ID3D12GraphicsCommandList* cmd, void* f, void*, ID3D12Resource* color, ID3D12Resource* depth,
        ID3D12Resource* motion, ID3D12Resource* out, unsigned w, unsigned h, unsigned dW, unsigned dH, unsigned mW, unsigned mH, unsigned dBx, unsigned dBy, unsigned mBx, unsigned mBy,
        int depthInv, int reset, float intensity, int style, float localStructure, float localTone, float skinStructure, int useAutoMask, float mvSx, float mvSy) {
    try {
        auto* F = (Feature*)f; int n = ++g_evalCalls;
        if (!F || !cmd || !color || !out) return 0;
        if (F->state == S_FAILED) return passThrough(cmd, color, out, w, h);   // NR off for this session (model guard, runtime failure): native frame
        setHost(F, intensity, style, localStructure, localTone, skinStructure, useAutoMask);
        std::lock_guard<std::mutex> lk(F->mu); F->lastUse = GetTickCount(); F->evaluated = true;
        if (F->state == S_SYNC) applyCond(F);   // the runtime applies it at the start of the next network run
        if ((w != F->W || h != F->H) && F->state != S_SYNC) { if (!F->warnedSize) { F->warnedSize = true; logf("evaluate: frame %ux%u, model %ux%u - native frame until the runtime has loaded", w, h, F->W, F->H); } return passThrough(cmd, color, out, w, h); }
        D3D12_RESOURCE_DESC cd = color->GetDesc(), od = out->GetDesc();
        // R9G9B9E5 output (RE Engine): sync path only, packed in a shader and copied in. Only when the host created the texture
        // with ALLOW_UNORDERED_ACCESS: then it is in the UAV state (OptiScaler-DLSSNR contract) and our UAV<->COPY_DEST barriers
        // are legal. Without the flag a UAV-state barrier removes the device, so that case stays native.
        const bool e5 = F->state == S_SYNC && od.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP && (od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        if (F->state == S_SYNC && od.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP && !e5) {
            if (!F->warnedFmt) { F->warnedFmt = true; logf("evaluate: R9G9B9E5 output without UAV flag (%llux%u) - native frame; try NR before super resolution", od.Width, od.Height); } return 0; }
        if (!(fmtOk(cd.Format) || (F->state == S_SYNC && cd.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP)) || !(fmtOk(od.Format) || e5) || cd.Width < w || cd.Height < h || od.Width < w || od.Height < h) {
            if (!F->warnedFmt) { F->warnedFmt = true; logf("evaluate: unsupported color/output format %d/%d (%llux%u / %llux%u) - only RGBA16F / R11G11B10_FLOAT / RGBA8_UNORM", (int)cd.Format, (int)od.Format, cd.Width, cd.Height, od.Width, od.Height); } return 0; }
        F->inFmt = (int)cd.Format; F->outFmt = (int)od.Format;
        if (reset == 1) { F->published = -1; F->resetGen++; }      // a real reset (first frame, resolution change, camera cut): no edit from before it survives. Only exactly 1 counts.
        if (n <= 3) logf("evaluate #%d state %d published %d", n, F->state.load(), F->published.load());

        if (F->state == S_SYNC) {                                   // same-frame path: everything below runs on the GPU in the host's list order
            Sync* s = F->sy; ID3D12GraphicsCommandList2* c2 = nullptr;
            if (FAILED(cmd->QueryInterface(IID_PPV_ARGS(&c2)))) { if (!s->busy++) logf("sync: host list has no WriteBufferImmediate (ID3D12GraphicsCommandList2) - native frame"); return passThrough(cmd, color, out, w, h); }
            c2->Release();                                          // the host holds the list; we only need the interface for this call
            if (reset == 1) { s->minSeq = s->seq + 1; s->part = 0; }  // fg/split: no edit from before a camera cut is reused; split: a half-done job is abandoned
            const bool sp = s->split;
            const unsigned seq = s->seq + 1; ++s->frm; const unsigned accF0 = s->accFrm, accT0 = s->accTo; const unsigned age = s->frm - s->lastNrFrm; const unsigned chain = reset != 1 && (age == 1 || (age <= 8 && s->accFrm + 1 == s->frm)) ? age : 0, histOk = s->histOn ? chain : 0;   // frames since the last NR output; > 1 needs an unbroken offset chain (acc) over the in-between frames
            const bool reuseOn = true;   // cap the edit on every NR frame: busy frames (and spin timeouts) reuse it   // cap the edit on NR frames, reuse it on the others
            const bool skip = g_nrEvery > 1 && reset != 1 && s->sinceNr + 1 < (unsigned)g_nrEvery && (int)(s->seq - s->minSeq) >= 0;   // nr_every: reuse frame (an edit since the last reset exists)
            auto enq = [&] { return sp ? p_extEnqueue3(F->ctx, seq, g_maxInflight, s->part == 0 && histOk ? g_hist : 0, (int)s->part + 1) : skip ? 1 : (s->histOn ? p_extEnqueue2(F->ctx, seq, g_maxInflight, histOk ? g_hist : 0) : p_extEnqueue(F->ctx, seq, g_maxInflight)); };   // HIP: wait flag_in >= seq, run, flag_out = seq (queued now, runs when the list below executes)
            int q = enq();
            // busy_mode=wait: HIP slots free up when the ext thread finishes a job of an EARLIER, already submitted list (or drops it after its own 500 ms
            // flag_in bound), never through the list we are recording -> a bounded CPU wait cannot deadlock. It throttles the game to the network rate.
            if (q == 1 && (sp || !skip) && !g_fg && g_busy == 0 && s->waitFails < 8) { const DWORD t0 = GetTickCount();
                while (q == 1 && GetTickCount() - t0 < (DWORD)g_busyWaitMs) { Sleep(0); q = enq(); }   // ponytail: yield-spin (one core for up to ~1 frame), an event from the runtime if CPU use matters
                if (q == 1 && ++s->waitFails == 8) logf("sync: busy_wait_ms %d ran out 8 times (host submits late?) - busy_mode reuse from now on", g_busyWaitMs);
                else if (q == 0) s->waitFails = 0; }
            if (q < 0) { seterr("nr_ext_enqueue failed (%d)", q); F->state = S_FAILED; return 0; }
            s->sinceNr = q == 0 ? 0 : s->sinceNr + 1;
            const bool haveNr = (int)(s->seq - s->minSeq) >= 0;   // an NR edit since the last reset exists to reuse
            if (q != 0 && !g_fg && !skip && !sp && !haveNr) { if (s->busy++ < 5 || s->busy % 100 == 0) logf("sync: frame not queued (HIP still busy with older frames, %u so far) - native frame", s->busy); return passThrough(cmd, color, out, w, h); }
            if (q == 0) s->seq = seq;
            if (q == 0 && !(sp && s->part != 0)) { s->frmOf[seq % 16] = s->frm; if (++s->ran <= 3 || s->ran % 1000 == 0) logf("sync: NR frame %u queued (frame %ux%u, %u native, %u reused so far)", s->ran, w, h, s->busy, s->reused); }
            else s->reused++;                                       // fg: HIP busy / nr_every: in-between frame -> live frame + last NR edit, the queue does not wait
            UINT k = (++s->rec % kSyncRing) * 4; auto cpu = s->heap->GetCPUDescriptorHandleForHeapStart(); auto gpu = s->heap->GetGPUDescriptorHandleForHeapStart();
            auto at = [&](UINT i) { D3D12_CPU_DESCRIPTOR_HANDLE c = cpu; c.ptr += (SIZE_T)(k + i) * s->incr; return c; }; gpu.ptr += (UINT64)k * s->incr;
            D3D12_SHADER_RESOURCE_VIEW_DESC sv = {}; sv.Format = cd.Format; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
            F->dev->CreateShaderResourceView(color, &sv, at(0));
            // mv reuse guides (janblade: vectors point current -> previous, raw * mvScale = pixels of the w x h frame; depth far-is-zero after `depthInv`)
            auto srvFmt = [](DXGI_FORMAT f) { switch (f) { case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
                case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
                case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
                case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
                case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT; case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT; default: return f; } };
            const DXGI_FORMAT mf = motion ? srvFmt(motion->GetDesc().Format) : DXGI_FORMAT_UNKNOWN, df = depth ? srvFmt(depth->GetDesc().Format) : DXGI_FORMAT_UNKNOWN;
            const bool mvOk = motion && mf != DXGI_FORMAT_UNKNOWN && mW && mH && mvSx == mvSx && mvSy == mvSy, mvOn = (g_mv || sp) && mvOk, hMv = (s->histOn || sp) && mvOk, dOn = (mvOn || hMv) && depth && df != DXGI_FORMAT_UNKNOWN && dW && dH;
            if (s->histOn && !mvOk && !s->warnedMv) { s->warnedMv = true; logf("history: no usable motion vectors from the host (motion %p %ux%u) - history unwarped (gates only)", (void*)motion, mW, mH); }
            if (g_mv && !mvOn && !s->warnedMv) { s->warnedMv = true; logf("mv_reuse: no usable motion vectors from the host (motion %p %ux%u) - plain reuse", (void*)motion, mW, mH); }
            sv.Format = mvOn || hMv ? mf : DXGI_FORMAT_R16G16_FLOAT; F->dev->CreateShaderResourceView(mvOn || hMv ? motion : nullptr, &sv, at(1));
            sv.Format = dOn ? df : DXGI_FORMAT_R32_FLOAT; F->dev->CreateShaderResourceView(dOn ? depth : nullptr, &sv, at(2));
            D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = e5 ? DXGI_FORMAT_R16G16B16A16_FLOAT : od.Format; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; F->dev->CreateUnorderedAccessView(e5 ? nullptr : out, nullptr, &uv, at(3));   // e5: null UAV, output goes through ob
            UINT cb[kCb] = { F->W, F->H, seq, (UINT)(g_fg ? g_fgWaitMs : g_spinMs) * 6000u /* ~0.16 us per poll on a 6900 XT */, (UINT)s->outOff, (UINT)s->flagOff, 0, w | (h << 16), reuseOn ? s->minSeq : 0u, 0, 0, (UINT)g_gate, s->frm,
                             (UINT)(g_fg ? (std::max)(g_fgMaxAge, g_nrEvery - 1) : 0x7FFF /* busy reuse: no age limit, no fade (the colour gate guards) */), (UINT)mvOn, 0, 0, mW | (mH << 16), mBx | (mBy << 16), dW | (dH << 16), dBx | (dBy << 16), (UINT)dOn | (depthInv ? 2u : 0u), 0, 0,
                             (UINT)s->histOff, s->padW | (s->padH << 16), (UINT)histOk, (UINT)hMv, 0, 0, 0, (UINT)s->histOut }; memcpy(&cb[30], &s->impScale, 4);
            cb[33] = g_hdr == 1 ? 1u : g_hdr == 2 && (cd.Format == DXGI_FORMAT_R16G16B16A16_FLOAT || cd.Format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) ? 2u | ((UINT)(g_hdrFrac * F->W * F->H) << 2) : 0u;   // hdrM | auto threshold << 2
            if (!F->pass) g_hdrP0 = cb[33] & 3; else if (g_hdr == 2) cb[33] = g_hdrP0 == 2 ? 3u : g_hdrP0;   // extra pass: its input is pass 0's output after the host's ClampProxy (saturated to [0,1]), it cannot detect HDR itself -> pass 0's decision (3 = follow hl)
            { const float hk = F->pass ? g_pass2Gain : 1.f, hx = F->pass ? 1.f : g_hdrExp * g_hdrBias, hkb = g_hdrKey * g_hdrBias; memcpy(&cb[34], &hk, 4); memcpy(&cb[35], &hx, 4); memcpy(&cb[36], &hkb, 4); memcpy(&cb[37], &g_hdrAdapt, 4); cb[39] = (UINT)g_compose; memcpy(&cb[40], &g_detail, 4); }   // extra pass: E = 1 (its input is ClampProxy's unit cube, not scene-linear; pass 0's E would squeeze it to T(E) ~ 0.25, near black)
            const float I = curIntensity(F), amp = I > 1.f ? I : 0.f; memcpy(&cb[38], &amp, 4);   // intensity > 1: the blend stays at 1, put() applies the reference's ratio compose (boost)
            float gain = (F->pass ? g_gain * g_pass2Gain : g_gain) * (std::min)(I, 1.f); memcpy(&cb[6], &gain, 4); memcpy(&cb[9], &g_gateLo, 4); memcpy(&cb[10], &g_gateHi, 4);   // gLo, gHi   // gain x intensity: NVIDIA's blend towards the NR result (up to 1; compose nv/ratio get it as their g)
            memcpy(&cb[15], &mvSx, 4); memcpy(&cb[16], &mvSy, 4); memcpy(&cb[22], &g_mvLo, 4); memcpy(&cb[23], &g_mvHi, 4); memcpy(&cb[28], &g_hLo, 4); memcpy(&cb[29], &g_hHi, 4);
            const bool teOn = s->tea && s->teb && (sp || w != F->W || h != F->H); memcpy(&cb[41], &g_eT, 4); memcpy(&cb[42], &g_eL, 4);   // edit stabilisation: split or resampled full-NR path
            cb[43] = teOn ? 1u | (UINT)g_eFl | (s->teFrm + 1 == s->frm && reset != 1 ? 8u : 0u) : 0u;
            ID3D12DescriptorHeap* hp[] = { s->heap }; cmd->SetDescriptorHeaps(1, hp); cmd->SetComputeRootSignature(s->rs);
            cmd->SetComputeRootDescriptorTable(0, gpu); cmd->SetComputeRootUnorderedAccessView(1, s->io->GetGPUVirtualAddress()); cmd->SetComputeRootUnorderedAccessView(2, s->st->GetGPUVirtualAddress());
            if (!g_hl) g_hl = sharedBuffer(F->dev, 65536, D3D12_HEAP_FLAG_NONE, nullptr);   // ponytail: one process-global latch, never freed; one device assumed
            cmd->SetComputeRootUnorderedAccessView(5, (g_hl ? g_hl : s->st)->GetGPUVirtualAddress());
            cmd->SetComputeRootUnorderedAccessView(4, s->st->GetGPUVirtualAddress());   // ob slot: unused unless e5; bound so the root signature is complete
            D3D12_RESOURCE_BARRIER uav = {}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;   // null resource = all UAV writes
            if (!F->pass && (cb[33] & 3)) { cmd->SetComputeRoot32BitConstants(3, kCb, cb, 0); cmd->SetPipelineState(s->lat); cmd->Dispatch(1, 1, 1); cmd->ResourceBarrier(1, &uav);   // hdr: latch + exposure (from last frame's meter) before anything reads them
                static DWORD eT = 0; static ID3D12Resource* eRb = nullptr; static uint8_t* eRbp = nullptr; const DWORD now = GetTickCount();
                if (g_hl && now - eT >= 1000) {   // ~1/s: log the previous copy (landed long ago), queue a new one of hl[0..8]   // ponytail: process-global, never freed
                    if (eRb) { uint32_t v[2]; memcpy(v, eRbp, 8); float e; memcpy(&e, &v[1], 4); logf("hdr: latched %u, exposure E %.4f (scene mean L ~%.3g)", v[0], e, e > 0 ? g_hdrKey * g_hdrBias / e : 0.0); }
                    else eRb = makeBuffer(F->dev, D3D12_HEAP_TYPE_READBACK, 256, &eRbp);
                    if (eRb) { D3D12_RESOURCE_BARRIER t = {}; t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; t.Transition.pResource = g_hl; t.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; t.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                               cmd->ResourceBarrier(1, &t); cmd->CopyBufferRegion(eRb, 0, g_hl, 0, 8); std::swap(t.Transition.StateBefore, t.Transition.StateAfter); cmd->ResourceBarrier(1, &t); }
                    eT = now; } }   // hdr auto: latch before anything reads it
            auto tr = [&](D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) { D3D12_RESOURCE_BARRIER t = {}; t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; t.Transition.pResource = s->sig; t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; t.Transition.StateBefore = a; t.Transition.StateAfter = b; cmd->ResourceBarrier(1, &t); };
            auto release = [&] { tr(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);   // input is in io (or: frame N+1 began): release the HIP stream's flag_in wait
                D3D12_WRITEBUFFERIMMEDIATE_PARAMETER wp = { s->sig->GetGPUVirtualAddress(), seq }; D3D12_WRITEBUFFERIMMEDIATE_MODE wm = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
                c2->WriteBufferImmediate(1, &wp, &wm); tr(D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON); };
            if (sp) {   // nr_split N: frame J = input + part 1 of job J, frames J+1..J+N-1 = parts 2..N; every frame shows the last finished job's edit, warped (age N-1 .. 2N-2)
                auto pass = [&](ID3D12PipelineState* p, UINT gx, UINT gy) { cmd->SetComputeRoot32BitConstants(3, kCb, cb, 0); cmd->SetPipelineState(p); cmd->Dispatch(gx, gy, 1); cmd->ResourceBarrier(1, &uav); };
                const UINT gx = (F->W + 7) / 8, gy = (F->H + 7) / 8;
                auto accP = [&] { if (chain) { cb[26] = chain; pass(s->acc, gx, gy); s->accFrm = s->frm; s->accTo = s->lastNrFrm; } };   // offsets from this frame to the last job's frame
                if (s->part == 0) {
                    if (q == 0) { pass(s->imp, gx, gy); if (s->histOn) pass(histOk > 1 ? s->hwp2 : s->hwp, (s->padW + 7) / 8, (s->padH + 7) / 8);
                                  if (s->np > 2) { if (chain) { cb[26] = chain; pass(s->anc, gx, gy); s->ancFrm = s->frm; s->ancTo = s->lastNrFrm; } else s->ancFrm = 0xFFFFFFF0u; }   // anchor: this job's frame -> the previous job's frame (shown until this job is done)
                                  release(); pass(s->spin, 1, 1); s->lastNrFrm = s->jobFrm = s->frm; s->part = 1; }   // the spin waits for part 1 only
                    else accP();   // HIP busy: keep the chain to the shown edit
                } else {
                    accP();
                    if (q == 0) { release(); pass(s->spin, 1, 1);
                                  if (++s->part == s->np) { cb[12] = s->jobFrm;   // last part done -> keep its P / edit, stamped with the job's own frame
                                      if (s->histOn) pass(s->hst, gx, gy); pass(s->cap, gx, gy); cb[12] = s->frm; s->doneFrm = s->jobFrm; s->part = 0; } }
                }
                // offsets from this frame to the shown edit's frame T: vectors to frame-1, then acc of frame-1 (-> accT0) and/or the anchor (ancFrm -> ancTo)
                const unsigned da = s->frm - s->doneFrm, T = s->doneFrm; const bool accOk = accF0 + 1 == s->frm, ancOk = s->np > 2 && s->ancTo == T;
                const unsigned fl = da <= 1 ? 0 : accOk && accT0 == T ? 2 : ancOk && s->ancFrm + 1 == s->frm ? 4 : ancOk && accOk && accT0 == s->ancFrm ? 6 : 8;   // 8: chain broken (HIP busy long), vectors only (colour gate guards)
                cb[32] = 1 | fl | ((da < 0x8000000u ? da : 0u) << 4);
                if (T != 0xFFFFFFF0u && da < 64) { s->ageSum += da; s->ageCnt++; s->ageMax = (std::max)(s->ageMax, da); s->loose += fl == 8; }
                if (s->ageCnt == 200) { logf("nr_split: edit age mean %.2f max %u frames, %u of 200 frames vectors-only", s->ageSum / 200.0, s->ageMax, s->loose); s->ageSum = s->ageCnt = s->ageMax = s->loose = 0; }
                if (teOn) { pass(s->tea, gx, gy); pass(s->teb, gx, gy); s->teFrm = s->frm; }
                cmd->SetComputeRoot32BitConstants(3, kCb, cb, 0);
            } else {
            if (g_fg && s->seq > (q == 0 ? 1u : 0u)) {              // fg: harvest the newest earlier frame if HIP finished it after its own (short) wait
                UINT hb[kCb]; memcpy(hb, cb, sizeof hb); hb[2] = q == 0 ? seq - 1 : s->seq; hb[3] = 1; hb[12] = s->frmOf[hb[2] % 16]; cmd->SetComputeRoot32BitConstants(3, kCb, hb, 0);
                cmd->SetPipelineState(s->spin); cmd->Dispatch(1, 1, 1); cmd->ResourceBarrier(1, &uav);
                cmd->SetPipelineState(s->cap); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav);
            }
            cmd->SetComputeRoot32BitConstants(3, kCb, cb, 0);
            if (q != 0) {                                           // reuse frame: st[0] = 0 forces the reuse/native branch of exp
                cb[3] = 0; cb[2] = s->seq + 1; cmd->SetComputeRoot32BitConstants(3, kCb, cb, 0);   // spin with 0 polls: flag < seq+1 -> st[0] = 0
                cmd->SetPipelineState(s->spin); cmd->Dispatch(1, 1, 1); cmd->ResourceBarrier(1, &uav);
                if (s->histOn && histOk) {          // two+-frame history: accumulate this frame's vectors onto the offsets to the last NR frame
                    cb[26] = histOk; cmd->SetComputeRoot32BitConstants(3, kCb, cb, 0); cmd->SetPipelineState(s->acc); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav); s->accFrm = s->frm; }
            } else {
                cmd->SetPipelineState(s->imp); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav);
                if (s->histOn) { cmd->SetPipelineState(histOk > 1 ? s->hwp2 : s->hwp); cmd->Dispatch((s->padW + 7) / 8, (s->padH + 7) / 8, 1); cmd->ResourceBarrier(1, &uav); s->lastNrFrm = s->frm; }   // history slot for the network
                release();
                cmd->SetPipelineState(s->spin); cmd->Dispatch(1, 1, 1); cmd->ResourceBarrier(1, &uav);   // wait (bounded) for the HIP stream's flag_out = seq
                if (s->histOn) { cmd->SetPipelineState(s->hst); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav); }   // keep this output for the next frame's history
                if (reuseOn) { cmd->SetPipelineState(s->cap); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav); }   // keep the edit (+ input, depth) for the reuse frames
                if (teOn) { for (auto* p : {s->tea, s->teb}) { cmd->SetPipelineState(p); cmd->Dispatch((F->W + 7) / 8, (F->H + 7) / 8, 1); cmd->ResourceBarrier(1, &uav); } s->teFrm = s->frm; }
            }
            }
            if (e5) {                                               // R9G9B9E5 has no typed UAV store: pack into a buffer, copy into the texture
                const UINT pitch = (w * 4 + 255) & ~255u;
                if (!s->ob || s->obW != w || s->obH != h) {
                    if (s->ob) s->ob->Release(); s->ob = nullptr; D3D12_HEAP_PROPERTIES hp = {D3D12_HEAP_TYPE_DEFAULT};
                    D3D12_RESOURCE_DESC bd = {}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = (UINT64)pitch * h; bd.Height = 1; bd.DepthOrArraySize = 1; bd.MipLevels = 1;
                    bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                    HRESULT hr = F->dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&s->ob));
                    if (FAILED(hr)) { s->ob = nullptr; logf("sync: R9G9B9E5 staging buffer failed (0x%08lx) - native frame", (unsigned long)hr); return passThrough(cmd, color, out, w, h); }
                    s->obW = w; s->obH = h; logf("sync: output is R9G9B9E5 - packed in a shader and copied (%ux%u)", w, h);
                }
                cmd->SetComputeRootUnorderedAccessView(4, s->ob->GetGPUVirtualAddress());
                cmd->SetPipelineState(s->expPk); cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
                D3D12_RESOURCE_BARRIER b[2] = {}; for (auto& x : b) { x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; }
                b[0].Transition.pResource = s->ob; b[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
                b[1].Transition.pResource = out; b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
                cmd->ResourceBarrier(2, b);
                D3D12_TEXTURE_COPY_LOCATION dl = {}, sl = {}; dl.pResource = out; dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                sl.pResource = s->ob; sl.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; sl.PlacedFootprint.Footprint = {DXGI_FORMAT_R9G9B9E5_SHAREDEXP, w, h, 1, pitch};
                D3D12_BOX bx = {0, 0, 0, w, h, 1}; cmd->CopyTextureRegion(&dl, 0, 0, 0, &sl, &bx);
                for (auto& x : b) std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
                cmd->ResourceBarrier(2, b);
                return 1;
            }
            cmd->SetComputeRootUnorderedAccessView(4, s->st->GetGPUVirtualAddress());   // unused by exp; bound so the root signature is complete
            cmd->SetPipelineState(s->exp); cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1); cmd->ResourceBarrier(1, &uav);
            return 1;
        }

        D3D12_BOX box = {0, 0, 0, w, h, 1};
        auto barrier = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
            D3D12_RESOURCE_BARRIER rb = {}; rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; rb.Transition.pResource = r;
            rb.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; rb.Transition.StateBefore = a; rb.Transition.StateAfter = b; cmd->ResourceBarrier(1, &rb); };
        auto footprint = [&](ID3D12Resource* buf, DXGI_FORMAT fmt) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = buf; l.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            l.PlacedFootprint.Footprint.Format = fmt; l.PlacedFootprint.Footprint.Width = w; l.PlacedFootprint.Footprint.Height = h;
            l.PlacedFootprint.Footprint.Depth = 1; l.PlacedFootprint.Footprint.RowPitch = F->pitch; return l; };
        auto tex = [&](ID3D12Resource* t) { D3D12_TEXTURE_COPY_LOCATION l = {}; l.pResource = t; l.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; return l; };

        if (F->state == S_IDLE) {                                   // start a job: grab this frame
            memset(F->rbp, 0xFF, (size_t)F->pitch * F->H);          // sentinel; the worker sees it disappear when the copy lands
            barrier(color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            auto dst = footprint(F->rb, cd.Format), src = tex(color); cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
            barrier(color, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            F->jobGen = F->resetGen; F->jobStart = GetTickCount(); F->state = S_WAIT_COPY;
        }
        int slot = F->published; if (slot >= 0) F->slotEval[slot] = (unsigned)n;
        if (slot < 0 && !F->delta) return passThrough(cmd, color, out, w, h);   // copy mode: no answer yet, host keeps the native frame (delta mode passes the live frame through instead)
        if (F->delta) {                                             // out = current colour + stale NR edit, computed on the GPU in the host's list
            UINT k = (UINT)(n % kRing) * kDesc; auto cpu = F->heap->GetCPUDescriptorHandleForHeapStart(); auto gpu = F->heap->GetGPUDescriptorHandleForHeapStart();
            auto at = [&](UINT i) { D3D12_CPU_DESCRIPTOR_HANDLE h = cpu; h.ptr += (SIZE_T)(k + i) * F->incr; return h; };
            D3D12_SHADER_RESOURCE_VIEW_DESC sv = {}; sv.Format = cd.Format; sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; sv.Texture2D.MipLevels = 1;
            F->dev->CreateShaderResourceView(color, &sv, at(0));
            D3D12_SHADER_RESOURCE_VIEW_DESC bv = {}; bv.Format = DXGI_FORMAT_R32_TYPELESS; bv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; bv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            bv.Buffer.NumElements = (UINT)((UINT64)F->pitch * F->H / 4); bv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            F->dev->CreateShaderResourceView(slot < 0 ? F->zero : F->up[slot], &bv, at(1)); F->dev->CreateShaderResourceView(slot < 0 ? F->zero : F->upOld[slot], &bv, at(2));
            D3D12_UNORDERED_ACCESS_VIEW_DESC uv = {}; uv.Format = od.Format; uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D; F->dev->CreateUnorderedAccessView(out, nullptr, &uv, at(3));
            ID3D12DescriptorHeap* hp[] = { F->heap }; cmd->SetDescriptorHeaps(1, hp); cmd->SetComputeRootSignature(F->rs); cmd->SetPipelineState(F->pso);
            gpu.ptr += (UINT64)k * F->incr; cmd->SetComputeRootDescriptorTable(0, gpu);
            UINT cb[6] = { w, h, F->pitch, (UINT)(g_gate && slot >= 0), 0, 0 }; memcpy(&cb[4], &g_gateLo, 4); memcpy(&cb[5], &g_gateHi, 4); cmd->SetComputeRoot32BitConstants(1, 6, cb, 0);
            cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            D3D12_RESOURCE_BARRIER ub = {}; ub.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; ub.UAV.pResource = out; cmd->ResourceBarrier(1, &ub);
            return 1;
        }
        barrier(out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        auto dst = tex(out), src = footprint(F->up[slot], od.Format); cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        barrier(out, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        return 1;
    } catch (...) { return 0; }
}

}   // extern "C"
// Runtime context + every D3D12 resource of F. Callers: release, the worker (released while loading), eviction (keepF).
static void freeRes(Feature* F) {
    if (F->ctx && p_destroy) p_destroy(F->ctx); F->ctx = nullptr;
    if (F->rb) F->rb->Release(); for (auto u : F->up) if (u) u->Release();
    for (auto u : F->upOld) if (u) u->Release(); if (F->zero) F->zero->Release();
    if (F->pso) F->pso->Release(); if (F->rs) F->rs->Release(); if (F->heap) F->heap->Release();
    freeSync(F->sy);   // ponytail: assumes the host released after its last list using us completed (INTERFACE.md: it parks handles); a still-spinning list ends by its bound
    F->rb = F->zero = nullptr; F->up[0] = F->up[1] = F->upOld[0] = F->upOld[1] = nullptr; F->pso = nullptr; F->rs = nullptr; F->heap = nullptr; F->sy = nullptr;
}
static void freeFeature(Feature* F) { try { freeRes(F); } catch (...) {} delete F; }
static void evictIdle(Feature* self) {
    std::lock_guard<std::mutex> g(g_featMu);   // ponytail: held while freeing (<= ~2 s); a release or create on another thread waits that long
    for (Feature* F : g_feats) {                 // release() unlists F under g_featMu first, so every F here is alive
        if (F == self || !F->evaluated) continue;   // never evaluated yet (e.g. pass 0 while the host creates its extra passes): not abandoned
        if (!F->pass) {   // pass 0 stays while any extra pass of the chain was evaluated recently
            bool chainLive = false;
            for (Feature* G : g_feats) if (G != F && G->pass && G->evaluated && GetTickCount() - G->lastUse < (DWORD)g_evictMs) chainLive = true;
            if (chainLive) continue;
        }
        std::unique_lock<std::mutex> lk(F->mu, std::try_to_lock);   // being evaluated right now: in use
        if (!lk.owns_lock() || GetTickCount() - F->lastUse < (DWORD)g_evictMs || F->state == S_LOADING || F->state == S_FAILED) continue;
        F->quit = true; if (F->th.joinable()) F->th.join();   // async worker loop ends; a sync worker has already returned
        logf("evict %p: idle %lu ms while a new feature loads - runtime and buffers freed, native frames if the host uses it again", (void*)F, GetTickCount() - F->lastUse.load());
        F->state = S_FAILED; freeRes(F);   // F itself stays valid for the host's later evaluate/release
    }
}
extern "C" {
__declspec(dllexport) void dlssnr_call_release(void* f) {
    auto* F = (Feature*)f; if (!F) return;
    try {
        logf("release %p", f); F->quit = true; g_live--;
        { std::lock_guard<std::mutex> g(g_featMu); g_feats.erase(std::remove(g_feats.begin(), g_feats.end(), F), g_feats.end()); }
        if (g_exiting) { if (F->th.joinable()) F->th.detach(); return; }   // process exit: never block, leak F
        if (F->state == S_LOADING) { if (F->th.joinable()) F->th.detach(); if (F->done.exchange(true)) freeFeature(F); return; }   // still loading: never block the game, the worker frees F when nr_create returns
        if (F->th.joinable()) F->th.join();
        freeFeature(F);
    } catch (...) {}
}
}
