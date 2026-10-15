// See nr_runtime.h. Dispatch logic mirrors net_run.cpp (same manifest/kernarg rebasing), minus per-frame process
// launch, module loading and arena upload/download.
#include <hip/hip_runtime.h>
#define NR_BUILD
#include "nr_runtime.h"
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <deque>
#include <thread>
#include <condition_variable>
#include <atomic>
#ifdef _WIN32
#include <windows.h>
#include <compressapi.h>
#endif
#include "nr_override.h"

struct Step { hipFunction_t fn; std::vector<uint8_t> ka; unsigned gx, gy, gz, thr; bool pre = false, post = false;   // pre = k_pre_block (its kernarg holds the strength fields)
              const NrOverride* ov = nullptr; std::vector<uint8_t> src; };   // ov: native replacement; src = original rebased kernarg it is built from
struct nr_ctx {
  std::map<std::string, NrOverride> ovr;   // see OVERRIDES.md; empty -> unchanged behaviour
  std::string dir; uint8_t* d = nullptr; size_t arena = 0; uint64_t base = 0, dev = 0;
  size_t off_src = 0, src_size = 0, off_dst = 0, dst_size = 0; int vit_lo = 0, vit_hi = 0, vit_every = 1; long frame = 0;
  std::vector<Step> steps; std::vector<hipModule_t> mods; unsigned mach = 0x36; std::deque<std::vector<uint8_t>> imgs; hipEvent_t e0 = nullptr, e1 = nullptr;
  // strength = the two f32 pairs of k_pre_block PreParams: +0x28 (LocalTone, LocalStructure) and +0x48 (a, b). Set from any thread, applied at the start of the next run.
  std::mutex mx; float str[4] = {0, 0, 1, 1}; float pend[4] = {0, 0, 1, 1}; bool dirty = false;
  // D3D12 interop (nr_ext_*): imported shared buffers, own non-blocking stream, one event per queued frame
  hipExternalMemory_t em_io = nullptr, em_sig = nullptr; uint8_t *io = nullptr, *sig = nullptr; size_t out_off = 0, flag_off = 0;
  // No GPU-side cross-API wait (hipStreamWaitValue32 hung the HIP queue forever when D3D12 never wrote flag_in -> TDR/PC freeze on 2026-10-08):
  // a CPU thread polls flag_in with a timeout and only then launches the frame.
  hipStream_t s = nullptr, ps = nullptr; uint32_t* hf = nullptr; std::thread th; std::condition_variable cv; struct Job { unsigned seq; int hv, part; }; std::deque<Job> q;
  std::atomic<bool> stop{false}; std::atomic<int> inflight{0};
  // temporal history (nr_run2 / nr_ext_enqueue2): own PAD_H x PAD_W x RGB32F buffer, pointed to by pre-block +0x40 and post-block +0x40 when valid
  float* hist = nullptr; uint64_t netout = 0; int hist_mask = 0; unsigned ph = 0, pw = 0; bool have_prev = false; int dbg_n = 0; size_t io_size = 0, hin = 0, hout = 0;
  std::vector<hipEvent_t> prof; std::vector<std::string> names;   // NR_PROF only
  int ksplit[2] = {0, 0};   // nr_ext_split: first dispatch of part 2 for a full / a skipping (vit_every) frame; 0 = no split
  std::atomic<bool> broken{false}; long checked = 0;   // output self-check (ext path): broken -> nr_ext_enqueue* returns -2, the DLL stays native
};
#ifndef NR_BUILD_ID
#define NR_BUILD_ID __DATE__ " " __TIME__
#endif
static void (*g_logcb)(const char*) = nullptr;   // nr_set_log: the DLL's log (dlssnr_nr.log); stderr when unset
static void rlog(const char* fmt, ...) { char b[512]; va_list a; va_start(a, fmt); vsnprintf(b, sizeof b, fmt, a); va_end(a); if (g_logcb) g_logcb(b); else fprintf(stderr, "%s\n", b); }
NR_API void nr_set_log(void (*cb)(const char*)) { g_logcb = cb; }
static const size_t kStrOff0 = 0x28, kStrOff1 = 0x48;
static const size_t PAD = 1 << 16;
#define FAIL(...) do { snprintf(err, errlen, __VA_ARGS__); nr_destroy(c); return nullptr; } while (0)
#define HK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { snprintf(err, errlen, "%s: %s", #x, hipGetErrorString(e_)); nr_destroy(c); return nullptr; } } while (0)

static bool rd(const std::string& p, std::vector<uint8_t>& v) {
  FILE* f = fopen(p.c_str(), "rb"); if (!f) return false;
#ifdef _WIN32
  _fseeki64(f, 0, SEEK_END); long long n = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);   // arena > 2 GB safe
#else
  fseeko(f, 0, SEEK_END); long long n = ftello(f); fseeko(f, 0, SEEK_SET);
#endif
  v.resize((size_t)n); bool ok = !n || fread(v.data(), 1, (size_t)n, f) == (size_t)n; fclose(f); return ok;
}
static std::string rel(const std::string& dir, const std::string& p) {   // relative module paths resolve against the data dir
  return p.size() > 1 && (p[1] == ':' || p[0] == '\\' || p[0] == '/') ? p : dir + "/" + p;
}
static bool upload_arena(nr_ctx* c) {   // pristine arena between two 0xA5 guard pads, exactly as net_run lays it out
  std::vector<uint8_t> a;
  if (!rd(c->dir + "/arena.bin", a)) {   // release packs ship arena.lzms (LZMS, Windows Compression API) instead
#ifdef _WIN32
    std::vector<uint8_t> z; if (!rd(c->dir + "/arena.lzms", z)) return false;
    COMPRESSOR_HANDLE h; SIZE_T n = 0; a.resize(c->arena);
    if (!CreateDecompressor(COMPRESS_ALGORITHM_LZMS, nullptr, &h)) return false;
    BOOL ok = Decompress(h, z.data(), z.size(), a.data(), a.size(), &n); CloseDecompressor(h);
    if (!ok || n != c->arena) return false;
#else
    return false;
#endif
  }
  if (a.size() != c->arena) return false;
  // weights.map: the release ships no vendor weights; copy them by record name from the WEIGHTS_HT resource of the
  // user's nvngx_dlssnr.dll (next to the data dir), so any DLL build carrying the same weights works.
  // Record: u64 name_len, name, u64, u64, u64 D, u32 1, data[D], 20-byte trailer (see rdna2/emu/weights_ht.py).
#ifdef _WIN32
  if (FILE* m = fopen((c->dir + "/weights.map").c_str(), "r")) {
    std::map<std::string, std::pair<const uint8_t*, uint64_t>> rec;
    HMODULE dll = nullptr;   // beside the data dir, or one level up (nr_data\<W>x<H>\ step dirs)
    for (const char* up : {"/../", "/../../"}) if (!dll) dll = LoadLibraryExA((c->dir + up + "nvngx_dlssnr.dll").c_str(), nullptr, LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
    HRSRC r = dll ? FindResourceA(dll, "WEIGHTS_HT", MAKEINTRESOURCEA(10)) : nullptr;
    const uint8_t* b = r ? (const uint8_t*)LockResource(LoadResource(dll, r)) : nullptr; uint64_t sz = r ? SizeofResource(dll, r) : 0, o = 8;
    while (b && o + 8 <= sz) {
      uint64_t L, D; memcpy(&L, b + o, 8); if (o + 8 + L + 28 > sz) break;
      std::string nm((const char*)b + o + 8, L); memcpy(&D, b + o + 8 + L + 16, 8);
      o += 8 + L + 28; if (o + D + 20 > sz) break; rec[nm] = {b + o, D}; o += D + 20;
    }
    unsigned long long want = 0, h = 0xcbf29ce484222325ull, ao, n; int cnt = 0; char nm[256];
    bool ok = !rec.empty() && fscanf(m, "fnv %llu %d", &want, &cnt) == 2;
    for (int i = 0; ok && i < cnt; i++) {
      ok = fscanf(m, "%llu %255s %llu", &ao, nm, &n) == 3 && rec.count(nm) && rec[nm].second == n && ao + n <= a.size();
      if (ok) memcpy(a.data() + ao, rec[nm].first, n);
      for (unsigned long long k = 0; ok && k < n; k++) h = (h ^ a[ao + k]) * 0x100000001b3ull;
    }
    fclose(m); if (dll) FreeLibrary(dll);
    if (!ok || h != want) return false;   // missing nvngx_dlssnr.dll or one with different weights
  }
#endif
  std::vector<uint8_t> host(a.size() + 2 * PAD, 0xA5); memcpy(host.data() + PAD, a.data(), a.size());
  return hipMemcpy(c->d, host.data(), host.size(), hipMemcpyHostToDevice) == hipSuccess;
}

// Kernels are built for gfx1030. Every RDNA2 chip (gfx1030-1036) runs the same ISA (LLVM's gfx10-3-generic), so other
// RDNA2 cards only need the ELF e_flags machine id relabelled at load time. Clang offload bundles are unwrapped first.
static unsigned rdna2_mach(const char* arch) {
  static const struct { const char* n; unsigned m; } t[] = {{"gfx1030", 0x36}, {"gfx1031", 0x37}, {"gfx1032", 0x38}, {"gfx1033", 0x39},
                                                            {"gfx1034", 0x3e}, {"gfx1035", 0x3d}, {"gfx1036", 0x45}};
  for (auto& e : t) if (!strncmp(arch, e.n, 7) && (arch[7] == 0 || arch[7] == ':')) return e.m;
  return 0;
}
static hipError_t load_mod(nr_ctx* c, const std::string& path, hipModule_t* m) {
  std::vector<uint8_t> b; if (!rd(path, b)) return hipErrorFileNotFound;
  if (b.size() > 32 && !memcmp(b.data(), "__CLANG_OFFLOAD_BUNDLE__", 24)) {   // u64 n, then n x (u64 off, u64 size, u64 idlen, id)
    uint64_t n, off, sz, il; size_t o = 32; memcpy(&n, b.data() + 24, 8); std::vector<uint8_t> elf;
    for (uint64_t i = 0; i < n && o + 24 <= b.size(); i++) {
      memcpy(&off, b.data() + o, 8); memcpy(&sz, b.data() + o + 8, 8); memcpy(&il, b.data() + o + 16, 8); o += 24;
      std::string id((const char*)b.data() + o, (size_t)il); o += (size_t)il;
      if (id.find("gfx1030") != std::string::npos && off + sz <= b.size()) elf.assign(b.begin() + off, b.begin() + off + sz);
    }
    if (elf.empty()) return hipErrorInvalidImage;
    b.swap(elf);
  }
  if (b.size() > 0x34 && b[0] == 0x7f && !memcmp(b.data() + 1, "ELF", 3) && b[0x30] == 0x36) b[0x30] = (uint8_t)c->mach;   // EF_AMDGPU_MACH, low byte
  c->imgs.push_back(std::move(b));   // kept alive for the module's lifetime
  return hipModuleLoadData(m, c->imgs.back().data());
}

NR_API nr_ctx* nr_create(const char* data_dir, int width, int height, int vit_every, char* err, int errlen) {
  nr_ctx* c = new nr_ctx; c->dir = data_dir; const bool bn = vit_every < -1; c->vit_every = bn ? -vit_every : vit_every < 1 ? 1 : vit_every;
  std::map<std::string, std::string> kv;   // meta.txt: key=value
  { FILE* f = fopen((c->dir + "/meta.txt").c_str(), "r"); if (!f) FAIL("no meta.txt in %s", data_dir);
    char l[256]; while (fgets(l, sizeof l, f)) { char* q = strchr(l, '='); if (q) { *q = 0; kv[l] = std::string(q + 1); } } fclose(f); }
  auto num = [&](const char* k) { return strtoull(kv[k].c_str(), nullptr, 10); };
  if ((int)num("width") != width || (int)num("height") != height) FAIL("data dir is %sx%s, asked for %dx%d", kv["width"].c_str(), kv["height"].c_str(), width, height);
  c->base = num("base"); c->arena = num("arena_size"); c->off_src = num("off_src"); c->src_size = num("src_size");
  c->off_dst = num("off_dst"); c->dst_size = num("dst_size"); c->vit_lo = (int)num("vit_lo"); c->vit_hi = (int)num("vit_hi");
  if (!bn && c->vit_every > 1 && num("off_vitres") == num("off_headb")) FAIL("vit_every>1 needs data generated with VIT_OWN=1");

  hipDeviceProp_t p{}; HK(hipGetDeviceProperties(&p, 0));
  { size_t fr = 0, tot = 0; hipMemGetInfo(&fr, &tot); int rv = 0, dv = 0; hipRuntimeGetVersion(&rv); hipDriverGetVersion(&dv);
    rlog("runtime: build %s | GPU %s (%s), %d CUs (HIP count, WGPs on RDNA), VRAM %zu MB total / %zu MB free | HIP runtime %d, driver %d",
         NR_BUILD_ID, p.name, p.gcnArchName, p.multiProcessorCount, tot >> 20, fr >> 20, rv, dv);
    const size_t need = c->arena + 2 * PAD + (64u << 20) + (256u << 20);   // arena + history/io (~64 MB) + headroom for the game
    if (fr && fr < need) FAIL("not enough free VRAM: NR needs about %zu MB, only %zu MB of %zu MB free - NR stays off (lower texture quality/resolution or close other GPU apps)", need >> 20, fr >> 20, tot >> 20); }
  c->mach = rdna2_mach(p.gcnArchName); if (!c->mach) FAIL("not an RDNA2 GPU (gfx1030-1036): %s", p.gcnArchName);
  HK(hipMalloc(&c->d, c->arena + 2 * PAD)); c->dev = (uint64_t)(c->d + PAD);
  if (!upload_arena(c)) FAIL("arena data missing/wrong size, nvngx_dlssnr.dll missing or wrong version next to %s, or upload failed", data_dir);

  std::vector<uint8_t> kab; if (!rd(c->dir + "/kernargs.bin", kab)) FAIL("no kernargs.bin");
  FILE* f = fopen((c->dir + "/manifest.txt").c_str(), "r"); if (!f) FAIL("no manifest.txt");
  { std::string m; if (!nr_load_overrides(c->dir + "/manifest.txt", c->ovr, m)) { fclose(f); FAIL("%s", m.c_str()); } }
  std::map<std::string, hipModule_t> mods; char line[1024];
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    std::string s(line); while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    std::vector<std::string> t; size_t pos = 0;   // '|' separated: module paths contain spaces
    for (;;) { size_t q = s.find('|', pos); if (q == std::string::npos) { t.push_back(s.substr(pos)); break; } t.push_back(s.substr(pos, q - pos)); pos = q + 1; }
    if (t.size() != 7 && t.size() != 8) { fclose(f); FAIL("bad manifest line: %s", line); }
    t[0] = rel(c->dir, t[0]);
    if (!mods.count(t[0])) { hipModule_t m; hipError_t e = load_mod(c, t[0], &m); if (e != hipSuccess) { fclose(f); FAIL("hipModuleLoad %s: %s", t[0].c_str(), hipGetErrorString(e)); } mods[t[0]] = m; c->mods.push_back(m); }
    Step st; st.pre = t[1].find("k_pre_block") != std::string::npos; st.post = t[1].find("k_post_block") != std::string::npos; hipError_t e = hipModuleGetFunction(&st.fn, mods[t[0]], t[1].c_str()); if (e != hipSuccess) { fclose(f); FAIL("no symbol %s", t[1].c_str()); }
    size_t off = strtoull(t[2].c_str(), nullptr, 10), len = strtoull(t[3].c_str(), nullptr, 10);
    if (off + len > kab.size()) { fclose(f); FAIL("kernarg out of range"); }
    st.ka.assign(kab.begin() + off, kab.begin() + off + len);
    for (size_t o = 0; o + 8 <= st.ka.size(); o += 8) {   // rebase arena pointers once, not per frame
      uint64_t v; memcpy(&v, st.ka.data() + o, 8);
      if (v >= c->base && v < c->base + c->arena) { v = c->dev + (v - c->base); memcpy(st.ka.data() + o, &v, 8); }
    }
    st.gx = atoi(t[4].c_str()); st.gy = atoi(t[5].c_str()); st.thr = atoi(t[6].c_str()); st.gz = t.size() == 8 ? atoi(t[7].c_str()) : 1;
    auto ov = c->ovr.find(t[1]);
    if (ov != c->ovr.end()) {   // swap in the native kernel; its kernarg is rebuilt from the original one (also after strength patches)
      NrOverride& o = ov->second; st.ov = &o;
      o.mod = rel(c->dir, o.mod);
      if (!mods.count(o.mod)) { hipModule_t m; hipError_t e = load_mod(c, o.mod, &m); if (e != hipSuccess) { fclose(f); FAIL("hipModuleLoad %s: %s", o.mod.c_str(), hipGetErrorString(e)); } mods[o.mod] = m; c->mods.push_back(m); }
      if (hipModuleGetFunction(&st.fn, mods[o.mod], o.sym.c_str()) != hipSuccess) { fclose(f); FAIL("no symbol %s", o.sym.c_str()); }
      st.gx = nr_grid(o.g[0], st.gx); st.gy = nr_grid(o.g[1], st.gy); st.gz = nr_grid(o.g[2], st.gz); st.thr = nr_grid(o.g[3], st.thr);
      st.src.swap(st.ka); if (!nr_apply_override(o, st.src, st.ka)) { fclose(f); FAIL("override for %s copies past the original kernarg", t[1].c_str()); }
    }
    if (st.pre && (st.ov ? st.src : st.ka).size() >= kStrOff1 + 8) {          // remember the data dir's own values as the current strength
      const auto& k = st.ov ? st.src : st.ka; memcpy(&c->str[0], k.data() + kStrOff0, 8); memcpy(&c->str[2], k.data() + kStrOff1, 8);
      memcpy(c->pend, c->str, sizeof c->str);
    }
    if (st.post) { const auto& k = st.ov ? st.src : st.ka; memcpy(&c->netout, k.data() + 0x10, 8); memcpy(&c->ph, k.data() + 0x20, 4); memcpy(&c->pw, k.data() + 0x24, 4); }
    c->names.push_back(st.ov ? st.ov->sym : t[1]); c->steps.push_back(std::move(st));
  }
  fclose(f);
  if (bn) {   // bottleneck reuse: also skip the c512 encoder stage + k_final_head before the ViT and k_dec_upsample + c512 decoder stage after it
    c->vit_lo -= 33; c->vit_hi += 33;   // (32 dispatches each side + 1). Their outputs live in own arena buffers that nothing later overwrites.
    if (c->vit_lo < 0 || c->vit_hi >= (int)c->names.size() || c->names[c->vit_lo].find("ffwd2") == std::string::npos || c->names[c->vit_hi].find("swin") == std::string::npos)
      FAIL("bottleneck reuse: unexpected dispatch layout around %d..%d", c->vit_lo, c->vit_hi);
  }
  if (c->pw) HK(hipMalloc(&c->hist, (size_t)c->pw * c->ph * 12));   // ~8 MB; filled after every frame so the first hist_valid frame already has history
  HK(hipEventCreate(&c->e0)); HK(hipEventCreate(&c->e1));
  if (getenv("NR_PROF")) { c->prof.resize(c->steps.size()); for (auto& e : c->prof) HK(hipEventCreate(&e)); }
  return c;
}

NR_API size_t nr_bytes(nr_ctx* c) { return c->src_size; }
NR_API int nr_hist_layout(nr_ctx* c, unsigned* pad_w, unsigned* pad_h) { if (!c || !c->pw) return 1; if (pad_w) *pad_w = c->pw; if (pad_h) *pad_h = c->ph; return 0; }

// History: before a frame, point pre/post +0x40 at the history buffer (valid) or null (= today's path, bit-exact); after it, keep the
// post-block output rgb (P space) as the next frame's unwarped history and optionally hand it to the DLL at io+hout.
static size_t hist_bytes(nr_ctx* c) { return (size_t)c->pw * c->ph * 12; }
static bool hist_begin(nr_ctx* c, hipStream_t s, int valid, const uint8_t* src) {
  bool v = valid && c->hist && (src || c->have_prev);
  if (v && src && hipMemcpyAsync(c->hist, src, hist_bytes(c), hipMemcpyDeviceToDevice, s) != hipSuccess) return false;
  // hist_valid: 1 = pre-block lanes + post lerp, 2 = post lerp only (stable on static frames), 3 = pre-block only (debug). Env NR_HIST_MASK overrides 1.
  static const int env = getenv("NR_HIST_MASK") ? atoi(getenv("NR_HIST_MASK")) : 3;
  const int mask = !v ? 0 : valid == 2 ? 2 : valid == 3 ? 1 : env;
  if (mask != c->hist_mask) {
    for (auto& st : c->steps) if (st.pre || st.post) {
      uint64_t p = (st.pre ? mask & 1 : mask & 2) ? (uint64_t)c->hist : 0; auto& k = st.ov ? st.src : st.ka;
      if (k.size() >= 0x48) { memcpy(k.data() + 0x40, &p, 8); if (st.ov) nr_apply_override(*st.ov, st.src, st.ka); } }
    c->hist_mask = mask;
  }
  return true;
}
static bool hist_end(nr_ctx* c, hipStream_t s) {   // RGBA f32 (16 B/px) -> RGB f32 (12 B/px), same padded layout
  if (!c->hist) return true;
  bool ok = hipMemcpy2DAsync(c->hist, 12, (void*)c->netout, 16, 12, (size_t)c->pw * c->ph, hipMemcpyDeviceToDevice, s) == hipSuccess;
  if (ok && c->io && c->hout) ok = hipMemcpy2DAsync(c->io + c->hout, 12, (void*)c->netout, 16, 12, (size_t)c->pw * c->ph, hipMemcpyDeviceToDevice, s) == hipSuccess;
  c->have_prev = ok; return ok;
}

static void apply_strength(nr_ctx* c) {                    // between frames, from the run thread: patch the pre-block kernarg copies
  std::lock_guard<std::mutex> g(c->mx); if (!c->dirty) return;
  for (auto& st : c->steps) { auto& k = st.ov ? st.src : st.ka;   // strength offsets are in the original layout
    if (st.pre && k.size() >= kStrOff1 + 8) { memcpy(k.data() + kStrOff0, &c->pend[0], 8); memcpy(k.data() + kStrOff1, &c->pend[2], 8); if (st.ov) nr_apply_override(*st.ov, st.src, st.ka); } }
  memcpy(c->str, c->pend, sizeof c->str); c->dirty = false;
}
NR_API int nr_set_strength(nr_ctx* c, float tone, float structure, float a, float b) {
  if (!c) return 1; if (!(tone >= 0 && tone <= 64 && structure >= 0 && structure <= 64 && a >= 0 && a <= 64 && b >= 0 && b <= 64)) return 2;   // rejects NaN and absurd values
  std::lock_guard<std::mutex> g(c->mx); c->pend[0] = tone; c->pend[1] = structure; c->pend[2] = a; c->pend[3] = b; c->dirty = true; return 0;
}
NR_API int nr_set_intensity(nr_ctx* c, float i) {          // user-facing 0..2 (measured on the real frame, see own/STRENGTH_RUNTIME.md): 0 = effect off, 1 = default, 2 = largest clean setting
  if (!(i >= 0)) return 2; if (i > 2.f) i = 2.f;                // beyond 2 the picture breaks up (blocky noise, brightness shift), so it is clamped
  if (i <= 1.f) return nr_set_strength(c, 0, 0, i, i);
  float k = i - 1.f; return nr_set_strength(c, 4 * k, 4 * k, 1 + 7 * k, 1 + 7 * k);
}
NR_API int nr_get_strength(nr_ctx* c, float* tone, float* structure, float* a, float* b) {   // the values in effect for the last/next run (pending ones once applied)
  if (!c) return 1; std::lock_guard<std::mutex> g(c->mx); const float* v = c->dirty ? c->pend : c->str;
  if (tone) *tone = v[0]; if (structure) *structure = v[1]; if (a) *a = v[2]; if (b) *b = v[3]; return 0;
}

static bool skipping(nr_ctx* c) { return c->vit_every > 1 && (c->frame % c->vit_every) != 0; }   // frame 0 always computes
static bool launch_range(nr_ctx* c, hipStream_t st, size_t lo, size_t hi) {   // the network's dispatches [lo, hi), in manifest order
  bool skip_vit = skipping(c);
  for (size_t i = lo; i < hi && i < c->steps.size(); i++) {
    if (c->dbg_n > 0 && (int)i >= c->dbg_n) break;
    if (skip_vit && (int)i >= c->vit_lo && (int)i < c->vit_hi) continue;
    Step& s = c->steps[i]; size_t sz = s.ka.size();
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, s.ka.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    if (hipModuleLaunchKernel(s.fn, s.gx, s.gy, s.gz, s.thr, 1, 1, 0, st, nullptr, cfg) != hipSuccess) return false;
    if (c->prof.size()) hipEventRecord(c->prof[i], st);
  }
  return true;
}
static bool launch_all(nr_ctx* c, hipStream_t st) { return launch_range(c, st, 0, c->steps.size()); }

NR_API int nr_run(nr_ctx* c, const void* in, void* out, float* gpu_ms) { return nr_run2(c, in, out, gpu_ms, 0); }
NR_API int nr_run2(nr_ctx* c, const void* in, void* out, float* gpu_ms, int hist_valid) {
  apply_strength(c);
  if (!hist_begin(c, nullptr, hist_valid, nullptr)) return 5;
  if (hipMemcpy(c->d + PAD + c->off_src, in, c->src_size, hipMemcpyHostToDevice) != hipSuccess) return 1;
  // k_export reads the previous contents of its destination (it behaves like a max/accumulate on many pixels): without clearing it, every frame after
  // the first carries bright single-pixel specks from the previous result (measured: 2010 vs 26 spikes>0.15 per frame). Always start from zeros, as frame 0 does.
  if (hipMemset(c->d + PAD + c->off_dst, 0, c->dst_size) != hipSuccess) return 1;
  hipEventRecord(c->e0);
  if (!launch_all(c, nullptr)) return 2;
  if (!hist_end(c, nullptr)) return 5;
  hipEventRecord(c->e1);
  if (hipEventSynchronize(c->e1) != hipSuccess) return 3;   // ponytail: no timeout like net_run's SWIN_TIMEOUT; add if a hang matters
  if (gpu_ms) hipEventElapsedTime(gpu_ms, c->e0, c->e1);
  if (c->prof.size() && c->frame == 5) {   // NR_PROF=1: per-dispatch GPU time of frame 5 to stderr (events between launches serialize nothing extra)
    float t; hipEvent_t prev = c->e0;
    for (size_t i = 0; i < c->steps.size(); i++) { hipEventElapsedTime(&t, prev, c->prof[i]); fprintf(stderr, "prof %3zu %.3f %s\n", i, t, c->names[i].c_str()); prev = c->prof[i]; }
  }
  if (hipMemcpy(out, c->d + PAD + c->off_dst, c->dst_size, hipMemcpyDeviceToHost) != hipSuccess) return 4;
  c->frame++; return 0;
}

// Debug (stage tracing): run only the first n dispatches (0 = all); copy arena bytes [off, off+size) to host.
NR_API int nr_dbg_steps(nr_ctx* c, int n) { if (!c) return 1; c->dbg_n = n; return 0; }
NR_API int nr_dbg_read(nr_ctx* c, unsigned long long off, unsigned long long size, void* dst) {
  if (!c || off + size > c->arena) return 1; return hipMemcpy(dst, c->d + PAD + off, size, hipMemcpyDeviceToHost) != hipSuccess;
}
NR_API int nr_reset(nr_ctx* c) { c->frame = 0; c->have_prev = false; return upload_arena(c) ? 0 : 1; }

static void ext_thread(nr_ctx* c);
#ifdef _WIN32
static bool import_buf(void* h, size_t alloc, size_t size, hipExternalMemory_t* em, uint8_t** p) {
  hipExternalMemoryHandleDesc md = {}; md.type = hipExternalMemoryHandleTypeD3D12Resource; md.handle.win32.handle = h; md.size = alloc; md.flags = hipExternalMemoryDedicated;
  if (hipImportExternalMemory(em, &md) != hipSuccess) return false;
  hipExternalMemoryBufferDesc bd = {}; bd.size = size;
  if (hipExternalMemoryGetMappedBuffer((void**)p, *em, &bd) != hipSuccess) { hipDestroyExternalMemory(*em); *em = nullptr; return false; }
  return true;
}
#endif
NR_API int nr_ext_open(nr_ctx* c, void* io, unsigned long long io_alloc, unsigned long long io_size, unsigned long long out_off,
                       unsigned long long flag_off, void* sig, unsigned long long sig_alloc, unsigned long long sig_size, char* err, int errlen) {
#ifdef _WIN32
  if (!c || c->s) { snprintf(err, errlen, "no context or already open"); return 1; }
  if (out_off < c->src_size || out_off + c->dst_size > flag_off || flag_off + 4 > io_size || sig_size < 4) { snprintf(err, errlen, "io layout too small"); return 2; }
  if (!import_buf(io, io_alloc, io_size, &c->em_io, &c->io) || !import_buf(sig, sig_alloc, sig_size, &c->em_sig, &c->sig)) {
    snprintf(err, errlen, "hipImportExternalMemory (D3D12Resource) failed"); nr_ext_close(c, 0); return 4; }
  c->out_off = out_off; c->flag_off = flag_off; c->io_size = io_size;
  if (hipStreamCreateWithFlags(&c->s, hipStreamNonBlocking) != hipSuccess || hipStreamCreateWithFlags(&c->ps, hipStreamNonBlocking) != hipSuccess ||
      hipHostMalloc((void**)&c->hf, 4) != hipSuccess) { snprintf(err, errlen, "hipStreamCreate failed"); nr_ext_close(c, 0); return 5; }
  c->th = std::thread(ext_thread, c);
  return 0;
#else
  snprintf(err, errlen, "D3D12 interop is Windows only"); return 9;
#endif
}
// One-off output sanity check (ext path, after the 10th finished frame): reads the io input and output (RGBA16F, ~2x7 MB once) and
// disables NR for the session if the output is non-finite or mostly pure black/white where the input is not.
// Env NR_SELFCHECK_TEST=1 corrupts that output first (test hook).
static float h2f_(uint16_t h) { uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31, m = h & 1023, u;
  if (e == 31) u = s | 0x7f800000u | (m << 13); else if (e) u = s | ((e + 112) << 23) | (m << 13); else { float f = m / 16777216.f; return s ? -f : f; }
  float f; memcpy(&f, &u, 4); return f; }
static void self_check(nr_ctx* c) {
  if (getenv("NR_SELFCHECK_TEST") && atoi(getenv("NR_SELFCHECK_TEST"))) hipMemsetAsync(c->io + c->out_off, 0xFF, c->dst_size, c->s);   // all NaN
  const size_t n = c->dst_size < c->src_size ? c->dst_size : c->src_size; std::vector<uint16_t> in(n / 2), out(n / 2);
  if (hipMemcpyAsync(in.data(), c->io, n, hipMemcpyDeviceToHost, c->s) != hipSuccess || hipMemcpyAsync(out.data(), c->io + c->out_off, n, hipMemcpyDeviceToHost, c->s) != hipSuccess ||
      hipStreamSynchronize(c->s) != hipSuccess) { rlog("selfcheck: readback failed - skipped"); return; }
  size_t px = 0, bad = 0, bwo = 0, bwi = 0; double mean = 0, dif = 0;
  auto bw = [](const float* v) { return (v[0] <= 0 && v[1] <= 0 && v[2] <= 0) || (v[0] >= 1 && v[1] >= 1 && v[2] >= 1); };
  for (size_t i = 0; i + 4 <= out.size(); i += 4 * 7) {   // every 7th pixel
    float o[3], a[3]; bool fin = true; for (int k = 0; k < 3; k++) { o[k] = h2f_(out[i + k]); a[k] = h2f_(in[i + k]); fin = fin && std::isfinite(o[k]); }
    px++; if (!fin) { bad++; continue; }
    bwo += bw(o); bwi += bw(a); for (int k = 0; k < 3; k++) { mean += o[k]; if (std::isfinite(a[k])) dif += fabs(o[k] - a[k]); }
  }
  const size_t ok = px - bad; const double fb = px ? (double)bad / px : 1, fo = px ? (double)bwo / px : 0, fi = px ? (double)bwi / px : 0;
  rlog("selfcheck: %zu px sampled, non-finite %.2f%%, black/white out %.1f%% (in %.1f%%), mean %.4f, mean |out-in| %.4f",
       px, 100 * fb, 100 * fo, 100 * fi, ok ? mean / (3 * ok) : 0.0, ok ? dif / (3 * ok) : 0.0);
  if (fb > 0.001 || fo - fi >= 0.30) { c->broken = true; rlog("selfcheck: NR output looks broken on this GPU - disabling (native frames for this session)"); }
}
static const int kExtWaitMs = 500;   // flag_in wait bound; longer than the D3D12 spin bound, so a late frame is dropped on both sides
static void ext_thread(nr_ctx* c) {
  for (;;) {
    unsigned seq; int hv, part;
    { std::unique_lock<std::mutex> g(c->mx); c->cv.wait(g, [&] { return c->stop || !c->q.empty(); }); if (c->stop) return; seq = c->q.front().seq; hv = c->q.front().hv; part = c->q.front().part; c->q.pop_front(); }
    bool ready = false;
#ifdef _WIN32
    DWORD t0 = GetTickCount();
    while (!c->stop && GetTickCount() - t0 < (DWORD)kExtWaitMs) {
      if (hipMemcpyAsync(c->hf, c->sig, 4, hipMemcpyDeviceToHost, c->ps) != hipSuccess || hipStreamSynchronize(c->ps) != hipSuccess) break;
      if ((int)(*c->hf - seq) >= 0) { ready = true; break; }
      Sleep(0);
    }
#endif
    if (ready) {   // late/never-executed frame: skip it; D3D12's bounded spin shows the native frame
      hipStream_t s = c->s; const size_t n = c->steps.size(), k = part ? (size_t)c->ksplit[skipping(c)] : n;   // part 1 = [0, k), part 2 = [k, n); the arena keeps everything in between
      bool ok = true;
      if (part != 2) { apply_strength(c);
        ok = hipMemcpyAsync(c->d + PAD + c->off_src, c->io, c->src_size, hipMemcpyDeviceToDevice, s) == hipSuccess &&
             hist_begin(c, s, hv, hv && c->hin ? c->io + c->hin : nullptr) &&
             hipMemsetAsync(c->d + PAD + c->off_dst, 0, c->dst_size, s) == hipSuccess &&   // k_export accumulates into its destination, see nr_run
             launch_range(c, s, 0, k); }
      if (part != 1)
        ok = ok && launch_range(c, s, part ? k : n, n) && hist_end(c, s) &&
             hipMemcpyAsync(c->io + c->out_off, c->d + PAD + c->off_dst, c->dst_size, hipMemcpyDeviceToDevice, s) == hipSuccess;
      ok = ok && hipStreamWriteValue32(s, c->io + c->flag_off, seq, 0) == hipSuccess;
      hipStreamSynchronize(s); if (part != 1) c->frame++;
      if (ok && part != 1 && ++c->checked == 10) self_check(c);
      if (!ok) fprintf(stderr, "nr_ext: frame %u launch failed\n", seq);
    }
    c->inflight--;
  }
}
NR_API int nr_ext_enqueue(nr_ctx* c, unsigned seq, int max_inflight) { return nr_ext_enqueue2(c, seq, max_inflight, 0); }
NR_API int nr_ext_history(nr_ctx* c, unsigned long long hist_in_off, unsigned long long hist_out_off) {
  if (!c || !c->io || !c->pw) return 1;
  size_t n = hist_bytes(c);
  for (unsigned long long o : {hist_in_off, hist_out_off}) if (o && (o < c->src_size || o + n > c->io_size || o % 4)) return 2;
  c->hin = hist_in_off; c->hout = hist_out_off; return 0;
}
NR_API int nr_ext_enqueue2(nr_ctx* c, unsigned seq, int max_inflight, int hist_valid) { return nr_ext_enqueue3(c, seq, max_inflight, hist_valid, 0); }
NR_API int nr_ext_enqueue3(nr_ctx* c, unsigned seq, int max_inflight, int hist_valid, int part) {
  if (!c || !c->s || part < 0 || part > 2 || (part && !c->ksplit[0])) return -1;
  if (c->broken) return -2;   // self-check failed: the DLL goes native for the session
  if (c->inflight >= (max_inflight < 1 ? 1 : max_inflight)) return 1;
  c->inflight++; { std::lock_guard<std::mutex> g(c->mx); c->q.push_back({seq, part == 2 ? 0 : hist_valid, part}); } c->cv.notify_one(); return 0;
}
// Split point: k where the dispatches before it take half the GPU time. Measured once (3 runs on the default stream, the 1st is warm-up; the
// arena only sees two extra frames on its pristine input before the first real one), or fixed by the caller. Skipping frames (vit_every)
// get their own k over the dispatches they run.
NR_API int nr_ext_split(nr_ctx* c, int* k_full, int* k_skip, float* ms) {
  if (!c || !c->s || !k_full || !k_skip) return 1;
  const int n = (int)c->steps.size(); std::vector<double> t(n, 0.0);
  if (*k_full <= 0 || *k_skip <= 0) {
    std::vector<hipEvent_t> ev(n + 1); for (auto& e : ev) if (hipEventCreate(&e) != hipSuccess) return 2;
    bool ok = true; const long f0 = c->frame; c->frame = 0;   // frame 0: every dispatch runs
    for (int r = 0; r < 3 && ok; r++) {
      ok = hipEventRecord(ev[0], c->s) == hipSuccess;
      for (int i = 0; i < n && ok; i++) ok = launch_range(c, c->s, i, i + 1) && hipEventRecord(ev[i + 1], c->s) == hipSuccess;
      ok = ok && hipStreamSynchronize(c->s) == hipSuccess;
      for (int i = 0; i < n && ok && r; i++) { float ms = 0; hipEventElapsedTime(&ms, ev[i], ev[i + 1]); t[i] += ms; }
    }
    c->frame = f0; for (auto& e : ev) hipEventDestroy(e); if (!ok) return 3;
  }
  auto half = [&](bool skip) { double tot = 0, acc = 0; for (int i = 0; i < n; i++) if (!(skip && i >= c->vit_lo && i < c->vit_hi)) tot += t[i];
    int best = n / 2; double bd = 1e30; for (int i = 0; i < n; i++) { if (fabs(acc - tot / 2) < bd) { bd = fabs(acc - tot / 2); best = i; } if (!(skip && i >= c->vit_lo && i < c->vit_hi)) acc += t[i]; }
    return best < 1 ? 1 : best; };
  if (*k_full <= 0) *k_full = half(false); if (*k_skip <= 0) *k_skip = c->vit_every > 1 ? half(true) : *k_full;
  if (*k_full >= n) *k_full = n - 1; if (*k_skip >= n) *k_skip = n - 1;
  c->ksplit[0] = *k_full; c->ksplit[1] = *k_skip;
  if (ms) for (int j = 0; j < 4; j++) { ms[j] = 0; for (int i = 0; i < n; i++) if ((i < (j < 2 ? *k_full : *k_skip)) == !(j & 1) && !(j >= 2 && c->vit_every > 1 && i >= c->vit_lo && i < c->vit_hi)) ms[j] += (float)(t[i] / 2); }
  return 0;
}
NR_API int nr_ext_close(nr_ctx* c, int) {
  if (!c) return 0;
  if (c->th.joinable()) { { std::lock_guard<std::mutex> g(c->mx); c->stop = true; c->q.clear(); } c->cv.notify_all(); c->th.join(); }   // bounded: one poll <= kExtWaitMs
  if (c->s) { hipStreamSynchronize(c->s); hipStreamDestroy(c->s); c->s = nullptr; }
  if (c->ps) { hipStreamDestroy(c->ps); c->ps = nullptr; }
  if (c->hf) { hipHostFree(c->hf); c->hf = nullptr; }
  if (c->em_io) hipDestroyExternalMemory(c->em_io); if (c->em_sig) hipDestroyExternalMemory(c->em_sig);
  c->em_io = c->em_sig = nullptr; c->io = c->sig = nullptr; c->hin = c->hout = 0; c->stop = false; c->inflight = 0; return 0;
}

NR_API void nr_destroy(nr_ctx* c) {
  if (!c) return;
  if (nr_ext_close(c, 2000)) return;   // GPU still busy with our memory: leak the context instead of freeing under it
  for (auto m : c->mods) hipModuleUnload(m);
  if (c->d) hipFree(c->d);
  if (c->hist) hipFree(c->hist);
  delete c;
}
