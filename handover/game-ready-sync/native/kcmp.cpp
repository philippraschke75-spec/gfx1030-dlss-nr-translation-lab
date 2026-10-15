// kcmp <data dir> <extra overrides> <color.bin> [dump dir]: per-dispatch A/B of the manifest kernel against a native
// replacement. Runs one frame in manifest order; every step whose symbol has an entry in <extra overrides> is run twice
// from the same arena state (manifest kernel = A, native = B), the whole arena is compared, the A state is kept so
// later steps stay isolated. Prints per step: differing bytes, span, GPU ms of A and B (mean of 5).
// With a dump dir: <dir>/s<step>_{in,a,b}.bin = the differing span before / after A / after B.
// build: hipcc -std=c++17 -O2 kcmp.cpp -lcabinet -o kcmp.exe (includes the runtime source for its loader)
#define NOMINMAX
#include "../runtime/nr_runtime.cpp"

int main(int argc, char** argv) {
  if (argc < 4) { printf("usage: kcmp datadir extra_overrides color.bin [dumpdir]\n"); return 9; }
  char err[512] = {}; std::vector<uint8_t> in; if (!rd(argv[3], in)) { printf("no %s\n", argv[3]); return 9; }
  int W = 0, H = 0; { FILE* f = fopen((std::string(argv[1]) + "/meta.txt").c_str(), "r"); char l[256];
    while (f && fgets(l, sizeof l, f)) { sscanf(l, "width=%d", &W); sscanf(l, "height=%d", &H); } if (f) fclose(f); }
  nr_ctx* c = nr_create(argv[1], W, H, 1, err, sizeof err); if (!c) { printf("create: %s\n", err); return 1; }
  std::map<std::string, NrOverride> ex; std::string msg;
  _putenv_s("NR_OVERRIDES", argv[2]); if (!nr_load_overrides("", ex, msg)) { printf("%s\n", msg.c_str()); return 1; }
  if (in.size() != c->src_size) { printf("color size %zu != %zu\n", in.size(), c->src_size); return 1; }
  hipMemcpy(c->d + PAD + c->off_src, in.data(), c->src_size, hipMemcpyHostToDevice); hipMemset(c->d + PAD + c->off_dst, 0, c->dst_size);
  uint8_t* bk; if (hipMalloc(&bk, c->arena) != hipSuccess) { printf("no backup memory\n"); return 1; }
  std::vector<uint8_t> A(c->arena), B(c->arena), I; hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  uint8_t* ar = c->d + PAD;
  auto run = [&](Step& s, int reps) {   // restore + launch reps times, returns mean kernel ms
    float tot = 0;
    for (int r = 0; r < reps; r++) {
      hipMemcpy(ar, bk, c->arena, hipMemcpyDeviceToDevice);
      size_t sz = s.ka.size(); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, s.ka.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
      hipEventRecord(e0); hipModuleLaunchKernel(s.fn, s.gx, s.gy, s.gz, s.thr, 1, 1, 0, nullptr, nullptr, cfg); hipEventRecord(e1);
      if (hipEventSynchronize(e1) != hipSuccess) { printf("kernel failed\n"); exit(3); }
      float t; hipEventElapsedTime(&t, e0, e1); tot += t;
    }
    return tot / reps;
  };
  std::map<std::string, hipModule_t> mods;
  for (int f = getenv("KCMP_WARM") ? atoi(getenv("KCMP_WARM")) : 0; f > 0; f--) {   // earlier frames (manifest kernels only): buffers hold last frame's data
    if (!launch_all(c, nullptr) || hipDeviceSynchronize() != hipSuccess) { printf("warm frame failed\n"); return 3; }
    hipMemset(c->d + PAD + c->off_dst, 0, c->dst_size);
  }
  for (size_t i = 0; i < c->steps.size(); i++) {
    Step& s = c->steps[i]; auto it = ex.find(c->names[i]);
    if (it == ex.end()) { c->dbg_n = 0; if (!launch_range(c, nullptr, i, i + 1) || hipDeviceSynchronize() != hipSuccess) { printf("step %zu failed\n", i); return 3; } continue; }
    NrOverride& o = it->second; Step n = s; std::string mp = rel(c->dir, o.mod);
    if (!mods.count(mp)) { hipModule_t m; if (load_mod(c, mp, &m) != hipSuccess) { printf("load %s\n", mp.c_str()); return 1; } mods[mp] = m; }
    if (hipModuleGetFunction(&n.fn, mods[mp], o.sym.c_str()) != hipSuccess) { printf("no symbol %s\n", o.sym.c_str()); return 1; }
    n.gx = nr_grid(o.g[0], s.gx); n.gy = nr_grid(o.g[1], s.gy); n.gz = nr_grid(o.g[2], s.gz); n.thr = nr_grid(o.g[3], s.thr);
    if (!nr_apply_override(o, s.ka, n.ka)) { printf("recipe past kernarg\n"); return 1; }
    hipMemcpy(bk, ar, c->arena, hipMemcpyDeviceToDevice);
    const float ta = run(s, 5); hipMemcpy(A.data(), ar, c->arena, hipMemcpyDeviceToHost);
    const float tb = run(n, 5); hipMemcpy(B.data(), ar, c->arena, hipMemcpyDeviceToHost);
    size_t nd = 0, lo = SIZE_MAX, hi = 0, nw = 0, wlo = SIZE_MAX, whi = 0; I.resize(c->arena); hipMemcpy(I.data(), bk, c->arena, hipMemcpyDeviceToHost);
    for (size_t k = 0; k < c->arena; k++) {
      if (A[k] != B[k]) { nd++; lo = std::min(lo, k); hi = k; }
      if (A[k] != I[k]) { nw++; wlo = std::min(wlo, k); whi = k; }
    }
    printf("step %3zu %-40s diff %zu B [%zu..%zu] | A wrote %zu B [%zu..%zu] | ms A %.4f B %.4f\n", i, c->names[i].c_str(), nd, nd ? lo : 0, hi, nw, nw ? wlo : 0, whi, ta, tb);
    if (argc > 4 && nw) {
      size_t a = wlo, b = std::min(whi + 1, wlo + ((size_t)256 << 20));
      auto dump = [&](const char* t, const std::vector<uint8_t>& v) { FILE* f = fopen((std::string(argv[4]) + "/s" + std::to_string(i) + "_" + t + ".bin").c_str(), "wb"); fwrite(v.data() + a, 1, b - a, f); fclose(f); };
      dump("in", I); dump("a", A); dump("b", B); printf("  dumped [%zu..%zu)\n", a, b);
    }
    if (argc > 4 && getenv("KCMP_SAVE") && strstr(getenv("KCMP_SAVE"), ("," + std::to_string(i) + ",").c_str())) {   // whole arena before the step (inputs for offline analysis)
      FILE* f = fopen((std::string(argv[4]) + "/arena_s" + std::to_string(i) + ".bin").c_str(), "wb"); fwrite(I.data(), 1, I.size(), f); fclose(f); }
    hipMemcpy(ar, A.data(), c->arena, hipMemcpyHostToDevice);
  }
  printf("done\n"); return 0;
}
