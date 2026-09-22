// Sequential multi-dispatch runner for a whole network forward pass.
//
// var_gpu_test.cpp dispatches exactly one kernel, which is right for a difftest but cannot run the
// 71-block schedule. This runs a script of dispatches against one shared arena, so activations flow
// from block to block through the arena exactly as the real host's buffers do.
//
// Block-to-block ordering uses plain host-side synchronisation. That is not a simplification: the
// vendor's k_flag_set/k_flag_wait kernels are referenced nowhere in the network driver or either
// per-block launcher, so they are not part of the forward pass (see VARPARAMS_HOST_CONTRACT.md).
//
// usage: net_run manifest.txt kernargs.bin arena.bin arena_base_hex
//
// manifest.txt: one dispatch per line, '#' comments and blank lines ignored:
//     <module.co>|<symbol>|<ka_offset>|<ka_size>|<gx>|<gy>|<threads>
// ('|' separated because module paths contain spaces.)
// kernargs.bin holds every dispatch's kernarg block back to back; ka_offset/ka_size select one.
// Every 8-byte aligned qword of a kernarg lying inside [arena_base, arena_base+size) is rebased to
// the device arena, the same rule var_gpu_test uses. arena.bin is overwritten with the final
// contents and the 64 KiB sentinel guards around the arena are checked at the end.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){std::printf("%s: %s\n",#x,hipGetErrorString(e)); return 2;} } while(0)

static std::vector<uint8_t> rd(const char* p){
  FILE*f=std::fopen(p,"rb"); if(!f){std::printf("cannot open %s\n",p);std::exit(9);}
  std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET);
  std::vector<uint8_t> v(n); if(n&&std::fread(v.data(),1,n,f)!=(size_t)n)std::exit(9);
  std::fclose(f); return v;
}
struct Step { std::string mod, sym; size_t off, len; unsigned gx, gy, thr; };

int main(int argc,char**argv){
  if(argc<5){ std::printf("usage: net_run manifest kernargs.bin arena.bin arena_base_hex\n"); return 1; }
  auto kab=rd(argv[2]); auto arena=rd(argv[3]);
  uint64_t base=std::strtoull(argv[4],nullptr,16);

  std::vector<Step> steps;
  { FILE* f=std::fopen(argv[1],"r"); if(!f){ std::printf("cannot open %s\n",argv[1]); return 9; }
    char line[1024];
    while(std::fgets(line,sizeof line,f)){
      if(line[0]=='#'||line[0]=='\n'||line[0]=='\r') continue;
      // '|' separated, not whitespace: module paths routinely contain spaces.
      std::string s(line);
      while(!s.empty() && (s.back()=='\n'||s.back()=='\r')) s.pop_back();
      std::vector<std::string> f; size_t pos=0;
      for(;;){ size_t q=s.find('|',pos); if(q==std::string::npos){ f.push_back(s.substr(pos)); break; }
               f.push_back(s.substr(pos,q-pos)); pos=q+1; }
      if(f.size()!=7){ std::printf("bad manifest line (%zu fields): %s\n",f.size(),line); return 1; }
      steps.push_back({f[0],f[1],(size_t)std::strtoull(f[2].c_str(),nullptr,10),
                       (size_t)std::strtoull(f[3].c_str(),nullptr,10),
                       (unsigned)atoi(f[4].c_str()),(unsigned)atoi(f[5].c_str()),(unsigned)atoi(f[6].c_str())});
    }
    std::fclose(f);
  }
  std::printf("net_run: %zu dispatches, arena %zu bytes\n",steps.size(),arena.size());

  hipDeviceProp_t p{}; CHECK(hipGetDeviceProperties(&p,0));
  if(std::strcmp(p.gcnArchName,"gfx1030")){ std::printf("not gfx1030: %s\n",p.gcnArchName); return 3; }

  const size_t PAD=1<<16; uint8_t* d; CHECK(hipMalloc(&d,arena.size()+2*PAD));
  std::vector<uint8_t> host(arena.size()+2*PAD,0xA5);
  std::memcpy(host.data()+PAD,arena.data(),arena.size());
  CHECK(hipMemcpy(d,host.data(),host.size(),hipMemcpyHostToDevice));
  uint64_t dev=(uint64_t)(d+PAD);

  std::map<std::string,hipFunction_t> fns;           // modules are reused across many blocks
  std::map<std::string,hipModule_t> mods;
  double tmo=getenv("SWIN_TIMEOUT")?atof(getenv("SWIN_TIMEOUT")):60;
  auto t_all=std::chrono::steady_clock::now();
  double total_ms=0;

  for(size_t i=0;i<steps.size();i++){
    const Step& s=steps[i];
    std::string key=s.mod+"|"+s.sym;
    if(!fns.count(key)){
      if(!mods.count(s.mod)){ hipModule_t m; CHECK(hipModuleLoad(&m,s.mod.c_str())); mods[s.mod]=m; }
      hipFunction_t f; CHECK(hipModuleGetFunction(&f,mods[s.mod],s.sym.c_str())); fns[key]=f;
    }
    if(s.off+s.len>kab.size()){ std::printf("step %zu kernarg out of range\n",i); return 1; }
    std::vector<uint8_t> ka(kab.begin()+s.off, kab.begin()+s.off+s.len);
    for(size_t o=0;o+8<=ka.size();o+=8){
      uint64_t v; std::memcpy(&v,ka.data()+o,8);
      if(v>=base && v<base+arena.size()){ v=dev+(v-base); std::memcpy(ka.data()+o,&v,8); }
    }
    size_t sz=ka.size();
    void* cfg[]={HIP_LAUNCH_PARAM_BUFFER_POINTER,ka.data(),HIP_LAUNCH_PARAM_BUFFER_SIZE,&sz,HIP_LAUNCH_PARAM_END};
    hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1); hipEventRecord(e0);
    CHECK(hipModuleLaunchKernel(fns[key],s.gx,s.gy,1,s.thr,1,1,0,nullptr,nullptr,cfg));
    hipEventRecord(e1);
    auto t0=std::chrono::steady_clock::now();
    while(hipEventQuery(e1)==hipErrorNotReady){
      if(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()>tmo){
        std::printf("TIMEOUT at step %zu (%s) after %.0f s\n",i,s.sym.c_str(),tmo);
        std::fflush(stdout); std::_Exit(5); }
    }
    hipError_t e=hipEventQuery(e1);
    if(e!=hipSuccess){ std::printf("step %zu (%s) failed: %s\n",i,s.sym.c_str(),hipGetErrorString(e)); return 6; }
    float ms; hipEventElapsedTime(&ms,e0,e1); total_ms+=ms;
    hipEventDestroy(e0); hipEventDestroy(e1);
    if(getenv("NET_RUN_VERBOSE")) std::printf("  [%3zu] %-44s %.3f ms\n",i,s.sym.c_str(),ms);
  }

  CHECK(hipMemcpy(host.data(),d,host.size(),hipMemcpyDeviceToHost));
  bool guard=true;
  for(size_t i=0;i<PAD;i++) guard&=host[i]==0xA5;
  for(size_t i=PAD+arena.size();i<host.size();i++) guard&=host[i]==0xA5;
  FILE*fo=std::fopen(argv[3],"wb"); std::fwrite(host.data()+PAD,1,arena.size(),fo); std::fclose(fo);
  double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-t_all).count();
  std::printf("net_run OK: %zu dispatches, %.3f ms GPU, %.2f s wall, arena_dev=0x%llx, guards %s\n",
              steps.size(),total_ms,wall,(unsigned long long)dev,guard?"intact":"CORRUPT");
  return guard?0:7;
}
