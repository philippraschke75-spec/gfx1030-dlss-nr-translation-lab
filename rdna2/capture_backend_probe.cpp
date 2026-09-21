// Isolated enumeration only. Does not load the research runtime or launch kernels.
#include <windows.h>
#include <cstdio>
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    auto h=LoadLibraryA(argv[1]);
    if(!h){std::printf("load failed %lu\n",GetLastError());return 3;}
    auto count=reinterpret_cast<int(*)(int*)>(GetProcAddress(h,"hipGetDeviceCount"));
    if(!count)return 4;
    int n=-1;
    std::puts("ENTER hipGetDeviceCount");std::fflush(stdout);
    int rc=count(&n);
    std::printf("RETURN hipGetDeviceCount rc=%d count=%d\n",rc,n);std::fflush(stdout);
    return rc==0?0:5;
}
