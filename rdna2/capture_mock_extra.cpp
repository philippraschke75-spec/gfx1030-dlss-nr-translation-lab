#include <windows.h>
extern "C" {
__declspec(dllexport) int hipEventCreateWithFlags(void** p, unsigned f) {
    if (!p || f!=0x1234) return 998; *p=reinterpret_cast<void*>(0x12345678); return 17;
}
__declspec(dllexport) int hipEventQuery(void* p) {
    return p==reinterpret_cast<void*>(0x12345678) ? 600 : 998;
}
__declspec(dllexport) int hipStreamCreateWithFlags(void** p, unsigned f) {
    if (!p || f!=0x5678) return 998; *p=reinterpret_cast<void*>(0x23456789); return 19;
}
__declspec(dllexport) int hipStreamSynchronize(void* p) {
    return p==reinterpret_cast<void*>(0x23456789) ? 23 : 998;
}
}
