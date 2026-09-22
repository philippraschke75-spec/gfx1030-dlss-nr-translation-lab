#include "capture_exception_trace.h"
#include <cassert>
#include <fstream>
#include <string>
static bool breakpoint_received=false;
static LONG CALLBACK test_handler(EXCEPTION_POINTERS* info) {
    if(info->ExceptionRecord->ExceptionCode==EXCEPTION_BREAKPOINT) {
        breakpoint_received=true;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
int main() {
    DeleteFileW(L"exception-observer-test.log");
    assert(capture_exception_trace::install(L"exception-observer-test.log"));
    bool caught=false;
    try { throw 42; } catch(int value) { caught=value==42; }
    assert(caught);
    // Exhaust ordinary logging, then verify a breakpoint is still recorded
    // and delivered to a later handler. Only this test consumes the exception.
    for(int i=0;i<20;++i) { try { throw i; } catch(int) {} }
    auto handler=AddVectoredExceptionHandler(0,test_handler);
    assert(handler);
    RaiseException(EXCEPTION_BREAKPOINT,0,0,nullptr);
    assert(breakpoint_received);
    assert(RemoveVectoredExceptionHandler(handler));
    std::ifstream file("exception-observer-test.log");
    std::string text((std::istreambuf_iterator<char>(file)),{});
    assert(text.find("code=0xe06d7363")!=std::string::npos);
    assert(text.find("frame address=")!=std::string::npos);
    assert(text.find("module=")!=std::string::npos);
    assert(text.find("code=0x80000003")!=std::string::npos);
    assert(text.find("utc=")!=std::string::npos);
    std::puts("PASS C++ and breakpoint logging; normal exception disposition preserved");
}
