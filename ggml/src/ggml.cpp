#include "ggml-impl.h"

#include <cstdio>

#include <cstdlib>
#include <exception>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    try { if (std::current_exception()) { throw; } } catch (const std::exception & e) { fprintf(stderr, "uncaught exception: %s\n", e.what()); } catch (...) { fprintf(stderr, "uncaught exception of unknown type\n"); }   // fork: Windows prints nothing otherwise
#if defined(_WIN32)
    {   // fork: a bad_alloc on Windows is mostly the commit limit (RAM + pagefile), not free RAM: the numbers beside the stack
        MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            fprintf(stderr, "memory: commit %.1f of %.1f GiB in use (%.1f GiB left), RAM %.1f of %.1f GiB free\n", (ms.ullTotalPageFile - ms.ullAvailPageFile) / 1073741824.0,
                    ms.ullTotalPageFile / 1073741824.0, ms.ullAvailPageFile / 1073741824.0, ms.ullAvailPhys / 1073741824.0, ms.ullTotalPhys / 1073741824.0);
        }
    }
#endif
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();
