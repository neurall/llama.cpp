// The planner's ranked mutexes (src/llama-lockrank-fork.h): a second lock of a mutex the thread holds aborts at once with its name (a std::mutex hangs there for
// good: build lp_not_under_xsplit, 2026-11-14, the server hung at start for 600 s with g_pp_mu taken again on one thread); a lock below a held rank logs one
// "moe-cache: INV lock order" line, which fails the row (run.py ENGINE_CHECK).
// build: cmake --build <dir> --target test-lockrank-fork;  run: ctest -R test-lockrank-fork   (no GPU, no model)
#include "ggml.h"
#include "../src/llama-lockrank-fork.h"

#include <cstdio>
#include <mutex>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

static std::string g_log;
static void log_cb(ggml_log_level, const char * text, void *) { g_log += text; }

static int g_fails = 0;
#define CHECK(c, ...) do { if (!(c)) { g_fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static llama_ranked_mutex outer(10, "outer_mu"), inner(20, "inner_mu");

int main() {
    ggml_log_set(log_cb, nullptr);
    {   // rising ranks: quiet
        std::lock_guard<llama_ranked_mutex> a(outer);
        std::lock_guard<llama_ranked_mutex> b(inner);
    }
    CHECK(g_log.empty(), "rising ranks logged: [%s]", g_log.c_str());
    {   // the same mutex again after its release: quiet
        { std::lock_guard<llama_ranked_mutex> a(inner); }
        { std::lock_guard<llama_ranked_mutex> a(inner); }
    }
    CHECK(g_log.empty(), "a lock after its release logged: [%s]", g_log.c_str());
    {   // falling ranks: one INV line with both names, then the locks work
        std::lock_guard<llama_ranked_mutex> b(inner);
        std::lock_guard<llama_ranked_mutex> a(outer);
    }
    CHECK(g_log.find("moe-cache: INV lock order") != std::string::npos && g_log.find("outer_mu") != std::string::npos && g_log.find("inner_mu") != std::string::npos,
          "falling ranks not logged with both names: [%s]", g_log.c_str());
    {   // another thread holds its own set: no false report across threads
        g_log.clear();
        std::lock_guard<llama_ranked_mutex> b(inner);
        std::thread t([] { std::lock_guard<llama_ranked_mutex> a(outer); });
        t.join();
    }
    CHECK(g_log.empty(), "a lock on another thread reported: [%s]", g_log.c_str());
    ggml_log_set(nullptr, nullptr);

    // a second lock on the thread that holds it: the process aborts (in a child), it does not hang
    fflush(stdout); fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        alarm(5);   // a hang ends the child by SIGALRM, which the parent reads as a failure
        std::lock_guard<llama_ranked_mutex> a(inner);
        std::lock_guard<llama_ranked_mutex> b(inner);
        _exit(0);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT, "a relock did not abort (status %d)", st);

    if (g_fails) {
        fprintf(stderr, "test-lockrank-fork: %d failed\n", g_fails);
        return 1;
    }
    printf("test-lockrank-fork: ok\n");
    return 0;
}
