// The planner's checked mutexes (src/llama-lockrank-fork.h). The order is learned from the locks the process takes: a lock taken while another is held
// records "held before taken"; a lock that closes a cycle of these (B under A once, A under B now, also through others and on other threads) is a
// deadlock two threads can hit, and logs one "moe-cache: INV lock order" line, which fails the row (run.py ENGINE_CHECK). No order is declared by hand:
// the hand-declared ranks of 2026-11-14 had g_dsf_mu and g_pp_mu the wrong way round and failed every row of their first run.
// A second lock of a mutex the thread holds aborts at once with its name (a std::mutex hangs there for good: build lp_not_under_xsplit, the server
// hung at start for 600 s with g_pp_mu taken again on one thread).
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

static bool told(const char * a, const char * b) {
    return g_log.find("moe-cache: INV lock order") != std::string::npos && g_log.find(a) != std::string::npos && g_log.find(b) != std::string::npos;
}

// one pair of mutexes per case: the learned order is process-wide, cases must not share edges
static llama_checked_mutex a1("a1_mu"), b1("b1_mu");
static llama_checked_mutex a2("a2_mu"), b2("b2_mu");
static llama_checked_mutex a3("a3_mu"), b3("b3_mu"), c3("c3_mu");
static llama_checked_mutex a4("a4_mu"), b4("b4_mu");
static llama_checked_mutex a5("a5_mu"), b5("b5_mu");
static llama_checked_mutex r6("r6_mu");

int main() {
    ggml_log_set(log_cb, nullptr);

    // one order, many times, in any declaration order: quiet (b1 is declared after a1 and taken first: no order is implied by names or declarations)
    for (int i = 0; i < 3; ++i) {
        std::lock_guard<llama_checked_mutex> x(b1);
        std::lock_guard<llama_checked_mutex> y(a1);
    }
    { std::lock_guard<llama_checked_mutex> y(a1); }   // alone: no order
    CHECK(g_log.empty(), "one order logged: [%s]", g_log.c_str());

    // the reverse of a seen order on the same thread: one INV line with both names, the lock is still taken
    g_log.clear();
    { std::lock_guard<llama_checked_mutex> x(a2); std::lock_guard<llama_checked_mutex> y(b2); }
    { std::lock_guard<llama_checked_mutex> y(b2); std::lock_guard<llama_checked_mutex> x(a2); }
    CHECK(told("a2_mu", "b2_mu"), "reverse order not logged with both names: [%s]", g_log.c_str());

    // a cycle through a third mutex: a3 then b3, b3 then c3, c3 then a3
    g_log.clear();
    { std::lock_guard<llama_checked_mutex> x(a3); std::lock_guard<llama_checked_mutex> y(b3); }
    { std::lock_guard<llama_checked_mutex> x(b3); std::lock_guard<llama_checked_mutex> y(c3); }
    CHECK(g_log.empty(), "a chain without a cycle logged: [%s]", g_log.c_str());
    { std::lock_guard<llama_checked_mutex> x(c3); std::lock_guard<llama_checked_mutex> y(a3); }
    CHECK(told("c3_mu", "a3_mu"), "a 3-mutex cycle not logged: [%s]", g_log.c_str());

    // the two orders on two threads, one after the other (the deadlock needs them at the same time; the check must not)
    g_log.clear();
    std::thread([] { std::lock_guard<llama_checked_mutex> x(a4); std::lock_guard<llama_checked_mutex> y(b4); }).join();
    std::thread([] { std::lock_guard<llama_checked_mutex> y(b4); std::lock_guard<llama_checked_mutex> x(a4); }).join();
    CHECK(told("a4_mu", "b4_mu"), "reverse order on another thread not logged: [%s]", g_log.c_str());

    // held on one thread, taken on another: no order between them (each thread holds its own set)
    g_log.clear();
    {
        std::lock_guard<llama_checked_mutex> x(b5);
        std::thread([] { std::lock_guard<llama_checked_mutex> y(a5); }).join();
    }
    { std::lock_guard<llama_checked_mutex> y(a5); std::lock_guard<llama_checked_mutex> x(b5); }
    CHECK(g_log.empty(), "a lock on another thread counted as an order: [%s]", g_log.c_str());
    ggml_log_set(nullptr, nullptr);

    // a second lock on the thread that holds it: the process aborts (in a child), it does not hang
    fflush(stdout); fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        alarm(5);   // a hang ends the child by SIGALRM, which the parent reads as a failure
        std::lock_guard<llama_checked_mutex> a(r6);
        std::lock_guard<llama_checked_mutex> b(r6);
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
