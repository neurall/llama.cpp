// The decode bell (ggml/src/ggml-bell-fork.h): how a card and the host meet once per MoE layer without a driver call.
// The card writes a layer's picks and the block each one has on the card into the layer's bell and counts its seq up; the host computes the picks that have no
// block and raises the flag; the card takes those rows. Here a thread plays the card.
// From the nsys trace of 2026-11-18 (Qwen3.8 IQ3_S, build b15843): a decode layer was two graph launches and two stream waits, 5.1 ms of a 19.2 ms token with no
// card computing; and about 3% of the picks are missing, so most layers have nothing for the host: such a layer must not wait for it.
#include "../ggml/src/ggml-bell-fork.h"

#undef NDEBUG
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static const int N_EMBD = 64, N_USED = 10, N_EXPERT = 128, N_LAYER = 48;

static float row_value(int layer, int id, float x0) { return 1000.0f * layer + id + x0; }   // what the host "computes" for a missing pick

int main() {
    std::vector<std::vector<char>> mem(N_LAYER, std::vector<char>(ggml_bell_size(N_USED, N_EMBD)));
    std::vector<ggml_bell *> bells;
    for (auto & m : mem) { bells.push_back(ggml_bell_init(m.data(), N_USED, N_EMBD)); }
    assert(ggml_bell_size(N_USED, N_EMBD) >= sizeof(ggml_bell) + sizeof(float) * (N_EMBD + N_USED * N_EMBD));

    {   // one layer by hand: picks 5 and 9 are not on the card, pick -1 is nobody's
        ggml_bell * b = bells[0];
        std::vector<int32_t> table(N_EXPERT, -1); table[3] = 7; table[4] = 0;
        const int32_t ids[N_USED] = { 3, 5, 4, 9, -1, 3, 3, 3, 3, 3 };
        std::vector<float> x(N_EMBD, 0.5f);
        int32_t slots[N_USED];
        const int n_miss = ggml_bell_publish(b, ids, table.data(), x.data(), slots);
        assert(n_miss == 2 && slots[0] == 7 && slots[1] == -1 && slots[2] == 0 && slots[3] == -1 && slots[4] == -1);
        assert(b->seq == 1 && ggml_bell_x(b)[N_EMBD - 1] == 0.5f);
        int32_t todo[N_USED];
        assert(ggml_bell_host_misses(b, todo) == 2);
        assert(todo[0] == -1 && todo[1] == 5 && todo[2] == -1 && todo[3] == 9 && todo[4] == -1);   // exactly the picks the card called missing
        assert(!ggml_bell_rows_ready(b));
        ggml_bell_rows(b)[1 * N_EMBD] = 11.0f; ggml_bell_rows(b)[3 * N_EMBD] = 33.0f;
        ggml_bell_host_done(b, 1);
        assert(ggml_bell_rows_ready(b));
        std::vector<float> down(N_USED * N_EMBD, -2.0f);
        assert(ggml_bell_rows_take(b, down.data()) == 2);
        assert(down[0] == -2.0f && down[1 * N_EMBD] == 11.0f && down[2 * N_EMBD] == -2.0f && down[3 * N_EMBD] == 33.0f && down[4 * N_EMBD] == -2.0f);   // the card's own rows stay
    }
    for (auto & m : mem) { memset(m.data(), 0, m.size()); }
    for (size_t l = 0; l < bells.size(); ++l) { bells[l] = ggml_bell_init(mem[l].data(), N_USED, N_EMBD); }

    // many tokens: the card thread and the host thread, 3% of the picks missing
    const int n_tokens = 400;
    std::atomic<long> card_miss{0}, host_miss{0}, bad{0}, waits{0};
    std::thread host([&] {
        std::vector<uint32_t> seen(N_LAYER, 0);
        for (int t = 0; t < n_tokens; ++t) {
            for (int l = 0; l < N_LAYER; ++l) {
                ggml_bell * b = bells[l];
                assert(ggml_bell_host_wait(b, seen[l], 10 * 1000 * 1000));
                seen[l] = b->seq;
                int32_t todo[N_USED];
                const int n = ggml_bell_host_misses(b, todo);
                host_miss += n;
                if (n == 0) { continue; }   // nothing to compute: the card does not wait, the host does not answer
                for (int i = 0; i < N_USED; ++i) {
                    if (todo[i] >= 0) { for (int j = 0; j < N_EMBD; ++j) { ggml_bell_rows(b)[i * N_EMBD + j] = row_value(l, todo[i], ggml_bell_x(b)[0]); } }
                }
                ggml_bell_host_done(b, seen[l]);
            }
        }
    });
    std::thread card([&] {
        std::mt19937 rng(7);
        std::vector<int32_t> table(N_EXPERT);
        std::vector<float> x(N_EMBD), down(N_USED * N_EMBD);
        for (int t = 0; t < n_tokens; ++t) {
            for (int l = 0; l < N_LAYER; ++l) {
                for (int e = 0; e < N_EXPERT; ++e) { table[e] = rng() % 100 < 3 ? -1 : e; }
                int32_t ids[N_USED], slots[N_USED];
                for (int i = 0; i < N_USED; ++i) { ids[i] = (int32_t) (rng() % N_EXPERT); }
                x[0] = (float) t;
                const int n_miss = ggml_bell_publish(bells[l], ids, table.data(), x.data(), slots);
                card_miss += n_miss;
                for (auto & v : down) { v = -2.0f; }
                waits += ggml_bell_rows_take(bells[l], down.data()) > 0;
                for (int i = 0; i < N_USED; ++i) {
                    const float want = slots[i] < 0 ? row_value(l, ids[i], (float) t) : -2.0f;
                    for (int j = 0; j < N_EMBD; ++j) { bad += down[i * N_EMBD + j] != want; }
                }
            }
        }
    });
    card.join(); host.join();
    printf("%d tokens x %d layers: %ld picks missing on the card, %ld computed by the host, %ld layers waited for the host, %ld wrong values\n",
           n_tokens, N_LAYER, card_miss.load(), host_miss.load(), waits.load(), bad.load());
    assert(card_miss == host_miss && card_miss > 0 && bad == 0);
    assert(waits < (long) n_tokens * N_LAYER);

    {   // a token with every pick on the card while the host sleeps: the card runs through all layers without it
        std::vector<int32_t> table(N_EXPERT); for (int e = 0; e < N_EXPERT; ++e) { table[e] = e; }
        std::vector<float> x(N_EMBD, 1.0f), down(N_USED * N_EMBD, -2.0f);
        const auto t0 = std::chrono::steady_clock::now();
        for (int l = 0; l < N_LAYER; ++l) {
            int32_t ids[N_USED], slots[N_USED];
            for (int i = 0; i < N_USED; ++i) { ids[i] = i; }
            assert(ggml_bell_publish(bells[l], ids, table.data(), x.data(), slots) == 0);
            assert(ggml_bell_rows_take(bells[l], down.data()) == 0);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        printf("a token with every pick on the card, nobody serving: %.3f ms\n", ms);
        assert(ms < 50.0);
    }
    {   // a card that never rings: the host gives up after its time and says so
        uint32_t seen = bells[0]->seq;
        assert(!ggml_bell_host_wait(bells[0], seen, 20 * 1000));
    }
    printf("ok\n");
    return 0;
}
