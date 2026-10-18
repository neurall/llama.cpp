// KV tier ladder decisions (qkv-ram-fallback), pure functions so tests/test-kvtier-fork.cpp can run them with no GPU and no model; llama-context.cpp (kv_fit_select) and
// llama-moecache-fork.cpp (llama_moe_cache_kv_ram_fork) call these.
//   tiers: 0 f16, 1 q8_0/q8_0, 2 q4_0/q4_0 (all converted in place in the pool), 3 RAM (whole layers' KV in host memory)
//   --moe kv-fallback=k8v8,k4v4,ram  the tiers allowed after f16 (default all three; "none": f16 only)
#pragma once
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// the tiers a kv-fallback list allows (after f16), ascending. Words are matched without case or blanks; a word that is none of them is left out of the tiers and, with `unknown`,
// reported (a typo like "RAM" or "q8" otherwise reads as "no fallback" and the refusal later prints the list as typed, which looks consistent with it); an empty list too.
static inline std::vector<int> kvx_allowed(const std::string & list, std::vector<std::string> * unknown = nullptr) {
    std::vector<int> allowed;
    if (list == "none") { return allowed; }
    if (list.empty() && unknown) { unknown->push_back("(empty list)"); }
    size_t i = 0;
    while (i <= list.size() && !list.empty()) {
        size_t j = list.find(',', i); if (j == std::string::npos) { j = list.size(); }
        std::string e = list.substr(i, j - i);
        e.erase(std::remove_if(e.begin(), e.end(), [](unsigned char c) { return c == ' ' || c == '\t'; }), e.end());
        for (char & c : e) { c = (char) tolower((unsigned char) c); }
        if (e == "k8v8") { allowed.push_back(1); } else if (e == "k4v4") { allowed.push_back(2); } else if (e == "ram") { allowed.push_back(3); }
        else if (unknown) { unknown->push_back(list.substr(i, j - i)); }
        i = j + 1;
    }
    std::sort(allowed.begin(), allowed.end());   // tried from the cheapest tier up whatever the order of the list
    allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
    return allowed;
}

// The tier a request of n_tokens needs: the lowest tier at or above `lo` that holds it (f16 first, then the allowed tiers in list order); -1 when none does.
// lo: min_tier when given (>= 0), else the tier above the cache's own (a request whose KV filled up goes higher, whatever the estimate said); the RAM tier repeats
// (a request that outgrows the layers moved so far moves more of them), so lo is at most 3. Caps: tokens each tier holds (llama_kv_fit_caps).
template <class C> static int kvx_select(const C & cp, const std::vector<int> & allowed, int kv_tier, int min_tier, size_t n_tokens) {
    const int lo = min_tier < 0 ? std::min(kv_tier + 1, 3) : min_tier;
    if (lo <= 0 && n_tokens <= cp.f16) { return 0; }
    for (int t : allowed) {
        if (t < lo) { continue; }
        if (n_tokens <= (t == 1 ? cp.k8v8 : t == 2 ? cp.k4v4 : cp.ram)) { return t; }
    }
    return -1;
}

// the RAM tier takes the pool's KV to q4_0 first when the list allows q4_0 and the cache is below it (the layers moved to host memory hold q4_0)
static inline bool kvx_ram_needs_q4(const std::vector<int> & allowed, int kv_tier) {
    return kv_tier < 2 && std::find(allowed.begin(), allowed.end(), 2) != allowed.end();
}

// cells a prompt's KV can grow to behind a lend0 and a fixed part F with combined row R while the lend grows by pc bytes per cell
static inline size_t kvx_cells(size_t tail_end, size_t lend0, size_t F, size_t R, size_t pc) {
    const size_t fl = lend0 + F;
    return R && tail_end > fl ? (tail_end - fl) / (R + pc) : 0;
}

// The host buffer of the RAM tier (allocated once): cells it holds for a budget of bytes when one cell of every layer takes r_all bytes and each of n_seg arrays is padded to
// 256 bytes (slack), at most cmax. 0 when the budget does not even cover the slack.
static inline size_t kvx_host_cells(size_t budget, size_t n_seg, size_t r_all, size_t cmax) {
    const size_t slack = (n_seg + 1) * 256;
    return r_all && budget > slack ? std::min(cmax, (budget - slack) / r_all) : 0;
}
// its size, and the place of each layer array in it: every array holds host_cells rows of its stride, padded to 256 bytes; returns the bytes used by all of them
static inline size_t kvx_host_size(size_t host_cells, size_t r_all, size_t n_seg) { return host_cells * r_all + (n_seg + 1) * 256; }
static inline size_t kvx_host_used(size_t host_cells, const std::vector<size_t> & strides) {
    size_t used = 0;
    for (size_t st : strides) { used += (host_cells * st + 255) / 256 * 256; }
    return used;
}
