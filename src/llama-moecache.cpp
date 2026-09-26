#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <cctype>
#include <map>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending
    std::vector<uint32_t> expert_count;   // expert id -> uses, halved every LLAMA_MOE_CACHE_HALVE_EVERY steps
    std::vector<uint32_t> win_count;      // expert id -> uses in the last 64 tokens
    std::deque<std::vector<int32_t>> recent; // ids of those tokens, oldest first
    // where up/gate/down live on disk (fd < 0: unknown, upload from host memory)
    int    src_fd[3]   = { -1, -1, -1 };
    size_t src_offs[3] = { 0, 0, 0 };

    std::vector<uint64_t> glob_count;     // expert id -> lifetime uses
    uint64_t              glob_max = 1;
    std::vector<bool>     sticky;         // expert id -> never evicted (most-used by lifetime count)
    std::vector<uint32_t> prompt_count;   // expert id -> uses in the last GPU-offloaded prompt batch(es)

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    done = false;
};

struct moe_cache {
    int32_t n_slots     = 0;
    int32_t max_inserts = 2;
    int32_t window      = 64; // recent-usage window in tokens (--moe-cache-window)

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;
    std::map<const ggml_tensor *, std::pair<size_t, int>> by_src; // host weight -> (layer, 0 up / 1 gate / 2 down)

    // big-batch prefill: experts served device-to-device from the cache vs over PCIe
    uint64_t n_src_hit   = 0;
    uint64_t n_src_query = 0;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::vector<std::thread> workers;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
    int                      in_flight = 0; // jobs popped by a worker, not yet in done
    std::condition_variable  dcv;           // signalled when a job lands in done

    // swap cost accounting (guarded by wmtx for the upload side)
    double   upload_us    = 0;  // EMA of one expert upload (up+gate+down)
    uint64_t n_uploads    = 0;
    int64_t  last_step_us = 0;
    double   step_us_busy = 0;  // EMA of token time while uploads were in flight
    double   step_us_idle = 0;  // EMA of token time with no uploads in flight
    size_t   rr           = 0;  // round-robin start layer for the swap budget
    int      last_budget  = 0;
    int      last_margin  = 0;

    // pinned staging per upload worker, for file-backed uploads (pread -> pinned -> DMA)
    std::vector<ggml_backend_buffer_t> staging;

    // hot-set profile: per-expert lifetime uses saved at shutdown, preloaded at the next start
    std::string profile;

    bool prompt_burst = false; // a GPU-offloaded prompt batch ran: upload its hottest experts at the next step
};

// Where a tensor's raw GGUF bytes live (recorded by the model loader).
bool file_location(const llama_model & model, const ggml_tensor * t, int * fd, size_t * off) {
    auto it = model.exps_file_locs.find(t);
    if (it == model.exps_file_locs.end()) {
        return false;
    }
    *fd  = it->second.fd;
    *off = it->second.offs;
    return true;
}

bool pread_full(int fd, void * dst, size_t n, size_t off) {
#ifdef _WIN32
    // no file locations are recorded on Windows; uploads use host memory
    GGML_UNUSED(fd); GGML_UNUSED(dst); GGML_UNUSED(n); GGML_UNUSED(off);
    return false;
#else
    uint8_t * d = (uint8_t *) dst;
    while (n > 0) {
        const ssize_t r = pread(fd, d, n, (off_t) off);
        if (r <= 0) {
            return false;
        }
        d += r; n -= (size_t) r; off += (size_t) r;
    }
    return true;
#endif
}


enum class policy { halve, window, hybrid, add };

policy get_policy() {
    static const policy p = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_POLICY");
        if (e && strcmp(e, "window") == 0) return policy::window;
        if (e && strcmp(e, "hybrid") == 0) return policy::hybrid;
        if (e && strcmp(e, "halve")  == 0) return policy::halve;
        return policy::add;
    }();
    return p;
}

// eviction score, in "uses" units so the pay-back margin applies to all policies
double score(const layer_state & ls, int32_t id) {
    switch (get_policy()) {
        case policy::window: return ls.win_count[id];
        case policy::hybrid: return ls.win_count[id] * ((double) ls.glob_count[id] / (double) ls.glob_max);
        case policy::add: {
            static const double k = [] {
                const char * w = getenv("LLAMA_MOE_CACHE_GLOBAL_WEIGHT");
                return w ? atof(w) : 16.0;
            }();
            return ls.win_count[id] + k * ((double) ls.glob_count[id] / (double) ls.glob_max);
        }
        case policy::halve:  return ls.expert_count[id];
    }
    return 0;
}

// Hot-set profile: which experts this model used, so the next start preloads them into VRAM
// instead of warming the cache over the first requests. Keyed by model name + file size.
// LLAMA_MOE_CACHE_PROFILE=<path> overrides the location, =0 disables.
std::string profile_path(const llama_model & model) {
    const char * e = getenv("LLAMA_MOE_CACHE_PROFILE");
    if (e) {
        return strcmp(e, "0") == 0 ? "" : e;
    }
#ifdef _WIN32
    const char * base = getenv("LOCALAPPDATA");
    const std::string dir = base ? std::string(base) + "\\llama.cpp" : "";
#else
    const char * xdg  = getenv("XDG_CACHE_HOME");
    const char * home = getenv("HOME");
    const std::string dir = xdg ? std::string(xdg) + "/llama.cpp" : home ? std::string(home) + "/.cache/llama.cpp" : "";
#endif
    if (dir.empty()) {
        return "";
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    std::string name = model.name;
    for (char & c : name) {
        if (!isalnum((unsigned char) c) && c != '-' && c != '.') {
            c = '_';
        }
    }
    return dir + "/moe-hot-" + name + "-" + std::to_string(model.size()) + ".bin";
}

constexpr uint32_t PROFILE_MAGIC = 0x484f454d; // "MOEH"

void profile_save(const moe_cache * mc) {
    if (mc->profile.empty() || mc->n_steps < 64) { // too little use to be worth keeping
        return;
    }
    FILE * f = fopen(mc->profile.c_str(), "wb");
    if (!f) {
        return;
    }
    const uint32_t hdr[2] = { PROFILE_MAGIC, (uint32_t) mc->layers.size() };
    fwrite(hdr, sizeof(hdr), 1, f);
    for (const auto & ls : mc->layers) {
        const uint32_t n = (uint32_t) ls.glob_count.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(ls.glob_count.data(), sizeof(uint64_t), n, f);
    }
    fclose(f);
}

// seed usage from the profile and queue the most-used experts of each layer for upload;
// the first step() waits for them and publishes them. Returns the number queued.
// Always-hot experts (formatting, common tokens) stay in VRAM: the most-used experts by lifetime
// count, up to LLAMA_MOE_CACHE_STICKY (default 0.3) of each layer's slots, are never evicted.
void refresh_sticky(layer_state & ls) {
    static const double frac = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_STICKY");
        return e ? atof(e) : 0.3;
    }();
    ls.sticky.assign(ls.glob_count.size(), false);
    const int32_t k = (int32_t) (frac * ls.pub.n_slots);
    if (k <= 0) {
        return;
    }
    std::vector<int32_t> ids;
    for (int32_t e = 0; e < (int32_t) ls.glob_count.size(); ++e) {
        if (ls.glob_count[e] > 0) {
            ids.push_back(e);
        }
    }
    const int32_t n = std::min<int32_t>(k, (int32_t) ids.size());
    std::partial_sort(ids.begin(), ids.begin() + n, ids.end(), [&](int32_t a, int32_t b) { return ls.glob_count[a] > ls.glob_count[b]; });
    for (int32_t i = 0; i < n; ++i) {
        ls.sticky[ids[i]] = true;
    }
}

int parse_layer_from_name(const char * name);

size_t profile_preload(moe_cache * mc, const llama_model & model) {
    std::vector<std::vector<uint64_t>> counts;
    FILE * f = mc->profile.empty() ? nullptr : fopen(mc->profile.c_str(), "rb");
    const char * from = mc->profile.c_str();
    bool ok = f != nullptr;
    if (f) {
        uint32_t hdr[2] = {};
        ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == PROFILE_MAGIC && hdr[1] == mc->layers.size();
        for (size_t il = 0; ok && il < mc->layers.size(); ++il) {
            uint32_t n = 0;
            ok = fread(&n, sizeof(n), 1, f) == 1 && n == mc->layers[il].glob_count.size();
            if (ok) {
                counts.emplace_back(n);
                ok = fread(counts.back().data(), sizeof(uint64_t), n, f) == n;
            }
        }
        fclose(f);
        if (!ok) {
            LLAMA_LOG_WARN("moe-cache: ignoring profile %s (other model or cache layout)\n", from);
        }
    }
    if (!ok && !model.moe_expert_usage.empty()) {
        // no local profile: the model's own usage table (GGUF key moe_cache.expert_usage)
        const size_t n_expert = mc->layers.empty() ? 0 : mc->layers[0].glob_count.size();
        counts.clear();
        ok = n_expert > 0;
        for (size_t il = 0; ok && il < mc->layers.size(); ++il) {
            const size_t layer = (size_t) parse_layer_from_name(mc->layers[il].pub.up_src->name);
            ok = mc->layers[il].glob_count.size() == n_expert && (layer + 1)*n_expert <= model.moe_expert_usage.size();
            if (ok) {
                const uint32_t * u = model.moe_expert_usage.data() + layer*n_expert;
                counts.emplace_back(u, u + n_expert);
            }
        }
        from = "GGUF moe_cache.expert_usage";
    }
    if (!ok) {
        return 0;
    }
    LLAMA_LOG_INFO("moe-cache: usage profile from %s\n", from);
    size_t queued = 0;
    std::lock_guard<std::mutex> lk(mc->wmtx);
    for (size_t il = 0; il < mc->layers.size(); ++il) {
        auto & ls = mc->layers[il];
        ls.glob_count = counts[il];
        ls.glob_max   = std::max<uint64_t>(1, *std::max_element(ls.glob_count.begin(), ls.glob_count.end()));
        refresh_sticky(ls);
        std::vector<int32_t> ids;
        for (int32_t e = 0; e < (int32_t) ls.glob_count.size(); ++e) {
            if (ls.glob_count[e] > 0) {
                ids.push_back(e);
            }
        }
        std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return ls.glob_count[a] > ls.glob_count[b]; });
        for (int32_t s = 0; s < std::min<int32_t>(ls.pub.n_slots, (int32_t) ids.size()); ++s) {
            ls.slot_in_flight[s] = true;
            mc->todo.push_back({ il, ids[s], s });
            queued++;
        }
    }
    mc->wcv.notify_all();
    return queued;
}

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    // big-batch prefill ids are observed too: they warm the cache before decode
    // (hits aren't counted there, the cache graph isn't built for big batches)
    const bool prefill = n_tokens > llama_moe_cache_max_batch();

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        if ((int32_t) ls->recent.size() >= mc->window) {
            for (int32_t old : ls->recent.front()) {
                ls->win_count[old]--;
            }
            ls->recent.pop_front();
        }
        ls->recent.emplace_back();
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            ls->expert_count[id]++;
            ls->win_count[id]++;
            ls->recent.back().push_back(id);
            ls->glob_max = std::max(ls->glob_max, ++ls->glob_count[id]);
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit += !prefill;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss += !prefill;
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

} // namespace

// Big-batch prefill offloads expert matmuls and copies the used experts to the
// GPU; experts published in this cache on that GPU are copied from their slot
// (device-to-device) instead. Called from the scheduler on the decode thread;
// published slots are stable until the next step(), which runs after a sync.
static bool moe_src_cb(const ggml_tensor * weight, int32_t expert, ggml_backend_dev_t dev,
                const void ** data, ggml_backend_buffer_t * buffer, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    auto it = mc->by_src.find(weight);
    if (it == mc->by_src.end()) {
        return false;
    }
    layer_state & ls = mc->layers[it->second.first];
    std::lock_guard<std::mutex> lock(mc->mtx);
    // GPU-offloaded prompt batches never reach the CPU observer (moe_obs_cb): record the prompt's
    // experts here, once per expert and batch (on up), so they warm the cache before decode
    if (it->second.second == 0 && expert >= 0 && expert < (int32_t) ls.expert_slot.size()) {
        ls.prompt_count[expert]++;
        mc->prompt_burst = true;
        ls.expert_count[expert]++;
        ls.glob_max = std::max(ls.glob_max, ++ls.glob_count[expert]);
        if (ls.expert_slot[expert] < 0 && std::find(ls.pending.begin(), ls.pending.end(), expert) == ls.pending.end()) {
            ls.pending.push_back(expert);
        }
    }
    ggml_tensor * c = it->second.second == 0 ? ls.pub.up_c : it->second.second == 1 ? ls.pub.gate_c : ls.pub.down_c;
    if (!c || !c->buffer || c->nb[2] != weight->nb[2] ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(c->buffer)) != dev) {
        return false;
    }
    mc->n_src_query++;
    if (expert < 0 || expert >= (int32_t) ls.expert_slot.size() || ls.expert_slot[expert] < 0) {
        return false;
    }
    mc->n_src_hit++;
    *data   = (const uint8_t *) c->data + (size_t) ls.expert_slot[expert]*c->nb[2];
    *buffer = c->buffer;
    return true;
}

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, int32_t prefetch_slots, int32_t window) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        if (n_slots == 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }
        mc->window = std::max(1, window);

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer) {
                continue;
            }
            {
                // host or CPU-repacked (non-host but CPU device) experts are cacheable
                ggml_backend_dev_t d = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.ffn_up_exps->buffer));
                const bool on_cpu = ggml_backend_buffer_is_host(l.ffn_up_exps->buffer) ||
                                    (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!on_cpu) {
                    continue; // experts already on a device: nothing to cache
                }
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        // n_slots < 0: fill each device's free VRAM (called after KV/compute buffers
        // exist), leaving a margin for the compute graph growing by the cache chain.
        // ponytail: one slot count per device, uniform over that device's layers.
        auto slots_for = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands) -> int32_t {
            if (n_slots > 0) {
                return n_slots;
            }
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            if (!dev) {
                return 0;
            }
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            const char * m = getenv("LLAMA_MOE_CACHE_MARGIN_MB");
            // default 384 MiB: measured peak growth after load is ~170 MiB per GPU (GLM, chat + 12k prefill)
            size_t margin = (size_t) (m ? atoll(m) : 384) * 1024 * 1024;
            // --prefetch-experts-slots: the scheduler lazily allocates N full expert
            // tensors on the device big batches are offloaded to
            if (prefetch_slots >= 2) {
                if (model.dev_offload < model.devices.size() && dev == model.devices[model.dev_offload].dev) {
                    size_t max_tensor = 0;
                    for (const auto & c : all) {
                        max_tensor = std::max({max_tensor, ggml_nbytes(c.l->ffn_up_exps), ggml_nbytes(c.l->ffn_gate_exps), ggml_nbytes(c.l->ffn_down_exps)});
                    }
                    margin += (size_t) std::min(prefetch_slots, 4) * max_tensor;
                }
            }
            size_t per_slot = 0;
            int64_t n_expert = 0;
            for (const auto & c : cands) {
                per_slot += c.l->ffn_up_exps->nb[2] + c.l->ffn_gate_exps->nb[2] + c.l->ffn_down_exps->nb[2];
                n_expert = c.l->ffn_up_exps->ne[2];
            }
            if (free <= margin + per_slot) {
                return 0;
            }
            // one extra (dummy) slot per layer is allocated on top
            const int64_t fit = (int64_t) ((free - margin) / per_slot) - 1;
            return (int32_t) std::max<int64_t>(0, std::min<int64_t>(fit, n_expert - 1));
        };

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            const int32_t slots = tables_only ? 0 : slots_for(buft, cands);
            if (!tables_only && slots <= 0) {
                LLAMA_LOG_WARN("%s: no room for MoE cache slots on %s\n", __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ls->pub.n_slots = slots;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], slots + 1);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], slots + 1);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], slots + 1);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            if (!tables_only) {
                // the scheduler places ops by the residency of weight buffers only;
                // without this the cache chain follows its neighbours onto the CPU
                ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            const int32_t ns = ls.pub.n_slots;
            ls.slot_expert.assign(ns, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(ns, 0);
            ls.slot_in_flight.assign(ns, false);
            ls.expert_count.assign(n_expert, 0);
            ls.win_count.assign(n_expert, 0);
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            for (int k = 0; k < 3; ++k) {
                if (!file_location(model, srcs[k], &ls.src_fd[k], &ls.src_offs[k])) {
                    ls.src_fd[k] = -1;
                }
            }
            ls.glob_count.assign(n_expert, 0);
            ls.sticky.assign(n_expert, false);
            ls.prompt_count.assign(n_expert, 0);

            std::vector<int32_t> dummy(n_expert, ns);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            mc->by_src[ls.pub.up_src]   = { (size_t) (&ls - mc->layers.data()), 0 };
            mc->by_src[ls.pub.gate_src] = { (size_t) (&ls - mc->layers.data()), 1 };
            mc->by_src[ls.pub.down_src] = { (size_t) (&ls - mc->layers.data()), 2 };
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        {
            size_t max_expert = 0;
            bool   any_file   = false;
            bool   any_repack = false; // host memory isn't the file layout: must upload from the file
            for (auto & ls : mc->layers) {
                max_expert = std::max(max_expert, ls.pub.up_src->nb[2] + ls.pub.gate_src->nb[2] + ls.pub.down_src->nb[2]);
                any_file   |= ls.src_fd[0] >= 0 && ls.src_fd[1] >= 0 && ls.src_fd[2] >= 0;
                any_repack |= !ggml_backend_buffer_is_host(ls.pub.up_src->buffer);
            }
            ggml_backend_dev_t gpu = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(mc->bufs.back()));
            // default: copy from the mmap'd host memory (measured faster); pread path is opt-in
            const char * pr = getenv("LLAMA_MOE_CACHE_PREAD");
            const bool use_pread = (pr && pr[0] == '1') || any_repack;
            ggml_backend_buffer_type_t hbuft = (use_pread && any_file && gpu) ? ggml_backend_dev_host_buffer_type(gpu) : nullptr;
            const char * nt = getenv("LLAMA_MOE_CACHE_UPLOAD_THREADS");
            const int n_workers = std::max(1, nt ? atoi(nt) : 1);
            for (int w = 0; w < n_workers; ++w) {
                ggml_backend_buffer_t b = hbuft ? ggml_backend_buft_alloc_buffer(hbuft, max_expert) : nullptr;
                mc->staging.push_back(b);
            }
            // weights loaded without mmap (--load-mode pin) sit in the GPU's pinned host buffer: direct DMA
            const bool src_pinned = gpu && ggml_backend_buffer_get_type(mc->layers[0].pub.up_src->buffer) == ggml_backend_dev_host_buffer_type(gpu);
            LLAMA_LOG_WARN("moe-cache: %d upload workers, %s\n", n_workers,
                    mc->staging[0] ? "pread -> pinned staging -> GPU" :
                    src_pinned     ? "from pinned host memory (direct DMA)" :
                                     "from pageable host memory (mmap; --load-mode pin: faster uploads)");
        }

        for (size_t w = 0; w < mc->staging.size(); ++w)
        mc->workers.emplace_back([mc, w]() {
            uint8_t * staging_ptr = mc->staging[w] ? (uint8_t *) ggml_backend_buffer_get_base(mc->staging[w]) : nullptr;
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                    mc->in_flight++;
                }
                auto & ls = mc->layers[j.layer_idx];
                const int64_t t0 = ggml_time_us();
                ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
                const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
                for (int k = 0; k < 3; ++k) {
                    const size_t sz = srcs[k]->nb[2];
                    if (staging_ptr && ls.src_fd[k] >= 0 &&
                            pread_full(ls.src_fd[k], staging_ptr, sz, ls.src_offs[k] + (size_t) j.expert*sz)) {
                        ggml_backend_tensor_set(dsts[k], staging_ptr, (size_t) j.slot*dsts[k]->nb[2], sz);
                    } else if (ggml_backend_buffer_is_host(srcs[k]->buffer)) {
                        upload_slice(dsts[k], srcs[k], j.expert, j.slot);
                    } else {
                        LLAMA_LOG_ERROR("moe-cache: can't upload repacked '%s' without its file location\n", srcs[k]->name);
                    }
                }
                const double dt = (double) (ggml_time_us() - t0);
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    mc->upload_us = mc->n_uploads++ ? 0.9*mc->upload_us + 0.1*dt : dt;
                    j.done = true;
                    mc->done.push_back(j);
                    mc->in_flight--;
                }
                mc->dcv.notify_all();
            }
        });

        mc->profile = profile_path(model);
        if (const size_t n = profile_preload(mc, model)) {
            LLAMA_LOG_INFO("moe-cache: preloading %zu experts (usage profile)\n", n);
        }

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        {
            const char * e = getenv("LLAMA_MOE_CACHE_PREFILL_D2D");
            if (!e || e[0] != '0') {
                ggml_backend_set_moe_src_callback(moe_src_cb, mc);
            }
        }
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_WARN("%s: MoE expert cache enabled: %zu layers, slots/layer %d..%d, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(),
                std::min_element(mc->layers.begin(), mc->layers.end(), [](auto & a, auto & b) { return a.pub.n_slots < b.pub.n_slots; })->pub.n_slots,
                std::max_element(mc->layers.begin(), mc->layers.end(), [](auto & a, auto & b) { return a.pub.n_slots < b.pub.n_slots; })->pub.n_slots,
                mc->max_inserts, vram/1024.0/1024.0);
    }();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step(int64_t n_tokens) {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    // LLAMA_MOE_CACHE_DETERMINISTIC=1 (benchmarks): publish exactly the uploads
    // scheduled by the previous step, in a fixed order, with a fixed swap budget
    // and margin, so a run no longer depends on upload timing and repeats its
    // output. Uploads still overlap the token's compute; the wait is only for
    // the ones not finished by then.
    static const bool det = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_DETERMINISTIC");
        return e && atoi(e) != 0;
    }();
    // wait for and publish the previous step's uploads (default; LLAMA_MOE_CACHE_WAIT=0
    // disables): a cache that is current every step beats the time the wait costs,
    // fixed-text decode +16%, hit rate 66% -> 77%
    static const bool wait_publish = det || [] {
        const char * e = getenv("LLAMA_MOE_CACHE_WAIT");
        return !e || atoi(e) != 0;
    }();

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::unique_lock<std::mutex> wlk(mc->wmtx);
        if (wait_publish) {
            mc->dcv.wait(wlk, [mc]() { return mc->todo.empty() && mc->in_flight == 0; });
            std::sort(mc->done.begin(), mc->done.end(), [](const upload_job & a, const upload_job & b) {
                return a.layer_idx != b.layer_idx ? a.layer_idx < b.layer_idx : a.slot < b.slot;
            });
        }
        std::lock_guard<std::mutex> lk(mc->mtx);
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            ls.slot_in_flight[j.slot]  = false;
            set_table_entry(ls.pub, j.expert, j.slot);
        }
        mc->done.clear();
    }

    // swap budget for this step, from measured costs: keep upload time within
    // LLAMA_MOE_CACHE_SWAP_FRAC of the token time, minus what is still queued
    static const double frac = [] {
        const char * f = getenv("LLAMA_MOE_CACHE_SWAP_FRAC");
        return f ? atof(f) : 0.25;
    }();
    int budget_total;
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        const int64_t now = ggml_time_us();
        if (mc->last_step_us) {
            const double dt = (double) (now - mc->last_step_us);
            if (dt < 5e6) { // ignore idle gaps between requests
                double & ema = mc->todo.empty() ? mc->step_us_idle : mc->step_us_busy;
                ema = ema ? 0.9*ema + 0.1*dt : dt;
            }
        }
        mc->last_step_us = now;
        const double step_us = std::max(mc->step_us_idle, mc->step_us_busy);
        if (mc->upload_us <= 0 || step_us <= 0) {
            budget_total = (int) mc->layers.size(); // bootstrap: ~1 per layer until measured
        } else {
            budget_total = (int) (frac*step_us*mc->workers.size()/mc->upload_us) - (int) mc->todo.size();
        }
        // static budget: LLAMA_MOE_CACHE_BUDGET swaps per step (deterministic mode: default 8)
        static const int fixed_budget = [] {
            const char * e = getenv("LLAMA_MOE_CACHE_BUDGET");
            return e ? atoi(e) : -1;
        }();
        if (fixed_budget >= 0 || det) {
            budget_total = fixed_budget >= 0 ? fixed_budget : 8;
        }
        budget_total = std::max(0, budget_total);
        mc->last_budget = budget_total;
    }

    // a swap must pay for itself: over the count's horizon the candidate saves
    // (count_c - count_v) CPU expert evals of t_cpu each, and costs one upload
    static const double cpu_gbs = [] {
        const char * g = getenv("LLAMA_MOE_CACHE_CPU_GBS");
        return g ? atof(g) : 50.0;
    }();
    uint32_t margin = 1;
    if (mc->upload_us > 0 && !mc->layers.empty()) {
        const auto & l0 = mc->layers[0].pub;
        const double bytes  = (double) (l0.up_src->nb[2] + l0.gate_src->nb[2] + l0.down_src->nb[2]);
        const double cpu_us = bytes / (cpu_gbs * 1e3); // GB/s -> bytes per us
        margin = (uint32_t) std::max(1.0, std::ceil(mc->upload_us / cpu_us));
    }
    // static pay-back margin: LLAMA_MOE_CACHE_MARGIN (deterministic mode: default 4)
    static const int fixed_margin = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_MARGIN");
        return e ? atoi(e) : -1;
    }();
    if (fixed_margin >= 0 || det) {
        margin = (uint32_t) (fixed_margin >= 0 ? fixed_margin : 4);
    }
    mc->last_margin = (int) margin;

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;
    if (mc->n_steps % 4096 == 0) { // follow the workload: re-pick the always-hot set from lifetime counts
        for (auto & ls : mc->layers) {
            refresh_sticky(ls);
        }
    }

    // 1b) after a long prompt: upload the prompt's most-used uncached experts in one burst (up to
    //     LLAMA_MOE_CACHE_BURST of each layer's slots, default 0.25), so decode starts with them cached
    //     instead of gaining ~2 per layer per step; the next step waits for them once
    // only once the prompt is done (a decode-size batch), not between the prompt's own chunks
    if (mc->prompt_burst && n_tokens <= llama_moe_cache_max_batch()) {
        static const double burst = [] {
            const char * e = getenv("LLAMA_MOE_CACHE_BURST");
            return e ? atof(e) : 0.25;
        }();
        size_t n_burst = 0;
        for (size_t li = 0; li < mc->layers.size(); ++li) {
            auto & ls = mc->layers[li];
            std::vector<int32_t> ids;
            for (int32_t e = 0; e < (int32_t) ls.prompt_count.size(); ++e) {
                if (ls.prompt_count[e] > 0 && ls.expert_slot[e] < 0) {
                    ids.push_back(e);
                }
            }
            std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return ls.prompt_count[a] > ls.prompt_count[b]; });
            const int32_t cap = std::min<int32_t>((int32_t) ids.size(), (int32_t) (burst * ls.pub.n_slots));
            for (int32_t i = 0; i < cap; ++i) {
                // victim: empty slot, else the non-sticky cached expert the prompt used least (a long
                // prompt touches nearly every expert), and only if the prompt used it less than this one
                int32_t slot = -1;
                uint32_t least = UINT32_MAX;
                for (int32_t sl = 0; sl < ls.pub.n_slots; ++sl) {
                    if (ls.slot_in_flight[sl]) {
                        continue;
                    }
                    const int32_t v = ls.slot_expert[sl];
                    if (v < 0) { slot = sl; least = 0; break; }
                    if (ls.sticky[v]) {
                        continue;
                    }
                    if (ls.prompt_count[v] < least) { least = ls.prompt_count[v]; slot = sl; }
                }
                if (slot < 0 || least >= ls.prompt_count[ids[i]]) {
                    break; // candidates are sorted: nothing left the prompt used more than what's cached
                }
                const int32_t victim = ls.slot_expert[slot];
                if (victim >= 0) {
                    ls.expert_slot[victim] = -1;
                    ls.slot_expert[slot]   = -1;
                    set_table_entry(ls.pub, victim, ls.pub.n_slots);
                }
                ls.slot_in_flight[slot] = true;
                std::lock_guard<std::mutex> wlk(mc->wmtx);
                mc->todo.push_back({li, ids[i], slot});
                n_burst++;
            }
            std::fill(ls.prompt_count.begin(), ls.prompt_count.end(), 0);
        }
        mc->prompt_burst = false;
        if (n_burst) {
            mc->wcv.notify_all();
            static const bool stats = getenv("LLAMA_MOE_CACHE_STATS") != nullptr;
            if (stats) {
                LLAMA_LOG_INFO("moe-cache: prompt burst: %zu experts\n", n_burst);
            }
        }
    }

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    const size_t n_layers = mc->layers.size();
    mc->rr = (mc->rr + 1) % n_layers;
    for (size_t k = 0; k < n_layers; ++k) {
        const size_t li = (mc->rr + k) % n_layers;
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        // hottest candidates first; the budget only throttles evictions,
        // filling empty slots is free
        std::sort(ls.pending.begin(), ls.pending.end(), [&](int32_t a, int32_t b) {
            return score(ls, a) > score(ls, b);
        });
        // swaps cost PCIe bandwidth and latency: cap evictions per layer per step,
        // and none at all while earlier uploads are still queued
        int & budget = budget_total;
        for (auto it = ls.pending.begin(); it != ls.pending.end(); ++it) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || score(ls, id) <= 0) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the least-called cached expert
            int32_t slot = -1;
            double best = 1e300;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.sticky[ls.slot_expert[s]]) {
                    continue;
                }
                const double c = score(ls, ls.slot_expert[s]);
                if (c < best) { best = c; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                if (score(ls, id) < score(ls, victim) + margin || budget-- <= 0) {
                    break; // candidates are sorted: nothing hotter than what's cached
                }
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, ls.pub.n_slots);
            }
            ls.slot_in_flight[slot] = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
        }
        ls.pending.clear();
    }
    mc->wcv.notify_all();

    // decay: counts halve every N steps, so recent use dominates old use
    static const uint64_t halve_every = [] {
        const char * h = getenv("LLAMA_MOE_CACHE_HALVE_EVERY");
        return (uint64_t) std::max(1, h ? atoi(h) : 64);
    }();
    if (mc->n_steps % halve_every == 0) {
        for (auto & ls : mc->layers) {
            for (auto & c : ls.expert_count) { c >>= 1; }
        }
    }

    static const bool stats = getenv("LLAMA_MOE_CACHE_STATS") != nullptr;
    if (stats && mc->n_steps % 64 == 0) {
        static uint64_t ph = 0, pm = 0;
        uint64_t h = 0, m = 0, filled = 0, inflight = 0, total = 0;
        for (auto & ls : mc->layers) {
            h += ls.n_hit; m += ls.n_miss; total += ls.pub.n_slots;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                filled   += ls.slot_expert[s] >= 0;
                inflight += ls.slot_in_flight[s];
            }
        }
        size_t queued;
        {
            std::lock_guard<std::mutex> wlk(mc->wmtx);
            queued = mc->todo.size();
        }
        const uint64_t dh = h - ph, dm = m - pm;
        ph = h; pm = m;
        LLAMA_LOG_WARN("moe-cache: step %" PRIu64 " filled %" PRIu64 "/%" PRIu64 " inflight %" PRIu64 " queued %zu window-hit %.1f%% | upload %.2f ms/expert, token %.1f ms idle / %.1f ms busy, swap budget %d, min count gain %d\n",
                mc->n_steps, filled, total, inflight, queued, dh + dm ? 100.0*dh/(dh + dm) : 0.0,
                mc->upload_us/1e3, mc->step_us_idle/1e3, mc->step_us_busy/1e3, mc->last_budget, mc->last_margin);
    }

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

bool llama_moe_cache_active() {
    return g_cache != nullptr;
}

int64_t llama_moe_cache_max_batch() {
    static const int64_t v = [] {
        const char * e = getenv("LLAMA_MOE_CACHE_MAX_BATCH");
        return (int64_t) (e ? atoi(e) : 31);
    }();
    return v;
}

void llama_moe_cache_free() {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }
    ggml_set_moe_obs_callback(nullptr, nullptr);
    ggml_backend_set_moe_src_callback(nullptr, nullptr);
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_all();
    for (auto & t : mc->workers) {
        t.join();
    }
    profile_save(mc);
    uint64_t h = 0, m = 0;
    for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
    LLAMA_LOG_WARN("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%, prefill experts from cache %" PRIu64 "/%" PRIu64 "\n",
            mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0, mc->n_src_hit, mc->n_src_query);
    for (auto * b : mc->staging) { if (b) { ggml_backend_buffer_free(b); } }
    for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
    for (auto * c : mc->ctxs) { ggml_free(c); }
    delete mc;
    g_cache     = nullptr;
    g_init_done = false;
}
