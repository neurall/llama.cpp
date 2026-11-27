#pragma once
// fork: GGUF files written by another tool for an architecture we have: their arch name, keys and tensor names mapped to ours at load, in memory (the file
// is never rewritten). User 2026-11-27: ERNIE 5.5 Flash as converted for Strata (Lowkey-Loki/Ernie-5.5-Flash-Preview-Custom-IQ4_XS-GGUF, arch "ernie55")
// against upstream PR 31925's converter ("ernie5_5"). Same model: the differences are names, where the MLA sizes are kept, the MLA marker in compress_ratios
// (-2 there, 0 here) and the VHA head mix kept as its two factors U, V (ours: the product I + V U^T; src/models/ernie5-5.cpp runs the factors as they are).
#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// our name of a file's architecture, nullptr when the file uses ours
static inline const char * gguf_compat_arch_fork(const std::string & arch) {
    return arch == "ernie55" ? "ernie5_5" : nullptr;
}

// our name of one of its tensors
static inline std::string gguf_compat_tensor_fork(const std::string & arch, const std::string & name) {
    if (arch != "ernie55") { return name; }
    static const struct { const char * from, * to; } ren[] = {
        { ".ffn_exp_scale.weight",   ".exp_scale.weight"       },
        { ".ffn_latent_norm.weight", ".ffn_routed_norm.weight" },
        { ".attn_vha_u.weight",      ".attn_vha.u"             },   // the factors under ATTN_VHA with their own suffix (no new tensor enum)
        { ".attn_vha_v.weight",      ".attn_vha.v"             },
    };
    for (const auto & r : ren) {
        const size_t n = strlen(r.from);
        if (name.size() > n && name.compare(name.size() - n, n, r.from) == 0) { return name.substr(0, name.size() - n) + r.to; }
    }
    return name;
}

static inline int64_t gguf_compat_int(const gguf_context * g, const std::string & key, int64_t dflt) {
    const int64_t i = gguf_find_key(g, key.c_str());
    if (i < 0) { return dflt; }
    switch (gguf_get_kv_type(g, i)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, i);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, i);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(g, i);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(g, i);
        default:               return dflt;
    }
}

// every "<from>." key copied to "<to>.", then the keys whose meaning differs set as ours reads them; general.architecture becomes ours
static inline void gguf_compat_kv_fork(gguf_context * g, const std::string & from, const char * to) {
    const std::string pf = from + ".";
    const int64_t n_kv = gguf_get_n_kv(g);   // the copies go to the end: only the file's own keys are walked
    for (int64_t i = 0; i < n_kv; ++i) {
        const std::string key = gguf_get_key(g, i);
        if (key.compare(0, pf.size(), pf) != 0) { continue; }
        const std::string nk = std::string(to) + "." + key.substr(pf.size());
        switch (gguf_get_kv_type(g, i)) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (g, nk.c_str(), gguf_get_val_u8  (g, i)); break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (g, nk.c_str(), gguf_get_val_i8  (g, i)); break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (g, nk.c_str(), gguf_get_val_u16 (g, i)); break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (g, nk.c_str(), gguf_get_val_i16 (g, i)); break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (g, nk.c_str(), gguf_get_val_u32 (g, i)); break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (g, nk.c_str(), gguf_get_val_i32 (g, i)); break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (g, nk.c_str(), gguf_get_val_u64 (g, i)); break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (g, nk.c_str(), gguf_get_val_i64 (g, i)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (g, nk.c_str(), gguf_get_val_f32 (g, i)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (g, nk.c_str(), gguf_get_val_f64 (g, i)); break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(g, nk.c_str(), gguf_get_val_bool(g, i)); break;
            case GGUF_TYPE_STRING:  { const std::string s = gguf_get_val_str(g, i); gguf_set_val_str(g, nk.c_str(), s.c_str()); break; }
            case GGUF_TYPE_ARRAY: {
                const enum gguf_type t = gguf_get_arr_type(g, i);
                const size_t n = gguf_get_arr_n(g, i);
                if (t == GGUF_TYPE_STRING) {
                    std::vector<std::string> s(n);
                    std::vector<const char *> p(n);
                    for (size_t j = 0; j < n; ++j) { s[j] = gguf_get_arr_str(g, i, j); p[j] = s[j].c_str(); }
                    gguf_set_arr_str(g, nk.c_str(), p.data(), n);
                } else {
                    const char * d = (const char *) gguf_get_arr_data(g, i);
                    std::vector<char> v(d, d + n * gguf_type_size(t));   // a copy: setting a key may move the array it came from
                    gguf_set_arr_data(g, nk.c_str(), t, v.data(), n);
                }
                break;
            }
            default: break;
        }
    }
    gguf_set_val_str(g, "general.architecture", to);
    if (from == "ernie55") {
        const std::string o = std::string(to) + ".attention.";
        const int64_t kv_lora = gguf_compat_int(g, pf + "attention.mla.kv_lora_rank", 512);
        const int64_t rope    = gguf_compat_int(g, pf + "attention.mla.rope_dimension_count", 64);
        const int64_t nope    = gguf_compat_int(g, pf + "attention.mla.key_length_nope", 192);
        const int64_t v_mla   = gguf_compat_int(g, pf + "attention.mla.value_length", 256);
        const int64_t head    = gguf_compat_int(g, pf + "attention.key_length", 512);   // theirs: the sliding-window layers' head size
        gguf_set_val_u32(g, (o + "kv_lora_rank").c_str(),       (uint32_t) kv_lora);
        gguf_set_val_u32(g, (o + "key_length_mla").c_str(),     (uint32_t) (nope + rope));
        gguf_set_val_u32(g, (o + "value_length_mla").c_str(),   (uint32_t) v_mla);
        gguf_set_val_u32(g, (o + "key_length_swa").c_str(),     (uint32_t) head);
        gguf_set_val_u32(g, (o + "value_length_swa").c_str(),   (uint32_t) gguf_compat_int(g, pf + "attention.value_length", head));
        gguf_set_val_u32(g, (o + "key_length").c_str(),         (uint32_t) (kv_lora + rope));   // ours: the full layers' cache row [kv latent | rope]
        gguf_set_val_u32(g, (o + "value_length").c_str(),       (uint32_t) kv_lora);
        const int64_t ci = gguf_find_key(g, (o + "compress_ratios").c_str());
        if (ci >= 0 && gguf_get_kv_type(g, ci) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, ci) == GGUF_TYPE_INT32) {
            const int32_t * d = (const int32_t *) gguf_get_arr_data(g, ci);
            std::vector<int32_t> r(d, d + gguf_get_arr_n(g, ci));
            for (auto & x : r) { x = x < 0 ? 0 : x; }   // theirs marks the MLA layers -2, ours 0
            gguf_set_arr_data(g, (o + "compress_ratios").c_str(), GGUF_TYPE_INT32, r.data(), r.size());
        }
    }
}
