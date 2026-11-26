#pragma once
// fork: one card or the layer split for a model that fits one card (common.cpp, common_moe_cache_auto_impl), tests/test-estimate-fork.cpp "place start"

#include "llama-estimate-fork.h"

#include <string>

// This start's estimate decides (place_estimate, from what this start reads: the weights the fitter leaves in RAM for the split, the RAM read and
// upload probes, the split's passes); a tie keeps the stock layer split. Nothing earlier starts measured is read. Until 2026-11-26 per-request records
// in the state file ([place <model>] .one / .split) decided once both arms had one (place_one_card_pick): Inkling 2 Mini's were written at a 100 W
// night power cap (runs.db 34395, 34403: 12.6 and 12.8 o/s), the split won on them by 0.7-5% and the release ran it at 41 o/s, one card runs 69.
struct place_start { bool one; std::string label, why; };
static inline place_start place_start_pick(const place_est_in & in, double n_p, double n_g) {
    const place_est e = place_estimate(in, n_p);
    const bool one = place_estimate_pick(e, n_p, n_g);
    return { one, std::string(one ? "one card" : "layer split") + " (estimated)", place_estimate_why(e, n_p, n_g) };
}
