// Unit test of the state file routing (src/llama-moestate-route-fork.h): the heat of a model (hot.<layer> keys) lives in a file of its own that every
// machine shares, everything else the engine learns (placement, tuned values, run records, probes) stays in the file of this hardware.
// No GPU, no model.
//
// build: cmake --build <dir> --target test-moestate-route-fork;  run: ctest -R test-moestate-route-fork
#include "../src/llama-moestate-route-fork.h"

#include <cstdio>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "test-moestate-route-fork: FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)

int main() {
    CHECK(moe_state_is_heat_key("hot.0"));
    CHECK(moe_state_is_heat_key("hot.46"));
    CHECK(!moe_state_is_heat_key("hotmap"));
    CHECK(!moe_state_is_heat_key("tuned.l2"));
    CHECK(!moe_state_is_heat_key("place.g2.a1.v3"));
    CHECK(!moe_state_is_heat_key("run.hit_pct"));
    CHECK(!moe_state_is_heat_key(""));

    // one write of mixed keys is split in two, order kept inside each part
    const moe_state_kv in = { { "hot.0", "1 2" }, { "run.knobs", "a" }, { "hot.1", "3" }, { "geom", "g" } };
    moe_state_kv heat, rest;
    moe_state_split(in, heat, rest);
    CHECK(heat.size() == 2 && heat[0].first == "hot.0" && heat[1].first == "hot.1");
    CHECK(rest.size() == 2 && rest[0].first == "run.knobs" && rest[1].first == "geom");

    // one section read from the two files: the hardware file's hot.* keys (stale, from before the split) lose to the heat file's
    const moe_state_kv hw    = { { "geom", "g" }, { "hot.0", "old" }, { "run.knobs", "a" } };
    const moe_state_kv heatf = { { "hot.0", "new" }, { "hot.1", "3" } };
    const moe_state_kv m = moe_state_merge(hw, heatf);
    CHECK(m.size() == 4);
    int n_hot0 = 0;
    for (const auto & p : m) { if (p.first == "hot.0") { n_hot0++; CHECK(p.second == "new"); } }
    CHECK(n_hot0 == 1);

    // both files empty / one side empty
    CHECK(moe_state_merge({}, {}).empty());
    CHECK(moe_state_merge(hw, {}).size() == 3);   // no heat file yet: the heat the hardware file still holds is read
    printf("test-moestate-route-fork: ok\n");
    return 0;
}
