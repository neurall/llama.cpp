        [](common_params & params, int value) { params.diffusion.steps = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-visual"},
        string_format("enable visual diffusion mode (show progressive generation) (default: %s)", params.diffusion.visual_mode ? "true" : "false"),
        [](common_params & params) { params.diffusion.visual_mode = true; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eps"}, "F",
        string_format("epsilon for timesteps (default: %.6f)", (double) params.diffusion.eps),
        [](common_params & params, const std::string & value) { params.diffusion.eps = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-algorithm"}, "N",
        string_format(
            "diffusion algorithm: 0=DIFFUSION_ALGORITHM_ORIGIN, 1=DIFFUSION_ALGORITHM_ENTROPY_BASED, "
            "2=DIFFUSION_ALGORITHM_MARGIN_BASED, 3=DIFFUSION_ALGORITHM_RANDOM, "
            "4=DIFFUSION_ALGORITHM_CONFIDENCE_BASED (default: %d)", params.diffusion.algorithm),
        [](common_params & params, int value) { params.diffusion.algorithm = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-alg-temp"}, "F",
        string_format("dream algorithm temperature (default: %.3f)", (double) params.diffusion.alg_temp),
        [](common_params & params, const std::string & value) { params.diffusion.alg_temp = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-block-length"}, "N",
        string_format("llada block length for generation (default: %d)", params.diffusion.block_length),
        [](common_params & params, int value) { params.diffusion.block_length = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-cfg-scale"}, "F",
        string_format("llada classifier-free guidance scale (default: %.3f)", (double) params.diffusion.cfg_scale),
        [](common_params & params, const std::string & value) { params.diffusion.cfg_scale = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-add-gumbel-noise"}, "F",
        string_format("add gumbel noise to the logits if temp > 0.0 (default: %s)", params.diffusion.add_gumbel_noise ? "true" : "false"),
        [](common_params & params, const std::string & value) { params.diffusion.add_gumbel_noise = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        { "-lr", "--learning-rate" }, "ALPHA",
        string_format("adamw or sgd optimizer alpha (default: %.2g); note: sgd alpha recommended ~10x (no momentum)", (double) params.lr.lr0),
        [](common_params & params, const std::string & value) { params.lr.lr0 = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg({ "-lr-min", "--learning-rate-min" }, "ALPHA",
        string_format("(if >0) final learning rate after decay (if -decay-epochs is set, default=%.2g)",
            (double) params.lr.lr_min),
        [](common_params & params, const std::string & value) { params.lr.lr_min = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-decay-epochs", "--learning-rate-decay-epochs"}, "ALPHA",
        string_format("(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default=%.2g)", (double) params.lr.decay_epochs),
        [](common_params & params, const std::string & value) { params.lr.decay_epochs = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-wd", "--weight-decay"}, "WD",
        string_format("adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: %.2g).", (double) params.lr.wd),
        [](common_params & params, const std::string & value) { params.lr.wd = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-val-split", "--val-split"}, "FRACTION",
        string_format("fraction of data to use as validation set for training (default: %.2g).", (double) params.val_split),
        [](common_params & params, const std::string & value) { params.val_split = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-epochs", "--epochs"}, "N",
        string_format("optimizer max # of epochs (default: %d)", params.lr.epochs),
        [](common_params & params, int epochs) { params.lr.epochs = epochs; }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-opt", "--optimizer"}, "sgd|adamw", "adamw or sgd",
        [](common_params & params, const std::string & name) {
            params.optimizer = common_opt_get_optimizer(name.c_str());
            if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_COUNT) {
                throw std::invalid_argument("invalid --optimizer, valid options: adamw, sgd");
            }
        }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"--check"},
        string_format("check rather than generate results (default: %s)", params.check ? "true" : "false"),
        [](common_params & params) {
            params.check = true;
        }
    ).set_examples({LLAMA_EXAMPLE_RESULTS}));
    add_opt(common_arg(
        {"--save-logits"},
        string_format("save final logits to files for verification (default: %s)", params.save_logits ? "true" : "false"),
        [](common_params & params) {
            params.save_logits = true;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--logits-output-dir"}, "PATH",
        string_format("directory for saving logits output files (default: %s)", params.logits_output_dir.c_str()),
        [](common_params & params, const std::string & value) {
            params.logits_output_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--tensor-filter"}, "REGEX",
        "filter tensor names for debug output (regex pattern, can be specified multiple times)",
        [](common_params & params, const std::string & value) {
            params.tensor_filter.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));

    // presets

    add_opt(common_arg(
        {"--embd-gemma-default"},
        string_format("use default EmbeddingGemma model (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
            params.model.hf_file = "embeddinggemma-300M-qat-Q4_0.gguf";
            params.port = 8011;
            params.n_ubatch = 2048;
            params.n_batch = 2048;
            params.n_parallel = 32;
            params.n_ctx = 2048*params.n_parallel;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-1.5b-default"},
        string_format("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-1.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-3b-default"},
        string_format("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-default"},
        string_format("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-spec"},
        string_format("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-14b-spec"},
        string_format("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-14b-q8_0.gguf";
            params.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-30b-default"},
        string_format("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF";
            params.model.hf_file = "qwen3-coder-30b-a3b-instruct-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--gpt-oss-20b-default"},
        string_format("use gpt-oss-20b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-20b-GGUF";
            params.model.hf_file = "gpt-oss-20b-mxfp4.gguf";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--gpt-oss-120b-default"},
        string_format("use gpt-oss-120b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-120b-GGUF";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-4b-default"},
        string_format("use Gemma 3 4B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-4b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-12b-default"},
        string_format("use Gemma 3 12B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-12b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-default"},
        string_format("enable default speculative decoding config"),
        [](common_params & params) {
            params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
            params.speculative.ngram_mod.n_match = 24;
            params.speculative.ngram_mod.n_min = 48;
            params.speculative.ngram_mod.n_max = 64;

            // TODO: not sure if this is a good config - explore more settings and potentially enable it
            //params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
            //params.speculative.ngram_map_k4v.size_n = 8;
            //params.speculative.ngram_map_k4v.size_m = 24;
            //params.speculative.ngram_map_k4v.min_hits = 2;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    return ctx_arg;
}

void common_params_add_preset_options(std::vector<common_arg> & args) {
    // arguments below won't be treated as CLI args, only preset options
    args.push_back(common_arg(
        {"load-on-startup"}, "NAME",
        "in server router mode, autoload this model on startup",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_LOAD_ON_STARTUP).set_preset_only());

    args.push_back(common_arg(
        {"stop-timeout"}, "SECONDS",
        "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
        [](common_params &, int) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_STOP_TIMEOUT).set_preset_only());

    args.push_back(common_arg(
        {"dedup-cache-models"}, "0|1",
        "in server router mode, hide a cached model from the model list when this preset resolves to the same model file",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS).set_preset_only());

    // args.push_back(common_arg(
    //     {"pin"},
    //     "in server router mode, do not unload this model if models_max is exceeded",
    //     [](common_params &) { /* unused */ }
    // ).set_preset_only());
}

// file: common/arg.h
#pragma once

#include "common.h"
#include "download.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

// pseudo-env variable to identify preset-only arguments
#define COMMON_ARG_PRESET_LOAD_ON_STARTUP    "__PRESET_LOAD_ON_STARTUP"
#define COMMON_ARG_PRESET_STOP_TIMEOUT       "__PRESET_STOP_TIMEOUT"
#define COMMON_ARG_PRESET_DEDUP_CACHE_MODELS "__PRESET_DEDUP_CACHE_MODELS"

//
// CLI argument parsing
//

struct common_arg {
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    std::set<enum llama_example> excludes = {};
    std::vector<const char *> args;
    std::vector<const char *> args_neg;  // for negated args like --no-xxx
    const char * value_hint   = nullptr; // help text or example for arg value
    const char * value_hint_2 = nullptr; // for second arg value
    const char * env          = nullptr;
    std::string help;
    bool is_sampling = false; // is current arg a sampling param?
    bool is_spec = false; // is current arg a speculative decoding param?
    bool is_preset_only = false; // is current arg preset-only (not treated as CLI arg)
    void (*handler_void)   (common_params & params) = nullptr;
    void (*handler_string) (common_params & params, const std::string &) = nullptr;
    void (*handler_str_str)(common_params & params, const std::string &, const std::string &) = nullptr;
    void (*handler_int)    (common_params & params, int) = nullptr;
    void (*handler_bool)   (common_params & params, bool) = nullptr;

    common_arg() = default;

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &)
    ) : args(args), value_hint(value_hint), help(help), handler_string(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, int)
    ) : args(args), value_hint(value_hint), help(help), handler_int(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::string & help,
        void (*handler)(common_params & params)
    ) : args(args), help(help), handler_void(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::initializer_list<const char *> & args_neg,
        const std::string & help,
        void (*handler)(common_params & params, bool)
    ) : args(args), args_neg(args_neg), help(help), handler_bool(handler) {}

    // support 2 values for arg
    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const char * value_hint_2,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &, const std::string &)
    ) : args(args), value_hint(value_hint), value_hint_2(value_hint_2), help(help), handler_str_str(handler) {}

    common_arg & set_examples(std::initializer_list<enum llama_example> examples);
    common_arg & set_excludes(std::initializer_list<enum llama_example> excludes);
    common_arg & set_env(const char * env);
    common_arg & set_sampling();
    common_arg & set_spec();
    common_arg & set_preset_only();
    bool in_example(enum llama_example ex);
    bool is_exclude(enum llama_example ex);
    bool get_value_from_env(std::string & output) const;
    bool has_value_from_env() const;
    std::string to_string() const;

    // for using as key in std::map
    bool operator<(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) < 0;
    }
    bool operator==(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) == 0;
    }

    // get all args and env vars (including negated args/env)
    std::vector<std::string> get_args() const;
    std::vector<std::string> get_env() const;
};

namespace common_arg_utils {
    bool is_truthy(const std::string & value);
    bool is_falsey(const std::string & value);
    bool is_autoy(const std::string & value);
}

struct common_params_context {
    enum llama_example ex = LLAMA_EXAMPLE_COMMON;
    common_params & params;
    std::vector<common_arg> options;
    void(*print_usage)(int, char **) = nullptr;
    common_params_context(common_params & params) : params(params) {}
};

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
// TODO: this function can load ggml backend (by calling llama_support_rpc)
//       this is a side-effect that should be avoided
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// load all backends and print the list of available (non-CPU) devices to stdout
void common_print_available_devices();

// parse input arguments from CLI into a map
bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map);

// populate preset-only arguments
// these arguments are not treated as command line arguments
// see: https://github.com/ggml-org/llama.cpp/issues/18163
void common_params_add_preset_options(std::vector<common_arg> & args);

struct common_models_handler {
    common_download_hf_plan plan;
    common_download_hf_plan plan_spec;
    common_download_opts opts;
};

// initialize downloading opts and hf_plan if needed, but does not download anything yet
common_models_handler common_models_handler_init(const common_params & params, llama_example curr_ex);

// check if the model is a preset repo (i.e. has a preset file)
bool common_models_handler_is_preset_repo(const common_models_handler & handler);

// download and update params with the downloaded model path
void common_models_handler_apply(common_models_handler & handler, common_params & params, common_download_callback * callback = nullptr);

// initialize argument parser context - used by test-arg-parser and preset
common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// file: common/build-info.h
#pragma once

#include <cstdio>

int llama_build_number(void);

const char * llama_commit(void);
const char * llama_compiler(void);

const char * llama_build_target(void);
const char * llama_build_info(void);

void llama_print_build_info(const char *, FILE * = stderr);

// file: common/chat-auto-parser-generator.cpp
#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "parsers/parsers.h"
#include "peg-parser.h"

#include <stdexcept>
#include <string>

using json = common_json;

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder & p, const generation_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs) {
    // Run differential analysis to extract template structure
    struct autoparser autoparser;
    autoparser.analyze_template(tmpl);
    return generate_parser(tmpl, inputs, autoparser);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs,
                                                  const autoparser &              autoparser) {
    // Create the result structure
    common_chat_params data;
    data.prompt            = common_chat_template_direct_apply(tmpl, inputs);
    data.generation_prompt = common_chat_template_generation_prompt(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = autoparser.preserved_tokens;
    data.additional_stops.insert(data.additional_stops.end(),
        autoparser.additional_stops.begin(), autoparser.additional_stops.end());

    std::string parser_generation_prompt = data.generation_prompt;

    if (inputs.continue_final_message != COMMON_CHAT_CONTINUATION_NONE && !inputs.continue_msg.empty()) {
        // Build up generation prompt manually
        const auto & msg = inputs.continue_msg;

        if (!autoparser.reasoning.start.empty()) {
            data.generation_prompt = data.generation_prompt.substr(0, data.generation_prompt.find(autoparser.reasoning.start));
            data.generation_prompt += autoparser.reasoning.start + msg.reasoning_content;
            if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
                data.generation_prompt += autoparser.reasoning.end;
            }
        }

        if (inputs.continue_final_message == COMMON_CHAT_CONTINUATION_CONTENT) {
            data.generation_prompt += msg.render_content();
        }

        data.prompt += data.generation_prompt;
    }

    auto parser = autoparser.build_parser(inputs, parser_generation_prompt);
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools =
        autoparser.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !autoparser.tools.format.section_start.empty() ? autoparser.tools.format.section_start :
                                                                                  autoparser.tools.format.per_call_start;

    bool has_response_format = !inputs.json_schema.empty() && inputs.json_schema.is_object();
    bool include_grammar = has_response_format || (has_tools &&
            ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
              inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
            if (autoparser.tools.format.openai_wrapper_trigger) {
                // model emits the OpenAI function wrapper, trigger on it
                data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "{\"type\": \"function\"," });
            }
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const generation_params & inputs, const std::string & generation_prompt) const {
    if (!analysis_complete) {
        throw std::invalid_argument("Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
    }
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

        ctx.extracting_reasoning = extract_reasoning && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;
        ctx.reasoning            = &reasoning;

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        auto parser = p.eps();

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
        bool pure_content        = reasoning.mode == reasoning_mode::NONE;

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            parser = ctx.reasoning_parser + p.space() + p.choice({
                p.literal("```json") + p.space() + response_format + p.space() + p.literal("```"),
                p.space() + response_format  + p.space()
            }) + p.end();
            pure_content = false;
        } else if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            parser = tools.build_parser(ctx);
            pure_content = false;
        } else {
            parser = content.build_parser(ctx);
        }
        const std::string reasoning_start = trim_whitespace(reasoning.start);
        return pure_content ? p.prefix(generation_prompt, reasoning_start) + parser : p.prefix(generation_prompt, reasoning_start) << parser;
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        if (!end.empty()) {
            if (!start.empty()) {
                // Standard tag-based: optional(<think>reasoning</think>)
                return p.optional(p.optspace(start) + p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
            }
            // Delimiter-style (empty start)
            return p.optional(p.reasoning(p.until(trim_whitespace(end))) + p.optspace(end));
        }
    }

    return p.eps();
}

common_peg_parser analyze_content::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        if (ctx.extracting_reasoning) {
            return ctx.reasoning_parser + start + p.content(p.until(end)) + end + p.end();
        }
        return p.content(p.until(start)) + start + p.content(p.until(end)) + end + p.end();
    }
    return ctx.reasoning_parser + p.content(p.rest()) + p.end();
}

common_peg_parser analyze_content::build_optional_wrapped(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        return p.optional(start + p.content(p.until(end)) + end);
    }
    return p.eps();
}

common_peg_parser analyze_tools::build_parser(parser_build_context & ctx) const {
    switch (format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(ctx);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(ctx);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(ctx);
        default:
            LOG_ERR("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. "
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or "
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n");
            return ctx.p.eps();
    }
}

common_peg_parser analyze_tools::build_tool_parser_json_native(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    // Build effective field names with dot notation if function_field is set
    std::string name_field = format.name_field;
    std::string args_field = format.args_field;

    if (!format.function_field.empty() && format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = format.function_field + "." + name_field;
        args_field = format.function_field + "." + args_field;
    }

    auto tools_parser = p.eps();
    if (format.section_start.empty() && !format.per_call_start.empty()) {
        auto single_tool_parser = p.standard_json_tools(
            format.per_call_start, format.per_call_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
        tools_parser = p.trigger_rule("tool-calls", p.one_or_more(single_tool_parser + p.space()));
    } else {
        tools_parser = p.standard_json_tools(
            format.section_start, format.section_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order, format.openai_wrapper_trigger);
    }

    // Handle content wrappers if present
    if (ctx.content && ctx.content->is_always_wrapped()) {
        auto wrapped_content = ctx.content->build_optional_wrapped(ctx);
        return ctx.reasoning_parser + wrapped_content + tools_parser + p.end();
    }

    std::string tool_start = "{";
    if (!format.section_start.empty()) {
        tool_start = format.section_start;
    } else if (!format.per_call_start.empty()) {
        tool_start = format.per_call_start;
    }

    return ctx.reasoning_parser + p.optional(p.content(p.until(tool_start))) + tools_parser + p.end();
}

common_peg_parser analyze_tools::build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                                    const common_peg_parser & call_id_section, bool have_call_id,
                                                    const common_peg_parser & args,
                                                    std::optional<common_peg_parser> atomic_peek) const {
    auto              open           = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix);
    bool              matched_atomic = false;
    common_peg_parser func_parser    = p.eps();

    if (!function.args_separator.empty()) {
        open = open + p.space() + p.literal(function.args_separator);
    }

    if (!function.name_suffix.empty()) {
        func_parser    = open + call_id_section + p.space() + args;
        matched_atomic = true;
    } else if (have_call_id) {
        func_parser    = p.atomic(open + call_id_section) + p.space() + args;
        matched_atomic = true;
    } else if (atomic_peek.has_value()) {
        func_parser    = p.atomic(open + call_id_section + p.space() + *atomic_peek) + args;
        matched_atomic = true;
    } else {
        func_parser = open + call_id_section + p.space() + args;
    }

    if (!function.close.empty()) {
        func_parser = func_parser + p.space() + p.tool_close(p.literal(function.close));
    } else if (!format.per_call_end.empty()) {
        // When there's no func_close but there is a per_call_end marker, use peek() to ensure
        // we only emit tool_close when we can actually see the closing marker. This prevents
        // premature closing during partial parsing when we've seen e.g. "</" which could be
        // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
        // Laguna (v4): the model may emit whitespace between the last </arg_value> and
        // </tool_call> even though the template renders them tight. Tolerate optional
        // leading space in the close lookahead so the tool call still closes.
        auto close_peek = arguments.tolerate_intertag_whitespace
                              ? p.peek(p.space() + p.literal(format.per_call_end))
                              : p.peek(p.literal(format.per_call_end));
        func_parser = func_parser + p.tool_close(close_peek);
    } else {
        func_parser = func_parser + p.tool_close(p.space());  // force this to process tool closing callbacks in mapper
    }
    if (!matched_atomic) {
        func_parser = p.atomic(func_parser);
    }
    return func_parser;
}

common_peg_parser analyze_tools::build_tool_parser_tag_json(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto   schema = common_chat_tool_parameters(func);

        // Build call_id parser based on position (if supported)
        bool have_call_id = false;
        common_peg_parser call_id_section = p.eps();
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
            have_call_id = true;
        }
        auto args_parser = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));
        if (!arguments.start.empty()) {
            args_parser = p.literal(arguments.start) + args_parser;
        }
        if (!arguments.end.empty()) {
            args_parser = args_parser + p.literal(arguments.end);
        }

        auto atomic_peek = !arguments.start.empty() ? std::optional(p.peek(p.literal(arguments.start))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_parser, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + tool_choice + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice + format.section_end);
        }
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

common_peg_parser analyze_tools::build_tool_parser_tag_tagged(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;

    auto until_suffix = p.rule("until-suffix", p.until(arguments.value_suffix));

    common_peg_parser tool_choice = p.choice();

    foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func = tool.at("function");
        std::string  name = func.at("name");

        // Build parser for each argument, separating required and optional
        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
        foreach_parameter(func, [&](const common_chat_schema_property & param, const common_chat_schema_document_ptr & doc) {
            auto arg =
                p.tool_arg(p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param.name)) +
                                           arguments.name_suffix) +
                           arguments.value_prefix +
                           (param.schema->may_be_string() ?
                                p.ac(p.tool_arg_string_value(until_suffix) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)), arguments.value_suffix) :
                                (p.tool_arg_json_value(p.schema(
                                    p.json(), "tool-" + name + "-arg-" + param.name + "-schema", doc, *param.schema)) +
                                    p.tool_arg_close(p.literal(arguments.value_suffix)))));

            auto named_arg = p.rule("tool-" + name + "-arg-" + param.name, arg);
            if (param.required) {
                required_parsers.push_back(named_arg);
            } else {
                optional_parsers.push_back(named_arg);
            }
        });

        // Build required arg sequence in definition order
        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        // Build optional args with flexible ordering
        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
        }

        if (!arguments.start.empty()) {
            args_seq = p.literal(arguments.start) + args_seq;
        }
        if (!arguments.end.empty()) {
            args_seq = args_seq + p.literal(arguments.end);
        }

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        bool have_call_id = false;
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            have_call_id = true;
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix)) + call_id.suffix);
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
        }

        // Only peek for an arg tag when there are required args that must follow.
        // When all args are optional, the model may emit no arg tags at all (#20650).
        auto atomic_peek = (!arguments.name_prefix.empty() && !required_parsers.empty()) ?
            std::optional(p.peek(p.literal(arguments.name_prefix))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_seq, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + p.space() + tool_choice + p.space() + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call) + p.space());
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.space());
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end) + p.space()));
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + p.space() + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + p.space() +
                                                         format.section_end);
        } else {
            tool_calls = p.trigger_rule(
                "tool-call", format.section_start + p.space() + tool_choice + p.space() + format.section_end);
        }
    }

    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + p.optional(p.content(content_before_tools)) + tool_calls + p.end();
}

}  // namespace autoparser

// file: common/chat-auto-parser-helpers.cpp
#include "chat-auto-parser-helpers.h"

#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "log.h"
#include "peg-parser.h"

#include <cctype>
#include <numeric>

std::string trim_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    if (start == str.length()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    return str.substr(start, end - start + 1);
}

std::string trim_leading_whitespace(const std::string & str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }

    return str.substr(start);
}

std::string trim_trailing_whitespace(const std::string & str) {
    if (str.empty()) {
        return "";
    }

    size_t end = str.length() - 1;
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end]))) {
        end--;
    }

    // If first char is also whitespace, return empty string
    if (end == 0 && std::isspace(static_cast<unsigned char>(str[0]))) {
        return "";
    }

    return str.substr(0, end + 1);
}

std::string trim_trailing_newlines(const std::string & str) {
    size_t end = str.length();
    while (end > 0 && str[end - 1] == '\n') {
        end--;
    }

    return str.substr(0, end);
}

static size_t common_prefix_len(const std::string & left, const std::string & right) {
    size_t prefix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (prefix_len < min_len && left[prefix_len] == right[prefix_len]) {
        prefix_len++;
    }
    return prefix_len;
}

static size_t common_suffix_len(const std::string & left, const std::string & right) {
    size_t suffix_len = 0;
    size_t min_len    = std::min(left.length(), right.length());
    while (suffix_len < min_len && left[left.length() - 1 - suffix_len] == right[right.length() - 1 - suffix_len]) {
        suffix_len++;
    }
    return suffix_len;
}

diff_split calculate_diff_split(const std::string & left, const std::string & right) {
    diff_split result;

    auto left_seg = segmentize_markers(left);
    auto right_seg = segmentize_markers(right);

    if (left_seg.empty()) {
        result.right = right;
        return result;
    }
    if (right_seg.empty()) {
        result.left = left;
        return result;
    }

    auto left_start = left_seg.begin();
    auto left_end = --left_seg.end();
    auto right_start = right_seg.begin();
    auto right_end = --right_seg.end();

    auto test = [&] () {
        return left_start != left_end && right_start != right_end;
    };

    bool left_fully_consumed = false;
    bool right_fully_consumed = false;

    while (test()) {
        bool advanced = false;
        if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_start++;
            advanced = true;
        }
        if (*left_end == *right_end) {
            result.suffix = left_end->value + result.suffix;
            if (left_start != left_end) {
                left_end--;
            } else {
                left_fully_consumed = true;
            }
            if (right_start != right_end) {
                right_end--;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if (!advanced) {
            break;
        }
    }

    if (left_start == left_end && right_start != right_end) {
        if (*left_start == *right_end) {
            result.suffix = right_end->value + result.suffix;
            right_end--;
            left_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(right_start->value);
            right_start++;
            left_fully_consumed = true;
        }
    } else if (right_start == right_end && left_start != left_end) {
        if (*left_end == *right_start) {
            result.suffix = left_end->value + result.suffix;
            left_end--;
            right_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_fully_consumed = true;
        }
    } else if (left_start == left_end && right_start == right_end && *left_start == *right_start && left_start->type == segment_type::MARKER) {
        result.prefix.append(right_start->value);
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    auto eat_segment = [](std::string str, const segment & seg) -> std::string { return std::move(str) + seg.value; };

    bool can_have_text_suffix = left_end->type == segment_type::TEXT && right_end->type == segment_type::TEXT;
    bool can_have_text_prefix = right_start->type == segment_type::TEXT && left_start->type == segment_type::TEXT;

    std::string remainder_left = std::accumulate(left_start, left_fully_consumed ? left_end : ++left_end, std::string(), eat_segment);
    std::string remainder_right = std::accumulate(right_start, right_fully_consumed ? right_end : ++right_end, std::string(), eat_segment);

    size_t suffix_len = can_have_text_suffix ? common_suffix_len(remainder_left, remainder_right) : 0;
    // avoid overlaps between prefix and suffix
    size_t prefix_len = can_have_text_prefix ? common_prefix_len(remainder_left.substr(0, remainder_left.size() - suffix_len),
        remainder_right.substr(0, remainder_right.size() - suffix_len)) : 0;

    result.prefix.append(remainder_left.substr(0, prefix_len));
    result.suffix = remainder_left.substr(remainder_left.length() - suffix_len, suffix_len) + result.suffix;
    result.left = remainder_left.substr(prefix_len, remainder_left.length() - prefix_len - suffix_len);
    result.right = remainder_right.substr(prefix_len, remainder_right.length() - prefix_len - suffix_len);

    if (result.left == "" && result.right == "") {
        // degenerate case, no diff
        result.prefix = left;
        result.suffix = "";
        // pick prefix = all as representation
    }

    // When left has no unique content (result.left is empty), left is entirely
    // shared with right. The simultaneous prefix/suffix segment matching can
    // incorrectly consume trailing segments of left as suffix when those same
    // segments also appear at the end of right (e.g. "\n" at the end of both
    // the shared content and the generation prompt). This rotates the diff.
    // Fix: if left is a prefix of right, enforce that directly.
    if (result.left.empty() && !result.right.empty() &&
            left.size() <= right.size() &&
            right.substr(0, left.size()) == left) {
        result.prefix = left;
        result.suffix = "";
        result.right  = right.substr(left.size());
    }

    return result;
}

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common prefix of left and right
    size_t common_prefix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_prefix_len < min_len && left[common_prefix_len] == right[common_prefix_len]) {
        common_prefix_len++;
    }

    // If there's no common prefix, return empty string
    if (common_prefix_len == 0) {
        return "";
    }

    // Find the common prefix in the full string
    std::string common_prefix = left.substr(0, common_prefix_len);
    size_t      pos           = full.find(common_prefix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

    // Return everything before the common prefix
    return full.substr(0, pos);
}

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right) {
    // Find the common suffix of left and right (compare from the end)
    size_t common_suffix_len = 0;
    size_t min_len           = std::min(left.length(), right.length());
    while (common_suffix_len < min_len &&
           left[left.length() - 1 - common_suffix_len] == right[right.length() - 1 - common_suffix_len]) {
        common_suffix_len++;
    }

    // If there's no common suffix, return empty string
    if (common_suffix_len == 0) {
        return "";
    }

    // Extract the common suffix
    std::string common_suffix = left.substr(left.length() - common_suffix_len);

    // Find the last occurrence of the common suffix in the full string
    size_t pos = full.rfind(common_suffix);

    // If not found, return empty string
    if (pos == std::string::npos) {
        return "";
    }

