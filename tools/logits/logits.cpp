// Dump the logits of one forward pass, so a port can be compared against a reference
// implementation on numbers instead of on generated text.
//
//   llama-logits -m model.gguf -p "prompt" -o logits.bin [--tokens 1,2,3] [-ngl N]
//
// Writes n_vocab float32 values for the LAST position of the prompt, and prints the top-10
// so a mismatch is visible without opening the file. Token ids can be given directly with
// --tokens, which is what a cross-implementation comparison needs: the two tokenizers must
// not be part of what is being compared.

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// --dump: the graph already names its intermediates through cb(); the scheduler callback is
// the existing way to read them. One file per node, float32, with a small header so the
// reader does not have to know the shapes in advance.
struct dump_state {
    std::vector<std::string> nomes;
    std::string              prefixo;
    int                      escritos = 0;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    dump_state * st = (dump_state *) user_data;
    const char * nome = ggml_get_name(t);
    bool quer = false;
    for (const std::string & n : st->nomes) {
        if (n == nome) { quer = true; break; }
    }
    if (ask) {
        return quer;
    }
    if (!quer) {
        return true;
    }

    const int64_t n = ggml_nelements(t);
    // A view is not contiguous: ggml_argsort_top_k hands back ne[0] = 6 over rows of
    // nb[1] = n_expert * 4, so a linear read of nbytes returns the wrong values with the
    // right shape -- which reads as a defect in the model rather than in the probe. Copy
    // row by row and let the stride say where each row starts.
    const size_t linha = ggml_row_size(t->type, t->ne[0]);
    std::vector<char> bruto(linha * (size_t) (t->ne[1]*t->ne[2]*t->ne[3]));
    {
        size_t dst = 0;
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                    const size_t off = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1];
                    ggml_backend_tensor_get(t, bruto.data() + dst, off, linha);
                    dst += linha;
                }
            }
        }
    }

    std::vector<float> vals((size_t) n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(vals.data(), bruto.data(), (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_I32) {
        // expert ids and other index tensors: the traits have no to_float, and widening
        // them is exact for anything an index can hold
        const int32_t * src = (const int32_t *) bruto.data();
        for (int64_t i = 0; i < n; i++) {
            vals[(size_t) i] = (float) src[i];
        }
    } else {
        const auto * tr = ggml_get_type_traits(t->type);
        if (!tr->to_float) {
            fprintf(stderr, "dump: %s has type %s, which cannot be read as float\n", nome, ggml_type_name(t->type));
            return true;
        }
        tr->to_float(bruto.data(), vals.data(), n);
    }

    char caminho[1024];
    snprintf(caminho, sizeof(caminho), "%s.%s.bin", st->prefixo.c_str(), nome);
    FILE * f = fopen(caminho, "wb");
    if (!f) {
        fprintf(stderr, "dump: cannot write %s\n", caminho);
        return true;
    }
    // header: magic, ne[4] as int64, then n floats
    const char magia[8] = { 'D','S','V','4','D','M','P','1' };
    fwrite(magia, 1, 8, f);
    fwrite(t->ne, sizeof(int64_t), 4, f);
    fwrite(vals.data(), sizeof(float), (size_t) n, f);
    fclose(f);
    st->escritos++;
    printf("logits: dump %-24s [%5lld %5lld %5lld %5lld] %s -> %s\n", nome,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            ggml_type_name(t->type), caminho);
    return true;
}

static void uso(const char * exe) {
    printf("usage: %s -m model.gguf [-p prompt | --tokens 1,2,3] -o out.bin\n", exe);
    printf("  -ngl N            layers on the GPU\n");
    printf("  --layers a,b,c    also dump the input of these layers (mean over the hc copies)\n");
    printf("  --dump n1,n2      also dump these graph nodes by name, e.g. attn_out-0,ffn_out-0\n");
    printf("  --tokens-file F   read the token ids from a file, for prompts too long for argv\n");
    printf("  -c N              context size (default: prompt length rounded up)\n");
    printf("  -ub N             ubatch size (default: the whole prompt, one forward)\n");
    printf("  --no-bos          do not prepend BOS when tokenizing -p\n");
}

int main(int argc, char ** argv) {
    std::string modelo, prompt, saida, lista_tokens, lista_camadas, lista_nos;
    int   ngl = 0, n_ctx = 0, n_ubatch = 0;
    bool  add_bos = true;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto proximo = [&](const char * nome) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", nome); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")                    modelo        = proximo("-m");
        else if (a == "-p")                    prompt        = proximo("-p");
        else if (a == "-o")                    saida         = proximo("-o");
        else if (a == "--tokens")              lista_tokens  = proximo("--tokens");
        else if (a == "-ngl")                  ngl           = atoi(proximo("-ngl").c_str());
        else if (a == "-c")                    n_ctx         = atoi(proximo("-c").c_str());
        else if (a == "--layers")              lista_camadas = proximo("--layers");
        else if (a == "--dump")                lista_nos     = proximo("--dump");
        else if (a == "--tokens-file") {
            // a long prompt does not fit in argv, and the selection logic of this model only
            // does real work above a thousand tokens
            const std::string caminho = proximo("--tokens-file");
            FILE * ft = fopen(caminho.c_str(), "rb");
            if (!ft) { fprintf(stderr, "cannot read %s\n", caminho.c_str()); return 1; }
            std::string conteudo;
            char buf[4096];
            size_t nlidos;
            while ((nlidos = fread(buf, 1, sizeof(buf), ft)) > 0) conteudo.append(buf, nlidos);
            fclose(ft);
            for (char & c : conteudo) { if (c == '\n' || c == '\r' || c == ' ') c = ','; }
            lista_tokens = conteudo;
        }
        else if (a == "-ub")                   n_ubatch      = atoi(proximo("-ub").c_str());
        else if (a == "--no-bos")              add_bos       = false;
        else { uso(argv[0]); return 1; }
    }

    if (modelo.empty() || saida.empty() || (prompt.empty() && lista_tokens.empty())) {
        uso(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    // expert streaming is a load-time property of the model, not of the context

    llama_model * model = llama_model_load_from_file(modelo.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", modelo.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens;
    if (!lista_tokens.empty()) {
        // ids given directly: the tokenizer stays out of the comparison
        size_t pos = 0;
        while (pos < lista_tokens.size()) {
            size_t virgula = lista_tokens.find(',', pos);
            if (virgula == std::string::npos) virgula = lista_tokens.size();
            const std::string campo = lista_tokens.substr(pos, virgula - pos);
            if (!campo.empty()) {
                tokens.push_back((llama_token) atoi(campo.c_str()));
            }
            pos = virgula + 1;
        }
    } else {
        const int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, add_bos, true);
        tokens.resize(n);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), add_bos, true) < 0) {
            fprintf(stderr, "tokenize failed\n");
            return 1;
        }
    }
    if (tokens.empty()) {
        fprintf(stderr, "no tokens\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx > 0 ? n_ctx : (uint32_t) tokens.size() + 8;
    cp.n_batch = (uint32_t) tokens.size();
    // one ubatch for the whole prompt: the reference does a single forward, and a prefill
    // split into 512-token chunks would carry compressor state across the split, so the
    // comparison would be measuring the framing instead of the model
    cp.n_ubatch = n_ubatch > 0 ? (uint32_t) n_ubatch : cp.n_batch;

    dump_state despejo;
    if (!lista_nos.empty()) {
        size_t pos = 0;
        while (pos < lista_nos.size()) {
            size_t virgula = lista_nos.find(',', pos);
            if (virgula == std::string::npos) virgula = lista_nos.size();
            despejo.nomes.push_back(lista_nos.substr(pos, virgula - pos));
            pos = virgula + 1;
        }
        despejo.prefixo         = saida;
        cp.cb_eval              = dump_cb;
        cp.cb_eval_user_data    = &despejo;
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    std::vector<uint32_t> camadas;
    if (!lista_camadas.empty()) {
        size_t pos = 0;
        while (pos < lista_camadas.size()) {
            size_t virgula = lista_camadas.find(',', pos);
            if (virgula == std::string::npos) virgula = lista_camadas.size();
            const uint32_t lid = (uint32_t) atoi(lista_camadas.substr(pos, virgula - pos).c_str());
            camadas.push_back(lid);
            llama_set_embeddings_layer_inp(ctx, lid, true);
            pos = virgula + 1;
        }
    }

    printf("logits: %zu tokens:", tokens.size());
    for (const llama_token t : tokens) {
        printf(" %d", t);
    }
    printf("\n");

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * lg = llama_get_logits_ith(ctx, -1);
    if (!lg) {
        fprintf(stderr, "no logits\n");
        return 1;
    }

    FILE * f = fopen(saida.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", saida.c_str());
        return 1;
    }
    fwrite(lg, sizeof(float), n_vocab, f);
    fclose(f);

    for (const uint32_t lid : camadas) {
        const float * h = llama_get_embeddings_layer_inp(ctx, lid);
        if (!h) {
            fprintf(stderr, "layer %u: no data\n", lid);
            continue;
        }

        const int n_embd = llama_model_n_embd(model);
        char nome[1024];
        snprintf(nome, sizeof(nome), "%s.layer%u.bin", saida.c_str(), lid);

        FILE * fl = fopen(nome, "wb");
        if (fl) {
            // the last position only: one vector per layer is enough to find where two
            // implementations start to disagree
            fwrite(h + (size_t)(tokens.size() - 1)*n_embd, sizeof(float), n_embd, fl);
            fclose(fl);
            printf("logits: layer %u -> %s (%d values)\n", lid, nome, n_embd);
        }
    }

    std::vector<int> ord(n_vocab);
    for (int i = 0; i < n_vocab; i++) ord[i] = i;
    std::partial_sort(ord.begin(), ord.begin() + 10, ord.end(),
            [&](int a, int b) { return lg[a] > lg[b]; });

    printf("logits: %d values written to %s\n", n_vocab, saida.c_str());
    printf("logits: top 10\n");
    for (int i = 0; i < 10; i++) {
        char buf[128];
        const int n = llama_token_to_piece(vocab, ord[i], buf, sizeof(buf) - 1, 0, true);
        buf[n > 0 ? n : 0] = 0;
        printf("  %2d. id %6d  %12.6f  %s\n", i + 1, ord[i], lg[ord[i]], buf);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
