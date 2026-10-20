/*
 * encode.cpp
 * Converts a natural-language question into an embedding vector using the same
 * BGE model that was used to embed the documents (see data_preprocess.cpp).
 */

#include "encode.hpp"

#include <algorithm>
#include <stdexcept>

namespace {
// BGE models support at most 512 tokens per sequence.
constexpr int kMaxTokens = 512;
}

QueryEncoder::QueryEncoder(const std::string& model_path) {
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        throw std::runtime_error("Unable to load embedding model: " + model_path);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 1024;

    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        llama_model_free(model);
        model = nullptr;
        throw std::runtime_error("Failed to create the llama_context for the embedding model");
    }

    vocab = llama_model_get_vocab(model);
}

QueryEncoder::~QueryEncoder() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
    }
    llama_backend_free();
}

std::vector<llama_token> QueryEncoder::tokenize(const std::string& text) {
    // The first call reports the required size as a negative value if the buffer is too small.
    std::vector<llama_token> tokens(text.length() + 2);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    }
    if (n < 0) {
        throw std::runtime_error("Failed to tokenize the query");
    }
    tokens.resize(std::min(n, kMaxTokens));
    return tokens;
}

std::vector<float> QueryEncoder::encode(const std::string& query) {
    std::vector<llama_token> tokens = tokenize(query);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    // Clear state left over from the previous query.
    llama_memory_clear(llama_get_memory(ctx), true);

    if (llama_encode(ctx, batch)) {
        throw std::runtime_error("Failed to encode the query");
    }

    float* embd = llama_get_embeddings_seq(ctx, 0);
    if (embd == nullptr) {
        throw std::runtime_error("Failed to get embeddings");
    }

    int n_embd = llama_model_n_embd(model);
    return std::vector<float>(embd, embd + n_embd);
}
