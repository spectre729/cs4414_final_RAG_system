#include "llm_generation.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
constexpr int kMaxNewTokens = 256;
constexpr int kNumThreads = 4;

// Small chat models sometimes keep going and invent the next turn; stop when they do.
bool hit_stop_sequence(const std::string& text) {
    return text.find("<|user|>") != std::string::npos ||
           text.find("<|system|>") != std::string::npos ||
           text.find("Question:") != std::string::npos;
}
}

LLMGenerator::LLMGenerator(const std::string& model_path, int context_size) {
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        throw std::runtime_error("Unable to load model: " + model_path);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_size;
    ctx_params.n_batch = context_size;
    ctx_params.n_threads = kNumThreads;
    ctx_params.n_threads_batch = kNumThreads;

    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        llama_model_free(model);
        model = nullptr;
        throw std::runtime_error("Failed to create the llama_context");
    }

    vocab = llama_model_get_vocab(model);
}

LLMGenerator::~LLMGenerator() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
    }
    llama_backend_free();
}

std::vector<llama_token> LLMGenerator::tokenize(const std::string& text) {
    // The first call reports the required size as a negative value if the buffer is too small.
    std::vector<llama_token> tokens(text.length() + 2);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    }
    if (n < 0) {
        throw std::runtime_error("Failed to tokenize the prompt");
    }
    tokens.resize(n);
    return tokens;
}

std::string LLMGenerator::generate(const std::string& prompt) {
    std::vector<llama_token> prompt_tokens = tokenize(prompt);
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // Each call starts from an empty KV cache.
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string output;
    for (int n_decode = 0; n_decode < kMaxNewTokens; ++n_decode) {
        if (llama_decode(ctx, batch)) {
            std::cerr << " [Context limit reached]" << std::endl;
            break;
        }

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            llama_sampler_free(smpl);
            throw std::runtime_error("Failed to convert token to piece during generation");
        }

        output.append(buf, n);
        if (hit_stop_sequence(output)) {
            break;
        }

        std::cout.write(buf, n);
        std::cout.flush();

        batch = llama_batch_get_one(&new_token_id, 1);
    }

    llama_sampler_free(smpl);
    return output;
}
