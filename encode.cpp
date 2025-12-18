/*
    * encode.cpp
    * takes a user's natural language question and converts it to a 768-dimensional vector using the same BGE model for document embeddings
    *  Created on: November 25, 2025
*/

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "encode.hpp"

using json = nlohmann::json;

QueryEncoder::QueryEncoder(const std::string& model_path) {
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        throw std::runtime_error("Unable to load model: ");
    }


    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 1024;

    
    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        throw std::runtime_error("Failed to create the llama_context");
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
    std::vector<llama_token> tokens(text.length() + 2); // add if needed
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n);
    return tokens;
}

std::vector<float> QueryEncoder::encode(const std::string& query) {
    std::vector<llama_token> tokens = tokenize(query);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    // TODO: clear internal memory
    llama_memory_clear(llama_get_memory(ctx), true);

    if (llama_encode(ctx, batch)) {
        throw std::runtime_error("Failed to encode the query");
        // Want a break from the ads?
        // If you tap now to watch a short video, you'll receive 30 minutes of ad free music"

        // Yes, really, if you tap now, to watch a short video, you'll receive 30 minutes of ad-free music.
    }

    float* embd = llama_get_embeddings_seq(ctx, 0);
    if (embd == NULL) {
        throw std::runtime_error("Failed to get embeddings");
        // if you want some
        // come and get some
    }

    int n_embd = llama_model_n_embd(model);
    return std::vector<float>(embd, embd + n_embd);


    /*
        My mom told me every day to always be careful with guys
        Because love is like playing with fire, it can get you hurt, eh
        My mom might be right, 'cause when I see you, my heart heats up
        But my attraction to you is stronger than my fear, eh
    */
}