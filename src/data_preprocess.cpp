/*
 * data_preprocess.cpp
 * Embeds every document in a corpus with a BGE model and writes the result to disk.
 *
 * Input:  JSON array of {"id": <int>, "text": <string>} (e.g. passages from MS MARCO)
 * Output: JSON array of {"id", "text", "embedding": [768 floats]}
 *
 * Usage: data_preprocess [documents.json] [preprocessed_documents.json] [bge-base-en-v1.5-f32.gguf]
 */

#include "llama.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {
constexpr int kEmbeddingDim = 768;
// BGE models support at most 512 tokens per sequence; longer documents are truncated.
constexpr int kMaxTokens = 512;
}

int main(int argc, char** argv) {
    const std::string input_path = argc > 1 ? argv[1] : "documents.json";
    const std::string output_path = argc > 2 ? argv[2] : "preprocessed_documents.json";
    const std::string model_path = argc > 3 ? argv[3] : "bge-base-en-v1.5-f32.gguf";

    std::ifstream f(input_path);
    if (!f.is_open()) {
        std::cerr << input_path << " could not be opened. Please check that the path is correct." << std::endl;
        return 1;
    }

    json data;
    try {
        data = json::parse(f);
    } catch (json::parse_error& e) {
        std::cerr << input_path << " could not be parsed: " << e.what() << std::endl;
        return 1;
    }

    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "Unable to load embedding model: " << model_path << std::endl;
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 512;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::cerr << "Failed to create the llama_context" << std::endl;
        llama_model_free(model);
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int n_embd = llama_model_n_embd(model);
    if (n_embd != kEmbeddingDim) {
        std::cerr << "Expected " << kEmbeddingDim << "-dimensional embeddings, but the model produces "
                  << n_embd << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    int status = 0;
    json output_data = json::array();

    for (const auto& item : data) {
        const int id = item.at("id");
        const std::string text = item.at("text");
        std::cout << "ID: " << id << " TEXT: " << text.substr(0, 40) << " ... " << std::endl;

        // The first call reports the required size as a negative value if the buffer is too small.
        std::vector<llama_token> tokens(text.size() + 2);
        int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
        if (n_tokens < 0) {
            tokens.resize(-n_tokens);
            n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
        }
        if (n_tokens < 0) {
            std::cerr << "Failed to tokenize document " << id << std::endl;
            status = 1;
            break;
        }
        n_tokens = std::min(n_tokens, kMaxTokens);

        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        if (llama_encode(ctx, batch)) {
            std::cerr << "Failed to encode document " << id << std::endl;
            status = 1;
            break;
        }

        const float* embd = llama_get_embeddings_seq(ctx, 0);
        if (embd == nullptr) {
            std::cerr << "Failed to get embeddings for document " << id << std::endl;
            status = 1;
            break;
        }

        json new_item;
        new_item["id"] = id;
        new_item["text"] = text;
        new_item["embedding"] = std::vector<float>(embd, embd + n_embd);
        output_data.push_back(new_item);
    }

    if (status == 0) {
        std::ofstream o(output_path);
        o << std::setw(2) << output_data << std::endl;
        std::cout << "Wrote " << output_data.size() << " embedded documents to " << output_path << std::endl;
    }

    llama_free(ctx);
    llama_model_free(model);
    return status;
}
