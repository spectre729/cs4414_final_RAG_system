#ifndef ENCODE_HPP
#define ENCODE_HPP

#include <string>
#include <vector>
#include "llama.h"

// Converts text into a dense embedding using a BGE embedding model (GGUF) via llama.cpp.
class QueryEncoder {
public:
    explicit QueryEncoder(const std::string& model_path);
    ~QueryEncoder();

    QueryEncoder(const QueryEncoder&) = delete;
    QueryEncoder& operator=(const QueryEncoder&) = delete;

    // Encodes a single query and returns its embedding (768 floats for bge-base).
    std::vector<float> encode(const std::string& query);

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    std::vector<llama_token> tokenize(const std::string& text);
};

#endif // ENCODE_HPP
