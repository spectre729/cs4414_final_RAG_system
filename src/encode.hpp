#ifndef ENCODE_HPP
#define ENCODE_HPP

#include <string>
#include <vector>
#include "llama.h"

class QueryEncoder {
public:
    QueryEncoder(const std::string& model_path);
    ~QueryEncoder();

    // Encode a single query and return its embedding
    std::vector<float> encode(const std::string& query);

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    std::vector<llama_token> tokenize(const std::string& text);
};


#endif // ENCODE_HPP