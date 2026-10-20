#ifndef LLM_GENERATION_HPP
#define LLM_GENERATION_HPP
#include <string>
#include <vector>
#include "llama.h"

class LLMGenerator {
public:
    LLMGenerator(const std::string& model_path, int context_size);
    ~LLMGenerator();
    std::string generate(const std::string& prompt);
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    std::vector<llama_token> tokenize(const std::string& text);
};


#endif // LLM_GENERATION_HPP