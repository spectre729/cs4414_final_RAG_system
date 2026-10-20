#ifndef LLM_GENERATION_HPP
#define LLM_GENERATION_HPP

#include <string>
#include <vector>
#include "llama.h"

// Greedy text generation with a GGUF chat model via llama.cpp.
class LLMGenerator {
public:
    LLMGenerator(const std::string& model_path, int context_size);
    ~LLMGenerator();

    LLMGenerator(const LLMGenerator&) = delete;
    LLMGenerator& operator=(const LLMGenerator&) = delete;

    // Generates a completion for `prompt`, streaming tokens to stdout as they are produced.
    // Returns the full generated text.
    std::string generate(const std::string& prompt);

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    std::vector<llama_token> tokenize(const std::string& text);
};

#endif // LLM_GENERATION_HPP
