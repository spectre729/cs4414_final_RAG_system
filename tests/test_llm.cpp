#include <iostream>
#include "llm_generation.hpp"

int main() {
    try {
        std::string model_path = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf";
        LLMGenerator generator(model_path);
        std::string prompt = "Question: What is the capital of France?\nAnswer:";
        std::string response = generator.generate(prompt);

        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Response: " << response << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}