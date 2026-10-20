#include "encode.hpp"
#include <iostream>

int main() {
    try {
        QueryEncoder encoder("bge-base-en-v1.5-f32.gguf");
        std::string query = "What is a quarterback?";
        std::vector<float> embedding = encoder.encode(query);

        std::cout << "Embedding size: " << embedding.size() << std::endl;
        std::cout << "First 10 values of the embedding:" << std::endl;
        for (size_t i = 0; i < 10 && i < embedding.size(); ++i) {
            std::cout << embedding[i] << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}