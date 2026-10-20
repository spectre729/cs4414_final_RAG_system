#include <iostream>
#include "encode.hpp"
#include "vector_db.hpp"

int main() {
    // 1. Initialize Components
    std::string model_path = "bge-base-en-v1.5-f32.gguf";
    std::string json_path = "preprocessed_documents.json";

    std::cout << "--- Initializing System ---" << std::endl;
    QueryEncoder encoder(model_path);
    VectorDatabase db(json_path);

    // 2. Define a query
    std::string query = "How do I make flan?";
    std::cout << "\nQuery: " << query << std::endl;

    // 3. Encode
    std::vector<float> query_vec = encoder.encode(query);
    
    // 4. Search & Retrieve (Top 3)
    std::vector<SearchResult> results = db.search(query_vec, 3);

    // 5. Display Results
    std::cout << "\n--- Top 3 Retrieved Documents ---" << std::endl;
    for (const auto& res : results) {
        std::cout << "[ID: " << res.id << "] (Dist: " << res.distance << ")\n";
        std::cout << "Text: " << res.text.substr(0, 100) << "..." << std::endl; // Print first 100 chars
        std::cout << "-----------------------------------" << std::endl;
    }

    return 0;
}