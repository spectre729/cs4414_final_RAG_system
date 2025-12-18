#ifndef VECTOR_DB_HPP
#define VECTOR_DB_HPP
#include <faiss/IndexFlat.h>
#include <map>
#include <vector>
#include <string>

struct SearchResult{
    int id;
    float distance;
    std::string text;
};

class VectorDatabase {
public:
    VectorDatabase(const std::string& json_path, bool use_ivf = false);
    std::vector<SearchResult> search(const std::vector<float>& query_embedding, int k);
    std::vector<std::vector<SearchResult>> search_batch(const std::vector<float>& query_embedding, int batch_size, int k);
    ~VectorDatabase();
private:
    int dimension = 768;
    faiss::Index* index = nullptr;

    std::map<long, int> faiss_id_to_doc_id;
    std::map<int, std::string> doc_id_to_text;
};


#endif // VECTOR_DB_HPP