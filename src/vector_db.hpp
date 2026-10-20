#ifndef VECTOR_DB_HPP
#define VECTOR_DB_HPP

#include <faiss/Index.h>

#include <memory>
#include <string>
#include <vector>

struct SearchResult {
    int id;            // document id from the input file
    float distance;    // L2 distance to the query (smaller is closer)
    std::string text;
};

// In-memory vector database backed by a FAISS index (flat or IVF).
class VectorDatabase {
public:
    // Loads documents and their embeddings from `json_path`, as produced by data_preprocess.
    // Each entry must contain "id", "text" and a 768-dimensional "embedding".
    // `nprobe` is the number of IVF clusters searched per query (ignored for the flat index);
    // higher values improve recall at the cost of speed, up to the total number of clusters.
    explicit VectorDatabase(const std::string& json_path, bool use_ivf = false, int nprobe = 10);

    // Returns the k nearest documents to a single query embedding.
    std::vector<SearchResult> search(const std::vector<float>& query_embedding, int k);

    // Searches `batch_size` queries at once. `query_embeddings` holds the query vectors
    // concatenated back to back (batch_size * dimension floats).
    std::vector<std::vector<SearchResult>> search_batch(const std::vector<float>& query_embeddings,
                                                        int batch_size, int k);

private:
    static constexpr int kDimension = 768;

    std::unique_ptr<faiss::Index> index;

    // Documents are stored in insertion order, so the FAISS id is the index into these vectors.
    std::vector<int> doc_ids;
    std::vector<std::string> doc_texts;
};

#endif // VECTOR_DB_HPP
