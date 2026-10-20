#include "vector_db.hpp"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {
// Number of IVF clusters; capped by the corpus size because k-means needs at least nlist points.
constexpr int kMaxIvfClusters = 100;
}

VectorDatabase::VectorDatabase(const std::string& json_path, bool use_ivf, int nprobe) {
    std::cout << "Loading embeddings from " << json_path << std::endl;
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error(json_path + " could not be opened. Please check that the path is correct.");
    }

    json data = json::parse(f);
    std::vector<float> all_embeddings;
    for (const auto& item : data) {
        std::vector<float> embedding = item.at("embedding").get<std::vector<float>>();
        if (static_cast<int>(embedding.size()) != kDimension) {
            throw std::runtime_error("Expected " + std::to_string(kDimension) +
                                     "-dimensional embeddings, got " + std::to_string(embedding.size()));
        }
        doc_ids.push_back(item.at("id").get<int>());
        doc_texts.push_back(item.at("text").get<std::string>());
        all_embeddings.insert(all_embeddings.end(), embedding.begin(), embedding.end());
    }
    const faiss::idx_t num_docs = static_cast<faiss::idx_t>(doc_ids.size());

    if (use_ivf) {
        std::cout << "Using IVF index (nprobe=" << std::max(1, nprobe) << ")" << std::endl;
        const int nlist = static_cast<int>(std::min<faiss::idx_t>(kMaxIvfClusters, std::max<faiss::idx_t>(1, num_docs)));
        auto* quantizer = new faiss::IndexFlatL2(kDimension);
        auto* ivf_index = new faiss::IndexIVFFlat(quantizer, kDimension, nlist, faiss::METRIC_L2);
        ivf_index->own_fields = true;  // the index now owns and frees the quantizer
        ivf_index->nprobe = std::max(1, std::min(nprobe, nlist));
        index.reset(ivf_index);

        if (num_docs > 0) {
            std::cout << "Training IVF index with " << num_docs << " vectors." << std::endl;
            index->train(num_docs, all_embeddings.data());
        }
    } else {
        std::cout << "Using Flat index" << std::endl;
        index = std::make_unique<faiss::IndexFlatL2>(kDimension);
    }

    if (num_docs > 0) {
        index->add(num_docs, all_embeddings.data());
    }
    std::cout << "Loaded " << num_docs << " documents into the vector database." << std::endl;
}

std::vector<SearchResult> VectorDatabase::search(const std::vector<float>& query_embedding, int k) {
    return search_batch(query_embedding, 1, k).front();
}

std::vector<std::vector<SearchResult>> VectorDatabase::search_batch(const std::vector<float>& query_embeddings,
                                                                    int batch_size, int k) {
    if (static_cast<long>(query_embeddings.size()) != static_cast<long>(batch_size) * kDimension) {
        throw std::invalid_argument("query_embeddings must contain batch_size * " +
                                    std::to_string(kDimension) + " floats");
    }

    std::vector<faiss::idx_t> labels(static_cast<size_t>(batch_size) * k);
    std::vector<float> distances(static_cast<size_t>(batch_size) * k);
    index->search(batch_size, query_embeddings.data(), k, distances.data(), labels.data());

    std::vector<std::vector<SearchResult>> results(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < k; ++i) {
            const size_t flat_index = static_cast<size_t>(b) * k + i;
            const faiss::idx_t label = labels[flat_index];
            if (label < 0) {
                continue;  // fewer than k results available
            }
            results[b].push_back({doc_ids[label], distances[flat_index], doc_texts[label]});
        }
    }
    return results;
}
