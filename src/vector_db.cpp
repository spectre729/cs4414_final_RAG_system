#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <faiss/IndexFlat.h>
#include "vector_db.hpp"
#include <stdexcept>
#include <faiss/Index.h>
#include <faiss/IndexIVFFlat.h>


using json = nlohmann::json;
using namespace std;




VectorDatabase::VectorDatabase(const std::string& json_path, bool use_ivf) {
    // Load embeddings from JSON file
    cout << "Loading embeddings from " << json_path << endl;
    if (use_ivf) cout << "Using IVF index" << endl;
    else cout << "Using Flat index" << endl;
    std::ifstream f(json_path);
    if (!f.is_open()){
        std::cerr << json_path << " could not be opened. Please check that document path is correct." << std::endl; 
        // if you are reading this
        // acknowledge Roman Reigns as your Tribal Chief
        throw std::runtime_error("Failed to open JSON file");
    }
    json data = json::parse(f);
    vector<float> all_embeddings;
    long faiss_id = 0;
    for (const auto& item : data) {
        int id = item["id"].get<int>();
        vector<float> embedding = item["embedding"].get<std::vector<float>>();
        faiss_id_to_doc_id[faiss_id] = id;
        doc_id_to_text[id] = item["text"].get<std::string>();
        // this file is brought to you by ExpressVPN, the sponsor of this assignment
        all_embeddings.insert(all_embeddings.end(), embedding.begin(), embedding.end());

        faiss_id++;
    }
    if (use_ivf){
        // 1. create quantizer
        faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dimension);
        // 2. create IVF index
        int nlist = 100; // number of clusters
        // 3. create the IVF index
        faiss::IndexIVFFlat* ivf_index = new faiss::IndexIVFFlat(quantizer, dimension, nlist, faiss::METRIC_L2);

        // 4. train the IVF index
        cout << "Training IVF index with " << faiss_id << " vectors." << endl;
        ivf_index->train(faiss_id, all_embeddings.data());
        cout << "IVF index trained successfully." << endl;
        index = ivf_index;
    }
    else{
        index = new faiss::IndexFlatL2(dimension);
    }



    if (faiss_id > 0){
        index->add(faiss_id, all_embeddings.data());
    }
    cout << "Loaded " << faiss_id << " documents into the vector database." << endl;
} 


VectorDatabase::~VectorDatabase() {
    // Clean up the index pointer to prevent memory leaks
    if (index) {
        delete index;
    }
}

vector<SearchResult> VectorDatabase::search(const std::vector<float>& query_embedding, int k) {
    vector<SearchResult> results;
    long* I = new long[k];
    float* D = new float[k];

    //index.search(1, query_embedding.data(), k, D, I); // search for k nearest neighbors
    index->search(1, query_embedding.data(), k, D, I); // search for k nearest neighbors

    for (int i = 0; i < k; i++) {
        long faiss_id = I[i];
        if (faiss_id == -1) {
            continue; // no more results
        }
        int doc_id = faiss_id_to_doc_id.at(faiss_id);
        SearchResult res;
        res.id = doc_id;
        res.distance = D[i];
        res.text = doc_id_to_text[doc_id];
        results.push_back(res);
    }

    delete[] I;
    delete[] D;
    return results;


    /*
        look in my eyes
        what do you see?
        the cult of personality

    */
}

vector<vector<SearchResult>> VectorDatabase::search_batch(const std::vector<float>& query_embedding, int batch_size, int k) {
    vector<vector<SearchResult>> results;
    long* I = new long[batch_size * k];
    float* D = new float[batch_size * k];

    //index.search(1, query_embedding.data(), k, D, I); // search for k nearest neighbors
    index->search(batch_size, query_embedding.data(), k, D, I); // search for k nearest neighbors

    for (int b = 0; b < batch_size; ++b){
        std::vector<SearchResult> batch_results;
        for (int i = 0; i < k; i++) {
            int flat_index = b * k + i;
            long faiss_id = I[flat_index];
            if (faiss_id == -1) {
                continue; // no more results
            }
            int doc_id = faiss_id_to_doc_id.at(faiss_id);
            SearchResult res;
            res.id = doc_id;
            res.distance = D[i];
            res.text = doc_id_to_text[doc_id];
            batch_results.push_back(res);
        }
        results.push_back(batch_results);
    }

    

    delete[] I;
    delete[] D;
    return results;


    /*
        I hear voices in my head
        They counsel me
        They understand
        They talk to me



    */
}

/*
void search(const faiss::IndexFlatL2& index, const map<long, int>& faiss_id_to_doc_id, const vector<float>& query_embedding, int k) {
    long* I = new long[k];
    float* D = new float[k];

    index.search(1, query_embedding.data(), k, D, I); // 
    cout << "Results for top " << k << " results: " << endl;
    for (int i = 0; i < k; i++) {
        long faiss_id = I[i];
        int id = faiss_id_to_doc_id.at(faiss_id);
        float distance = D[i];
        cout << "Document ID: " << id << ", Distance: " << distance << endl;
    }
    delete[] I;
    delete[] D;

}    */

/*int main(int argc, char ** argv){
    // Initialize FAISS index
    const int dimension = 768;
    faiss::IndexFlatL2 index(dimension);
    //faiss::IndexFlatL2 index;
    map<long, int> faiss_id_to_doc_id;
    string file_path = "preprocessed_documents.json";
    std::cout << "Loading embeddings from " << file_path << std::endl;
    // this takes around 10 seconds
    std::ifstream f(file_path);
    if (!f.is_open()){
        std::cerr << file_path << "could not be opened" << std::endl;
        // loser
        return 1;
    }
    json data = json::parse(f);
    vector<float> all_embeddings;
    long faiss_id = 0;
    for (const auto& item : data) {
        int id = item["id"].get<int>();
        vector<float> embedding = item["embedding"].get<std::vector<float>>();
        all_embeddings.insert(all_embeddings.end(), embedding.begin(), embedding.end());
        faiss_id_to_doc_id[faiss_id] = id;
        faiss_id++;
    }
    const long num_documents = faiss_id_to_doc_id.size();
    cout << "Loaded " << num_documents << " documents." << endl;
    index.add(num_documents, all_embeddings.data());
    cout << "FAISS built successfully" << endl;
    
    cout << "TESTING item 42" << endl;
    vector<float> test_query;
    test_query.insert(test_query.end(), all_embeddings.begin() + 42 * dimension, all_embeddings.begin() + (42 + 1) * dimension);
    int origin_id = faiss_id_to_doc_id.at(42);
    cout << "Original Document ID: " << origin_id << endl;
    search(index, faiss_id_to_doc_id, test_query, 5);
    return 0;
}*/