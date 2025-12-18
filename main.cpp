#include <iostream>
#include <string>
#include "encode.hpp"
#include "vector_db.hpp"
#include <stdexcept>
#include <vector>
#include <sstream>
#include "llm_generation.hpp"
#include <chrono>
#include <bits/stdc++.h>
#include <map>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


std::string create_prompt(const std::string& query, const std::vector<SearchResult>& results){
    std::stringstream ss;

    /*
    ss << "Question: " << query << "\n";
    ss << "Context from relevant documents (Top documents): ";
    for (const auto& res : results) {
        ss << "\n[Document ID: " << res.id << "] " << res.text;
    }
    ss << "\nAnswer the question based only on the provided context.";*/


    // TinyLlama Chat Format
    // <|system|>You are a helpful AI.</s><|user|>Context... Question...</s><|assistant|>
    
   // Use the exact TinyLlama chat format with NO extra spaces
   
   ss << "<|system|>\n"
       << "You are a helpful assistant. Answer the question using only the context provided.\n" 
       // << "Be extremely concise. Answer in one sentence if possible.\n" // <--- ADD THIS
       << "</s>\n"
       << "<|user|>\n"
       << "Context:\n";
    
    for (const auto& res : results) {
        ss << res.text << "\n";
    }
    
    ss << "\nQuestion: " << query << "\n"
       << "</s>\n"
       << "<|assistant|>\n"; // End exactly here so the model starts writing the answer immediately
       
    return ss.str();
    
    // Kevin Patullo is a terrorist to the city of Philadelphia
    // how do you blow a 21 point lead...to the cowboys
}

/*
    Returns the query given an ID
*/
std::vector<std::string> getTestQueries(const std::string& json_path){
    std::ifstream f(json_path);
    if (!f.is_open()){
        std::cerr << json_path << " could not be opened. Please check that document path is correct." << std::endl; 
        throw std::runtime_error("Failed to open JSON file");
    }
    std::vector<std::string> queries;
    json data = json::parse(f);
    std::vector<int> id_list =  {0, 2, 4, 17, 20, 43, 53, 69}; // list of ids with the queries to test
    std::map<int, std::string> found_map;
    for (const auto& item : data){
        int id = item["id"].get<int>();
        //if (std::find(id_list.begin(), id_list.end(), id) != id_list.end()){
        //    found_map[id] = item["text"].get<std::string>();
        //} 
        //if (found_map.size() >= id_list.size()){
        //    break;
        //}
        found_map[id] = item["text"].get<std::string>();

    }

    for (int id : id_list){
        if (found_map.count(id) > 0){
            queries.push_back(found_map[id]);
        }
    }
    std::cout << "Loaded " << queries.size() << " test queries." << std::endl;
    return queries;

}

// Helper to load ALL queries from the JSON file
std::vector<std::string> load_all_queries(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()){
        std::cerr << json_path << " could not be opened." << std::endl; 
        throw std::runtime_error("Failed to open JSON file");
    }
    std::vector<std::string> queries;
    json data = json::parse(f);
    
    // Load every query available in the file
    for (const auto& item : data){
        queries.push_back(item["text"].get<std::string>());
    }
    std::cout << "Loaded " << queries.size() << " unique queries from file." << std::endl;
    return queries;
}

void run_batch_benchmark(QueryEncoder& encoder, VectorDatabase& db, const std::string& query_path) {
    std::cout << "\n--- Starting Batch Size Benchmark (k=1) ---" << std::endl;
    
    // 1. Load Queries
    std::vector<std::string> unique_queries = load_all_queries(query_path);
    if (unique_queries.empty()) {
        std::cerr << "No queries found!" << std::endl;
        return;
    }

    // 2. Prepare a large dataset by repeating queries
    // We need enough queries to ensure even the largest batch (128) runs at least a few times
    // Let's create a pool of 512 queries (4 batches of 128)
    int target_total_queries = 512;
    std::vector<std::string> test_queries;
    
    while (test_queries.size() < target_total_queries) {
        test_queries.insert(test_queries.end(), unique_queries.begin(), unique_queries.end());
    }
    // Trim to exact size
    test_queries.resize(target_total_queries);
    std::cout << "expanded test set to " << test_queries.size() << " queries for benchmarking.\n" << std::endl;

    // 3. Define Batch Sizes
    // The README typically asks for 1, 4, 8, 16, 32, 64, 128
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};
    int k = 1;

    // 4. Run Benchmark
    std::ofstream csv_file("batch_benchmark_results.csv");
    csv_file << "BatchSize,AvgTimePerBatch(ms),AvgTimePerQuery(ms),Throughput(QPS)\n";

    for (int b : batch_sizes) {
        double total_search_time = 0.0;
        int batches_run = 0;

        // Process the entire test_queries vector in chunks of size 'b'
        for (size_t i = 0; i < test_queries.size(); i += b) {
            
            // Handle last partial batch if necessary (though we resized to 512, so 128 divides evenly)
            int current_batch_size = std::min((int)b, (int)(test_queries.size() - i));
            
            // A. Encode (We don't time this for the SEARCH benchmark)
            std::vector<float> batch_flat;
            for (size_t j = i; j < i + current_batch_size; j++) {
                std::vector<float> query_vec = encoder.encode(test_queries[j]);
                batch_flat.insert(batch_flat.end(), query_vec.begin(), query_vec.end());
            }

            // B. Time the Search
            auto start_search = std::chrono::high_resolution_clock::now();
            auto batch_results = db.search_batch(batch_flat, current_batch_size, k);
            auto end_search = std::chrono::high_resolution_clock::now();

            double batch_duration = std::chrono::duration<double, std::milli>(end_search - start_search).count();
            
            total_search_time += batch_duration;
            batches_run++;
        }

        // 5. Calculate Metrics
        double avg_time_per_batch = total_search_time / batches_run;
        double avg_time_per_query = total_search_time / test_queries.size();
        double throughput_qps = 1000.0 / avg_time_per_query; // 1000 ms / time_per_query_ms

        // Print to Console
        std::cout << "Batch Size: " << b 
                  << " | Avg Batch Time: " << avg_time_per_batch << " ms"
                  << " | Avg Query Time: " << avg_time_per_query << " ms" 
                  << " | QPS: " << throughput_qps << std::endl;

        // Save to CSV
        csv_file << b << "," << avg_time_per_batch << "," << avg_time_per_query << "," << throughput_qps << "\n";
    }

    csv_file.close();
    std::cout << "\nBenchmark complete. Data saved to batch_benchmark_results.csv" << std::endl;
}

// simulate baseline

void run_baseline(QueryEncoder& encoder, VectorDatabase& db, LLMGenerator& generator, const std::string& query_path, int batch_size){
    std::cout << "Running in Benchmark Mode" << std::endl;
    std::vector<std::string> test_queries = getTestQueries(query_path);

    std::ofstream csv_file("benchmark_results.csv");
    csv_file << "Query,Encoding Time (ms),Search Time (ms),Generation Time (ms),Total Time (ms)\n";

    

    std::vector<int> ann_values = {1};

    std::vector<int> batch_sizes = {1,2,4,8,16,32,64,128};

    for (int b : batch_sizes){
        for (int k : ann_values){
            csv_file << "\n"; // separate different k values
            csv_file << "k=" << k << "\n";
            for (size_t i = 0; i < test_queries.size(); i += b){
                // prepare batch

                std::vector<float> batch_flat;
                int current_batch_size = 0;
                for (size_t j = i; j < i + b && j < test_queries.size(); j++){
                    std::vector<float> query_vec = encoder.encode(test_queries[j]);
                    batch_flat.insert(batch_flat.end(), query_vec.begin(), query_vec.end());
                    current_batch_size++;
                }
                std::cout << "Searching batch of size " << current_batch_size << " with k=" << k << std::endl;
                auto start_search = std::chrono::high_resolution_clock::now();
                auto batch_results = db.search_batch(
                    batch_flat, current_batch_size, k);
                auto end_search = std::chrono::high_resolution_clock::now();

                double time_search = std::chrono::duration<double, std::milli>(end_search- start_search).count();
                double time_per_query = time_search / current_batch_size;

                //csv_file << "Batch starting at query index " << i << ", , ," << time_per_query << "," << time_search << "\n"; 
                //std::cout << "Batch search complete. Batch size: "<< b <<" Average time per query: " << time_per_query << " ms" <<std::endl;
                // csv_file " Batch starting at query index " << i << ", , ," << time_per_query << "," << time_search << "\n";
                


            
                /*
                auto start_enc = std::chrono::high_resolution_clock::now();
                std::vector<float> query_vec = encoder.encode(query);
                auto end_enc = std::chrono::high_resolution_clock::now();
                double time_enc = std::chrono::duration<double, std::milli>(end_enc - start_enc).count();

                auto start_search = std::chrono::high_resolution_clock::now();
                std::vector<SearchResult> results = db.search(query_vec, k);
                auto end_search = std::chrono::high_resolution_clock::now();
                double time_search = std::chrono::duration<double, std::milli>(end_search- start_search).count();
        
                std::string prompt = create_prompt(query, results);
                auto start_gen = std::chrono::high_resolution_clock::now();
                std::string response = generator.generate(prompt);
                auto end_gen = std::chrono::high_resolution_clock::now();
                double time_gen = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();

                std::cout << "\n--- Response ---" << std::endl;
                std::cout << response << std::endl;
                std::cout << "----------------" << std::endl;

                std::cout << "\n[Metrics] Encoding: " << time_enc << "ms | Search: " << time_search << "ms | Generation: " << time_gen << "ms" << std::endl;

                double total_time = time_enc + time_search + time_gen;
                csv_file << "\"" << query << "\"," << time_enc << "," << time_search << "," << time_gen << "," << total_time << "\n";*/
                
            }
        
    }
        

        csv_file << "\n\nBatch Size = " << b << "\n";
    }
    
    csv_file.close();
    std::cout << "Benchmarking complete! Results saved to benchmark_results.csv" << std::endl;
}


int main(int argc, char* argv[]){

    // parameter guide:
    // argv[0] = boolean that controls whether to use IVF index or not
    // argv[1] = which llm model number to use
    // argv[2] = batch size, default to 1 if not provided :)


    // 1. Initialize Components

    std::vector<std::string> llm_models = {
        "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "qwen2-1_5b-instruct-q4_0.gguf"
    };
    int llm_model_num = 0; // default to first model
    bool ivf = true; // default to not using IVF index
    int batch_size = 1; // default batch size

    /*
    if (argc == 1){
        std::cerr << "Please provide whether to use IVF index (0 or 1) and LLM model number (0, 1, or 2). You provided: " << argc << " arguments" << std::endl;
        return 1;
    }
    else if (argc == 3 || argc == 4){
        ivf = argv[1] == std::string("1") ? true : false;
        llm_model_num = std::stoi(argv[2]);
        if (llm_model_num < 0 || llm_model_num >= llm_models.size()){
            std::cerr << "Invalid LLM model number. Please provide 0, 1, or 2." << std::endl;
            return 1;
        }
        if (argc == 4){
            batch_size = std::stoi(argv[3]);
        }
        if (batch_size <= 0){
            std::cerr << "Invalid batch size. Please provide a positive integer." << std::endl;
            return 1;
        }   
    }
    else{
        std::cerr << "Invalid number of arguments. Please provide whether to use IVF index (0 or 1) and LLM model number (0, 1, or 2) You provided: " << argc << " arguments" << std::endl;
        return 1;
    }*/

    std::cout << "Using LLM Model Number: " << llm_model_num << " (" << llm_models[llm_model_num] << ")" << std::endl;
    std::cout << "Using IVF Index: " << (ivf ? "Yes" : "No") << std::endl;

    std::string model_path = "bge-base-en-v1.5-f32.gguf";
    std::string json_path = "preprocessed_documents.json";
    std::string llm_model_path = llm_models[llm_model_num];
    std::string query_path = "queries.json";




    std::cout << "--- Initializing System --- Please wait. Perhaps you should get up and touch some grass??" << std::endl;
    
    std::cout << "Loading all the queries from " << query_path << std::endl;
    // load all queries 
    std::ifstream f(json_path);
    if (!f.is_open()){
        std::cerr << json_path << " could not be opened. Please check that document path is correct." << std::endl; 
        throw std::runtime_error("Failed to open JSON file");
    }
    std::cout << "Loading Query Encoder Model from " << model_path << std::endl;
    QueryEncoder encoder(model_path);
    std::cout << "Loading Vector Database from " << json_path << std::endl;
    VectorDatabase db(json_path, ivf);
    std::cout << "Loading LLM Model from " << llm_model_path << std::endl;
    int context_size = 4096;
    if (llm_model_num == 0){
        std::cout << "Using context size of 2048 for TinyLlama" << std::endl;
        context_size = 2048;
    }
    
    LLMGenerator generator(llm_model_path, context_size);

    std::cout << "System Initialization Complete! Type 'exit' or 'quit' to stop so you can go get some coffee" << std::endl;

    // please help pay for my tuition :)

    bool benchmark_mode = false;
    //run_baseline(encoder, db, generator, query_path, batch_size);
    bool use_ivf = true; 

    std::cout << "--- Initializing Benchmark System ---" << std::endl;
    
    try {
        QueryEncoder encoder(model_path);
        VectorDatabase db(json_path, use_ivf);
        
        // We don't need the LLM for the vector search benchmark
        // Pass dummy args if your run_batch_benchmark signature needed them, 
        // but the one above only needs encoder and db.
        
        run_batch_benchmark(encoder, db, query_path);
        
        } catch (const std::exception& e) {
            std::cerr << "Critical Error: " << e.what() << std::endl;
            return 1;
        }

    return 0;



    /*
    if (benchmark_mode){
        std::cout << "Running in Benchmark Mode" << std::endl;
        std::vector<std::string> test_queries = getTestQueries(query_path);

        std::ofstream csv_file("benchmark_results.csv");
        csv_file << "Query,Encoding Time (ms),Search Time (ms),Generation Time (ms),Total Time (ms)\n";
        for (const auto& query : test_queries){
            std::cout << "\n--- Query: " << query << " ---" << std::endl;

            auto start_enc = std::chrono::high_resolution_clock::now();
            std::vector<float> query_vec = encoder.encode(query);
            auto end_enc = std::chrono::high_resolution_clock::now();
            double time_enc = std::chrono::duration<double, std::milli>(end_enc - start_enc).count();

            auto start_search = std::chrono::high_resolution_clock::now();
            std::vector<SearchResult> results = db.search(query_vec, 3);
            auto end_search = std::chrono::high_resolution_clock::now();
            double time_search = std::chrono::duration<double, std::milli>(end_search- start_search).count();
    
            std::string prompt = create_prompt(query, results);
            auto start_gen = std::chrono::high_resolution_clock::now();
            std::string response = generator.generate(prompt);
            auto end_gen = std::chrono::high_resolution_clock::now();
            double time_gen = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();

            std::cout << "\n--- Response ---" << std::endl;
            std::cout << response << std::endl;
            std::cout << "----------------" << std::endl;

            std::cout << "\n[Metrics] Encoding: " << time_enc << "ms | Search: " << time_search << "ms | Generation: " << time_gen << "ms" << std::endl;

            double total_time = time_enc + time_search + time_gen;
            csv_file << "\"" << query << "\"," << time_enc << "," << time_search << "," << time_gen << "," << total_time << "\n";
            
        }
        csv_file.close();
        std::cout << "Benchmarking complete! Results saved to benchmark_results.csv" << std::endl;
        return 0;
    }*/
    bool run_user_loop = false;
    /*
    while (run_user_loop){
        // get stuck in a loop asking for user input until the user gets tired or something idk
        std::cout << "\nEnter your query: ";
        std::string query;
        if (!std::getline(std::cin, query)){
            break; // EOF or error
        }
        if (query == "exit" || query == "quit"){
            break; // user wants to exit
        }
        if (query.empty()){
            std::cout << "Please enter a non-empty query...I thought it was a bit obvious to not type a non-empty query but you do you I guess" << std::endl;
            continue;
        }

        // Ruby-chan?
        // hai
        // nani ga suki?
        // Chokominto yori mo a-na-ta

        // Ayumu-chan?
        // hai
        // nani ga suki?
        // Sutoroberii fureibaa yori mo a-na-ta

        // Shiki-chan?
        // Hai!
        // Nani ga suki?
        // Kukkii and kuriimu yori mo a-na-ta

        // Minna?
        // Hai!
        // Nani ga suki?
        // Mochiron daisuki aisukuriimu

        std::cout << "Thinking really really hard about your query...please be patient I'm not GPT 5" << std::endl;

        // 1. Measuring Encoding process
        auto start_enc = std::chrono::high_resolution_clock::now();
        std::vector<float> query_vec = encoder.encode(query);
        auto end_enc = std::chrono::high_resolution_clock::now();
        double time_enc = std::chrono::duration<double, std::milli>(end_enc - start_enc).count();


        // search time
        auto start_search = std::chrono::high_resolution_clock::now();
        std::vector<SearchResult> results = db.search(query_vec, 3);
        auto end_search = std::chrono::high_resolution_clock::now();
        double time_search = std::chrono::duration<double, std::milli>(end_search - start_search).count();
        // std::cout << "Retrieved " << results.size() << " documents." << std::endl;


        // negligible aka irrelevant like the new york jets
        std::string prompt = create_prompt(query, results);

        // measure generation time
        auto start_gen = std::chrono::high_resolution_clock::now();
        std::string response = generator.generate(prompt);
        auto end_gen = std::chrono::high_resolution_clock::now();
        double time_gen = std::chrono::duration<double, std::milli>(end_gen - start_gen).count();

        std::cout << "\n--- Response ---" << std::endl;
        std::cout << response << std::endl;
        std::cout << "----------------" << std::endl;


        std::cout << "\n[Metrics] Encoding: " << time_enc << "ms | Search: " << time_search << "ms | Generation: " << time_gen << "ms" << std::endl;
    }*/

    // https://www.jimbobobbytree.com/

    return 0;

    /*
    all right yall that's gonna do it for this assignment
    be sure to subscribe to my channel

    and my second channel
    and my third channel
    and my vlog channel
    and my gaming channel
    and my streaming channel
    and my clips channel
    and my
    and my 
    and my
    and my
    and my fitness channel
    and my music channel 
    and my react channel
    and my shorts channel
    and my podcast channel 
    and my deluxe channel
    and my backup channel
    and my channel channel

    and my extras channel that I don't even upload on but still got a million subscribers on anyway
    */
}