/*
 * main.cpp
 * Command-line entry point for the RAG system.
 *
 *   rag_system                 interactive question answering (retrieve + generate)
 *   rag_system --benchmark     measure vector-search throughput for several batch sizes
 *
 * Run `rag_system --help` for all options.
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "encode.hpp"
#include "llm_generation.hpp"
#include "vector_db.hpp"

using json = nlohmann::json;

namespace {

struct Config {
    std::string embed_model_path = "bge-base-en-v1.5-f32.gguf";
    std::string llm_model_path = "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf";
    std::string documents_path = "preprocessed_documents.json";
    std::string queries_path = "queries.json";
    std::string benchmark_output_path = "batch_benchmark_results.csv";
    int context_size = 2048;
    int top_k = 3;
    int nprobe = 10;
    bool use_ivf = true;
    bool benchmark = false;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Modes:\n"
        << "  (default)             Interactive question answering\n"
        << "  --benchmark           Benchmark batched vector search (does not load the LLM)\n\n"
        << "Options:\n"
        << "  --docs PATH           Embedded documents (default: preprocessed_documents.json)\n"
        << "  --queries PATH        Queries used by --benchmark (default: queries.json)\n"
        << "  --embed-model PATH    BGE embedding model (default: bge-base-en-v1.5-f32.gguf)\n"
        << "  --llm PATH            Chat model, TinyLlama prompt format (default: tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf)\n"
        << "  --ctx N               LLM context size in tokens (default: 2048)\n"
        << "  --top-k N             Documents retrieved per question (default: 3)\n"
        << "  --flat                Use an exact flat index instead of IVF\n"
        << "  --nprobe N            IVF clusters searched per query (default: 10)\n"
        << "  --output PATH         CSV file written by --benchmark (default: batch_benchmark_results.csv)\n"
        << "  -h, --help            Show this help\n";
}

// Returns false if the program should exit (after --help or on a usage error).
bool parse_args(int argc, char** argv, Config& config, int& exit_code) {
    auto value_of = [&](int& i, const std::string& flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value for " + flag);
        }
        return argv[++i];
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                exit_code = 0;
                return false;
            } else if (arg == "--benchmark") {
                config.benchmark = true;
            } else if (arg == "--flat") {
                config.use_ivf = false;
            } else if (arg == "--docs") {
                config.documents_path = value_of(i, arg);
            } else if (arg == "--queries") {
                config.queries_path = value_of(i, arg);
            } else if (arg == "--embed-model") {
                config.embed_model_path = value_of(i, arg);
            } else if (arg == "--llm") {
                config.llm_model_path = value_of(i, arg);
            } else if (arg == "--output") {
                config.benchmark_output_path = value_of(i, arg);
            } else if (arg == "--ctx") {
                config.context_size = std::stoi(value_of(i, arg));
            } else if (arg == "--top-k") {
                config.top_k = std::stoi(value_of(i, arg));
            } else if (arg == "--nprobe") {
                config.nprobe = std::stoi(value_of(i, arg));
            } else {
                throw std::invalid_argument("Unknown option: " + arg);
            }
        }
        if (config.context_size <= 0 || config.top_k <= 0 || config.nprobe <= 0) {
            throw std::invalid_argument("--ctx, --top-k and --nprobe must be positive integers");
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        exit_code = 1;
        return false;
    }
    return true;
}

// Builds a TinyLlama chat prompt: retrieved documents as context, followed by the question.
// The prompt ends right after the assistant tag so the model starts answering immediately.
std::string create_prompt(const std::string& query, const std::vector<SearchResult>& results) {
    std::stringstream ss;
    ss << "<|system|>\n"
       << "You are a helpful assistant. Answer the question using only the context provided.\n"
       << "</s>\n"
       << "<|user|>\n"
       << "Context:\n";

    for (const auto& res : results) {
        ss << res.text << "\n";
    }

    ss << "\nQuestion: " << query << "\n"
       << "</s>\n"
       << "<|assistant|>\n";
    return ss.str();
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point start,
                  std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Loads the "text" field of every entry in a JSON array of {"id", "text"} objects.
std::vector<std::string> load_queries(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        throw std::runtime_error(json_path + " could not be opened. Please check that the path is correct.");
    }

    std::vector<std::string> queries;
    for (const auto& item : json::parse(f)) {
        queries.push_back(item.at("text").get<std::string>());
    }
    std::cout << "Loaded " << queries.size() << " queries from " << json_path << "." << std::endl;
    return queries;
}

// Measures vector-search latency and throughput for increasing batch sizes (k = 1).
// Encoding time is excluded so the numbers reflect the search stage only.
void run_batch_benchmark(QueryEncoder& encoder, VectorDatabase& db, const Config& config) {
    std::cout << "\n--- Batch Size Benchmark (k=1) ---" << std::endl;

    std::vector<std::string> unique_queries = load_queries(config.queries_path);
    if (unique_queries.empty()) {
        throw std::runtime_error("No queries found in " + config.queries_path);
    }

    // Repeat the queries until there are enough for the largest batch to run several times.
    const size_t target_total_queries = 512;
    std::vector<std::string> test_queries;
    while (test_queries.size() < target_total_queries) {
        test_queries.insert(test_queries.end(), unique_queries.begin(), unique_queries.end());
    }
    test_queries.resize(target_total_queries);
    std::cout << "Expanded test set to " << test_queries.size() << " queries.\n" << std::endl;

    const std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};
    const int k = 1;

    std::ofstream csv_file(config.benchmark_output_path);
    if (!csv_file.is_open()) {
        throw std::runtime_error("Could not write " + config.benchmark_output_path);
    }
    csv_file << "BatchSize,AvgTimePerBatch(ms),AvgTimePerQuery(ms),Throughput(QPS)\n";

    for (int b : batch_sizes) {
        double total_search_ms = 0.0;
        int batches_run = 0;

        for (size_t i = 0; i < test_queries.size(); i += b) {
            const size_t batch_end = std::min(test_queries.size(), i + b);

            std::vector<float> batch_flat;
            for (size_t j = i; j < batch_end; ++j) {
                std::vector<float> query_vec = encoder.encode(test_queries[j]);
                batch_flat.insert(batch_flat.end(), query_vec.begin(), query_vec.end());
            }

            auto start = std::chrono::high_resolution_clock::now();
            db.search_batch(batch_flat, static_cast<int>(batch_end - i), k);
            auto end = std::chrono::high_resolution_clock::now();

            total_search_ms += elapsed_ms(start, end);
            ++batches_run;
        }

        const double avg_time_per_batch = total_search_ms / batches_run;
        const double avg_time_per_query = total_search_ms / test_queries.size();
        const double throughput_qps = 1000.0 / avg_time_per_query;

        std::cout << "Batch Size: " << b
                  << " | Avg Batch Time: " << avg_time_per_batch << " ms"
                  << " | Avg Query Time: " << avg_time_per_query << " ms"
                  << " | QPS: " << throughput_qps << std::endl;
        csv_file << b << "," << avg_time_per_batch << "," << avg_time_per_query << "," << throughput_qps << "\n";
    }

    std::cout << "\nBenchmark complete. Results saved to " << config.benchmark_output_path << std::endl;
}

// Reads questions from stdin and answers each one with retrieval-augmented generation.
void run_interactive(QueryEncoder& encoder, VectorDatabase& db, LLMGenerator& generator, const Config& config) {
    std::cout << "Ready. Type a question, or 'exit' to quit." << std::endl;

    std::string query;
    while (true) {
        std::cout << "\nEnter your query: " << std::flush;
        if (!std::getline(std::cin, query) || query == "exit" || query == "quit") {
            break;
        }
        if (query.empty()) {
            std::cout << "Please enter a non-empty query." << std::endl;
            continue;
        }

        auto start_enc = std::chrono::high_resolution_clock::now();
        std::vector<float> query_vec = encoder.encode(query);
        auto end_enc = std::chrono::high_resolution_clock::now();

        auto start_search = std::chrono::high_resolution_clock::now();
        std::vector<SearchResult> results = db.search(query_vec, config.top_k);
        auto end_search = std::chrono::high_resolution_clock::now();

        std::cout << "Retrieved " << results.size() << " document(s):";
        for (const auto& res : results) {
            std::cout << " [id " << res.id << ", dist " << res.distance << "]";
        }
        std::cout << std::endl;

        std::string prompt = create_prompt(query, results);

        std::cout << "\n--- Response ---" << std::endl;
        auto start_gen = std::chrono::high_resolution_clock::now();
        generator.generate(prompt);
        auto end_gen = std::chrono::high_resolution_clock::now();
        std::cout << "\n----------------" << std::endl;

        std::cout << "[Metrics] Encoding: " << elapsed_ms(start_enc, end_enc)
                  << " ms | Search: " << elapsed_ms(start_search, end_search)
                  << " ms | Generation: " << elapsed_ms(start_gen, end_gen) << " ms" << std::endl;
    }
}

}  // namespace

int main(int argc, char** argv) {
    Config config;
    int exit_code = 0;
    if (!parse_args(argc, argv, config, exit_code)) {
        return exit_code;
    }

    try {
        std::cout << "--- Initializing system (this may take a moment) ---" << std::endl;
        std::cout << "Loading query encoder from " << config.embed_model_path << std::endl;
        QueryEncoder encoder(config.embed_model_path);
        VectorDatabase db(config.documents_path, config.use_ivf, config.nprobe);

        if (config.benchmark) {
            run_batch_benchmark(encoder, db, config);
        } else {
            std::cout << "Loading LLM from " << config.llm_model_path << std::endl;
            LLMGenerator generator(config.llm_model_path, config.context_size);
            run_interactive(encoder, db, generator, config);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
