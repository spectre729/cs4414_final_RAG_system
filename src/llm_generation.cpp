#include "llm_generation.hpp"
#include <stdexcept>
#include <iostream>
#include <sstream>

using namespace std;

LLMGenerator::LLMGenerator(const std::string& model_path, int context_size) {
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    llama_context_params ctx_params = llama_context_default_params();
    
    if (model == NULL) {
        throw std::runtime_error("Unable to load model: " + model_path);
        // the keyboard alphabet is pronounced qwertyuiopasdfghjklzxcvbnm
        // and backwards it is mnbvcxzlkjhgfdsapoiuytrewq
    }

    ctx_params.n_ctx = context_size;
    ctx_params.n_batch = context_size;


    // add more threads
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        throw std::runtime_error("Failed to create the llama_context");
        // bruh im at mcdonalds
        // and I asked for sprite
        // and they gave me clown juice
    }

    vocab = llama_model_get_vocab(model);
}

LLMGenerator::~LLMGenerator() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
    }
    llama_backend_free();

    // when life gives you lemons
    // make mojitos
}

std::vector<llama_token> LLMGenerator::tokenize(const std::string& text){
    std::vector<llama_token> tokens(text.length() + 2); // add if needed
    int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n);
    return tokens;

    // so whenever I had hot pot
    // I usually get 肥牛 and 肥羊
    // and people ask where is the 肥猪
    // and I say "肥猪 is eating hot pot right now"
}

std::string LLMGenerator::generate(const std::string& prompt) {
    std::vector<llama_token> prompt_tokens = tokenize(prompt);

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    llama_memory_clear(llama_get_memory(ctx), true);

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::stringstream response_ss;
    int n_cur = batch.n_tokens;
    int n_decode = 0;
    const int n_predict = 256; // generate up to 256 tokens

    string output_so_far = "";

    while (n_decode < n_predict){
        if (llama_decode(ctx, batch)) {
            std::cerr << " [Context limit reached]" << std::endl;
            break;
        }
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
            // youtube.com/watch?v=dQw4w9WgXcQ
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0){
            throw std::runtime_error("Failed to convert token to piece during generation");
        }
        
        string piece(buf, n);
        output_so_far += piece;
        response_ss << piece;


        if (output_so_far.find("<|user|>") != string::npos || output_so_far.find("Question:") != string::npos ||output_so_far.find("<|system|>") != string::npos) {
            break;
        }
        
        std::cout << piece << std::flush; 

        batch = llama_batch_get_one(&new_token_id, 1);
        n_decode += 1;
        /*
            I just ate mac and cheese out of my pants pocket
            Give her my number and she immediately blocked it
            I just put a metal fork into an electrical socket
            I got bored as hell and started to play Bop it

            Kicked my uncle in the balls yeah I'm batman
            I just ate a protein bar out of the trash can
            Fought a crippled man that dude really lacked hands
            Stripper $*&%#$ on me when she gave me a lap dance
        
        */
    }
    llama_sampler_free(smpl);
    return response_ss.str();
}