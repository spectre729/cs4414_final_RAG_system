#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>


using json = nlohmann::json;
using namespace std;


int main(int argc, char ** argv) {

    // reads MS MARCO dataset
    std::ifstream f("documents.json");
    if (!f.is_open()){
        std::cerr << "documents.json could not be opened. Please check that document path is correct." << std::endl;
        // loser
        return 1;
    }

    json data;
    try{
        data = json::parse(f);
    }
    catch (json::parse_error& e){
        std::cerr << "documents.json could not be parse" << e.what() << std::endl;
        // double loser
        return 1;
    }


    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 512;

    // adrenaline in my soul
    // something something cody rhodes
    // ????? 
    // WHOAAAAAAAAAAAAAAAAAAAA

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file("bge-base-en-v1.5-f32.gguf", model_params);
    
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
        // triple loser
    }
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        // so i straight up ctrl c ctrl v it from recitation 9...
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);


    json output_data = json::array();

    for (const auto& item: data){
        int id = item.at("id");
        // at this point, i got tired of typing std:: everytime
        string text = item.at("text");
        cout << "ID: " << id << " TEXT: " << text.substr(0,40) << " ... " << endl;

        vector<llama_token> tokens(text.size());
        int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
        tokens.resize(n_tokens);

        llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
        // consider using a batch for processing multiple documents

        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            // multi loser HAHAHAHAHAHAHAHAHAHAHA
            return 1;
        }

        float * embd = llama_get_embeddings_seq(ctx, 0);
        int n_embd = llama_model_n_embd(model);
        if (embd == NULL){
            fprintf(stderr, "%s : failed to get embeddings\n", __func__);
            return 1;
        }
        if (n_embd != 768){
            fprintf(stderr, "%s : Embedding is not of length 768...go put the fries in the bag bro.");
            return 1;
        }

        vector<float> embedding_vector(embd, embd + n_embd);
        json new_item;
        new_item["id"] = id;
        new_item["text"] = text;
        new_item["embedding"] = embedding_vector;
        output_data.push_back(new_item);

        // 4 letters
        // 1 word
        // ugh ugh
        // YEET
        }
       


    // add embedding logic

    std::ofstream o("preprocessed_documents.json");
    o << std::setw(2) << output_data << std::endl;
    o.close();

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}