# CS4414 Fall 2025: HW3, RAG (Retrieval Augmented Generation)

4 weeks of homework.  One (long) assignment document.

# Due dates:
- Part 1: Nov 15.  (1 weeks)
- Part 2: Nov 25.  (1.5 weeks)
- Part 3: Dec 5. (1 week)


# Overview

This homework spans four weeks and consists of three parts. In Part 1, you will build one of the core components of a Retrieval-Augmented Generation (RAG) system: a vector database for semantic document search. In Part 2, you will integrate this component into a complete RAG pipeline that performs end-to-end retrieval and generation. Finally, in Part 3, you will analyze your system’s performance, identify bottlenecks, and explore optimizations.

Start early! If you finish early, that will be great and you can relax.  But this is way better than starting late and not finishing at all.  When it is time to turn something in, you should upload what you have at that point. When it’s time to submit, upload what you have. There's no starter code, and you're the architect to build your own system from the ground up. 

## Programing language: 
You may implement this project in either C++ or Python. Choose the language you are more comfortable with.  You can even use a mix if you like (there are several packages.  Python “bindings” is probably the most popular but all of them are heavily used).


## What You'll Build and Why It Matters

By completing this assignment, you'll:
- Build a complete Retrieval-Augmented Generation (RAG) system, the same technology powering ChatGPT's web search, enterprise AI assistants, and modern search engines
- Work with industry-standard tools (FAISS, llama.cpp) used by companies like Meta, Google, and OpenAI
- Learn to measure and optimize real AI systems—skills that translate directly to ML engineering roles
- Understand the critical tradeoffs between accuracy, speed, and cost that every production system faces

Think of this as your portfolio project. When you finish, you'll have built something genuinely impressive that you can demo in interviews!

# Background

## The Problem RAG Solves
Imagine asking ChatGPT: "What's our company's Q4 revenue target?"
ChatGPT won’t be able to answer because it was never trained on your company's internal data. Even if it was trained on public information about your company, that data is now months or years out of date. 

This is where Retrieval-Augmented Generation (RAG) comes in. Instead of relying solely on what the model memorized during training, RAG systems:
1.	Retrieve relevant information from a knowledge base (your documents, databases, files)
2.	Augment the user's question with this fresh, factual context
3.	Generate a response grounded in real information
It's like taking an open-book exam instead of a closed-book one. The model is still doing the thinking, but now it can just look up the answer in the reference materials, then format it in a way that matches what people with this kind of query tend to like.



## What is RAG?
Retrieval-Augmented Generation (RAG) is a framework that improves language model output by grounding the output generative stage using relevant external information. When a user submits a question, the system first retrieves related content from a knowledge base and then provides that content as context to the large language model (LLM). The model, now “aware” of the retrieved material, generates a more factual and contextually grounded response.  The technique is very widely used and important ([NVIDIA post](https://www.nvidia.com/en-us/glossary/retrieval-augmented-generation/)).

In contrast, although modern LLMs like ChatGPT, Llama, and Qwen can produce high-quality answers, there is a limit to how much they can memorize during training.  As a result, they are prone to either hallucinating or provide outdated information if they lack sufficient context. 

Research has shown that providing high-quality, directly-relevant context during generation can substantially improve model performance ([Google blog](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/?utm_source=chatgpt.com)).  RAG addresses this by:
- Grounding responses in facts: Retrieving relevant documents ensures the model has accurate information ([RankRAG](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf?utm_source=chatgpt.com));
- Extending knowledge: Injecting external context allows the model to answer questions about information outside its training data ([openAI post](https://help.openai.com/en/articles/8868588-retrieval-augmented-generation-rag-and-semantic-search-for-gpts?utm_source=chatgpt.com));
- Improving reliability: Carefully selected high-quality context often yields better accuracy than simply adding more data during training or letting the LLM itself search directly in the knowledge base ([microsoft research post](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/?utm_source=chatgpt.com)).

In effect, a RAG system pairs the LLM with a database that supports a vector index to enable approximate match.  Now the LLM can generate a vector search request and the index-search algorithm will hunt through the preloaded repository of documents for K documents that constitute a representative set of the closest matches to the query in the high-dimensional space that we use for these tasks (K nearest neighbors is costly and in any case, the task is approximate, so we use an approximate search: ANN, and assume that if there are a lot of matching documents, a random sample will be fine).  

Homework 1 was actually a first exposure to how this form of indexing is implemented, so you already know a little about it!


## How RAG Works: A Real Example

![Figure1](figure1.png)
Figure 1: RAG pipeline


Let's say you're building an AI assistant for Cornell students. In Figure1, someone asks: “What’s Cornell’s policy on course add/drop for 2025?”  
Here's what your RAG system does behind the scenes:

### Step 1. Document retrieval:
- 1.1: Encode the question into a 768-dimensional vector that captures its semantic meaning (not just keywords, but what the person is actually asking about)
- 1.2: Search a vector database of Cornell policy documents to find the chunks most directly related to (“most similar to”) this question vector.  Perhaps this will match against document chunks extracted from academic calendars, add deadlines, and drop procedures
- 1.3: Retrieve the actual text of those relevant document chunks

### Step2: Create an augmented prompt combining the original question with the retrieved context:

```
Question: What's Cornell's policy on course add/drop for 2025?
Context from relevant documents:
1. "Students can add courses through the first five weeks..."
2. "Drop without 'W' grade is permitted until the last four..."
3. "Late drops require instructor and college approval..."
Based on the context above, provide a detailed answer.
```

### Step 3: Generate the response using an LLM that now has all the facts it needs:
"According to Cornell's 2025 academic calendar, students can add courses through the first two weeks of the semester and drop courses without a 'W' until the fourth week. After that date, a late drop requires instructor and college approval."
The answer is accurate, current, and traceable to real documents. In contrast, without this context the LLM might have suffered from hallucination, or used outdated information.


Where does the database itself come from?  Before any of the searching occurs, it has to be built.  So someone collects high quality documents, perhaps by crawling the Cornell web site, and runs them through a specialized form of ML that identifies chunks of text that might be important in various ways.  Each of these chunks of texts is then embedded into that same 768-dimensional space, and the resulting (vector, document-URL) pair are stored in the ANN index.

Notice that we didn’t even need to store a copy of the documents themselves.  The entire RAG infrastructure focuses on embeddings and then the document URLs can be used to retrieve the actual documents at the step where response generation occurs.  This is especially helpful if a document might sometimes be updated after we chunked and indexed it: the generative LLM will use the most current version.


## Key Concepts in RAG

### Embedding Vector
How can we tell whether two texts are truly related in meaning? Traditional information retrieval relies on lexical matching, comparing literal words in a query and document. This works when phrasing overlaps but fails when two sentences convey the same idea with different words—for instance, “How can I speed up my computer?” and “Improving system performance can make your PC run faster.”
To capture this kind of semantic similarity, we use embedding-based retrieval. A trained encoder model (such as BAAI/bge-base-en-v1.5, which outputs 768-dimensional vectors) maps each document into a high-dimensional vector space where related texts are close together. All document embeddings are stored in a vector database. When a new query arrives, the system encodes it into a vector and compares it with stored embeddings using similarity measures such as cosine similarity or L2 distance. If two pieces of text are semantically relevant, their embeddings will have high similarity—allowing the system to retrieve top K related documents even when no words match exactly.

### Vector Database and Vector search
A vector database stores document embeddings and enables efficient similarity search. When a new query arrives, its embedding is compared with all stored vectors to find the most relevant documents. An exhaustive comparison gives perfect accuracy but scales poorly for large datasets, so real systems use Approximate Nearest Neighbor (ANN) search to trade a little accuracy for major speed gains. Popular structures include graph-based (HNSW, GraphRAG) and tree-based (IVF) indexes. The KDD tree we built in HW1 used a tree-based data structure for such search, which can also be used to build such vector search index. In this homework, you will use the FAISS library to build such a database and experiment with both flat and IVF search.

### LLM Generation 
Large Language Models such as Qwen, ChatGPT, and Llama are transformer-based models that generate text token by token. Their performance grows with model size and hardware resources, but smaller models can still run effectively on CPUs. In this project you will use TinyLlama-1.1B-Chat-v0.3, a compact model that can run locally. When given relevant retrieved context, LLMs produce far more accurate and grounded responses than when generating from memory alone—an effect you will observe directly in your experiments.


# Part 1. Vector Database (Due Nov 15)

The vector database is obviously a key component in a RAG pipeline. It's used for retrieval of relevant documents and is a bit like the "memory" AI will recall after “understanding” your query.  

In real applications, companies build vector databases over their proprietary information, such as internal documents, customer data, product manuals, code repositories. This lets them build AI assistants that can answer questions about their specific knowledge, not just general information from the internet.

For this assignment, you'll work with a subset of [MS MARCO](https://microsoft.github.io/msmarco/) (Microsoft MAchine Reading COmprehension), a large-scale dataset designed for question-answering research. It contains passages from web documents along with real user queries.  One thing to know about MS Marco is that it was artificially fuzzed by taking each document and chunking it, but then tossing in some extra context words to prevent a problem called overfitting.  So a single query might actually match multiple embeddings that really point to the same chunk.  The way we will handle this centers on the URL: if two embeddings have the same URL, we would know that they refer to the same document, and since we want K “diverse” documents, we would only include it once into our search results.  This is logic you’ll probably need: if you omit it, instead of K documents you may notice that your solution retrieves the same document K times!  Below we use document “ids” rather than URLs to make it easier to work with them.  An id is just a unique number.  One document would often result in multiple chunks with the same id, hence multiple embeddings with the same id.

## Getting Started: Download Your Dataset
You'll work with approximately 10,000 document chunks—enough to build an interesting system without overwhelming your laptop.

Download the dataset of document:
```
wget -c https://cornell.box.com/shared/static/ffwnimisulvjyzp1t9u3201uj3p0w3bk.json -O documents.json
```

Download the dataset of query:
```
wget -c https://cornell.box.com/shared/static/eg115icq6oz2shnq63vzjw3776v30mwy.json -O queries.json
```

This file contains documents in plain text, one per line. Your job is to transform these natural language texts into searchable vectors. Let's get started!

## Step 1: Data Preprocessing—From Text to Vectors (encode):
First step is to get the vector embedding out of the documents in the MSMARCO dataset. You are going to process the data from natural language text to higher dimensional embedding via a pre-trained encoder model. This step that transfers the text into embedding is called encoding. It is done by running a pre-trained encoder model. In this assignment we will be using BGE encoder model (you can find the model checkpoint from [huggingface](https://huggingface.co/BAAI/bge-base-en-v1.5), for people using llama.cpp library with cpp API, you can download model checkpoint using [this](https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf?utm_source=chatgpt.com) by running this command
```
wget -c "https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-f32.gguf?download=true" -O bge-base-en-v1.5-f32.gguf
```

The output embedding vectors have dimension 768,
In HW1, this is provided to you, for HW3, we will only provide you with the MSMARCO dataset, and you are going to write code to process the dataset into vectors, and store the preprocessed results in a json file.

### What You Need to Build?
Create data_preprocess.py (or .cpp) that does three things:
1. Reads the MS MARCO dataset
2. Encodes each document through the BGE model
3. Outputs a JSON file with this exact format (same as what we provided you in HW1):

```
[
  {
    "id": 0,
    "text": "We have been feeding our back yard squirrels for the fall and winter and we noticed that a few of them have missing fur. One has a patch missing down his back and under both arms. Also another has some missing on his whole chest. They are all eating and seem to have a good appetite.",
    "embedding": [
        -0.0185089111328125,
        -0.046875,
        …
        -0.0135040283203125,
        0.031494140625
    ]
  },
  {
    ...
  },
  ...
]
```


Important details:
- Each embedding is exactly 768 numbers (floats)
- Keep the id field—you'll need it to map results back to documents later
- Save as preprocessed_documents.json


## Step2: Building a vector database:

You are building a fast similarity search engine using [FAISS](https://github.com/facebookresearch/faiss.git). FAISS is an open-sourced library that implements a vector database, by Facebook AI Research (now Meta AI). It provides API to build a vector database (vector search engine, or as FAISS library named it, an index) It is able to read the embeddings into vector and construct a vector search index based on all the embeddings via its constructor: `faiss::IndexFlatL2 index(d)` in its CPP API, or calling `faiss.IndexFlatL2(d)`
Please learn how to use its APIs from this [tutorial](https://github.com/facebookresearch/faiss/tree/main/tutorial). 

### Installing FAISS
Python: 
```
pip install --user faiss-cpu
```
C++: Follow the official installation guide. You'll need CMake and a C++11 compiler

### Understanding FAISS Search Output
FAISS search is designed to handle batches of queries efficiently. You can search with one query or many queries at once.

Inputs:
- query_embeddings: Matrix of shape (batch_size, 768) - one or more query vectors
- k: Number of nearest neighbors to return

Outputs:
When you search, FAISS returns two arrays:
Distance matrix(D): How far each result is from your query
- Shape: (1, k) for a single query ; (batch_size, k) if the input is a batch of query
- The smaller the number is meaning two vectors are more similar
- A distance of 0 means perfect match
- Uses L2 (Euclidean) distance for IndexFlatL2

Indices matrix(I): The document IDs of the results
- Shape: (1, k) for a single query; (batch_size, k) if the input is a batch of query
- These are the index of the position of the embedding in the matrix you form (For example, if your stored embeddings were stacked in order [e₀, e₁, e₂, …], then an index of 42 means that the 43rd embedding in your dataset was one of the top-k most similar vectors to your query.)


**Example of single query:**
```
// Search with one query (batch_size = 1) 
query_embedding = encoder.encode("What causes squirrels to lose fur?") 
query_embedding = query_embedding.reshape(1, 768) # Shape: (1, 768)
D, I = index.search(query_embedding, k=5)
// distances matrix D: [[0.0000, 245.34, 312.56, 389.23, 421.88]]
// indices matrix I:   [[42,     1337,  891,    2043,   156   ]]
// Interpretation:
// The embedding at position 42 (in your original embedding array) is the closest match with distance 0
// The embedding at position 1337 is the second closest with distance 245.34
// And so on...
```


**For this assignment**: In Part 1, you'll primarily use single-query search (batch_size=1). In Part 3, you will explore batch search for performance optimization!


### What You Need to Build?
Create vector_db.py (or .cpp) that:
1.	Loads embeddings from preprocessed_documents.json
2.	Builds a FAISS index
3.	Implements a search function




## Step 3: Testing Your Vector Database
Before moving on, let's validate everything works!
1.	Pick one of your document embeddings (say, document ID 42)
2.	Use it as a search query
3.	The top result should be document 42 itself with distance ≈ 0

**Why?** A document should always be most similar to itself. If this doesn't work, something's wrong with your index.


## Part 1 Deliverables: What to Submit
### Code files:
1.	data_preprocess.py (or .cpp)
- Reads MS MARCO dataset
- Encodes documents with BGE
- Outputs preprocessed_documents.json
2.	vector_db.py (or .cpp)
- Loads preprocessed embeddings
- Builds FAISS index
- Implements search function
3.	Build files (C++ only):
- CMakeLists.txt OR Makefile OR build.sh
- Include compilation instructions in a README


### Demonstration video:
**Part1 recording (in .mp4 format, using zoom recording)**
1.	Running data_preprocess to generate the JSON file
2.	Running vector_db to build the index, and execute the test search on a query embedding
3.	Displaying the output (distances and indices arrays)

Please remember to submit all of the above files, especially the recording, since it will help us to reproduce your results when grading. Missing files would lead to points deduction.



# Part 2. RAG Pipeline (Due Nov 25)

Now let’s everything together to build a working AI assistant!
By the end of Part 2, you'll have a system where you can type a question and get back an AI-generated answer grounded in your document collection. This is the same architecture powering enterprise AI assistants, customer support bots, and internal knowledge bases at tech companies.

**The Five Components You'll Build:**
1.	Query Encoder - Converts user questions to vectors
2.	Vector Search - Finds relevant documents (using your Part 1 code)
3.	Document Retrieval - Fetches the actual text
4.	Prompt Augmentation - Combines question + context
5.	LLM Generation - Generates the final answer

Let's build each component step by step.

## Component 1: Query Encoder
First, write an **encoder** module that converts user queries into embeddings using the same BGE model as in Part 1. Next, use your FAISS index to retrieve the top 3 most relevant document IDs. Retrieve their full text using your stored mapping between document IDs and text (either kept in memory or loaded from file).

**What it does:** Takes a user's natural language question and converts it to a 768-dimensional vector using the same BGE model you used for documents.

**What You Need to Build?**
1.	A file encode.py (or encode.cpp) that onverts user questions into 768-dimensional vectors.
2.	Load the BGE-base-en-v1.5 encoder model
3.	Use it to convert the input query text into numpy array of shape (768,)


## Component 2: Vector Search:
You are going to use your Part 1 FAISS index to find the top-K most relevant documents for a query vector. (we are going to use top_k =3 for part2)
In this part of the assignment, you can assume that the documents are fixed and provided to you, such that you can preprocess them offline and directly load them into the vector search index, using the script from Part 1. You do NOT need to re-encode the documents into embeddings, could simply load your preprocessed_documents.json file that you created in Part 1.

**What You Need to Build?**

vector_db.py (or vector_db.cpp) that loads the preprocessed documents and perform similarity search.
1.	Load preprocessed_documents.json
2.	Build FAISS index with all document embeddings
3.	Provide search(query_embedding, tpp_k) function
4.	Return distances and document IDs


## Component 3: Document Retrieval
From Component2, we can get the top_k related index of the query embedding, but not the text. You need to map IDs back to documents. Now the vector search provided us with the top K related documents’ index. In order to form the context for the query to feed to the next stage of LLM generation in the RAG pipeline, we need to retrieve the actual documents corresponding to the ID. 
To do this, there are two options. One is to use an in-memory dictionary. You load the preprocessed_documents.json into memory could store it in a Python dictionary or an std::map using cpp. Then using the ID to get the document texts. In-memory read is fast as long as it can fit into the memory. The other one is to use file-based lookup, where you load the document texts on demand from the file system. This method is commonly used in the case where there are billion-scale documents. ([DiskANN](https://www.microsoft.com/en-us/research/publication/diskann-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node/) paper in Neurip2019 explores this idea) We are going to use the first option in part2, and explore the second option in part 3.

**Before** adding LLM complexity, make sure your retrieval works correctly. This is crucial!
Now you are able to see preliminarily about the correctness of your pipeline so far. Type in a question from the queries.json, and retrieve the related documents. Check if the retrieved documents make sense. If it does, then you can move on to the next step.


## Componet 4: Prompt Augmentation
LLMs are sensitive to how information is presented. A clear structure helps the model distinguish between the question and the reference material. 
Component4 combines the user's question with retrieved documents into a structured prompt. You can form a new prompt using the format of : `query + " Top documents:" + top1_doc_text + top2_doc_text + top3_doc_text`.


## Component 5: LLM Generation
For generation step, you will take the augmented prompt and generate a natural language response. We provided several options of models you can select from, you can choose one of them to incorperate in your pipeline. In Part3, you will explore more on other options to learn about the tradeoffs.

- **TinyLlama-1.1B-Chat-v0.3**, a surprisingly capable small model with 1.1B parameters (takes 0.7-0.8 GB of memory) and could run on CPU.

You can download the model checkpoint to your local directory via the command
```
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf
```

- **LLama3.2-3B**, a slightly larger and more recent model. Since we want to run the model on CPU, we use Q4_K_M (4-bit K-quant medium) quantization, which reduced the original model size and achieved a balance between its speed and response quality.

You can learn more about the model information from its huggingface checkpoint: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF 

You can donwload the model checkpoint to your local directory via the command
```
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```


- **Qwen2-1.5B-instruct**, Qwen2-1.5B a model that is a lot more capable and good at question answering type of tasks. The base model (without "Instruct") means that the model is trained on large text corpora to predict the next token and learn the general language patter. "Instruct" indeicates that this model has been fine-tuned for instruction-following tasks. It undergoes additioanl supervised fine-tuning (SFT) with data look like "Here is an instruction: ... Produce the answer: ..." That helps the model handle direct instructions, user queries, prompts etc.

You can learn more about the model information from its huggingface checkpoint: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF?utm_source=chatgpt.com 

You can download the model checkpoint to your local directory via the command
```
wget “https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_0.gguf”   -O qwen2-1_5b-instruct-q4_0.gguf
```

- **Qwen2-7B-Instruct**, Qwen2-7B-Instruct is a larger model, with 7 Billion parameters. It has more parameters, deeper and wider layers. It can capture more complex relationships, improving reasoning, coding, multilingual and general performance than Qwen2-1.5B. Note that this model is a lot larger and requires more memory to store and run. Although it is runable on CPU, you are not required to run this model, especially if your local machine is not as powerful to suppport this.

You can download the model checkpoint to your local directory via the command
```
wget "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_0.gguf"      -O qwen2-7b-instruct-q4_0.gguf
```

**Generation with LLM model:**
If you are going to use CPP API to call the LLM generation model, you can refer to the notes and demo we taught on [recitation 9](https://github.com/aliciayuting/CS4414Demo/tree/main/recitation9), that shows how to link and compile the C++ program with the llama.cpp library.
You can also use Python API of this model, we showed in [recitation 11](https://github.com/aliciayuting/CS4414Demo/tree/main/recitation11). 

To install the Python dependency you can run:
```
pip install --user llama-cpp-python
```


## Component 6: The Main Program—Your Interactive RAG System
Congratulations! Now you have all the components in RAG pipeline implemented.
This is where everything comes together! You'll create an interactive command-line interface where users can ask questions and get AI-powered answers.

Create a main program (main.cpp or main.py) that orchestrates the entire RAG workflow in an interactive command-line interface. The program should:
1.	Initialize the system by loading the preprocessed document embeddings from preprocessed_documents.json (generated in Part 1)
2.	Accept user queries through an interactive command-line prompt
3.	Execute the RAG pipeline for each query:
- Encode the user's query into an embedding vector
- Retrieve the top-k most relevant document chunks using vector search
- Construct a context-augmented prompt combining the query with retrieved documents
- Generate a response using the LLM with the augmented context
4.	Display results by printing the LLM-generated response to the console
5.	Loop continuously, returning to the command prompt to accept the next query until the user exits

## Part2 What to submit: Show us your working system
Code files:
1.	main.py (or main.cpp) - The complete interactive system
2.	encode.py (or .cpp) - Query encoding component
3.	vector_db.py (or .cpp) - Vector search (can reuse from Part 1)
4.	llm_generation.py (or .cpp) - LLM inference wrapper
5.	Document retrieval code (if separate from vector_db)
6.	Build files (C++ only): CMakeLists.txt OR Makefile OR build.sh

Demonstration video:

A recording(in .mp4 format, using zoom recording) of you running the full pipeline try with 2 queries.

Please remember to submit all of the above files, especially the recording, since it will help us to reproduce your results. Missing files would lead to a points deduction.

# Part 3. System Analysis and Optimizations (Due Dec 5)

**System Analysis by components:**
In this class, we learned that the most important step for understanding and optimizing a system entails detailed benchmarking and performance measurement. We would like you to write a small essay about your findings.  Report the distribution of latency breakdown for each component in your RAG pipeline (document retrieval, question augmentation, LLM generation). Use plots to show the distribution clearly of each step and between steps.

**Opmizing components:**
You can do more than the minimum but every solution MUST answer the following questions:
-	What is the time breakdown of different components in the system? Do you observe the major bottleneck of your system? Are there places you can optimize?
-	For AI system, not only do we care about system performance but also about accuracy. What are some tradeoffs in this system you can think of? Try at least two of the below optimization methods, report the change in performance, and discuss the tradeoffs you observe.
    -	A different encoder model that generates embeddings of dimensions greater than 768 or smaller than 768 of BGE encoder model we used in Part 1
    -	ANN search method top-K number, what if we set the top-K number of ANN search to be greater than 3 as we used in Part 2, or smaller than 3 by only selecting top 1
    -	LLM generation model, what happens if we use a smaller model or larger model? Do you observe it running faster or slower, giving more reasonable results or less reasonable? (You are also free to try with other models that are not in above lists provided, even OpenAI API call, there are limits in the number of calls one can make in a given day, but connect your pipeline with it and give a few tries to see how that response compares to the results from the TinyLlama-1.1B-Chat-v0.3-GGUF model we used in Part 2)



**Vector Search optimizations:**
The size of the documents could grow to millions or trillions, taking GB to TB memory. As the amount of data for vector search scales up, the step of vector search could become a dominant factor. Below are some ideas to help scale out this vector search step you can try. Test out below ideas and report the performance numbers you observe.
1.	Batching is a commonly used way to optimize system performance. Try batching at the vector search step, and plot the system performance, throughput, latency, with different batch sizes, ranging from 1,4,8,16,32,64,128. Analyze the performance numbers you observe, can you explain why?
2.	Instead of using Flat search, try with IVFFlat search as vector database engine.



## Part3 What to submit:
The writeup you submit should include
1.  system performance measurements breakdown by components
2.  Answer the question based on experiments with other configurations or models for different components in the pipeline
3.  Testing of Vector search optimization ideas and report of the performance measurements
