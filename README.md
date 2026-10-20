# RAG System in C++

A local Retrieval-Augmented Generation (RAG) pipeline written in C++17. It embeds a document corpus, indexes it for fast similarity search, and answers questions by retrieving the most relevant passages and feeding them to a small local language model. Everything runs on CPU with no external services.

Originally built as a course project for CS 4414 (Fall 2025).

**Status:** a working demo, not production software. The pipeline runs end to end and retrieval is sound, but answer quality is limited and several production concerns are unaddressed. See [Limitations](#limitations).

## How it works

```
                 offline                                    online
 documents.json ──► data_preprocess ──► preprocessed_documents.json
                    (BGE embeddings)              │
                                                  ▼
 question ──► QueryEncoder ──► VectorDatabase (FAISS) ──► top-k passages ──► LLMGenerator ──► answer
              (BGE, 768-d)     flat or IVF index           prompt template    (TinyLlama)
```

| Component | Source | Role |
|---|---|---|
| Document embedding | [src/data_preprocess.cpp](src/data_preprocess.cpp) | One-off step that embeds every document with `bge-base-en-v1.5` (via llama.cpp) |
| Query encoder | [src/encode.cpp](src/encode.cpp) | Embeds a question with the same model |
| Vector database | [src/vector_db.cpp](src/vector_db.cpp) | FAISS index (exact flat or approximate IVF), single and batched search |
| Generator | [src/llm_generation.cpp](src/llm_generation.cpp) | Greedy decoding with a GGUF chat model, streamed to the terminal |
| CLI | [src/main.cpp](src/main.cpp) | Interactive Q&A and a batched-search benchmark |

## Requirements

- A C++17 compiler and CMake 3.16+
- [llama.cpp](https://github.com/ggml-org/llama.cpp), installed to a prefix (default `~/opt_dev`)
- [FAISS](https://github.com/facebookresearch/faiss), a BLAS library (e.g. OpenBLAS), and [nlohmann/json](https://github.com/nlohmann/json)

Typical setup on Debian 13 or Ubuntu 25.10 or newer (older Ubuntu releases, including 24.04, do not package FAISS, so build it from source or use Docker below):

```bash
sudo apt install build-essential cmake libfaiss-dev libopenblas-dev libomp-dev nlohmann-json3-dev

git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_INSTALL_PREFIX=$HOME/opt_dev
cmake --build llama.cpp/build -j
cmake --install llama.cpp/build
```

Or skip the setup entirely and use [Docker](#docker).

## Build

```bash
cmake -S . -B build                 # add -DLLAMA_PREFIX=/path if llama.cpp is not in ~/opt_dev
cmake --build build -j
```

This produces `build/rag_system`, `build/data_preprocess`, and three smoke-test programs (`test_encode`, `test_llm`, `test_pipeline`; disable with `-DRAG_BUILD_TESTS=OFF`).

## Quickstart

Run these from the repository root. A small sample corpus is included in [data/sample/](data/sample/).

```bash
# 1. Download the embedding model (~440 MB) and the chat model (~670 MB)
./scripts/download_models.sh

# 2. Embed the sample documents
./build/data_preprocess data/sample/documents.json preprocessed_documents.json

# 3. Ask questions (--flat: with only 12 documents, IVF would retrieve just one)
./build/rag_system --flat
```

At the prompt, try `What is a quarterback?`. Each answer is preceded by the retrieved document ids and distances, and followed by encoding, search and generation times. Type `exit` to quit.

## Docker

The [Dockerfile](Dockerfile) builds llama.cpp and this project on Debian 13 into a self-contained image, so no other dependencies are needed. You need Docker running, and about 1.5 GB of disk for the image (333 MB) and models (1.1 GB), plus extra build cache while building. It was tested with 8 GB of memory available to Docker. The build takes a few minutes.

```bash
git clone <repository-url> && cd <repository-directory>   # all commands below start from the repo root
docker build -t rag-system .

mkdir -p models && cd models
docker run --rm -v "$PWD:/m" -w /m rag-system /app/scripts/download_models.sh
docker run --rm -v "$PWD:/m" -w /m rag-system \
    data_preprocess /app/data/sample/documents.json preprocessed_documents.json bge-base-en-v1.5-f32.gguf
docker run --rm -it -v "$PWD:/m" -w /m rag-system rag_system --flat
```

The models are downloaded into `models/` on your machine (the directory is mounted at `/m`), so they are reused across runs. Type a question at the prompt, or `exit` to quit. The sample data is baked into the image at `/app/data/sample`.

To run the search benchmark instead (from the same `models/` directory); the CSV is written there too:

```bash
docker run --rm -v "$PWD:/m" -w /m rag-system rag_system --benchmark --queries /app/data/sample/queries.json
```

The image has been built and run on arm64 (Apple silicon). It is built without CPU-specific optimizations, so it should also build on x86-64, but that has not been tested.

## Usage

```
rag_system [options]

  --benchmark           Benchmark batched vector search (does not load the LLM)
  --docs PATH           Embedded documents (default: preprocessed_documents.json)
  --queries PATH        Queries used by --benchmark (default: queries.json)
  --embed-model PATH    BGE embedding model (default: bge-base-en-v1.5-f32.gguf)
  --llm PATH            Chat model (default: tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf)
  --ctx N               LLM context size in tokens (default: 2048)
  --top-k N             Documents retrieved per question (default: 3)
  --flat                Use an exact flat index instead of IVF
  --nprobe N            IVF clusters searched per query (default: 10)
  --output PATH         CSV written by --benchmark (default: batch_benchmark_results.csv)
```

Benchmark the search stage on the sample queries:

```bash
./build/rag_system --benchmark --queries data/sample/queries.json
```

This encodes 512 queries and times vector search at batch sizes 1 through 128, printing latency and queries/second and writing them to a CSV.

## Using your own data

`data_preprocess` expects a JSON array of passages (the original project used [MS MARCO](https://microsoft.github.io/msmarco/)):

```json
[{"id": 0, "text": "First passage..."}, {"id": 1, "text": "Second passage..."}]
```

Passages longer than 512 tokens are truncated, which is the BGE model's limit. The output file adds a 768-dimensional `embedding` to each entry. The queries file for `--benchmark` uses the same `{"id", "text"}` format.

## Design notes

- **Index choice.** `IndexFlatL2` gives exact results; `IndexIVFFlat` (up to 100 clusters, `--nprobe` of them searched per query) trades some recall for faster search on large corpora; raise `--nprobe` for better recall. IVF is the default; use `--flat` for exact search. On very small corpora, such as the 12-document sample, IVF's clusters hold about one document each, so it retrieves fewer than `--top-k` passages and `--flat` is the better choice.
- **Prompt format.** The prompt template in `create_prompt` ([src/main.cpp](src/main.cpp)) follows TinyLlama's chat format. Other models need their own template.
- **Vectors are not normalized** and search uses L2 distance.

## Limitations

This was good enough for the class project and works as a quick demo, but it is not something to push toward production as it stands. The known issues are listed below so they are not a surprise.

**Answer quality**
- The default chat model, TinyLlama-1.1B at 4-bit quantization, is small. In testing, its answers often start out relevant and then drift: it may repeat the retrieved passage instead of answering, continue with invented follow-up questions, or mix in unrelated context.
- Generation is greedy with a fixed 256-token limit and no repetition penalty, and the model does not reliably emit an end-of-answer token. Output is cut off by a hard limit and by a few stop strings in [src/llm_generation.cpp](src/llm_generation.cpp), so a stray fragment such as `Question` can appear at the end of an answer.
- The prompt does not tell the model what to do when the context lacks the answer, so it can answer from its own (unreliable) knowledge. Questions outside the corpus can produce confident but wrong answers.
- The prompt template is hard-coded for TinyLlama. An attempt to switch to the ChatML format did not improve results with this model. Other models need their own template, and `--llm` only swaps the model file.
- A stronger instruction-following model (for example Qwen2-1.5B-Instruct or Llama-3.2-3B) is the most likely single improvement, but has not been evaluated here.

**Retrieval**
- Embeddings are not normalized and search uses L2 distance, although BGE models are trained for cosine similarity. Rankings may be better with normalization or an inner-product metric; this has not been tested.
- Exactly `--top-k` passages are always returned, however distant. On a small corpus this pads the prompt with irrelevant text. There is no distance cutoff.
- Documents are embedded as given, with no chunking, and are truncated at 512 tokens (the BGE limit).
- With IVF, results depend on `--nprobe`. FAISS defaults to 1, which on a tiny corpus returns fewer passages than `--top-k`; this project defaults to 10 and the quickstart uses `--flat`.

**Engineering**
- There is no evaluation of retrieval or answer quality, and no automated tests. The programs in `tests/` are smoke tests that print output for a human to check, and need the models and data.
- All embeddings are loaded from one JSON file into memory at startup, and the index is rebuilt (and, for IVF, retrained) on every run. This does not scale to large corpora.
- It is a single-process command-line program: no server, concurrency, request limits, logging or monitoring. The benchmark measures only the search stage, and on the 12-document sample its numbers are not meaningful.
- Only the Docker build on arm64 (Apple silicon) has been run. The native Linux build instructions and x86-64 are untested.

## Repository layout

```
src/        library and executable sources
tests/      smoke tests that exercise each component (require models and data)
data/sample sample documents and queries
scripts/    model download helper
```

`cs4414_final.md` is the original course assignment handout.
