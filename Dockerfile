# Reproducible build and runtime environment for the RAG pipeline.
#
#   docker build -t rag-system .
#   docker run --rm -it -v "$PWD/models:/models" -v "$PWD/work:/work" rag-system
#
# See the README for the full walkthrough.

FROM debian:trixie AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libfaiss-dev libopenblas-dev libomp-dev nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# llama.cpp, installed to /opt/llama (the project's LLAMA_PREFIX).
ARG LLAMA_CPP_REF=master
RUN git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp \
    && git -C /tmp/llama.cpp checkout ${LLAMA_CPP_REF} \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build \
        -DCMAKE_INSTALL_PREFIX=/opt/llama -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF \
    && cmake --build /tmp/llama.cpp/build -j"$(nproc)" \
    && cmake --install /tmp/llama.cpp/build \
    && rm -rf /tmp/llama.cpp

COPY CMakeLists.txt /src/CMakeLists.txt
COPY src /src/src
COPY tests /src/tests
RUN cmake -S /src -B /src/build -DLLAMA_PREFIX=/opt/llama -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /src/build -j"$(nproc)"


FROM debian:trixie

# Debian's FAISS is a static library, so it is already linked into the binaries.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/llama /opt/llama
COPY --from=build /src/build/rag_system /src/build/data_preprocess \
                  /src/build/test_encode /src/build/test_llm /src/build/test_pipeline /usr/local/bin/
COPY scripts /app/scripts
COPY data/sample /app/data/sample

ENV LD_LIBRARY_PATH=/opt/llama/lib
# Models live in /models (mount a volume to keep them between runs); outputs go in /work.
WORKDIR /work
CMD ["bash"]
