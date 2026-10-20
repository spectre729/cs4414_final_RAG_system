cmake_minimum_required(VERSION 3.16)
project(rag_system LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Dependencies ----
# llama.cpp is expected to be installed under LLAMA_PREFIX (headers in include/, libraries in lib/).
# Override with -DLLAMA_PREFIX=/path/to/install at configure time.
set(LLAMA_PREFIX "$ENV{HOME}/opt_dev" CACHE PATH "Install prefix of llama.cpp")
list(APPEND CMAKE_PREFIX_PATH "${LLAMA_PREFIX}")

option(RAG_BUILD_TESTS "Build the smoke-test executables" ON)

find_package(Threads REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(BLAS REQUIRED)
find_library(FAISS_LIBRARY faiss REQUIRED)
find_path(FAISS_INCLUDE_DIR faiss/Index.h REQUIRED)
find_package(OpenMP REQUIRED)  # FAISS is built with OpenMP

# Shared setup applied to every executable below.
function(rag_configure target)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${LLAMA_PREFIX}/include"
    "${LLAMA_PREFIX}/include/ggml"
  )
  target_link_directories(${target} PRIVATE "${LLAMA_PREFIX}/lib")
  target_link_libraries(${target} PRIVATE
    llama ggml ggml-base ggml-cpu
    nlohmann_json::nlohmann_json
    Threads::Threads
    ${CMAKE_DL_LIBS} m
  )
  # Lets the binary find libllama at runtime without LD_LIBRARY_PATH.
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH   "${LLAMA_PREFIX}/lib"
    INSTALL_RPATH "${LLAMA_PREFIX}/lib"
  )
endfunction()

# Targets that use the FAISS vector database. OpenMP must come after FAISS on the link line,
# because a static libfaiss.a leaves its OpenMP symbols for the linker to resolve afterwards.
function(rag_use_faiss target)
  target_include_directories(${target} PRIVATE "${FAISS_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${FAISS_LIBRARY}" ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
endfunction()

# ---- Libraries ----
add_library(rag_encode STATIC src/encode.cpp)
add_library(rag_llm    STATIC src/llm_generation.cpp)
add_library(rag_db     STATIC src/vector_db.cpp)
foreach(lib rag_encode rag_llm rag_db)
  rag_configure(${lib})
endforeach()
rag_use_faiss(rag_db)

# ---- Executables ----
add_executable(data_preprocess src/data_preprocess.cpp)
rag_configure(data_preprocess)

add_executable(rag_system src/main.cpp)
rag_configure(rag_system)
rag_use_faiss(rag_system)
target_link_libraries(rag_system PRIVATE rag_encode rag_llm rag_db)

# ---- Smoke tests (need the model files and preprocessed documents to run) ----
if(RAG_BUILD_TESTS)
  add_executable(test_encode tests/test_encode.cpp)
  rag_configure(test_encode)
  target_link_libraries(test_encode PRIVATE rag_encode)

  add_executable(test_llm tests/test_llm.cpp)
  rag_configure(test_llm)
  target_link_libraries(test_llm PRIVATE rag_llm)

  add_executable(test_pipeline tests/test_pipeline.cpp)
  rag_configure(test_pipeline)
  rag_use_faiss(test_pipeline)
  target_link_libraries(test_pipeline PRIVATE rag_encode rag_db)
endif()
