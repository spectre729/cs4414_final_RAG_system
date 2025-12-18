cmake_minimum_required(VERSION 3.16)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
project(llama_simple CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Where llama.cpp was installed ----
# Change this if you installed elsewhere, or pass -DLLAMA_PREFIX=/path at configure time.
set(LLAMA_PREFIX "$ENV{HOME}/opt_dev" CACHE PATH "Install prefix of llama.cpp")
list(APPEND CMAKE_PREFIX_PATH "${LLAMA_PREFIX}")

set(LLAMA_INCLUDE_DIR "${LLAMA_PREFIX}/include")
set(GGML_INCLUDE_DIR  "${LLAMA_PREFIX}/include/ggml")
set(LLAMA_LIB_DIR     "${LLAMA_PREFIX}/lib")



# ---- Target ----
add_executable(data_preprocess data_preprocess.cpp)

# add_executable(vector_db vector_db.cpp)

#add_executable(test_encode encode.cpp test_encode.cpp)

#add_executable(test_pipeline test_pipeline.cpp encode.cpp vector_db.cpp)

#add_executable(test_llm test_llm.cpp llm_generation.cpp)

add_executable(rag_system main.cpp encode.cpp vector_db.cpp llm_generation.cpp)


target_include_directories(data_preprocess PRIVATE
  "${LLAMA_INCLUDE_DIR}"
  "${GGML_INCLUDE_DIR}"
)

#target_include_directories(vector_db PRIVATE
#  "${LLAMA_INCLUDE_DIR}"
#  "${GGML_INCLUDE_DIR}"

#)

#target_include_directories(test_encode PRIVATE
#"${LLAMA_INCLUDE_DIR}"
#  "${GGML_INCLUDE_DIR}"

#)

#target_include_directories(test_pipeline PRIVATE
#"${LLAMA_INCLUDE_DIR}"
#  "${GGML_INCLUDE_DIR}"

#)

#target_include_directories(test_llm PRIVATE
#"${LLAMA_INCLUDE_DIR}"
#  "${GGML_INCLUDE_DIR}"

#)

target_include_directories(rag_system PRIVATE
"${LLAMA_INCLUDE_DIR}"
  "${GGML_INCLUDE_DIR}"

)


# Prefer per-target link directories over global link_directories
target_link_directories(data_preprocess PRIVATE "${LLAMA_LIB_DIR}")
# target_link_directories(vector_db PRIVATE "${LLAMA_LIB_DIR}")

#target_link_directories(test_encode PRIVATE "${LLAMA_LIB_DIR}")
#target_link_directories(test_pipeline PRIVATE "${LLAMA_LIB_DIR}")
#target_link_directories(test_llm PRIVATE "${LLAMA_LIB_DIR}")
target_link_directories(rag_system PRIVATE "${LLAMA_LIB_DIR}")


# Threads (pthread)
find_package(Threads REQUIRED)

# find_package(FAISS REQUIRED)

# Link the same libs as your working g++ command (CPU-only build of ggml)
target_link_libraries(data_preprocess PRIVATE
  llama
  ggml
  ggml-base
  ggml-cpu
  Threads::Threads
  dl
  m
)

#target_link_libraries(vector_db PRIVATE
#  llama
#  ggml
#  ggml-base
#  ggml-cpu
#  Threads::Threads
#  faiss
##  openblas
#  dl
#  m
  
  
#)

#target_link_libraries(test_encode PRIVATE
#  llama
#  ggml
#  ggml-base
#  ggml-cpu
#  Threads::Threads
#  dl
#  m
#)

#target_link_libraries(test_pipeline PRIVATE
#  llama
#  ggml
#  ggml-base
#  ggml-cpu
#  Threads::Threads
#  dl
#  m 
#  faiss
#  openblas
#  )


##target_link_libraries(test_llm PRIVATE
 # llama
#  ggml
#  ggml-base
#  ggml-cpu
#  Threads::Threads
#  dl
#  m
# )

target_link_libraries(rag_system PRIVATE
  llama
  ggml
  ggml-base
  ggml-cpu
  Threads::Threads
  dl
  m
  faiss
  openblas

)



# Optional: add OpenMP if available (harmless if not found)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(data_preprocess PRIVATE OpenMP::OpenMP_CXX)
  #target_link_libraries(test_encode PRIVATE OpenMP::OpenMP_CXX)
  #target_link_libraries(test_pipeline PRIVATE OpenMP::OpenMP_CXX)
  #target_link_libraries(test_llm PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(rag_system PRIVATE OpenMP::OpenMP_CXX)
  #target_link_libraries(vector_db PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- RPATH so the binary runs without LD_LIBRARY_PATH ----
set_target_properties(data_preprocess PROPERTIES
  BUILD_RPATH   "${LLAMA_LIB_DIR}"
  INSTALL_RPATH "${LLAMA_LIB_DIR}"
)

#set_target_properties(test_encode PROPERTIES
#  BUILD_RPATH   "${LLAMA_LIB_DIR}"
#  INSTALL_RPATH "${LLAMA_LIB_DIR}"
#)
#set_target_properties(test_pipeline PROPERTIES
#  BUILD_RPATH   "${LLAMA_LIB_DIR}"
#  INSTALL_RPATH "${LLAMA_LIB_DIR}"
#)
##set_target_properties(test_llm PROPERTIES
#  BUILD_RPATH   "${LLAMA_LIB_DIR}"
#  INSTALL_RPATH "${LLAMA_LIB_DIR}"
#)
set_target_properties(rag_system PROPERTIES
  BUILD_RPATH   "${LLAMA_LIB_DIR}"
  INSTALL_RPATH "${LLAMA_LIB_DIR}"
)


#set_target_properties(vector_db PROPERTIES
#  BUILD_RPATH   "${LLAMA_LIB_DIR}"
#  INSTALL_RPATH "${LLAMA_LIB_DIR}"
#)

