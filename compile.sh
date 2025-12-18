g++ -std=c++20 data_preprocess.cpp \
  -I"$HOME/opt_dev/include" \
  -L"$HOME/opt_dev/lib" \
  -Wl,-rpath,"$HOME/opt_dev/lib" \
  -lllama -lggml -lggml-base -lggml-cpu \
  -lpthread -ldl -lm \
  -o data_preprocess