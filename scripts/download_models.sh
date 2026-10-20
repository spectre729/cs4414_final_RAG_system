#!/usr/bin/env bash
# Downloads the GGUF models used by the default configuration into the current directory.
set -euo pipefail

BGE_URL="https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-f32.gguf"
TINYLLAMA_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"

for url in "$BGE_URL" "$TINYLLAMA_URL"; do
  file="$(basename "$url")"
  if [[ -f "$file" ]]; then
    echo "$file already exists, skipping"
  else
    echo "Downloading $file"
    curl -L --fail -o "$file" "$url"
  fi
done
