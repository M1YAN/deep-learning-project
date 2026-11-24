#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

snap=https://snap.stanford.edu/data/amazon/productGraph/categoryFiles
model=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41

download() (
    local url=$1 dest=$2 temporary
    [[ -s "$dest" ]] && exit 0
    mkdir -p -- "$(dirname -- "$dest")"
    temporary=$(mktemp "$dest.XXXXXX")
    trap 'rm -f -- "$temporary"' EXIT
    curl --fail --location --silent --show-error --retry 3 --connect-timeout 20 \
        --max-time 600 --output "$temporary" "$url"
    mv -f -- "$temporary" "$dest"
)

download "$snap/ratings_Beauty.csv" data/raw/Beauty/ratings_Beauty.csv &
ratings_pid=$!
download "$snap/meta_Beauty.json.gz" data/raw/Beauty/meta_Beauty.json.gz &
metadata_pid=$!
(
    for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt model.safetensors; do
        download "$model/$file" "data/models/all-MiniLM-L6-v2/$file"
    done
) &
model_pid=$!
status=0
for pid in "$ratings_pid" "$metadata_pid" "$model_pid"; do
    wait "$pid" || status=$?
done
exit "$status"
