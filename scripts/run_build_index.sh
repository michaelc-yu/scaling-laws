
echo "Building rag index with faiss + bm25..."

python3 rag/build_index.py \
  --retrieval_manifest data/splits/retrieval_manifest.jsonl \
  --deduped_data data/processed/wikipedia_deduped.jsonl \
  --out_dir data/rag/retrieval_index \
  --chunk_len 512 \
  --stride 256

echo "Done! Finished building rag index"

