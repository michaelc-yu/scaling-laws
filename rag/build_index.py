#!/usr/bin/env python3
import os, json, argparse, gc, pickle, hashlib
import numpy as np
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import tiktoken
from rank_bm25 import BM25Okapi
import yaml

# safe mode to avoid random segfaults on CPU/macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1); torch.set_num_interop_threads(1)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def chunk_text(text, enc, chunk_len=512, stride=256):
    ids = enc.encode(text)
    out = []
    if chunk_len <= 0: return out
    if stride <= 0: stride = chunk_len
    for s in range(0, len(ids), stride):
        e = s + chunk_len
        window = ids[s:e]
        if not window: break
        out.append(enc.decode(window))
        if e >= len(ids): break
    return out

def embed_texts(texts, model, tokenizer, batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval(); model.to(device)
    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding for FAISS"):
            batch = texts[i:i+batch_size]
            # defensive cleaning
            batch = [t if isinstance(t, str) else str(t) for t in batch]
            batch = [t for t in batch if t.strip()]
            if not batch:
                continue
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            outputs = model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype("float32")
            embs.append(batch_emb)
            del inputs, outputs
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.vstack(embs) if embs else np.zeros((0, model.config.hidden_size), dtype="float32")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_manifest", required=True, help="JSONL of allowed doc IDs (retrieval partition)")
    ap.add_argument("--deduped_data", required=True, help="Full deduped JSONL file")
    ap.add_argument("--out_dir", required=True, help="Directory to write indices/artifacts")
    ap.add_argument("--chunk_len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--bm25_lower", action="store_true", help="Lowercase for BM25 tokenization")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    faiss_path = os.path.join(args.out_dir, "faiss.index")
    bm25_path  = os.path.join(args.out_dir, "bm25.pkl")
    meta_path  = os.path.join(args.out_dir, "chunks_meta.jsonl")
    stats_path = os.path.join(args.out_dir, "stats.yaml")

    if (not args.overwrite) and all(os.path.exists(p) for p in [faiss_path, bm25_path, meta_path, stats_path]):
        print("Artifacts already exist, use --overwrite to rebuild.")
        return

    # load allowed IDs
    allowed_ids = {str(o["id"]) for o in read_jsonl(args.retrieval_manifest)}
    print(f"Loaded {len(allowed_ids):,} allowed IDs")

    # select + chunk retrieval docs
    enc = tiktoken.get_encoding("cl100k_base")
    chunks, bm25_docs, meta = [], [], []
    def tok_bm25(s): return (s.lower() if args.bm25_lower else s).split()
    n_docs, n_kept_docs = 0, 0

    for obj in tqdm(read_jsonl(args.deduped_data), desc="Selecting & chunking"):
        n_docs += 1
        doc_id = str(obj.get("id"))
        if doc_id not in allowed_ids: continue
        text = obj.get("text") or ""
        if not text.strip(): continue
        n_kept_docs += 1
        pieces = chunk_text(text, enc, args.chunk_len, args.stride)
        for p in pieces:
            cid = len(chunks)
            chunks.append(p)
            bm25_docs.append(tok_bm25(p))
            meta.append({
                "chunk_id": cid,
                "doc_id": doc_id,
                "len_chars": len(p),
                "sha1": hashlib.sha1(p.encode("utf-8")).hexdigest(),
            })

    if not chunks:
        raise RuntimeError("No chunks collected — check manifest/corpus alignment.")

    # dense embeddings -> FAISS
    print(f"Loading encoder: {args.encoder}")
    model_tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    model = AutoModel.from_pretrained(args.encoder)
    embs = embed_texts(chunks, model, model_tok, batch_size=16)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss.write_index(index, faiss_path)

    # BM25
    bm25 = BM25Okapi(bm25_docs)
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # write metadata and stats
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    stats = {
        "total_docs_scanned": n_docs,
        "retrieval_docs_kept": n_kept_docs,
        "n_chunks": len(chunks),
        "chunk_len_tokens": args.chunk_len,
        "stride_tokens": args.stride,
        "encoder": args.encoder,
        "bm25_lower": bool(args.bm25_lower),
        "faiss_index": os.path.basename(faiss_path),
        "bm25_index": os.path.basename(bm25_path),
        "avg_chunk_chars": float(np.mean([m["len_chars"] for m in meta])),
        "dim": int(dim),
    }
    with open(stats_path, "w") as f:
        yaml.safe_dump(stats, f, sort_keys=False)

    print("Built hybrid indices")
    print(f"FAISS: {faiss_path}")
    print(f"BM25: {bm25_path}")
    print(f"Meta: {meta_path}")
    print(f"Stats: {stats_path}")

if __name__ == "__main__":
    main()
