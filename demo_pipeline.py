import argparse
import logging
import sys
import json
import os
from pathlib import Path
from typing import List

import numpy as np

# Set the fallback API key that you provided earlier
os.environ["GEMINI_API_KEY"] = "AIzaSyC7Gp25ncAxHnqv8o_b1d9ZtK52gpd9-1w"

# Suppress verbose logs
logging.basicConfig(level=logging.WARNING)

from core.embeddings import get_embedding_model
from core.llm import get_llm_client
from core.s02_hybrid_search import RetrievedDoc
from core.utils import build_source_label, make_youtube_url
from data.chunked.eval_retrieval import (
    load_eval_index,
    dense_search,
    sparse_search,
    rrf_fuse,
)

def build_retrieved_doc(chunk_id: str, score: float, chunk_meta: dict) -> RetrievedDoc:
    """Helper to build a RetrievedDoc object from raw index metadata."""
    url = make_youtube_url(chunk_meta.get("video_id", ""), chunk_meta.get("start_time", 0))
    label = build_source_label(chunk_meta)
    
    return RetrievedDoc(
        chunk_id=chunk_id,
        chunk_text=chunk_meta.get("chunk_text", ""),
        vi_text=chunk_meta.get("vi_text", ""),
        title=chunk_meta.get("title", ""),
        video_id=chunk_meta.get("video_id", ""),
        course=chunk_meta.get("course", ""),
        start_time=chunk_meta.get("start_time", 0.0),
        end_time=chunk_meta.get("end_time", 0.0),
        url=url,
        source_label=label,
        score=score,
        retrieval_method="hybrid",
        metadata=chunk_meta
    )

def main():
    parser = argparse.ArgumentParser(description="Full RAG Pipeline Demo")
    parser.add_argument(
        "--query", type=str, required=True,
        help="The question you want to ask in Vietnamese"
    )
    parser.add_argument(
        "--index", type=Path, default=Path("index/eval/t072"),
        help="Path to the eval index"
    )
    parser.add_argument(
        "--jsonl", type=Path, default=Path("data/chunked/transcript_v3_t072.jsonl"),
        help="Path to the original JSONL to load chunk text"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" 🚀 STARTING FULL RAG PIPELINE DEMO")
    print("=" * 60)
    
    # 0. Load raw chunk texts
    print(f"[0/5] Loading raw texts from {args.jsonl}...")
    chunk_text_lookup = {}
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    c = json.loads(line)
                    chunk_text_lookup[c["chunk_id"]] = c["chunk_text"]
                except: pass
    
    # 1. Load Embedder
    print("[1/5] Loading Embedding Model (bge-m3)...")
    embedder = get_embedding_model()
    
    # 2. Load Index
    print(f"[2/5] Loading FAISS and BM25 Indexes from {args.index}...")
    try:
        index_data = load_eval_index(args.index)
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)
        
    faiss_index = index_data["faiss_index"]
    chunk_ids = index_data["chunk_ids"]
    chunk_meta = index_data["chunk_meta"]
    bm25 = index_data["bm25"]
    bm25_chunk_ids = index_data["bm25_chunk_ids"]
    
    # Build a lookup dict for metadata
    meta_lookup = {meta["chunk_id"]: meta for meta in chunk_meta}

    # 3. Search
    print("[3/5] Searching for the best chunk...")
    query_vec = np.array(embedder.embed_documents([args.query])[0], dtype=np.float32)
    
    dense_results = dense_search(query_vec, faiss_index, chunk_ids)
    sparse_results = sparse_search(args.query, bm25, bm25_chunk_ids)
    hybrid_results = rrf_fuse(dense_results, sparse_results)
    
    # Get top 3
    top_results = hybrid_results[:3]
    
    docs: List[RetrievedDoc] = []
    for cid, score in top_results:
        meta = meta_lookup.get(cid)
        if meta:
            meta["chunk_text"] = chunk_text_lookup.get(cid, "Text not found.")
            docs.append(build_retrieved_doc(cid, score, meta))
            
    print(f"      ✅ Found best chunk: {docs[0].chunk_id}")
    print("-" * 60)
    print(f"CONTEXT (English Chunk):\n{docs[0].chunk_text}")
    print("-" * 60)

    # 4. Generate Answer with LLM
    print("[4/5] Sending to Gemini to generate Vietnamese answer...")
    llm = get_llm_client()
    answer = llm.answer(args.query, docs)
    
    print("\n" + "=" * 60)
    print(" 🤖 GEMINI'S FINAL ANSWER")
    print("=" * 60)
    print(f"Câu hỏi: {args.query}\n")
    print(answer.text_vi)
    print("\n" + "=" * 60)
    
if __name__ == "__main__":
    main()
