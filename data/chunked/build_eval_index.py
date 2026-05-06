"""
data/chunked/build_eval_index.py
═══════════════════════════════════════════════════════════════════════
Build FAISS (dense) + BM25 (sparse) indexes for retrieval evaluation.

This is Script 1 of the retrieval-eval pipeline:

    transcript_v3.jsonl ──► [THIS SCRIPT] ──► index/eval/<run_name>/
                                                  ├── index.faiss
                                                  ├── chunk_ids.pkl
                                                  ├── chunk_meta.pkl
                                                  └── bm25_bundle.pkl

Uses raw faiss-cpu IndexFlatIP (NOT LangChain FAISS) because the eval
pipeline only needs chunk_id ↔ vector-position mapping, not full
LangChain Document objects. Simpler, faster, fewer moving parts.

The embedding model is bge-m3 (multilingual) loaded via core/embeddings.py
— the same model used for production retrieval. This ensures eval results
reflect real retrieval quality.

Usage
─────
    # Build index for a specific chunking strategy
    python -m data.chunked.build_eval_index \\
        --input data/chunked/transcript_v3_t072.jsonl \\
        --run-name t072

    # Different threshold output
    python -m data.chunked.build_eval_index \\
        --input data/chunked/transcript_v3_t055.jsonl \\
        --run-name t055

Run from the project root so module imports resolve.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Project root on sys.path so `from core...` works when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.embeddings import get_embedding_model  # noqa: E402
from core.utils import setup_logging               # noqa: E402

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
DEFAULT_INPUT = _PROJECT_ROOT / "data" / "chunked" / "transcript_v3.jsonl"
DEFAULT_INDEX_ROOT = _PROJECT_ROOT / "index" / "eval"


# ════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """Load chunked JSONL. Each line must have at least chunk_id and chunk_text."""
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: bad JSON, skipping: %s", line_num, e)
                continue

            if not record.get("chunk_text"):
                logger.warning("Line %d: missing chunk_text, skipping", line_num)
                continue

            chunks.append(record)

    logger.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


# ════════════════════════════════════════════════════════════════
# BUILD FAISS INDEX
# ════════════════════════════════════════════════════════════════

def build_faiss_index(
    chunks: List[Dict[str, Any]],
    embedder,
    batch_size: int = 32,
) -> tuple:
    """
    Embed all chunk texts with bge-m3 and build FAISS IndexFlatIP.

    bge-m3 returns normalized vectors (via core/config.py normalize_embeddings=True),
    so inner product == cosine similarity. IndexFlatIP is the right choice.

    Returns:
        (faiss_index, chunk_ids, chunk_metas)
        - faiss_index: faiss.IndexFlatIP with N vectors of dim D
        - chunk_ids: list[str] of length N (position i = vector i)
        - chunk_metas: list[dict] of length N (metadata for eval lookup)
    """
    texts = [c["chunk_text"] for c in chunks]
    n_total = len(texts)

    logger.info("Embedding %d chunks with bge-m3 (batch_size=%d)...", n_total, batch_size)
    logger.info("This will take 30-60 minutes on CPU for ~11,500 chunks.")

    all_embeddings = []
    t0 = time.time()

    for i in tqdm(range(0, n_total, batch_size), desc="Embedding chunks"):
        batch_texts = texts[i : i + batch_size]
        batch_vecs = embedder.embed_documents(batch_texts)
        all_embeddings.extend(batch_vecs)

    elapsed = time.time() - t0
    logger.info("Embedding complete: %d vectors in %.1fs (%.1f chunks/s)",
                n_total, elapsed, n_total / elapsed if elapsed > 0 else 0)

    # Convert to numpy and build FAISS index
    matrix = np.array(all_embeddings, dtype=np.float32)
    dim = matrix.shape[1]
    logger.info("Vector dimension: %d", dim)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    logger.info("FAISS IndexFlatIP built: %d vectors", index.ntotal)

    # Build ordered lists for position → chunk_id mapping
    chunk_ids = [c.get("chunk_id", f"unknown_{i}") for i, c in enumerate(chunks)]
    chunk_metas = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "video_id": c.get("video_id", ""),
            "title": c.get("title", ""),
            "course": c.get("course", ""),
            "start_time": c.get("start_time", 0.0),
            "end_time": c.get("end_time", 0.0),
            "token_count": c.get("token_count", 0),
            "chunk_type": c.get("chunk_type", ""),
        }
        for c in chunks
    ]

    return index, chunk_ids, chunk_metas


# ════════════════════════════════════════════════════════════════
# BUILD BM25 INDEX
# ════════════════════════════════════════════════════════════════

def build_bm25(chunks: List[Dict[str, Any]]) -> dict:
    """
    Build BM25Okapi from chunk texts.

    Tokenization: whitespace + lowercase — same as the production retriever
    in core/s02_hybrid_search.py. For VI queries this will underperform
    (English chunks, Vietnamese query tokens won't match), but it's a
    baseline. The dense path handles cross-lingual via bge-m3.

    Returns a bundle dict with:
        - bm25: the BM25Okapi object
        - chunk_ids: ordered list matching BM25 internal doc indices
        - tokenized_corpus: the tokenized texts (for debugging)
    """
    texts = [c.get("chunk_text", "") for c in chunks]
    chunk_ids = [c.get("chunk_id", f"unknown_{i}") for i, c in enumerate(chunks)]

    logger.info("Building BM25 index on %d texts...", len(texts))
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    logger.info("BM25 index built: %d documents", len(tokenized))

    return {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "tokenized_corpus": tokenized,
    }


# ════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════

def save_index(
    out_dir: Path,
    faiss_index,
    chunk_ids: List[str],
    chunk_metas: List[Dict],
    bm25_bundle: dict,
) -> None:
    """Save all index artifacts to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # FAISS index
    faiss_path = out_dir / "index.faiss"
    faiss.write_index(faiss_index, str(faiss_path))
    logger.info("Saved FAISS index: %s", faiss_path)

    # Chunk IDs (position i = FAISS vector i)
    ids_path = out_dir / "chunk_ids.pkl"
    with open(ids_path, "wb") as f:
        pickle.dump(chunk_ids, f)
    logger.info("Saved chunk IDs: %s", ids_path)

    # Chunk metadata (for eval lookup — course, video_id, times, etc.)
    meta_path = out_dir / "chunk_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(chunk_metas, f)
    logger.info("Saved chunk metadata: %s", meta_path)

    # BM25 bundle (bm25 object + chunk_ids + tokenized corpus)
    bm25_path = out_dir / "bm25_bundle.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_bundle, f)
    logger.info("Saved BM25 bundle: %s", bm25_path)

    logger.info("All artifacts saved to %s", out_dir)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 eval indexes from chunked JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m data.chunked.build_eval_index \\\n"
            "      --input data/chunked/transcript_v3_t072.jsonl \\\n"
            "      --run-name t072\n"
        ),
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Chunked JSONL file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Name for this index run (e.g. 't072', 'no_merge'). Output goes to index/eval/<run-name>/",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Embedding batch size (default: 32, reduce to 16 if OOM)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only index the first N chunks (for quick testing)",
    )
    parser.add_argument(
        "--filter-test-queries", type=Path, default=None,
        help="Path to test_queries.jsonl. Only indexes videos that appear in these queries (massive speedup for testing)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    out_dir = DEFAULT_INDEX_ROOT / args.run_name
    logger.info("=" * 60)
    logger.info("BUILD EVAL INDEX")
    logger.info("  Input : %s", args.input)
    logger.info("  Output: %s", out_dir)
    logger.info("  Batch : %d", args.batch_size)
    logger.info("=" * 60)

    # 1. Load chunks
    chunks = load_chunks(args.input)
    if not chunks:
        logger.error("No chunks loaded. Aborting.")
        sys.exit(1)

    if args.filter_test_queries:
        if not args.filter_test_queries.exists():
            logger.error("Test queries file not found: %s", args.filter_test_queries)
            sys.exit(1)
        # Extract target video IDs
        target_vids = set()
        with args.filter_test_queries.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        q = json.loads(line)
                        if "expected_video_id" in q:
                            target_vids.add(q["expected_video_id"])
                    except json.JSONDecodeError:
                        pass
        logger.info("Filtering chunks to match %d videos from test queries...", len(target_vids))
        chunks = [c for c in chunks if c.get("video_id") in target_vids]

    if args.limit:
        chunks = chunks[:args.limit]
        logger.info("Limited to first %d chunks for fast testing.", len(chunks))
    
    if not chunks:
        logger.error("No chunks left after filtering! Aborting.")
        sys.exit(1)

    # 2. Load embedder (bge-m3 via core/embeddings.py)
    logger.info("Loading embedding model (bge-m3)...")
    embedder = get_embedding_model()

    # 3. Build FAISS index
    faiss_index, chunk_ids, chunk_metas = build_faiss_index(
        chunks, embedder, batch_size=args.batch_size
    )

    # 4. Build BM25 index
    bm25_bundle = build_bm25(chunks)

    # 5. Save everything
    save_index(out_dir, faiss_index, chunk_ids, chunk_metas, bm25_bundle)

    # Summary
    courses = set(m["course"] for m in chunk_metas if m.get("course"))
    logger.info("")
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("  Chunks indexed : %d", len(chunk_ids))
    logger.info("  FAISS vectors  : %d", faiss_index.ntotal)
    logger.info("  Courses        : %s", ", ".join(sorted(courses)))
    logger.info("  Output dir     : %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
