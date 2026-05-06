"""
data/chunked/eval_retrieval.py
═══════════════════════════════════════════════════════════════════════
Evaluate retrieval quality: run test queries against FAISS/BM25 indexes,
compute Hit@k and MRR for dense, sparse, and hybrid (RRF) paths.

This is Script 3 of the retrieval-eval pipeline:

    test_queries.jsonl + index/eval/<run>/ ──► [THIS SCRIPT]
                                                     │
                                              eval_report.json
                                              + console table

This is the DECISION TOOL: whichever chunking strategy yields the
highest Hit@5 / MRR wins.

Success criteria (from the leader's plan):
    Hit@5 ≥ 0.70  — reasonable for a working RAG demo
    Hit@5 < 0.40  — something is fundamentally broken
    MRR   ≥ 0.50  — right chunk usually in top 2, good UX

Usage
─────
    # Evaluate single index
    python -m data.chunked.eval_retrieval \\
        --queries data/chunked/test_queries.jsonl \\
        --index index/eval/t072

    # Compare multiple chunking strategies
    python -m data.chunked.eval_retrieval \\
        --queries data/chunked/test_queries.jsonl \\
        --compare index/eval/t055 index/eval/t072 index/eval/no_merge

Run from the project root so module imports resolve.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

# Project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.embeddings import get_embedding_model  # noqa: E402
from core.utils import setup_logging               # noqa: E402

logger = logging.getLogger(__name__)

# Force UTF-8 on Windows console for table rendering
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

# Default paths
DEFAULT_QUERIES = _PROJECT_ROOT / "data" / "chunked" / "test_queries.jsonl"

# RRF parameters — match core/config.py
RRF_ALPHA = 0.6          # dense weight
RRF_K = 60               # RRF constant (standard in IR literature)
TOP_K_RETRIEVAL = 20     # top-k for each path before fusion


# ════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class RetrievalMetrics:
    """Aggregated metrics for one retrieval path on one index."""
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr: float = 0.0
    n_queries: int = 0

    def add_rank(self, rank: Optional[int]) -> None:
        """
        Add the rank of the expected chunk in the retrieved list.
        rank is 1-based, or None if not found in top-k.
        """
        self.n_queries += 1
        if rank is not None:
            if rank <= 1:
                self.hit_at_1 += 1
            if rank <= 3:
                self.hit_at_3 += 1
            if rank <= 5:
                self.hit_at_5 += 1
            if rank <= 10:
                self.hit_at_10 += 1
            self.mrr += 1.0 / rank

    def finalize(self) -> None:
        """Normalize counts to rates."""
        if self.n_queries == 0:
            return
        self.hit_at_1 /= self.n_queries
        self.hit_at_3 /= self.n_queries
        self.hit_at_5 /= self.n_queries
        self.hit_at_10 /= self.n_queries
        self.mrr /= self.n_queries

    def to_dict(self) -> Dict[str, float]:
        return {
            "hit@1": round(self.hit_at_1, 4),
            "hit@3": round(self.hit_at_3, 4),
            "hit@5": round(self.hit_at_5, 4),
            "hit@10": round(self.hit_at_10, 4),
            "mrr": round(self.mrr, 4),
            "n_queries": self.n_queries,
        }


# ════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════

def load_queries(path: Path) -> List[Dict[str, Any]]:
    """Load test queries from JSONL."""
    queries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    logger.info("Loaded %d test queries from %s", len(queries), path)
    return queries


def load_eval_index(index_dir: Path) -> Dict[str, Any]:
    """
    Load FAISS index + BM25 bundle + metadata from an eval index directory.

    Expected files:
        index.faiss      — FAISS IndexFlatIP
        chunk_ids.pkl    — ordered chunk IDs (position i = vector i)
        chunk_meta.pkl   — ordered metadata dicts
        bm25_bundle.pkl  — {bm25, chunk_ids, tokenized_corpus}
    """
    faiss_path = index_dir / "index.faiss"
    ids_path = index_dir / "chunk_ids.pkl"
    meta_path = index_dir / "chunk_meta.pkl"
    bm25_path = index_dir / "bm25_bundle.pkl"

    for p in [faiss_path, ids_path, meta_path, bm25_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing index file: {p}")

    faiss_index = faiss.read_index(str(faiss_path))
    with open(ids_path, "rb") as f:
        chunk_ids = pickle.load(f)
    with open(meta_path, "rb") as f:
        chunk_meta = pickle.load(f)
    with open(bm25_path, "rb") as f:
        bm25_bundle = pickle.load(f)

    logger.info(
        "Loaded index from %s: %d vectors, %d chunk_ids",
        index_dir, faiss_index.ntotal, len(chunk_ids),
    )
    return {
        "faiss_index": faiss_index,
        "chunk_ids": chunk_ids,
        "chunk_meta": chunk_meta,
        "bm25": bm25_bundle["bm25"],
        "bm25_chunk_ids": bm25_bundle["chunk_ids"],
    }


# ════════════════════════════════════════════════════════════════
# RETRIEVAL PATHS
# ════════════════════════════════════════════════════════════════

def dense_search(
    query_vec: np.ndarray,
    faiss_index,
    chunk_ids: List[str],
    k: int = TOP_K_RETRIEVAL,
) -> List[Tuple[str, float]]:
    """
    FAISS inner-product search.

    Returns list of (chunk_id, score) sorted by score descending.
    """
    query_vec = query_vec.reshape(1, -1).astype(np.float32)
    scores, indices = faiss_index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunk_ids):
            continue
        results.append((chunk_ids[idx], float(score)))
    return results


def sparse_search(
    query_text: str,
    bm25,
    chunk_ids: List[str],
    k: int = TOP_K_RETRIEVAL,
) -> List[Tuple[str, float]]:
    """
    BM25 keyword search.

    Tokenization: whitespace lowercase — matches production retriever.
    For VI queries against EN chunks this will underperform, but that's
    expected — the dense path handles cross-lingual via bge-m3.
    """
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        if idx < len(chunk_ids) and scores[idx] > 0:
            results.append((chunk_ids[idx], float(scores[idx])))
    return results


def rrf_fuse(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    alpha: float = RRF_ALPHA,
    k_rrf: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.

    score(doc) = α / (rank_dense + k) + (1-α) / (rank_sparse + k)

    Same formula as core/s02_hybrid_search.py._rrf_fuse().
    """
    rrf_scores: Dict[str, float] = {}

    for rank, (cid, _score) in enumerate(dense_results):
        contrib = alpha / (rank + 1 + k_rrf)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contrib

    for rank, (cid, _score) in enumerate(sparse_results):
        contrib = (1 - alpha) / (rank + 1 + k_rrf)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contrib

    # Sort by RRF score descending
    sorted_results = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return sorted_results


def find_rank(results: List[Tuple[str, float]], target_chunk_id: str) -> Optional[int]:
    """
    Find 1-based rank of target_chunk_id in results.
    Returns None if not found.
    """
    for i, (cid, _score) in enumerate(results):
        if cid == target_chunk_id:
            return i + 1
    return None


# ════════════════════════════════════════════════════════════════
# EVALUATION CORE
# ════════════════════════════════════════════════════════════════

def evaluate_index(
    queries: List[Dict[str, Any]],
    index_data: Dict[str, Any],
    embedder,
) -> Dict[str, Any]:
    """
    Run all queries against one index and compute metrics.

    Returns dict with:
        - dense_metrics, sparse_metrics, hybrid_metrics: RetrievalMetrics
        - per_course: {course: {dense: metrics, sparse: metrics, hybrid: metrics}}
        - per_query: list of per-query results for debugging
    """
    dense_metrics = RetrievalMetrics()
    sparse_metrics = RetrievalMetrics()
    hybrid_metrics = RetrievalMetrics()

    per_course: Dict[str, Dict[str, RetrievalMetrics]] = defaultdict(
        lambda: {
            "dense": RetrievalMetrics(),
            "sparse": RetrievalMetrics(),
            "hybrid": RetrievalMetrics(),
        }
    )

    per_query: List[Dict[str, Any]] = []

    # Batch-embed all queries at once for efficiency
    query_texts = [q["query_vi"] for q in queries]
    logger.info("Embedding %d queries...", len(query_texts))
    query_vecs = np.array(
        embedder.embed_documents(query_texts), dtype=np.float32
    )

    faiss_index = index_data["faiss_index"]
    chunk_ids = index_data["chunk_ids"]
    bm25 = index_data["bm25"]
    bm25_chunk_ids = index_data["bm25_chunk_ids"]

    for i, query in enumerate(queries):
        expected_cid = query["expected_chunk_id"]
        course = query.get("course", "unknown")
        query_vi = query["query_vi"]

        # Dense path
        dense_results = dense_search(query_vecs[i], faiss_index, chunk_ids)
        dense_rank = find_rank(dense_results, expected_cid)

        # Sparse path
        sparse_results = sparse_search(query_vi, bm25, bm25_chunk_ids)
        sparse_rank = find_rank(sparse_results, expected_cid)

        # Hybrid path (RRF)
        hybrid_results = rrf_fuse(dense_results, sparse_results)
        hybrid_rank = find_rank(hybrid_results, expected_cid)

        # Record metrics
        dense_metrics.add_rank(dense_rank)
        sparse_metrics.add_rank(sparse_rank)
        hybrid_metrics.add_rank(hybrid_rank)

        per_course[course]["dense"].add_rank(dense_rank)
        per_course[course]["sparse"].add_rank(sparse_rank)
        per_course[course]["hybrid"].add_rank(hybrid_rank)

        per_query.append({
            "query_id": query.get("query_id", f"q{i}"),
            "query_vi": query_vi[:80],
            "expected_chunk_id": expected_cid,
            "course": course,
            "dense_rank": dense_rank,
            "sparse_rank": sparse_rank,
            "hybrid_rank": hybrid_rank,
        })

    # Finalize
    dense_metrics.finalize()
    sparse_metrics.finalize()
    hybrid_metrics.finalize()

    for course_metrics in per_course.values():
        for m in course_metrics.values():
            m.finalize()

    return {
        "dense": dense_metrics,
        "sparse": sparse_metrics,
        "hybrid": hybrid_metrics,
        "per_course": dict(per_course),
        "per_query": per_query,
    }


# ════════════════════════════════════════════════════════════════
# RENDERING
# ════════════════════════════════════════════════════════════════

def render_metrics_table(
    metrics: Dict[str, RetrievalMetrics],
    title: str,
) -> str:
    """Render a metrics table for dense/sparse/hybrid paths."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)
    lines.append(f"  {'Path':<10} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8} {'N':>6}")
    lines.append(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")

    for path_name, m in metrics.items():
        lines.append(
            f"  {path_name:<10} "
            f"{m.hit_at_1:>8.3f} "
            f"{m.hit_at_3:>8.3f} "
            f"{m.hit_at_5:>8.3f} "
            f"{m.hit_at_10:>8.3f} "
            f"{m.mrr:>8.3f} "
            f"{m.n_queries:>6}"
        )

    return "\n".join(lines)


def render_per_course(
    per_course: Dict[str, Dict[str, RetrievalMetrics]],
) -> str:
    """Render per-course breakdown for hybrid path."""
    lines = []
    lines.append("")
    lines.append("-" * 70)
    lines.append("  PER-COURSE BREAKDOWN (hybrid path)")
    lines.append("-" * 70)
    lines.append(f"  {'Course':<30} {'Hit@5':>8} {'MRR':>8} {'N':>6}")
    lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6}")

    for course in sorted(per_course.keys()):
        m = per_course[course]["hybrid"]
        lines.append(
            f"  {course:<30} "
            f"{m.hit_at_5:>8.3f} "
            f"{m.mrr:>8.3f} "
            f"{m.n_queries:>6}"
        )

    return "\n".join(lines)


def render_verdict(hybrid: RetrievalMetrics) -> str:
    """Render overall verdict based on success criteria."""
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  VERDICT")
    lines.append("=" * 70)

    h5 = hybrid.hit_at_5
    mrr = hybrid.mrr

    if h5 >= 0.70:
        lines.append(f"  Hit@5 = {h5:.3f} >= 0.70  -- GOOD. Working RAG demo quality.")
    elif h5 >= 0.40:
        lines.append(f"  Hit@5 = {h5:.3f} (0.40-0.70)  -- NEEDS IMPROVEMENT.")
        lines.append("    Consider: different threshold, better query generation, or check embedder.")
    else:
        lines.append(f"  Hit@5 = {h5:.3f} < 0.40  -- BROKEN. Investigate embedder or queries.")

    if mrr >= 0.50:
        lines.append(f"  MRR   = {mrr:.3f} >= 0.50  -- GOOD. Right chunk usually in top 2.")
    else:
        lines.append(f"  MRR   = {mrr:.3f} < 0.50  -- Right chunk often ranked low.")

    return "\n".join(lines)


def render_comparison_table(
    results: Dict[str, Dict[str, RetrievalMetrics]],
) -> str:
    """Render side-by-side comparison table for multiple runs."""
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  COMPARISON: HYBRID PATH ACROSS CHUNKING STRATEGIES")
    lines.append("=" * 80)

    run_names = list(results.keys())
    header = f"  {'Metric':<12}"
    for name in run_names:
        header += f" {name:>14}"
    lines.append(header)
    lines.append(f"  {'-'*12}" + f" {'-'*14}" * len(run_names))

    metric_names = [
        ("Hit@1", "hit_at_1"),
        ("Hit@3", "hit_at_3"),
        ("Hit@5", "hit_at_5"),
        ("Hit@10", "hit_at_10"),
        ("MRR", "mrr"),
    ]

    for display_name, attr_name in metric_names:
        row = f"  {display_name:<12}"
        values = []
        for name in run_names:
            m = results[name]["hybrid"]
            val = getattr(m, attr_name)
            values.append(val)
            row += f" {val:>14.3f}"
        # Mark the best
        if values:
            best_idx = values.index(max(values))
            # Re-render with marker
            row = f"  {display_name:<12}"
            for j, name in enumerate(run_names):
                val = values[j]
                marker = " *" if j == best_idx and len(values) > 1 else "  "
                row += f" {val:>12.3f}{marker}"
        lines.append(row)

    lines.append("")
    lines.append("  (* = best)")

    # Determine winner
    if len(run_names) > 1:
        best_run = max(run_names, key=lambda n: results[n]["hybrid"].hit_at_5)
        best_h5 = results[best_run]["hybrid"].hit_at_5
        best_mrr = results[best_run]["hybrid"].mrr
        lines.append("")
        lines.append(f"  WINNER: {best_run} (Hit@5={best_h5:.3f}, MRR={best_mrr:.3f})")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════

def save_report(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Save evaluation report as JSON."""
    report = {}
    for run_name, result in results.items():
        report[run_name] = {
            "dense": result["dense"].to_dict(),
            "sparse": result["sparse"].to_dict(),
            "hybrid": result["hybrid"].to_dict(),
            "per_course": {
                course: {
                    path: metrics.to_dict()
                    for path, metrics in course_data.items()
                }
                for course, course_data in result["per_course"].items()
            },
            "per_query": result.get("per_query", []),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", output_path)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality: Hit@k and MRR for chunking strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single index evaluation\n"
            "  python -m data.chunked.eval_retrieval \\\n"
            "      --queries data/chunked/test_queries.jsonl \\\n"
            "      --index index/eval/t072\n\n"
            "  # Compare multiple chunking strategies\n"
            "  python -m data.chunked.eval_retrieval \\\n"
            "      --queries data/chunked/test_queries.jsonl \\\n"
            "      --compare index/eval/t055 index/eval/t072\n"
        ),
    )
    parser.add_argument(
        "--queries", type=Path, default=DEFAULT_QUERIES,
        help=f"Test queries JSONL (default: {DEFAULT_QUERIES})",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--index", type=Path, default=None,
        help="Single index directory to evaluate",
    )
    group.add_argument(
        "--compare", type=Path, nargs="+", default=None,
        help="Multiple index directories to compare side-by-side",
    )

    parser.add_argument(
        "--report", type=Path, default=None,
        help="Output path for JSON report (default: eval_report.json in first index dir)",
    )
    parser.add_argument(
        "--show-queries", action="store_true",
        help="Print exactly what rank the correct chunk got for each individual query",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.queries.exists():
        logger.error("Queries file not found: %s", args.queries)
        sys.exit(1)

    # Determine which index directories to evaluate
    if args.index:
        index_dirs = [args.index]
    else:
        index_dirs = args.compare

    for d in index_dirs:
        if not d.exists():
            logger.error("Index directory not found: %s", d)
            sys.exit(1)

    # Load queries
    queries = load_queries(args.queries)
    if not queries:
        logger.error("No queries loaded. Aborting.")
        sys.exit(1)

    # Load embedder (shared across all runs — bge-m3)
    logger.info("Loading embedding model (bge-m3)...")
    embedder = get_embedding_model()

    # Evaluate each index
    all_results: Dict[str, Dict[str, Any]] = {}
    for index_dir in index_dirs:
        run_name = index_dir.name
        logger.info("")
        logger.info("=" * 60)
        logger.info("EVALUATING: %s", run_name)
        logger.info("=" * 60)

        index_data = load_eval_index(index_dir)
        t0 = time.time()
        result = evaluate_index(queries, index_data, embedder)
        elapsed = time.time() - t0

        logger.info("Evaluation complete in %.1fs", elapsed)
        all_results[run_name] = result

        # Print per-run results
        print(render_metrics_table(
            {"dense": result["dense"], "sparse": result["sparse"], "hybrid": result["hybrid"]},
            f"RESULTS: {run_name}",
        ))
        
        if args.show_queries:
            print("\n  INDIVIDUAL QUERY RESULTS (Hybrid Rank):")
            print("  ------------------------------------------------------------")
            for i, q in enumerate(result["per_query"]):
                rank = q["hybrid_rank"]
                rank_str = str(rank) if rank is not None else "NOT FOUND"
                marker = "✅" if rank == 1 else ("⚠️" if rank and rank <= 5 else "❌")
                
                # Get the full query and context from the original queries list
                full_query = queries[i]["query_vi"]
                context = queries[i].get("source_text", "").strip()
                
                print(f"  {marker} Rank {rank_str:<5} | {q['expected_chunk_id']}")
                print(f"  Q: {full_query}")
                print(f"  A: {context}")
                print("  ------------------------------------------------------------")
            
        print(render_per_course(result["per_course"]))
        print(render_verdict(result["hybrid"]))

    # Comparison table (if multiple runs)
    if len(all_results) > 1:
        print(render_comparison_table(all_results))

    # Save report
    report_path = args.report
    if report_path is None:
        report_path = index_dirs[0].parent / "eval_report.json"
    save_report(all_results, report_path)


if __name__ == "__main__":
    main()
