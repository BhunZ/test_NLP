import argparse
import json
import time
from pathlib import Path
from typing import Any


# Project-relative paths (was hardcoded /Users/carwyn/... — now portable).
# evaluation/metrics_eval.py → repo root is parents[1]
# Also fixed: faiss_index_072 → faiss_index_072_n (the current rebuilt index)
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = str(_REPO_ROOT / "data" / "chunked" / "transcript_v3_t072.jsonl")
DEFAULT_FAISS_INDEX_PATH = str(_REPO_ROOT / "indexes" / "faiss_index_072_n")
DEFAULT_BM25_PATH = str(_REPO_ROOT / "indexes" / "bm25_072.pkl")
DEFAULT_K_VALUES = [1, 3, 6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval and rerank metrics."
    )
    parser.add_argument(
        "--queries-file",
        required=True,
        help=(
            "JSON or JSONL file. Each item needs query/query_vi and either "
            "relevant_doc_ids, relevant_chunk_ids, or graded_relevance."
        ),
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--faiss-index-path", default=DEFAULT_FAISS_INDEX_PATH)
    parser.add_argument("--bm25-path", default=DEFAULT_BM25_PATH)
    parser.add_argument("--k-retrieve", type=int, default=20)
    parser.add_argument("--k-final", type=int, default=6)
    parser.add_argument("--k-rerank", type=int, default=3)
    parser.add_argument("--k-values", default="1,3,6")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--weight-bm25", type=float, default=0.2)
    parser.add_argument("--weight-dense", type=float, default=0.8)
    parser.add_argument("--fusion-method", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save per-query results as JSON.",
    )
    return parser.parse_args()


def load_eval_queries(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if file_path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("Queries file must contain a list or {'queries': [...]}.")
    return data


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("--k-values must contain at least one integer.")
    return values


def get_query_text(item: dict[str, Any]) -> str:
    for key in ("query", "query_vi", "question", "question_vi"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Missing query text in item: {item}")


def get_relevance_map(item: dict[str, Any]) -> dict[str, float]:
    graded = item.get("graded_relevance") or item.get("relevance")
    if isinstance(graded, dict):
        return {str(key): float(value) for key, value in graded.items()}

    relevance: dict[str, float] = {}
    for field in ("relevant_doc_ids", "relevant_chunk_ids", "relevant_ids"):
        values = item.get(field)
        if isinstance(values, list):
            for value in values:
                relevance[str(value)] = 1.0

    if not relevance:
        raise ValueError(
            "Each query needs relevant_doc_ids, relevant_chunk_ids, "
            "relevant_ids, or graded_relevance."
        )
    return relevance


def doc_identifier_candidates(doc: Any) -> list[str]:
    metadata = getattr(doc, "metadata", {}) or {}
    candidates = [
        metadata.get("chunk_id"),
        metadata.get("doc_id"),
        metadata.get("url"),
    ]
    return [str(value) for value in candidates if value is not None]


def relevance_for_doc(doc: Any, relevance_map: dict[str, float]) -> float:
    return max(
        (relevance_map.get(candidate, 0.0) for candidate in doc_identifier_candidates(doc)),
        default=0.0,
    )


def relevance_list(docs: list[Any], relevance_map: dict[str, float]) -> list[float]:
    return [relevance_for_doc(doc, relevance_map) for doc in docs]


def hit_at_k(relevances: list[float], k: int) -> float:
    return 1.0 if any(score > 0 for score in relevances[:k]) else 0.0


def precision_at_k(relevances: list[float], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for score in relevances[:k] if score > 0)
    return hits / k


def recall_at_k(relevances: list[float], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    hits = sum(1 for score in relevances[:k] if score > 0)
    return hits / total_relevant


def average_precision_at_k(relevances: list[float], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, score in enumerate(relevances[:k], start=1):
        if score > 0:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(total_relevant, k)


def reciprocal_rank_at_k(relevances: list[float], k: int) -> float:
    for rank, score in enumerate(relevances[:k], start=1):
        if score > 0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: list[float], k: int) -> float:
    dcg = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        gain = (2.0**rel) - 1.0
        discount = 1.0 / _log2(index + 1)
        dcg += gain * discount
    return dcg


def ndcg_at_k(relevances: list[float], ideal_relevances: list[float], k: int) -> float:
    ideal_dcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


def _log2(value: int) -> float:
    import math

    return math.log(value, 2)


def unpack_result(result: Any) -> list[Any]:
    if isinstance(result, dict):
        return result.get("documents", [])
    return result


def evaluate_docs(
    docs: list[Any],
    relevance_map: dict[str, float],
    k_values: list[int],
) -> dict[str, float]:
    relevances = relevance_list(docs, relevance_map)
    ideal_relevances = list(relevance_map.values())
    total_relevant = sum(1 for score in relevance_map.values() if score > 0)

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = hit_at_k(relevances, k)
        metrics[f"precision@{k}"] = precision_at_k(relevances, k)
        metrics[f"recall@{k}"] = recall_at_k(relevances, total_relevant, k)
        metrics[f"map@{k}"] = average_precision_at_k(relevances, total_relevant, k)
        metrics[f"mrr@{k}"] = reciprocal_rank_at_k(relevances, k)
        metrics[f"dcg@{k}"] = dcg_at_k(relevances, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevances, ideal_relevances, k)
    return metrics


def doc_ids(docs: list[Any]) -> list[str]:
    ids = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        ids.append(str(metadata.get("chunk_id") or metadata.get("doc_id") or ""))
    return ids


def mean_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    metric_keys = sorted(
        key
        for row in rows
        for key in row[prefix].keys()
    )
    summary = {}
    for key in metric_keys:
        values = [row[prefix][key] for row in rows]
        summary[key] = sum(values) / len(values) if values else 0.0
    return summary


def print_summary(title: str, metrics: dict[str, float]) -> None:
    print(f"\n=== {title} ===")
    for key, value in metrics.items():
        print(f"{key:>12}: {value:.4f}")


def main() -> None:
    args = parse_args()
    eval_items = load_eval_queries(args.queries_file)
    if not eval_items:
        raise ValueError("No eval queries found.")

    k_values = parse_k_values(args.k_values)

    from hybrid_improved import build_hybrid_retriever

    retrieval_k_final = max(max(k_values), args.k_final)
    retrieval_retriever = build_hybrid_retriever(
        data_path=args.data_path,
        faiss_index_path=args.faiss_index_path,
        bm25_path=args.bm25_path,
        k_retrieve=max(args.k_retrieve, retrieval_k_final),
        k_final=retrieval_k_final,
        k_rerank=retrieval_k_final,
        reranker_model=args.reranker_model,
        use_reranker=False,
        weight_bm25=args.weight_bm25,
        weight_dense=args.weight_dense,
        fusion_method=args.fusion_method,
        enable_tracing=True,
    )

    rerank_retriever = None
    if not args.no_reranker:
        rerank_retriever = build_hybrid_retriever(
            data_path=args.data_path,
            faiss_index_path=args.faiss_index_path,
            bm25_path=args.bm25_path,
            k_retrieve=args.k_retrieve,
            k_final=args.k_final,
            k_rerank=args.k_rerank,
            reranker_model=args.reranker_model,
            use_reranker=True,
            weight_bm25=args.weight_bm25,
            weight_dense=args.weight_dense,
            fusion_method=args.fusion_method,
            enable_tracing=True,
        )

    rows = []
    for index, item in enumerate(eval_items, start=1):
        query = get_query_text(item)
        relevance_map = get_relevance_map(item)

        t0 = time.time()
        retrieval_docs = unpack_result(retrieval_retriever(query))
        retrieval_latency_ms = (time.time() - t0) * 1000

        retrieval_metrics = evaluate_docs(retrieval_docs, relevance_map, k_values)

        rerank_docs = []
        rerank_metrics = {}
        rerank_latency_ms = None
        if rerank_retriever is not None:
            t0 = time.time()
            rerank_docs = unpack_result(rerank_retriever(query))
            rerank_latency_ms = (time.time() - t0) * 1000
            rerank_metrics = evaluate_docs(rerank_docs, relevance_map, k_values)

        row = {
            "id": item.get("id", index),
            "query": query,
            "retrieval": retrieval_metrics,
            "retrieval_doc_ids": doc_ids(retrieval_docs),
            "retrieval_latency_ms": round(retrieval_latency_ms, 1),
            "rerank": rerank_metrics,
            "rerank_doc_ids": doc_ids(rerank_docs),
            "rerank_latency_ms": (
                round(rerank_latency_ms, 1) if rerank_latency_ms is not None else None
            ),
        }
        rows.append(row)

        print(
            f"[{index}/{len(eval_items)}] {query} | "
            f"retrieval hit@{k_values[-1]}={retrieval_metrics[f'hit@{k_values[-1]}']:.0f}"
        )

    print_summary("Retrieval without rerank", mean_metrics(rows, "retrieval"))
    if rerank_retriever is not None:
        print_summary("After rerank", mean_metrics(rows, "rerank"))

    avg_retrieval_latency = sum(row["retrieval_latency_ms"] for row in rows) / len(rows)
    print(f"\nretrieval latency avg: {avg_retrieval_latency:.1f} ms/query")
    if rerank_retriever is not None:
        latencies = [row["rerank_latency_ms"] for row in rows if row["rerank_latency_ms"] is not None]
        print(f"rerank latency avg   : {sum(latencies) / len(latencies):.1f} ms/query")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved per-query results to {output_path}")


if __name__ == "__main__":
    main()
