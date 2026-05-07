import argparse
from pathlib import Path
from typing import Any


DEFAULT_DATA_PATH = "/Users/carwyn/Downloads/transcript_v3_t072.jsonl"
DEFAULT_FAISS_INDEX_PATH = "indexes/faiss_index_072"
DEFAULT_BM25_PATH = "indexes/bm25_072.pkl"
DEFAULT_QUERY = "Gradient descent la gi?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test hybrid_improved + query_writer retrieval flow."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Vietnamese or English query to test.",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to chunked transcript jsonl file.",
    )
    parser.add_argument(
        "--faiss-index-path",
        default=DEFAULT_FAISS_INDEX_PATH,
        help="Path to FAISS index directory.",
    )
    parser.add_argument(
        "--bm25-path",
        default=DEFAULT_BM25_PATH,
        help="Path to BM25 pickle file.",
    )
    parser.add_argument("--k-retrieve", type=int, default=20)
    parser.add_argument("--k-final", type=int, default=6)
    parser.add_argument("--k-rerank", type=int, default=3)
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model name.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranker for faster testing.",
    )
    parser.add_argument(
        "--no-query-writer",
        action="store_true",
        help="Use the original query directly instead of query_writer.",
    )
    parser.add_argument(
        "--query-writer-model",
        default="qwen2.5:3b",
        help="Ollama model used by query_writer.",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Optional cache file for rewritten queries.",
    )
    parser.add_argument(
        "--weight-bm25",
        type=float,
        default=0.2,
        help="BM25 weight for hybrid_improved weighted fusion.",
    )
    parser.add_argument(
        "--weight-dense",
        type=float,
        default=0.8,
        help="Dense/FAISS weight for hybrid_improved weighted fusion.",
    )
    return parser.parse_args()


def build_queries(args: argparse.Namespace) -> dict[str, str]:
    if args.no_query_writer:
        return {"original": args.query}

    try:
        from query_writer import rewrite_vietnamese_query
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import query_writer dependencies. "
            "Install them or run with --no-query-writer."
        ) from exc

    rewritten = rewrite_vietnamese_query(
        vietnamese_query=args.query,
        model=args.query_writer_model,
        cache_file=args.cache_file,
    )

    queries = {
        "original": rewritten["original"],
        "q1_literal": rewritten["q1"],
        "q2_expanded": rewritten["q2"],
    }
    return queries


def unpack_result(result: Any) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(result, dict):
        return result.get("documents", []), result.get("trace", {})
    return result, {}


def print_documents(docs: list[Any], trace: dict[str, Any]) -> None:
    fusion_details = trace.get("fusion_details", {})

    for i, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get("doc_id", "")
        score = fusion_details.get(doc_id, {}).get("final")
        rerank_score = doc.metadata.get("rerank_score")

        print(f"\n--- Document {i} ---")
        print(f"doc_id: {doc_id}")
        if score is not None:
            print(f"fusion_score: {score}")
        if rerank_score is not None:
            print(f"rerank_score: {rerank_score}")
        print("title:", doc.metadata.get("title"))
        print("url:", doc.metadata.get("url"))
        print("metadata:", doc.metadata)
        print("content:")
        print(doc.page_content[:700].replace("\n", " ") + "...")


def print_trace(trace: dict[str, Any]) -> None:
    if not trace:
        return

    print("\nTrace:")
    print("num_bm25_results:", trace.get("num_bm25_results"))
    print("num_faiss_results:", trace.get("num_faiss_results"))
    print("num_fused:", trace.get("num_fused"))
    print("weights:", trace.get("weights"))
    print("fusion_scores:", trace.get("fusion_scores"))


def main() -> None:
    args = parse_args()

    if not Path(args.data_path).exists():
        print(f"Warning: data path does not exist yet: {args.data_path}")
        print("If FAISS/BM25 indexes already exist, loading may still work.")

    queries = build_queries(args)

    print("\nQueries:")
    for name, query in queries.items():
        print(f"- {name}: {query}")

    from hybrid_improved import build_hybrid_retriever

    retriever = build_hybrid_retriever(
        data_path=args.data_path,
        faiss_index_path=args.faiss_index_path,
        bm25_path=args.bm25_path,
        k_retrieve=args.k_retrieve,
        k_final=args.k_final,
        k_rerank=args.k_rerank,
        reranker_model=args.reranker_model,
        use_reranker=not args.no_reranker,
        weight_bm25=args.weight_bm25,
        weight_dense=args.weight_dense,
        enable_tracing=True,
    )

    combined_docs = {}

    for name, query in queries.items():
        print(f"\n================ {name}: {query} ================")
        result = retriever(query)
        docs, trace = unpack_result(result)
        print(f"Retrieved docs: {len(docs)}")
        print_documents(docs, trace)
        print_trace(trace)

        for doc in docs:
            key = doc.metadata.get("doc_id") or id(doc)
            combined_docs[key] = doc

    print("\n================ Combined unique docs ================")
    print(f"Unique docs across all queries: {len(combined_docs)}")
    for i, doc in enumerate(combined_docs.values(), 1):
        print(
            f"{i}. doc_id={doc.metadata.get('doc_id')} "
            f"title={doc.metadata.get('title')} "
            f"rerank_score={doc.metadata.get('rerank_score')}"
        )


if __name__ == "__main__":
    main()
