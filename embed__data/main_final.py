import argparse
from pathlib import Path
from typing import Any
 
 
DEFAULT_DATA_PATH = "/Users/carwyn/Downloads/transcript_v3_t072.jsonl"
DEFAULT_FAISS_INDEX_PATH = "indexes/faiss_index_072_n"
DEFAULT_BM25_PATH = "indexes/bm25_072.pkl"
DEFAULT_QUERY = "principle maximum likelihood là gì ?"
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid RAG test with weighted (VI) + RRF (EN) retrieval."
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
    parser.add_argument("--k-rerank", type=int, default=6)
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
    # VI weights (original query)
    parser.add_argument(
        "--weight-bm25-vi",
        type=float,
        default=0.3,
        help="BM25 weight for the Vietnamese original query.",
    )
    parser.add_argument(
        "--weight-dense-vi",
        type=float,
        default=0.7,
        help="Dense weight for the Vietnamese original query.",
    )
    # EN fusion method (q2_expanded)
    parser.add_argument(
        "--fusion-method-en",
        choices=["rrf", "weighted"],
        default="rrf",
        help="Fusion method for English query (q2_expanded). "
             "Use 'rrf' or 'weighted' with balanced 0.5/0.5.",
    )
    parser.add_argument(
        "--global-rerank-query",
        choices=["auto", "original", "q1_literal", "q2_expanded"],
        default="auto",
        help=(
            "Query used for final global rerank. "
            "auto uses q2_expanded, then q1_literal, then original."
        ),
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
    return {
        "original": rewritten["original"],
        "q1_literal": rewritten["q1"],
        "q2_expanded": rewritten["q2"],
    }
 
 
def unpack_result(result: Any) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(result, dict):
        return result.get("documents", []), result.get("trace", {})
    return result, {}
 
 
def print_documents(docs: list[Any], trace: dict[str, Any]) -> None:
    fusion_details = trace.get("fusion_details", {})
    for i, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get("chunk_id") or doc.metadata.get("doc_id", "")
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
        print("content:")
        print(doc.page_content[:700].replace("\n", " ") + "...")
 
 
def print_trace(trace: dict[str, Any]) -> None:
    if not trace:
        return
    print("\nTrace:")
    print("num_bm25_results:", trace.get("num_bm25_results"))
    print("num_faiss_results:", trace.get("num_faiss_results"))
    print("num_fused:", trace.get("num_fused"))
    print("fusion_method:", trace.get("fusion_method"))
    print("weights:", trace.get("weights"))


def choose_global_rerank_query(
    queries: dict[str, str],
    mode: str,
) -> tuple[str, str]:
    if mode != "auto":
        return mode, queries.get(mode) or queries.get("original", "")

    for name in ("q2_expanded", "q1_literal", "original"):
        query = queries.get(name)
        if query:
            return name, query

    return "original", ""
 
 
def main() -> None:
    args = parse_args()
 
    if not Path(args.data_path).exists():
        print(f"Warning: data path does not exist yet: {args.data_path}")
        print("If FAISS/BM25 indexes already exist, loading may still work.")
 
    queries = build_queries(args)
 
    print("\nQueries:")
    for name, query in queries.items():
        print(f"  - {name}: {query}")
 
    from hybrid_improved import build_hybrid_retriever
 
    # =========================================================
    # RETRIEVER 1 — Vietnamese original query
    # Weighted fusion: BM25 0.3 / Dense 0.7
    # Tiếng Việt → BM25 yếu hơn vì tokenization kém, dense mạnh hơn
    # =========================================================
    print("\n[Build] Weighted retriever (VI): "
          f"BM25={args.weight_bm25_vi} / Dense={args.weight_dense_vi}")
 
    vi_retriever = build_hybrid_retriever(
        data_path=args.data_path,
        faiss_index_path=args.faiss_index_path,
        bm25_path=args.bm25_path,
        k_retrieve=args.k_retrieve,
        k_final=args.k_final,
        k_rerank=args.k_rerank,
        reranker_model=args.reranker_model,
        use_reranker=False,          # rerank globally at the end
        weight_bm25=args.weight_bm25_vi,
        weight_dense=args.weight_dense_vi,
        fusion_method="weighted",
        enable_tracing=True,
    )
 
    # =========================================================
    # RETRIEVER 2 — English query (q2_expanded)
    # RRF hoặc Weighted 0.5/0.5
    # Tiếng Anh → BM25 tốt hơn, cân bằng với dense
    # =========================================================
    en_fusion = args.fusion_method_en
    en_bm25_w = 0.5 if en_fusion == "weighted" else None
    en_dense_w = 0.5 if en_fusion == "weighted" else None
 
    print(f"[Build] EN retriever: fusion={en_fusion}"
          + (f", BM25={en_bm25_w} / Dense={en_dense_w}"
             if en_fusion == "weighted" else ""))
 
    en_retriever = build_hybrid_retriever(
        data_path=args.data_path,
        faiss_index_path=args.faiss_index_path,
        bm25_path=args.bm25_path,
        k_retrieve=args.k_retrieve,
        k_final=args.k_final,
        k_rerank=args.k_rerank,
        reranker_model=args.reranker_model,
        use_reranker=False,          # rerank globally at the end
        weight_bm25=en_bm25_w,
        weight_dense=en_dense_w,
        fusion_method=en_fusion,
        enable_tracing=True,
    )
 
    # =========================================================
    # RETRIEVE — route each query to the right retriever
    # =========================================================
    combined_docs: dict[str, Any] = {}
 
    for name, query in queries.items():
        print(f"\n{'='*16} {name}: {query} {'='*16}")
 
        if name in {"q1_literal", "q2_expanded"}:
            print(f"  → EN retriever (fusion={en_fusion})")
            result = en_retriever(query)
        else:
            print(f"  → VI retriever (weighted {args.weight_bm25_vi}/{args.weight_dense_vi})")
            result = vi_retriever(query)
 
        docs, trace = unpack_result(result)
        print(f"Retrieved docs: {len(docs)}")
        print_documents(docs, trace)
        print_trace(trace)
 
        # merge unique docs, track which queries retrieved each doc
        for doc in docs:
            key = (
                doc.metadata.get("chunk_id")
                or doc.metadata.get("doc_id")
                or id(doc)
            )
            if key not in combined_docs:
                doc.metadata["retrieved_by"] = [name]
                combined_docs[key] = doc
            else:
                existing_retrieved_by = combined_docs[key].metadata.get("retrieved_by", [])
                if name not in existing_retrieved_by:
                    existing_retrieved_by.append(name)
                combined_docs[key].metadata["retrieved_by"] = existing_retrieved_by
 
    # =========================================================
    # COMBINED
    # =========================================================
    final_docs = list(combined_docs.values())
    print(f"\n{'='*16} Combined unique docs {'='*16}")
    print(f"Total unique docs: {len(final_docs)}")
 
    # =========================================================
    # GLOBAL RERANK (on original query — most natural)
    # =========================================================
    if not args.no_reranker:
        print("\nRunning GLOBAL reranker...")
        from re_ranker import build_reranker, rerank_documents

        rerank_query_name, rerank_query = choose_global_rerank_query(
            queries,
            args.global_rerank_query,
        )
        print(f"Global rerank query ({rerank_query_name}): {rerank_query}")

        reranker = build_reranker(args.reranker_model)
        final_docs = rerank_documents(
            query=rerank_query,
            docs=final_docs,
            reranker=reranker,
            top_k=args.k_rerank,
        )
 
    # =========================================================
    # FINAL OUTPUT
    # =========================================================
    print(f"\n{'='*16} FINAL RERANKED DOCS {'='*16}")
    for i, doc in enumerate(final_docs, 1):
        print(f"\n--- Final Document {i} ---")
        print("chunk_id:", doc.metadata.get("chunk_id"))
        print("title:", doc.metadata.get("title"))
        print("retrieved_by:", doc.metadata.get("retrieved_by"))
        print("rerank_score:", doc.metadata.get("rerank_score"))
        print("\ncontent:")
        print(doc.page_content[:700].replace("\n", " ") + "...")
 
 
if __name__ == "__main__":
    main()
 
