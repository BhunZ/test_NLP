import numpy as np
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from bm25_store import build_or_load_bm25
from re_ranker import build_reranker, rerank_documents
from vectorstore_builder import build_vectorstore


def build_hybrid_retriever(
    data_path: str,
    faiss_index_path: str = "indexes/faiss_index_072_n",
    bm25_path: str = "indexes/bm25_072.pkl",
    k_retrieve: int = 20,
    k_final: int = 20,
    k_rerank: int = 3,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    use_reranker: bool = True,
    weight_bm25: float | None = None,
    weight_dense: float | None = None,
    fusion_method: str = "rrf",
    rrf_k: int = 60,
    enable_tracing: bool = True,
):
    """Build hybrid retriever supporting both RRF and weighted fusion."""
    if fusion_method not in {"rrf", "weighted"}:
        raise ValueError("fusion_method must be 'rrf' or 'weighted'")

    if fusion_method == "weighted":
        if weight_bm25 is None or weight_dense is None:
            raise ValueError("Weighted fusion requires both weight_bm25 and weight_dense")
        if abs((weight_bm25 + weight_dense) - 1.0) >= 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {weight_bm25 + weight_dense}")
    else:
        if weight_bm25 is not None or weight_dense is not None:
            print(" Note: weight_bm25 and weight_dense are ignored in RRF mode.")
        weight_bm25 = None
        weight_dense = None

    print("\n🔧 Hybrid Retriever Config:")
    print(f" - Fusion method: {fusion_method}")
    if fusion_method == "weighted":
        print(f" - BM25 weight : {weight_bm25:.1%}")
        print(f" - Dense weight: {weight_dense:.1%}")
    else:
        print(f" - RRF k       : {rrf_k}")
    print(f" - Retrieve k  : {k_retrieve}")
    print(f" - Final k     : {k_final}")
    print(f" - Rerank k    : {k_rerank}")
    print(f" - Tracing     : {'✓' if enable_tracing else '✗'}")

    vectorstore = build_vectorstore(data_path, faiss_index_path)
    bm25_retriever = build_or_load_bm25(data_path, bm25_path, k=k_retrieve)

    reranker = None
    if use_reranker:
        print(f" - Loading reranker: {reranker_model}")
        reranker = build_reranker(reranker_model)

    def hybrid_retriever(query: str) -> dict | list[Document]:
        bm25_results = bm25_retriever.invoke(query)
        faiss_results = _faiss_search_with_similarity(vectorstore, query, k_retrieve)

        if fusion_method == "weighted":
            bm25_norm = _normalize_scores(
                [float(doc.metadata.get("score", 0)) for doc in bm25_results]
            )
            faiss_norm = _normalize_scores(
                [float(doc.metadata.get("score", 0.5)) for doc in faiss_results]
            )
        else:
            bm25_norm = None
            faiss_norm = None

        fusion_scores: dict[str, float] = {}
        fusion_details: dict[str, dict] = {}
        doc_map: dict[str, Document] = {}

        for i, doc in enumerate(bm25_results):
            key = _doc_key(doc)
            if not key:
                continue

            doc_map[key] = doc
            if fusion_method == "rrf":
                contrib = 1.0 / (rrf_k + i + 1)
                raw = None
            else:
                contrib = float(weight_bm25 * bm25_norm[i])
                raw = float(bm25_norm[i])

            fusion_scores[key] = fusion_scores.get(key, 0.0) + contrib
            fusion_details[key] = {
                "bm25_score": raw,
                "bm25_rank": i + 1,
                "bm25_contrib": contrib,
                "dense_score": None,
                "dense_rank": None,
                "dense_contrib": 0.0,
            }

        for i, doc in enumerate(faiss_results):
            key = _doc_key(doc)
            if not key:
                continue

            doc_map[key] = doc
            if fusion_method == "rrf":
                contrib = 1.0 / (rrf_k + i + 1)
                raw = None
            else:
                contrib = float(weight_dense * faiss_norm[i])
                raw = float(faiss_norm[i])

            fusion_scores[key] = fusion_scores.get(key, 0.0) + contrib

            if key not in fusion_details:
                fusion_details[key] = {
                    "bm25_score": None,
                    "bm25_rank": None,
                    "bm25_contrib": 0.0,
                    "dense_score": raw,
                    "dense_rank": i + 1,
                    "dense_contrib": contrib,
                }
            else:
                fusion_details[key].update(
                    {
                        "dense_score": raw,
                        "dense_rank": i + 1,
                        "dense_contrib": contrib,
                    }
                )

        sorted_keys = sorted(fusion_scores, key=lambda x: -fusion_scores[x])
        fused_docs = [doc_map[key] for key in sorted_keys[:k_final]]
        
        if use_reranker and reranker:

            fused_docs = rerank_documents(
                query=query,
                docs=fused_docs,
                reranker=reranker,
                top_k=k_rerank,
            )
        else:
            fused_docs = fused_docs[:k_rerank]

        if enable_tracing:
            trace = {
                "fusion_method": fusion_method,
                "num_bm25_results": len(bm25_results),
                "num_faiss_results": len(faiss_results),
                "num_fused": len(fused_docs),
                "weights": (
                    {"bm25": weight_bm25, "dense": weight_dense}
                    if fusion_method == "weighted"
                    else None
                ),
                "fusion_scores": {
                    key: round(fusion_scores[key], 4)
                    for key in sorted_keys[:k_final]
                },
                "fusion_details": {
                    key: {
                        **fusion_details[key],
                        "final": round(fusion_scores[key], 4),
                    }
                    for key in sorted_keys[:k_final]
                },
            }
            return {"documents": fused_docs, "trace": trace}

        return fused_docs

    print(" Hybrid Retriever ready!\n")
    return hybrid_retriever


def _doc_key(doc: Document) -> str | None:
    metadata = doc.metadata or {}
    key = metadata.get("chunk_id") or metadata.get("doc_id")
    if key is not None:
        metadata["doc_id"] = str(key)
        return str(key)
    return None


def _faiss_search_with_similarity(vectorstore, query: str, k: int) -> list[Document]:
    results = vectorstore.similarity_search_with_score(query, k=k)
    docs = []

    for doc, raw_score in results:
        metadata = dict(doc.metadata)
        metadata["score"] = _faiss_score_to_similarity(vectorstore, float(raw_score))
        if metadata.get("chunk_id"):
            metadata["doc_id"] = str(metadata["chunk_id"])

        docs.append(Document(page_content=doc.page_content, metadata=metadata))

    return docs


def _faiss_score_to_similarity(vectorstore, raw_score: float) -> float:
    strategy = getattr(vectorstore, "distance_strategy", None)
    if strategy == DistanceStrategy.MAX_INNER_PRODUCT:
        return raw_score
    if strategy == DistanceStrategy.COSINE:
        return 1.0 - raw_score

    return 1.0 / (1.0 + max(raw_score, 0.0))


def _normalize_scores(scores: list[float]) -> np.ndarray:
    scores = np.array(scores)
    if len(scores) == 0:
        return scores

    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score - min_score < 1e-8:
        return np.ones_like(scores) * 0.5

    return (scores - min_score) / (max_score - min_score)
