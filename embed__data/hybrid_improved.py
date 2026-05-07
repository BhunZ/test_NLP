import numpy as np
from langchain_core.documents import Document
from vectorstore_builder import build_vectorstore
from bm25_store import build_or_load_bm25
from re_ranker import build_reranker, rerank_documents


def build_hybrid_retriever(
    data_path: str,
    faiss_index_path: str = "indexes/faiss_index_072",
    bm25_path: str = "indexes/bm25_072.pkl",
    k_retrieve: int = 20,
    k_final: int = 6,
    k_rerank: int = 3,
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    use_reranker: bool = True,
    # New: Weighted fusion parameters
    weight_bm25: float = 0.2,
    weight_dense: float = 0.8,
    enable_tracing: bool = True,
):
    """
    Build hybrid retriever with weighted fusion.
    
    Args:
        weight_bm25: Weight for BM25 scores (default 0.2)
        weight_dense: Weight for dense/FAISS scores (default 0.8)
        enable_tracing: Enable detailed tracing for debugging
        
    Returns:
        hybrid_retriever: Function that takes query and returns {
            "documents": [...],
            "trace": {...}  # Only if enable_tracing=True
        }
    """
    
    # Ensure weights sum to 1
    assert abs((weight_bm25 + weight_dense) - 1.0) < 0.01, \
        f"Weights must sum to 1.0, got {weight_bm25 + weight_dense}"
    
    print(f"🔧 Hybrid Retriever Config:")
    print(f"   - BM25 weight: {weight_bm25:.1%}")
    print(f"   - Dense weight: {weight_dense:.1%}")
    print(f"   - Retrieve k: {k_retrieve}")
    print(f"   - Fusion k: {k_final}")
    print(f"   - Rerank k: {k_rerank}")
    print(f"   - Tracing: {'✓' if enable_tracing else '✗'}")
    
    # Build components
    vectorstore = build_vectorstore(data_path, faiss_index_path)
    
    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_retrieve}
    )
    
    bm25_retriever = build_or_load_bm25(data_path, bm25_path, k=k_retrieve)
    
    reranker = None
    if use_reranker:
        print(f" Loading reranker: {reranker_model}")
        reranker = build_reranker(reranker_model)
    
    # Main retrieval function
    def hybrid_retriever(query: str) -> dict | list[Document]:
        """
        Hybrid retrieval with weighted fusion.
        
        Returns:
            If enable_tracing: {"documents": [...], "trace": {...}}
            Otherwise: list[Document] (backward compatible)
        """
        
        # === STEP 1: Retrieve from both systems ===
        bm25_results = bm25_retriever.invoke(query)
        faiss_results = faiss_retriever.invoke(query)
        
        # === STEP 2: Normalize scores to [0, 1] ===
        # BM25 scores are raw frequencies → need normalization
        bm25_scores_raw = [float(doc.metadata.get("score", 0)) 
                          for doc in bm25_results]
        
        # FAISS similarity scores (usually already [0, 1] range)
        faiss_scores_raw = [float(doc.metadata.get("score", 0.5)) 
                           for doc in faiss_results]
        
        # Normalize to [0, 1]
        bm25_norm = _normalize_scores(bm25_scores_raw)
        faiss_norm = _normalize_scores(faiss_scores_raw)
        
        # === STEP 3: Weighted fusion ===
        fusion_scores = {}
        doc_map = {}
        fusion_details = {}  # For tracing
        
        # Add BM25 results
        for i, doc in enumerate(bm25_results):
            key = doc.metadata.get("doc_id")
            if not key:
                continue
            
            doc_map[key] = doc
            bm25_contribution = weight_bm25 * bm25_norm[i]
            fusion_scores[key] = bm25_contribution
            fusion_details[key] = {
                "bm25_score": bm25_norm[i],
                "bm25_contribution": bm25_contribution,
                "dense_score": 0.0,
                "dense_contribution": 0.0,
            }
        
        # Add FAISS results
        for i, doc in enumerate(faiss_results):
            key = doc.metadata.get("doc_id")
            if not key:
                continue
            
            doc_map[key] = doc
            dense_contribution = weight_dense * faiss_norm[i]
            fusion_scores[key] = fusion_scores.get(key, 0) + dense_contribution
            
            if key in fusion_details:
                fusion_details[key]["dense_score"] = faiss_norm[i]
                fusion_details[key]["dense_contribution"] = dense_contribution
            else:
                fusion_details[key] = {
                    "bm25_score": 0.0,
                    "bm25_contribution": 0.0,
                    "dense_score": faiss_norm[i],
                    "dense_contribution": dense_contribution,
                }
        
        # Sort by final score
        sorted_keys = sorted(fusion_scores, key=lambda x: -fusion_scores[x])
        fused_docs = [doc_map[key] for key in sorted_keys[:k_final]]
        
        # === STEP 4: Reranking ===
        if use_reranker and reranker:
            fused_docs = rerank_documents(
                query=query,
                docs=fused_docs,
                reranker=reranker,
                top_k=k_rerank
            )
        else:
            fused_docs = fused_docs[:k_rerank]
        
        # === STEP 5: Return results ===
        if enable_tracing:
            trace = {
                "num_bm25_results": len(bm25_results),
                "num_faiss_results": len(faiss_results),
                "num_fused": len(fused_docs),
                "weights": {
                    "bm25": weight_bm25,
                    "dense": weight_dense,
                },
                "fusion_scores": {
                    key: round(fusion_scores[key], 4)
                    for key in sorted_keys[:k_final]
                },
                "fusion_details": {
                    key: {
                        "bm25": round(fusion_details[key]["bm25_score"], 4),
                        "bm25_contrib": round(fusion_details[key]["bm25_contribution"], 4),
                        "dense": round(fusion_details[key]["dense_score"], 4),
                        "dense_contrib": round(fusion_details[key]["dense_contribution"], 4),
                        "final": round(fusion_scores[key], 4),
                    }
                    for key in sorted_keys[:k_final]
                }
            }
            
            return {
                "documents": fused_docs,
                "trace": trace
            }
        else:
            # Backward compatible: just return documents
            return fused_docs
    
    print(f" Hybrid Retriever ready!\n")
    return hybrid_retriever


def _normalize_scores(scores: list[float]) -> np.ndarray:
    """
    Normalize scores to [0, 1] range using min-max scaling.
    Handles edge cases (all zeros, single value, etc.)
    """
    scores = np.array(scores)
    
    if len(scores) == 0:
        return scores
    
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # Handle case where all scores are the same
    if max_score - min_score < 1e-8:
        return np.ones_like(scores) * 0.5
    
    # Min-max normalization
    normalized = (scores - min_score) / (max_score - min_score)
    return normalized


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Configuration
    DATA_PATH = "data/chunks.jsonl"
    FAISS_INDEX = "indexes/faiss_index_072"
    BM25_PATH = "indexes/bm25_072.pkl"
    
    # Build retriever with weighted fusion
    retriever = build_hybrid_retriever(
        data_path=DATA_PATH,
        faiss_index_path=FAISS_INDEX,
        bm25_path=BM25_PATH,
        k_retrieve=20,
        k_final=6,
        k_rerank=3,
        weight_bm25=0.2,
        weight_dense=0.8,
        enable_tracing=True,
    )
    
    # Test query
    test_query = "What is transformer attention mechanism"
    result = retriever(test_query)
    
    print(f"\n Results for: '{test_query}'")
    print(f"Retrieved: {len(result['documents'])} documents\n")
    
    for i, doc in enumerate(result['documents'], 1):
        print(f"{i}. {doc.metadata.get('title', 'N/A')} "
              f"(Score: {result['trace']['fusion_details'].get(doc.metadata.get('doc_id', ''), {}).get('final', 0)})")
    
    # Print fusion details
    print("\n Fusion Score Breakdown:")
    for doc_id, details in result['trace']['fusion_details'].items():
        print(f"\n  Doc {doc_id}:")
        print(f"    BM25:  {details['bm25']:.4f} × {result['trace']['weights']['bm25']:.1%} = {details['bm25_contrib']:.4f}")
        print(f"    Dense: {details['dense']:.4f} × {result['trace']['weights']['dense']:.1%} = {details['dense_contrib']:.4f}")
        print(f"    Final: {details['final']:.4f}")