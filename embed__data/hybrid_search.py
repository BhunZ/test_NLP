from langchain_core.documents import Document
from vectorstore_builder import build_vectorstore
from bm25_store import build_or_load_bm25


def build_hybrid_retriever(
    data_path: str,
    faiss_index_path: str = "indexes/faiss_index",
    bm25_path: str = "indexes/bm25.pkl",
    k_retrieve: int = 20,   #  FIX: retrieve nhiều hơn để recall tốt
    k_final: int = 6,       #  FIX: tách biệt k retrieve vs k trả về
):
    """
    Hybrid Retriever dùng RRF (Reciprocal Rank Fusion).
    Không cần tune bm25_weight — RRF tự cân bằng 2 nguồn.
    """
    # Build / Load FAISS
    vectorstore = build_vectorstore(data_path, faiss_index_path)
    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_retrieve}
    )

    # Build / Load BM25
    bm25_retriever = build_or_load_bm25(data_path, bm25_path, k=k_retrieve)

    def hybrid_retriever(query: str) -> list[Document]:
        bm25_results = bm25_retriever.invoke(query)   # list[Document]
        faiss_results = faiss_retriever.invoke(query)  # list[Document]

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        RRF_K = 60  #  FIX: dùng RRF thay weighted fusion, stable hơn

        # BM25 scoring
        for rank, doc in enumerate(bm25_results):
            key = doc.metadata.get("doc_id")
            if not key:
                continue
            # FIX: không overwrite doc đã có
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)

        # FAISS scoring
        for rank, doc in enumerate(faiss_results):
            key = doc.metadata.get("doc_id")
            if not key:
                continue
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)

        # Sắp xếp và trả về top k_final Documents
        sorted_keys = sorted(scores, key=lambda x: -scores[x])

        #  FIX: dùng 'key' thay 'k' trong comprehension — tránh shadow variable
        return [doc_map[key] for key in sorted_keys[:k_final]]

    print(f"Hybrid Retriever ready! (RRF, k_retrieve={k_retrieve}, k_final={k_final})")
    return hybrid_retriever