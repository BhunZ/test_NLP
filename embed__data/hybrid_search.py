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
):
    
    # FAISS
    vectorstore = build_vectorstore(data_path, faiss_index_path)

    faiss_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k_retrieve}
    )

    # BM25
    bm25_retriever = build_or_load_bm25(
        data_path,
        bm25_path,
        k=k_retrieve
    )

    # Build reranker 1 lần duy nhất
    reranker = None
    if use_reranker:
        print(f"Loading reranker: {reranker_model}")
        reranker = build_reranker(reranker_model)

    def hybrid_retriever(query: str) -> list[Document]:

        # Retrieve
        bm25_results = bm25_retriever.invoke(query)
        faiss_results = faiss_retriever.invoke(query)

        # RRF fusion
        scores = {}
        doc_map = {}

        RRF_K = 60

        for rank, doc in enumerate(bm25_results):
            key = doc.metadata.get("doc_id")
            if not key:
                continue

            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)

        for rank, doc in enumerate(faiss_results):
            key = doc.metadata.get("doc_id")
            if not key:
                continue

            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank + 1)

        sorted_keys = sorted(scores, key=lambda x: -scores[x])

        fused_docs = [doc_map[key] for key in sorted_keys[:k_final]]

        # Rerank
        if use_reranker and reranker:
            fused_docs = rerank_documents(
                query=query,
                docs=fused_docs,
                reranker=reranker,
                top_k=k_rerank
            )

        return fused_docs

    print(
        f"Hybrid Retriever ready! "
        f"(retrieve={k_retrieve}, fusion={k_final}, rerank={k_rerank})"
    )

    return hybrid_retriever