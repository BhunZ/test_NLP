

# from hybrid_search import build_hybrid_retriever

# if __name__ == "__main__":
#     retriever = build_hybrid_retriever(
#         data_path="/Users/carwyn/Downloads/transcript_v3_t072.jsonl",
#         k_retrieve=20,     # BM25 + FAISS mỗi cái lấy 20
#         k_final=6,         # sau RRF giữ lại 6
#         k_rerank=3,        # sau rerank trả về 3 tốt nhất
#         reranker_model="BAAI/bge-reranker-v2-m3",  # hoặc gemma / minicpm
#         use_reranker=True,
#     )
#     # Test
#     docs = retriever("What is gradient descent?")
#     for i, doc in enumerate(docs):
#         print(f"\n--- Document {i+1} ---")
#         print(doc.page_content[:300] + "...")
#         print("Metadata:", doc.metadata)
#     docs = retriever("What is gradient descent?")
#     print("Retrieved docs:", len(docs))

from hybrid_improved import build_hybrid_retriever

retriever = build_hybrid_retriever(
    data_path        = "data/chunks.jsonl",
    faiss_index_path = "indexes/faiss_index_072",
    bm25_path        = "indexes/bm25_072.pkl",
    embedding_device = "cpu",   # ← CPU ổn vì index đã build sẵn
)