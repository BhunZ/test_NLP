from FlagEmbedding import FlagReranker, FlagLLMReranker, LayerWiseFlagLLMReranker
from langchain_core.documents import Document


def build_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
    """
    Chọn reranker phù hợp theo model_name.
    
    - bge-reranker-v2-m3         → FlagReranker         (nhẹ, nhanh, multilingual)
    - bge-reranker-v2-gemma      → FlagLLMReranker       (mạnh hơn, chậm hơn)
    - bge-reranker-v2-minicpm-layerwise → LayerWiseFlagLLMReranker (có thể chọn layer)
    """
    if "minicpm-layerwise" in model_name:
        reranker = LayerWiseFlagLLMReranker(
            model_name,
            use_fp16=use_fp16,
            cutoff_layers=[28]  # layer 28/40 — balance giữa speed và accuracy
        )
    elif "gemma" in model_name:
        reranker = FlagLLMReranker(model_name, use_fp16=use_fp16)
    else:
        # mặc định: bge-reranker-v2-m3
        reranker = FlagReranker(model_name, use_fp16=use_fp16)

    return reranker


def rerank_documents(
    query: str,
    docs: list[Document],
    reranker,
    top_k: int = 3
) -> list[Document]:
    """Rerank list[Document] theo query, trả về top_k tốt nhất."""
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.compute_score(pairs, normalize=True)  # normalize → [0, 1]

    # Gắn score vào metadata để debug/trace
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = round(float(score), 4)

    # Sắp xếp giảm dần theo score
    ranked = sorted(zip(scores, docs), key=lambda x: -x[0])
    return [doc for _, doc in ranked[:top_k]]