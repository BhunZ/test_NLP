import pickle
from pathlib import Path
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers.bm25 import default_preprocessing_func
from langchain_core.documents import Document

from vectorstore_builder import load_documents


class BM25PickleRetriever:
    """Adapter for older BM25 pickle files saved as {'bm25': BM25Okapi, 'docs': [...]}."""

    def __init__(self, bm25: Any, docs: list[Document], k: int = 20) -> None:
        self.bm25 = bm25
        self.docs = docs
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        tokens = default_preprocessing_func(query)
        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[: self.k]

        results = []
        for index in ranked_indices:
            doc = self.docs[index]
            metadata = dict(doc.metadata)
            metadata["score"] = float(scores[index])
            results.append(Document(page_content=doc.page_content, metadata=metadata))

        return results


def _as_bm25_retriever(obj: Any, k: int) -> Any:
    if isinstance(obj, dict) and "bm25" in obj and "docs" in obj:
        return BM25PickleRetriever(obj["bm25"], obj["docs"], k=k)

    obj.k = k
    return obj


def build_or_load_bm25(
    data_path: str,
    save_path: str = "indexes/bm25_072.pkl",
    k: int = 20  #  FIX: dùng k_retrieve lớn hơn để recall tốt hơn
) -> BM25Retriever:
    """Build BM25 hoặc load từ file pickle"""
    save_path = Path(save_path)

    if save_path.exists():
        print(f"Loading BM25 from {save_path}")
        with open(save_path, "rb") as f:
            return _as_bm25_retriever(pickle.load(f), k)

    print("Building new BM25 index...")
    docs = load_documents(data_path)
    bm25 = BM25Retriever.from_documents(docs, k=k)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 saved to {save_path}")
    return bm25
