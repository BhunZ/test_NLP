import pickle
from pathlib import Path
from langchain_community.retrievers import BM25Retriever
from vectorstore_builder import load_documents


def build_or_load_bm25(
    data_path: str,
    save_path: str = "indexes/bm25.pkl",
    k: int = 20  #  FIX: dùng k_retrieve lớn hơn để recall tốt hơn
) -> BM25Retriever:
    """Build BM25 hoặc load từ file pickle"""
    save_path = Path(save_path)

    if save_path.exists():
        print(f"Loading BM25 from {save_path}")
        with open(save_path, "rb") as f:
            retriever = pickle.load(f)
            retriever.k = k  # cập nhật k phòng trường hợp thay đổi
            return retriever

    print("Building new BM25 index...")
    docs = load_documents(data_path)
    bm25 = BM25Retriever.from_documents(docs, k=k)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"BM25 saved to {save_path}")
    return bm25