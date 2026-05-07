
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(path: str) -> list[Document]:
    """Load chunks từ file jsonl, gắn doc_id vào metadata"""
    docs = []
    with open(path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip() == "":
                continue
            chunk = json.loads(line)
            doc = Document(
                page_content=chunk["chunk_text"],
                metadata={
                    #  thêm doc_id để dùng làm key trong hybrid search
                    "doc_id": f"{chunk.get('chunk_index', 'unknown')}",
                    "video_id": chunk.get("video_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "title": chunk.get("title"),
                    "course": chunk.get("course"),
                    "playlist_id": chunk.get("playlist_id"),
                    "published_at": chunk.get("published_at"),
                    "start_time": chunk.get("start_time"),
                    "end_time": chunk.get("end_time"),
                    "duration": chunk.get("duration"),
                    "source": chunk.get("source"),
                    "chunk_type": chunk.get("chunk_type"),
                    "token_count": chunk.get("token_count"),
                    "url": chunk.get("url")
                }
            )
            docs.append(doc)
    print(f"Loaded {len(docs)} documents from {path}")
    return docs


def build_vectorstore(
    data_path: str,
    index_path: str = "indexes/faiss_index_072",
    embedding_model: str = "BAAI/bge-m3"
):
    """Build hoặc load FAISS vectorstore"""
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64
        }
    )

    index_dir = Path(index_path)
    if index_dir.exists():
        print(f"Loading existing FAISS index from {index_path}")
        #  FIX: không load docs thừa khi index đã tồn tại
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Chỉ load docs khi thực sự cần build index mới
    print("Building new FAISS index...")
    docs = load_documents(data_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(index_dir))
    print(f"FAISS index saved to {index_path}")
    return vectorstore

