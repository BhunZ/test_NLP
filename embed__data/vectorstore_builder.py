
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def make_document_id(chunk: dict, fallback_index: int) -> str:
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    video_id = chunk.get("video_id", "unknown_video")
    chunk_index = chunk.get("chunk_index", fallback_index)
    return f"{video_id}:chunk:{chunk_index}"


def load_documents(path: str) -> list[Document]:
    """Load chunks từ file jsonl, gắn doc_id vào metadata"""
    docs = []
    with open(path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip() == "":
                continue
            chunk = json.loads(line)
            doc_id = make_document_id(chunk, idx)
            doc = Document(
                page_content=chunk["chunk_text"],
                metadata={
                    # doc_id must be globally unique; chunk_index repeats across videos.
                    "doc_id": doc_id,
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
    _validate_unique_doc_ids(docs, path)
    print(f"Loaded {len(docs)} documents from {path}")
    return docs


def _validate_unique_doc_ids(docs: list[Document], path: str) -> None:
    seen = set()
    duplicates = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id in seen:
            duplicates.append(doc_id)
        seen.add(doc_id)

    if duplicates:
        sample = ", ".join(str(item) for item in duplicates[:5])
        raise ValueError(
            f"Duplicate doc_id values found while loading {path}: {sample}. "
            "Use globally unique chunk_id values before rebuilding indexes."
        )


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
