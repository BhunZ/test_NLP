from langchain_huggingface import HuggingFaceEmbeddings
from load_langchain import load_documents
from langchain_community.vectorstores import FAISS


def build_vectorstore(data_path: str):
    # 1. Load docs từ file khác
    docs = load_documents(data_path)

    # 2. Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 64,        # tránh OOM trên CPU
                    "show_progress_bar": True
                }
    )

    # 3. Tạo vector DB bằng faiss 
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore