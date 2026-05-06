"""
core/config.py – Cấu hình toàn cục cho RAG pipeline.
"""
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Cấu hình toàn cục."""

    # ── Đường dẫn ───────────────────────────────────────────────────────────
    ROOT_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Path(__file__).parent.parent / "embed/files"
    INDEX_DIR: Path = Path(__file__).parent.parent / "index"

    # ── Input file ───────────────────────────────────────────────────────────
    # File JSONL chứa semantic chunks đã qua xử lý từ Member 1
    raw_data_path: Path = DATA_DIR / "transcripts_enhanced.jsonl"

    # ── Output files (persist sau khi build) ─────────────────────────────────
    faiss_index_dir: Path = INDEX_DIR / "faiss"  # LangChain FAISS lưu toàn bộ vào folder
    bm25_path: Path = INDEX_DIR / "bm25.pkl"
    chunk_texts_path: Path = INDEX_DIR / "chunk_texts.pkl"

    # ── Embedding model ──────────────────────────────────────────────────────
    # BAAI/bge-m3: đa ngôn ngữ (VN↔EN), context 8192 tokens, output dim 1024
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Device: "cpu" hoặc "cuda" (nếu có GPU)
    # ⚠️ MX450 2GB dễ bị OOM và swap RAM qua PCIe nên phải dùng batch_size = 1
    device: str = "cuda"

    def __post_init__(self):
        pass # removed force-cpu override to allow cuda

    # Normalize embeddings cho cosine similarity via inner product (dùng cho FAISS)
    normalize_embeddings: bool = True

    # ── Embedding batch size ─────────────────────────────────────────────────
    # Giảm xuống 1 (hoặc 2) cho GPU 2GB để ko bị swap RAM chậm
    embedding_batch_size: int = 1

    # ── Search parameters ────────────────────────────────────────────────────
    top_k_dense: int = 20  # lấy top-K từ FAISS trước khi rerank
    top_k_bm25: int = 20  # lấy top-K từ BM25
    top_k_final: int = 5  # trả về K kết quả cuối cùng

    # Hybrid alpha: alpha * dense_score + (1-alpha) * bm25_score
    hybrid_alpha: float = 0.6

    # ── LangChain setup ──────────────────────────────────────────────────────
    # LangChain FAISS dùng mặc định InnerProduct (IP) metric
    # (nếu normalize_embeddings=True thì IP ≈ cosine similarity)
    faiss_metric_type: str = "ip"  # "l2" hoặc "ip"

    # ── LLM (answer generation, Stage 6) ─────────────────────────────────────
    # Provider chọn LLM cho bước trả lời cuối:
    #   "gemini" — Google Gemini (free tier, mặc định, đề xuất cho dự án này)
    #   "ollama" — local Qwen 2.5 / Llama (cần cài Ollama trên máy)
    #   "groq"   — Groq cloud (free tier, nhanh)
    llm_provider: str = "gemini"

    # Model name riêng cho từng provider:
    #   gemini: "gemini-2.5-flash" (mới nhất, nhanh, free tier rộng)
    #           hoặc "gemini-1.5-flash" (ổn định, free tier rộng)
    #   ollama: "qwen2.5:7b", "qwen2.5:14b", "llama3.3:70b"
    #   groq:   "llama-3.3-70b-versatile", "llama-3.1-8b-instant"
    llm_model: str = "gemini-2.5-flash"

    # Số tokens tối đa cho câu trả lời (None = để model tự quyết)
    llm_max_output_tokens: int = 1024

    # Temperature: 0.0 = deterministic, 1.0 = creative
    # 0.3 là sweet spot cho RAG (đủ tự nhiên, không hallucinate)
    llm_temperature: float = 0.3


def get_settings() -> Settings:
    """
    Lấy cấu hình toàn cục.

    Có thể override từ environment variables:
        export EMBEDDING_MODEL=BAAI/bge-m3
        export DEVICE=cuda
        export BATCH_SIZE=64
        export LLM_PROVIDER=gemini       # hoặc ollama / groq
        export LLM_MODEL=gemini-2.5-flash
    """
    import os

    cfg = Settings()

    # Override từ env nếu có
    if env_model := os.getenv("EMBEDDING_MODEL"):
        cfg.embedding_model_name = env_model

    if env_device := os.getenv("DEVICE"):
        cfg.device = env_device

    if env_batch := os.getenv("BATCH_SIZE"):
        cfg.embedding_batch_size = int(env_batch)

    if env_index := os.getenv("INDEX_DIR"):
        cfg.INDEX_DIR = Path(env_index)

    # LLM overrides
    if env_provider := os.getenv("LLM_PROVIDER"):
        cfg.llm_provider = env_provider.lower().strip()

    if env_llm_model := os.getenv("LLM_MODEL"):
        cfg.llm_model = env_llm_model

    if env_temp := os.getenv("LLM_TEMPERATURE"):
        cfg.llm_temperature = float(env_temp)

    return cfg
