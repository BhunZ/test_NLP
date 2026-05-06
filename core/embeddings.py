"""
core/embeddings.py – Khởi tạo embedding model.
"""
import logging
import torch
from sentence_transformers import SentenceTransformer
from .config import get_settings

logger = logging.getLogger(__name__)


class _EmbedderWrapper:
    """
    Thin wrapper quanh SentenceTransformer để cung cấp API
    embed_documents() tương thích với LangChain.
    """

    def __init__(self, model: SentenceTransformer, normalize: bool, batch_size: int):
        self._model = model
        self._normalize = normalize
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
            show_progress_bar=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_model():
    """
    Khởi tạo embedding model (bge-m3).

    bge-m3 được tối ưu cho:
      - Multilingual (VN, EN, ...)
      - Dense retrieval (semantic search)
      - Normalized embeddings (cosine similarity)

    GPU ≤ 2GB (MX450): load CPU → fp16 → move CUDA
    để tránh OOM (fp32 ~1.66GB, fp16 ~0.83GB).

    Output:
        Object có .embed_documents() và .embed_query()
    """
    cfg = get_settings()

    if cfg.device == "cuda":
        logger.info("Loading model trên CPU trước, sẽ convert fp16 → CUDA...")
        model = SentenceTransformer(cfg.embedding_model_name, device="cpu")
        model.half()  # fp32 → fp16 (~0.83GB)
        model.to(torch.device("cuda"))
        logger.info("Model đã move sang GPU (fp16). VRAM ~0.83GB.")
        batch_size = cfg.embedding_batch_size  # nhỏ để tránh OOM
    else:
        model = SentenceTransformer(cfg.embedding_model_name, device="cpu")
        batch_size = cfg.embedding_batch_size

    return _EmbedderWrapper(model, cfg.normalize_embeddings, batch_size)

