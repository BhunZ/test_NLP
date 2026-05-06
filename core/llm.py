"""
core/llm.py — LLM Client for the answer-generation step (Stage 6).

Architecture
────────────
    HybridRetriever ──► List[RetrievedDoc]
                              │
                              ▼
                  LLMClient.answer(query, sources)
                              │
                              ▼
                          LLMAnswer
                          ├─ text_vi      (Vietnamese answer)
                          ├─ citations    ([0, 2, 3] — indices into sources)
                          └─ confidence   ("high" | "medium" | "low")

The interface is provider-agnostic. Set LLM_PROVIDER env var or
cfg.llm_provider to swap implementations:

    LLM_PROVIDER=gemini  (default — Google Gemini, free tier)
    LLM_PROVIDER=ollama  (local Qwen 2.5 — requires Ollama installed)
    LLM_PROVIDER=groq    (Groq cloud — free tier, fast)

Setup for Gemini (the default)
──────────────────────────────
    pip install google-genai
    # Get a free key at https://aistudio.google.com/apikey
    export GEMINI_API_KEY=your_key_here

Quick usage
───────────
    from core.llm import get_llm_client
    from core.s02_hybrid_search import HybridRetriever

    retriever = HybridRetriever()
    llm = get_llm_client()

    sources = retriever.retrieve("Cơ chế attention hoạt động như thế nào?", k=5)
    answer = llm.answer("Cơ chế attention hoạt động như thế nào?", sources)

    print(answer.text_vi)
    print("Citations:", answer.citations)
    print("Confidence:", answer.confidence)
"""
from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.config import get_settings
from core.s02_hybrid_search import RetrievedDoc

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# DATA MODEL
# ════════════════════════════════════════════════════════════════

@dataclass
class LLMAnswer:
    """Kết quả từ LLM, đóng gói cho frontend."""

    text_vi: str                                # nội dung trả lời (tiếng Việt)
    citations: List[int] = field(default_factory=list)
    """0-indexed indices into the sources list that were cited (e.g. [1] → 0)."""

    confidence: str = "medium"                  # "high" | "medium" | "low"
    raw_response: str = ""                      # text gốc trước khi parse
    provider: str = ""                          # "gemini" | "ollama" | "groq"
    model: str = ""                             # tên model cụ thể đã dùng

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_vi": self.text_vi,
            "citations": self.citations,
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
        }


# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_VI = """Bạn là trợ giảng AI chuyên về NLP và Học máy, dạy bằng tiếng Việt cho sinh viên.

NHIỆM VỤ:
- Đọc các đoạn trích (NGUỒN) từ bài giảng Stanford được cung cấp.
- Trả lời câu hỏi của sinh viên bằng tiếng Việt một cách rõ ràng, dễ hiểu.
- Khi sử dụng thông tin từ một nguồn, đánh dấu trích dẫn bằng [n] (ví dụ: [1], [2], [3]).
  Số n tương ứng với số thứ tự của nguồn.
- Nếu các nguồn KHÔNG đủ thông tin để trả lời, hãy nói rõ điều đó thay vì bịa ra.
  Ví dụ: "Các nguồn không đề cập đến X. Tôi không thể trả lời chính xác."

PHONG CÁCH:
- Giải thích như đang dạy sinh viên năm nhất — đơn giản, có ví dụ cụ thể.
- GIỮ NGUYÊN thuật ngữ tiếng Anh chuyên ngành (transformer, attention, gradient descent,
  embedding, fine-tuning, ...) thay vì dịch máy móc sang tiếng Việt.
- Trích dẫn nguồn cho MỌI tuyên bố cụ thể, không chỉ ở cuối.
- Trả lời ngắn gọn (3-6 câu) trừ khi câu hỏi yêu cầu giải thích sâu.

ĐỊNH DẠNG TRẢ LỜI:
- Markdown đơn giản (gạch đầu dòng, **in đậm** thuật ngữ chính).
- Không lặp lại câu hỏi.
- Không thêm phần "Tài liệu tham khảo" cuối — citations [n] là đủ."""


USER_PROMPT_TEMPLATE = """NGUỒN:
{sources_block}

CÂU HỎI: {query}

Trả lời bằng tiếng Việt, sử dụng [n] để trích dẫn nguồn."""


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def format_sources(sources: List[RetrievedDoc]) -> str:
    """
    Định dạng list[RetrievedDoc] thành block text đánh số cho LLM prompt.

    Output ví dụ:
        [1] Course: CS25_Transformers
            Video: Stanford CS25 V1 Introduction to Transformers
            Time: 02:45
            Content: Self-attention lets every position attend...

        [2] Course: CS224N_NLP
            ...
    """
    parts = []
    for i, doc in enumerate(sources, start=1):
        parts.append(
            f"[{i}] Course: {doc.course or 'Unknown'}\n"
            f"    Video: {doc.title or 'Unknown'}\n"
            f"    Time: {doc.source_label or '—'}\n"
            f"    Content: {doc.chunk_text}"
        )
    return "\n\n".join(parts)


def parse_citations(answer_text: str, max_index: int) -> List[int]:
    """
    Trích các marker [n] trong câu trả lời, trả về list chỉ số 0-indexed,
    unique, theo thứ tự xuất hiện.

    [1] → 0, [2] → 1, ... và bỏ qua những số ngoài phạm vi 1..max_index.
    """
    matches = re.findall(r"\[(\d+)\]", answer_text)
    citations: List[int] = []
    seen = set()
    for m in matches:
        idx = int(m) - 1
        if 0 <= idx < max_index and idx not in seen:
            citations.append(idx)
            seen.add(idx)
    return citations


def estimate_confidence(sources: List[RetrievedDoc]) -> str:
    """
    Heuristic confidence dựa trên top retrieval score.

    Vì bge-m3 normalized cosine và RRF score có scale khác nhau, thresholds
    này nên được điều chỉnh sau khi có eval harness (xem plan, Step 0).
    Hiện tại dùng giá trị tạm thời.

    high   ≥ 0.80
    medium ≥ 0.60
    low    < 0.60 (hoặc không có source)
    """
    if not sources:
        return "low"
    top_score = sources[0].score
    if top_score >= 0.80:
        return "high"
    if top_score >= 0.60:
        return "medium"
    return "low"


# ════════════════════════════════════════════════════════════════
# ABSTRACT BASE
# ════════════════════════════════════════════════════════════════

class LLMClient(ABC):
    """
    Interface cho mọi LLM provider.

    Mỗi subclass triển khai phương thức answer() để gọi API của provider
    tương ứng. Toàn bộ logic format prompt và parse citation được tái sử dụng
    qua các helper bên trên.
    """

    provider_name: str = "abstract"

    @abstractmethod
    def answer(self, query: str, sources: List[RetrievedDoc]) -> LLMAnswer:
        """Sinh câu trả lời tiếng Việt từ query + retrieved sources."""
        raise NotImplementedError


# ════════════════════════════════════════════════════════════════
# GEMINI (default)
# ════════════════════════════════════════════════════════════════

class GeminiClient(LLMClient):
    """
    Google Gemini provider (free tier).

    Mặc định dùng gemini-2.5-flash:
      - Free tier ~1500 requests/day (kiểm tra giới hạn hiện tại tại
        https://ai.google.dev/pricing)
      - Hỗ trợ tiếng Việt tốt, multimodal, context 1M tokens
      - Latency thấp, đủ cho UI realtime

    SDK: google-genai (mới, thay thế google-generativeai cũ)
        pip install google-genai
    """

    provider_name = "gemini"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        try:
            from google import genai           # type: ignore
            from google.genai import types     # type: ignore
        except ImportError as e:
            raise ImportError(
                "GeminiClient yêu cầu package 'google-genai'.\n"
                "Cài đặt: pip install google-genai"
            ) from e

        cfg = get_settings()

        # API key: ưu tiên tham số → GEMINI_API_KEY → GOOGLE_API_KEY
        self._api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "Không tìm thấy API key cho Gemini.\n"
                "Lấy key miễn phí tại: https://aistudio.google.com/apikey\n"
                "Sau đó: export GEMINI_API_KEY=your_key_here"
            )

        self._model_name = model or cfg.llm_model
        self._temperature = (
            temperature if temperature is not None else cfg.llm_temperature
        )
        self._max_output_tokens = max_output_tokens or cfg.llm_max_output_tokens

        # Tạo client + cấu hình generation một lần ở init
        self._client = genai.Client(api_key=self._api_key)
        self._gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_VI,
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
        )
        logger.info(
            "GeminiClient khởi tạo: model=%s, temperature=%.2f",
            self._model_name,
            self._temperature,
        )

    def answer(self, query: str, sources: List[RetrievedDoc]) -> LLMAnswer:
        # Edge case: không có nguồn nào → trả lời mặc định, không gọi API
        if not sources:
            return LLMAnswer(
                text_vi=(
                    "Xin lỗi, tôi không tìm thấy nguồn nào phù hợp với "
                    "câu hỏi của bạn trong các bài giảng Stanford. "
                    "Bạn có thể thử diễn đạt lại câu hỏi không?"
                ),
                citations=[],
                confidence="low",
                provider=self.provider_name,
                model=self._model_name,
            )

        sources_block = format_sources(sources)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            sources_block=sources_block,
            query=query,
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                config=self._gen_config,
                contents=user_prompt,
            )
            text = (response.text or "").strip()
        except Exception as e:
            logger.exception("Gemini API call thất bại: %s", e)
            return LLMAnswer(
                text_vi=(
                    "Đã xảy ra lỗi khi gọi mô hình LLM. "
                    "Vui lòng thử lại sau ít phút."
                ),
                citations=[],
                confidence="low",
                provider=self.provider_name,
                model=self._model_name,
            )

        return LLMAnswer(
            text_vi=text,
            citations=parse_citations(text, max_index=len(sources)),
            confidence=estimate_confidence(sources),
            raw_response=text,
            provider=self.provider_name,
            model=self._model_name,
        )


# ════════════════════════════════════════════════════════════════
# PLACEHOLDER PROVIDERS (sẽ triển khai sau)
# ════════════════════════════════════════════════════════════════

class OllamaClient(LLMClient):
    """
    Local Qwen 2.5 / Llama via Ollama (chưa triển khai).

    Khi cần triển khai:
      - Cài Ollama: https://ollama.com
      - ollama pull qwen2.5:7b
      - ollama serve
      - Gọi http://localhost:11434/api/generate
    """

    provider_name = "ollama"

    def answer(self, query: str, sources: List[RetrievedDoc]) -> LLMAnswer:
        raise NotImplementedError(
            "OllamaClient chưa triển khai. Hiện tại dùng GeminiClient (mặc định)."
        )


class GroqClient(LLMClient):
    """
    Groq cloud (chưa triển khai).

    Khi cần triển khai (chủ yếu cho Step 3 — LLM boundary detection
    chunking offline):
      - Lấy free key tại https://console.groq.com
      - pip install groq
      - Model: llama-3.3-70b-versatile
    """

    provider_name = "groq"

    def answer(self, query: str, sources: List[RetrievedDoc]) -> LLMAnswer:
        raise NotImplementedError(
            "GroqClient chưa triển khai. Hiện tại dùng GeminiClient (mặc định)."
        )


# ════════════════════════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════════════════════════

_PROVIDERS = {
    "gemini": GeminiClient,
    "ollama": OllamaClient,
    "groq": GroqClient,
}


def get_llm_client(provider: Optional[str] = None, **kwargs: Any) -> LLMClient:
    """
    Factory: trả về LLMClient phù hợp với provider được chọn.

    Thứ tự ưu tiên cho provider:
      1. Tham số `provider` truyền vào trực tiếp
      2. env var LLM_PROVIDER
      3. cfg.llm_provider (mặc định "gemini")

    kwargs sẽ được forward cho constructor của provider (ví dụ
    api_key=..., model=...).

    Ví dụ:
        llm = get_llm_client()                          # gemini, mặc định
        llm = get_llm_client("gemini", model="gemini-1.5-flash")
        llm = get_llm_client("ollama")                  # NotImplementedError hiện tại
    """
    cfg = get_settings()
    provider = (provider or cfg.llm_provider).lower().strip()

    if provider not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"LLM provider '{provider}' không hỗ trợ. "
            f"Provider khả dụng: {available}"
        )

    cls = _PROVIDERS[provider]
    return cls(**kwargs)
