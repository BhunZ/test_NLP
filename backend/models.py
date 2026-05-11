from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=6, ge=1, le=20)
    rerank: bool = False
    llm_provider: Literal["groq", "mistral"] = "groq"
    course_filter: Optional[str] = None  # accepted for v1; no-op unless implemented
    enable_rewrite: bool = False


class AskRewrites(BaseModel):
    q1: Optional[str] = None
    q2: Optional[str] = None


class AskAnswer(BaseModel):
    text_vi: str
    citations: List[int] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


class SourceChunk(BaseModel):
    chunk_id: str
    rank: int
    score: Optional[float] = None

    video_id: str
    video_title: Optional[str] = None
    course: Optional[str] = None

    youtube_url: str
    start_seconds: int
    end_seconds: Optional[int] = None
    timestamp_display: str
    thumbnail_url: str

    text_en: str
    retrieved_by: List[str] = Field(default_factory=list)


class AskMeta(BaseModel):
    total_chunks_retrieved: int
    retrieval_latency_ms: int
    answer_latency_ms: int
    total_latency_ms: int
    models: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    query: str
    detected_lang: Literal["vi", "en", "unknown"] = "unknown"
    rewrites: Optional[AskRewrites] = None
    answer: AskAnswer
    sources: List[SourceChunk]
    meta: AskMeta


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    indexes_loaded: bool
    uptime_s: int
