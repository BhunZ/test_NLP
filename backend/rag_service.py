from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document

# Import language detection module
from .language_detection import detect_language, _detect_lang, _detect_and_correct_language

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    # Best-effort load of repo .env for GROQ_API_KEY / MISTRAL_API_KEY
    for candidate in (_REPO_ROOT / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


def _format_timestamp(seconds: float) -> str:
    s = int(max(0, seconds))
    mm = s // 60
    ss = s % 60
    return f"{mm:02d}:{ss:02d}"


def _youtube_thumbnail(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"


def _youtube_url(video_id: str, start_seconds: int) -> str:
    return f"https://youtu.be/{video_id}?t={start_seconds}"


def _parse_citation_numbers(text: str) -> List[int]:
    # Extract [1], [2] ... in-order, unique
    out: List[int] = []
    for m in re.finditer(r"\[(\d{1,3})\]", text):
        n = int(m.group(1))
        if n not in out:
            out.append(n)
    return out


@dataclass
class ServiceConfig:
    data_path: str
    faiss_path: str
    bm25_path: str
    default_llm_models: Dict[str, str]


def default_config_from_pipeline() -> ServiceConfig:
    # Reuse the same defaults as pipeline/ask.py
    from pipeline import ask as ask_mod  # type: ignore

    return ServiceConfig(
        data_path=str(getattr(ask_mod, "DEFAULT_DATA_PATH")),
        faiss_path=str(getattr(ask_mod, "DEFAULT_FAISS_PATH")),
        bm25_path=str(getattr(ask_mod, "DEFAULT_BM25_PATH")),
        default_llm_models={
            "groq": getattr(ask_mod, "GROQ_DEFAULT_MODEL"),
            "mistral": getattr(ask_mod, "MISTRAL_DEFAULT_MODEL"),
        },
    )


class RagService:
    def __init__(self, cfg: Optional[ServiceConfig] = None) -> None:
        _load_env()
        self.cfg = cfg or default_config_from_pipeline()

        self._retriever = None
        self._indexes_loaded = False
        self._started = time.time()

        # Lazy import: keep module import cheap for tooling.
        self._ask_mod = None

    @property
    def indexes_loaded(self) -> bool:
        return self._indexes_loaded

    def uptime_s(self) -> int:
        return int(time.time() - self._started)

    def _ensure_loaded(self, rerank: bool, top_k: int) -> None:
        if self._retriever is not None:
            return

        # Stage 7 retriever
        from pipeline.stage_07_retrieve.hybrid_retriever import (  # type: ignore
            build_hybrid_retriever,
        )

        # NOTE: build_hybrid_retriever uses k_rerank as output size
        self._retriever = build_hybrid_retriever(
            data_path=self.cfg.data_path,
            faiss_index_path=self.cfg.faiss_path,
            bm25_path=self.cfg.bm25_path,
            k_retrieve=20,
            k_final=top_k,
            k_rerank=top_k,
            use_reranker=rerank,
            weight_bm25=0.3,
            weight_dense=0.7,
            enable_tracing=False,
            fusion_method="weighted",
        )
        self._indexes_loaded = True

        # Import ask module once; contains call_groq/call_mistral + prompt format.
        from pipeline import ask as ask_mod  # type: ignore

        self._ask_mod = ask_mod

    def _maybe_rewrite(self, query: str, enable_rewrite: bool) -> Tuple[Dict[str, str], Optional[Dict[str, str]]]:
        if not enable_rewrite:
            return {"original": query}, None

        # Stage 6 requires Ollama. If unavailable, fail soft.
        try:
            from pipeline.stage_06_query_rewrite.query_writer import (  # type: ignore
                rewrite_vietnamese_query,
            )
        except Exception:
            return {"original": query}, None

        try:
            rewritten = rewrite_vietnamese_query(vietnamese_query=query, model="qwen2.5:3b", cache_file=None)
            q1 = rewritten.get("q1") or ""
            q2 = rewritten.get("q2") or ""
            out = {"original": rewritten.get("original", query)}
            if q1:
                out["q1_literal"] = q1
            if q2:
                out["q2_expanded"] = q2
            return out, {"q1": q1 or None, "q2": q2 or None}
        except Exception:
            return {"original": query}, None

    def _dedupe_and_label(self, docs_by_variant: Dict[str, List[Document]]) -> List[Document]:
        """
        Merge docs from multiple query variants, annotating metadata.retrieved_by.
        Keep stable order by first-seen (variant order, then within variant order).
        """
        seen: Dict[str, Document] = {}
        order: List[str] = []

        for variant_label, docs in docs_by_variant.items():
            for d in docs:
                md = d.metadata or {}
                key = str(md.get("chunk_id") or md.get("doc_id") or id(d))
                if key not in seen:
                    md = dict(md)
                    md["retrieved_by"] = [variant_label]
                    seen[key] = Document(page_content=d.page_content, metadata=md)
                    order.append(key)
                else:
                    rb = seen[key].metadata.setdefault("retrieved_by", [])
                    if variant_label not in rb:
                        rb.append(variant_label)

        return [seen[k] for k in order]

    async def ask_stream(
        self,
        query: str,
        top_k: int,
        rerank: bool,
        llm_provider: str,
        enable_rewrite: bool,
        course_filter: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        t_start = time.time()

        yield {"event": "stage", "data": {"id": "init", "label": "Đang khởi tạo..."}}

        self._ensure_loaded(rerank=rerank, top_k=top_k)
        assert self._retriever is not None
        assert self._ask_mod is not None

        if enable_rewrite:
            yield {"event": "stage", "data": {"id": "rewrite", "label": "Đang dịch câu hỏi..."}}
        
        rewrites_full, rewrites_slim = self._maybe_rewrite(query, enable_rewrite=enable_rewrite)
        
        if rewrites_slim:
            yield {"event": "rewrites", "data": rewrites_slim}

        # Retrieval
        yield {"event": "stage", "data": {"id": "retrieve", "label": "Đang tìm chunks..."}}
        t_retr = time.time()
        docs_by_variant: Dict[str, List[Document]] = {}

        if "q2_expanded" in rewrites_full:
            docs_by_variant["q2_expanded"] = list(self._retriever(rewrites_full["q2_expanded"]))
            docs_by_variant["original"] = list(self._retriever(rewrites_full["original"]))
        else:
            docs_by_variant["original"] = list(self._retriever(rewrites_full["original"]))

        docs = self._dedupe_and_label(docs_by_variant)

        if course_filter:
            docs = [d for d in docs if (d.metadata or {}).get("course") == course_filter]

        docs = docs[:top_k]
        retrieval_latency_ms = int((time.time() - t_retr) * 1000)

        sources = []
        for idx, d in enumerate(docs, 1):
            md = d.metadata or {}
            video_id = str(md.get("video_id") or "unknown")
            start_time = float(md.get("start_time") or 0.0)
            end_time = md.get("end_time")
            start_seconds = int(max(0, start_time))
            end_seconds = int(end_time) if end_time is not None else None
            url = str(md.get("url") or _youtube_url(video_id, start_seconds))
            sources.append({
                "chunk_id": str(md.get("chunk_id") or md.get("doc_id") or f"chunk:{idx}"),
                "rank": idx,
                "score": (float(md.get("score")) if md.get("score") is not None else None),
                "video_id": video_id,
                "video_title": md.get("title"),
                "course": md.get("course"),
                "youtube_url": url,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "thumbnail_url": _youtube_thumbnail(video_id),
                "text_en": (d.page_content or "").strip(),
                "timestamp_display": _format_timestamp(start_time),
                "retrieved_by": list(md.get("retrieved_by") or []),
            })
        
        yield {"event": "sources", "data": sources}

        # Merge context
        yield {"event": "stage", "data": {"id": "merge", "label": "Đang tổng hợp..."}}
        try:
            from pipeline.stage_08_merge_context.context_to_jsonl import (
                group_docs_by_video,
                merge_grouped_docs,
            )
            grouped = group_docs_by_video(docs)
            merged_contexts = merge_grouped_docs(grouped)
        except Exception:
            merged_contexts = []

        # LLM Answer streaming
        yield {"event": "stage", "data": {"id": "answer", "label": "Đang viết câu trả lời..."}}
        llm_model = self.cfg.default_llm_models.get(llm_provider, "")
        
        from backend.streaming import stream_groq, stream_mistral
        
        full_answer = ""
        t_ans = time.time()
        
        stream_fn = stream_groq if llm_provider == "groq" else stream_mistral
        corrected_query, detected_lang, _ = _detect_and_correct_language(query)
        
        try:
            async for token in stream_fn(
                question=corrected_query,
                contexts=merged_contexts,
                model=llm_model,
                system_prompt=self._ask_mod.SYSTEM_PROMPT_VI,
                build_user_prompt_fn=self._ask_mod.build_user_prompt,
                detected_lang=detected_lang
            ):
                full_answer += token
                yield {"event": "token", "data": {"text": token}}
        except Exception as e:
            yield {"event": "error", "data": {"message": f"Lỗi LLM: {str(e)}"}}
            return

        answer_latency_ms = int((time.time() - t_ans) * 1000)
        citations = _parse_citation_numbers(full_answer)
        confidence = "high" if citations and len(citations) >= 2 else ("medium" if citations else "low")
        total_latency_ms = int((time.time() - t_start) * 1000)

        yield {
            "event": "done",
            "data": {
                "answer": {
                    "text_vi": full_answer,
                    "citations": citations,
                    "confidence": confidence,
                },
                "meta": {
                    "total_chunks_retrieved": len(docs),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "answer_latency_ms": answer_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "models": {"llm": llm_model, "provider": llm_provider},
                    "retrieval": {"top_k": top_k, "rerank": rerank, "course_filter": course_filter},
                }
            }
        }

    def ask(
        self,
        query: str,
        top_k: int,
        rerank: bool,
        llm_provider: str,
        enable_rewrite: bool,
        course_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        t_start = time.time()

        self._ensure_loaded(rerank=rerank, top_k=top_k)
        assert self._retriever is not None
        assert self._ask_mod is not None

        rewrites_full, rewrites_slim = self._maybe_rewrite(query, enable_rewrite=enable_rewrite)

        # Retrieval: for Phase 1, run only original unless rewrite enabled.
        t_retr = time.time()
        docs_by_variant: Dict[str, List[Document]] = {}

        # If rewriting produced q2_expanded, prefer it for retrieval; keep original too.
        if "q2_expanded" in rewrites_full:
            docs_by_variant["q2_expanded"] = list(self._retriever(rewrites_full["q2_expanded"]))  # type: ignore[misc]
            docs_by_variant["original"] = list(self._retriever(rewrites_full["original"]))  # type: ignore[misc]
        else:
            docs_by_variant["original"] = list(self._retriever(rewrites_full["original"]))  # type: ignore[misc]

        docs = self._dedupe_and_label(docs_by_variant)

        # Optional course filter (no-op for now unless explicitly used)
        if course_filter:
            docs = [d for d in docs if (d.metadata or {}).get("course") == course_filter]

        docs = docs[:top_k]
        retrieval_latency_ms = int((time.time() - t_retr) * 1000)

        # Stage 8 merge: used for answer prompt only, not for source cards.
        t_merge = time.time()
        try:
            from pipeline.stage_08_merge_context.context_to_jsonl import (  # type: ignore
                group_docs_by_video,
                merge_grouped_docs,
            )
            grouped = group_docs_by_video(docs)
            merged_contexts = merge_grouped_docs(grouped)
        except Exception:
            merged_contexts = []
        _ = t_merge  # reserved for future meta

        # Stage 9 answer
        t_ans = time.time()
        llm_model = self.cfg.default_llm_models.get(llm_provider, "")
        corrected_query, detected_lang, _ = _detect_and_correct_language(query)
        answer_text = self._ask_mod.call_llm(llm_provider, llm_model, corrected_query, merged_contexts, detected_lang)
        answer_latency_ms = int((time.time() - t_ans) * 1000)

        citations = _parse_citation_numbers(answer_text)
        confidence = "high" if citations and len(citations) >= 2 else ("medium" if citations else "low")

        sources: List[Dict[str, Any]] = []
        for idx, d in enumerate(docs, 1):
            md = d.metadata or {}
            video_id = str(md.get("video_id") or "unknown")
            start_time = float(md.get("start_time") or 0.0)
            end_time = md.get("end_time")
            start_seconds = int(max(0, start_time))
            end_seconds = int(end_time) if end_time is not None else None

            url = str(md.get("url") or _youtube_url(video_id, start_seconds))
            sources.append(
                {
                    "chunk_id": str(md.get("chunk_id") or md.get("doc_id") or f"chunk:{idx}"),
                    "rank": idx,
                    "score": (float(md.get("score")) if md.get("score") is not None else None),
                    "video_id": video_id,
                    "video_title": md.get("title"),
                    "course": md.get("course"),
                    "youtube_url": url,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "timestamp_display": _format_timestamp(start_time),
                    "thumbnail_url": _youtube_thumbnail(video_id),
                    "text_en": (d.page_content or "").strip(),
                    "retrieved_by": list(md.get("retrieved_by") or []),
                }
            )

        total_latency_ms = int((time.time() - t_start) * 1000)

        return {
            "query": query,
            "detected_lang": _detect_lang(query),
            "rewrites": rewrites_slim,
            "answer": {
                "text_vi": answer_text,
                "citations": citations,
                "confidence": confidence,
            },
            "sources": sources,
            "meta": {
                "total_chunks_retrieved": len(docs),
                "retrieval_latency_ms": retrieval_latency_ms,
                "answer_latency_ms": answer_latency_ms,
                "total_latency_ms": total_latency_ms,
                "models": {"llm": llm_model, "provider": llm_provider},
                "retrieval": {"top_k": top_k, "rerank": rerank, "course_filter": course_filter},
            },
        }

