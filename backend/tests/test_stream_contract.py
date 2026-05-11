import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from backend.rag_service import RagService, ServiceConfig


class TestStreamContract(unittest.IsolatedAsyncioTestCase):
    async def test_done_event_contains_retrieval_meta_when_no_tokens(self):
        service = RagService(
            ServiceConfig(
                data_path="unused",
                faiss_path="unused",
                bm25_path="unused",
                default_llm_models={"groq": "fake-model", "mistral": "fake-model"},
            )
        )
        service._ask_mod = SimpleNamespace(
            SYSTEM_PROMPT_VI="system",
            build_user_prompt=lambda question, contexts: "prompt",
        )
        service._retriever = lambda query: [
            Document(
                page_content="context",
                metadata={"chunk_id": "c1", "video_id": "v1", "start_time": 3},
            )
        ]
        service._ensure_loaded = lambda rerank, top_k: None  # type: ignore[assignment]

        async def empty_stream(**kwargs):
            if False:  # pragma: no cover
                yield ""

        with patch("backend.streaming.stream_groq", empty_stream):
            events = []
            async for event in service.ask_stream(
                query="test question",
                top_k=3,
                rerank=False,
                llm_provider="groq",
                enable_rewrite=False,
            ):
                events.append(event)

        done_events = [e for e in events if e["event"] == "done"]
        self.assertEqual(len(done_events), 1)

        done_payload = done_events[0]["data"]
        self.assertEqual(done_payload["answer"]["text_vi"], "")
        self.assertIn("retrieval", done_payload["meta"])
        self.assertEqual(done_payload["meta"]["retrieval"]["top_k"], 3)
        self.assertFalse(done_payload["meta"]["retrieval"]["rerank"])


if __name__ == "__main__":
    unittest.main()
