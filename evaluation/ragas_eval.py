"""
embed__data/evaluate_ragas.py
═══════════════════════════════════════════════════════════════════════
RAGAS-style retrieval evaluation — context-side metrics only.

Evaluates the chunking + retrieval pipeline WITHOUT requiring an LLM
answer-generation step. This is the "before LLM" evaluation your team
lead asked for.

THREE METRICS (custom implementation, not the official `ragas` library)
──────────────────────────────────────────────────────────────────────

  1. Context Precision     — Of the retrieved contexts, how many were
                              actually useful for answering the question?
                              Measured at rank-aware MAP-style.
                              Range: 0..1, higher is better.

  2. Context Recall        — Did retrieval bring back enough info to
                              cover the ground-truth answer? We extract
                              atomic claims from the answer and check
                              each one against the retrieved contexts.
                              Range: 0..1, higher is better.

  3. Context Relevancy     — How much of each retrieved context is
                              actually relevant vs filler? Counts the
                              proportion of relevant sentences.
                              Range: 0..1, higher is better.

All three use Gemini (gemini-2.5-flash) as the judge. The judge LLM
reads the question / context / ground-truth and outputs a structured
verdict — see "AI judges AI" explanation in conversation history.

PIPELINE
────────

  Phase A — generate test queries (one-time, ~5 min)
    Sample chunks → Gemini generates (question_vi, answer_vi) pair
    per chunk → save to test_queries.jsonl

  Phase B — evaluate
    For each query: call existing hybrid retriever → get top-k contexts
    → run 3 RAGAS metrics via Gemini → aggregate scores

USAGE
─────

    # Phase A (one-time): generate 50 test queries
    export GEMINI_API_KEY=your_key_here
    python embed__data/evaluate_ragas.py --generate-queries 50

    # Phase B: evaluate
    python embed__data/evaluate_ragas.py

    # Quick test on first 5 queries
    python embed__data/evaluate_ragas.py --limit 5

    # Skip reranker for faster eval
    python embed__data/evaluate_ragas.py --no-reranker

    # Custom output
    python embed__data/evaluate_ragas.py --output reports/ragas_t072.json

    # Compare two chunking strategies (run twice, then diff the outputs)
    python embed__data/evaluate_ragas.py \\
        --data ../data/chunked/transcript_v3_t055.jsonl \\
        --faiss ../indexes/faiss_index_055 \\
        --bm25 ../indexes/bm25_055.pkl \\
        --output reports/ragas_t055.json

This file lives in embed__data/ so it can import hybrid_improved.py
directly. Run from project root:  cd <project>  &&  python embed__data/evaluate_ragas.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time

# Optional: load env vars from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    def _load_dotenv(*_args, **_kwargs):  # type: ignore
        return False
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# Make Windows console handle unicode in our reports
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# Make sibling embed__data/ modules importable regardless of CWD
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# DEFAULTS
# ════════════════════════════════════════════════════════════════

_PROJECT_ROOT = _THIS_DIR.parent
# Defaults match the leader's new pipeline (main_final.py):
#   - data: t072 chunked file (has chunk_id field, used by all newer indexes)
#   - faiss: faiss_index_072_n (the rebuilt index, note the _n suffix)
#   - bm25: bm25_072.pkl (rebuilt 2026-05-09)
DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "chunked" / "transcript_v3_t072.jsonl"
DEFAULT_FAISS_PATH = _PROJECT_ROOT / "indexes" / "faiss_index_072_n"
DEFAULT_BM25_PATH = _PROJECT_ROOT / "indexes" / "bm25_072.pkl"
DEFAULT_QUERIES_PATH = _PROJECT_ROOT / "data" / "chunked" / "test_queries.jsonl"
DEFAULT_REPORT_PATH = _PROJECT_ROOT / "data" / "chunked" / "ragas_report.json"

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_RPM_DELAY = 4.0   # seconds between calls — keeps us under 15 RPM free tier

# Groq defaults — much higher free quota than Gemini.
# Free tier (as of 2026): llama-3.3-70b-versatile = ~1,000 RPD,
# llama-3.1-8b-instant + qwen-qwq-32b = ~14,400 RPD.
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_RPM_DELAY = 2.0     # Groq tolerates ~30 RPM on free tier

# ════════════════════════════════════════════════════════════════
# DATA CLASSES
# ════════════════════════════════════════════════════════════════

@dataclass
class TestQuery:
    """A test record: question + reference answer."""
    query_id: str
    question: str
    ground_truth: str
    source_chunk_id: str
    source_video_id: str
    source_video_title: str
    source_start_time: float


@dataclass
class QueryEval:
    """Per-query evaluation result."""
    query_id: str
    question: str
    n_contexts: int
    # ── RAGAS metrics (LLM judge) ─────────────────────────────────
    context_precision: float
    context_recall: float
    context_relevancy: float
    # ── Rank metrics (deterministic, no LLM) ──────────────────────
    # Computed by checking if source_chunk_id appears in retrieved top-N.
    # source_rank is None when the chunk wasn't found in top-N at all.
    hit_at_1: int = 0
    hit_at_3: int = 0
    hit_at_5: int = 0
    hit_at_6: int = 0
    hit_at_10: int = 0
    hit_at_20: int = 0
    # Precision@K — distinct from Context Precision (LLM-judged)
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    # MRR / MAP
    reciprocal_rank: float = 0.0          # 1 / source_rank over top-20
    reciprocal_rank_at_3: float = 0.0     # MRR@3 — capped at top-3
    average_precision: float = 0.0        # AP per query (mean → MAP)
    # nDCG@K (binary relevance, single ground-truth doc)
    ndcg_at_3: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    source_rank: Optional[int] = None
    n_retrieved_for_rank: int = 0
    # ── Semantic metrics (deterministic, embedder-based) ──────────
    max_query_context_sim:   float = 0.0
    mean_query_context_sim:  float = 0.0
    semantic_precision_at_k: float = 0.0
    semantic_recall:         float = 0.0
    # ── Context Entities Recall (LLM-judged) ─────────────────────
    context_entities_recall: float = 0.0
    n_entities_extracted:    int = 0
    n_entities_found:        int = 0
    # ── Ops ──────────────────────────────────────────────────────
    retrieval_latency_ms: float = 0.0     # wall time spent in retrieval calls
    # ── Bookkeeping ──────────────────────────────────────────────
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    judge_notes: Dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
# GEMINI CLIENT (lightweight wrapper)
# ════════════════════════════════════════════════════════════════

class GeminiJudge:
    """Thin wrapper around google-genai for RAGAS judging."""

    def __init__(self, model: str = GEMINI_MODEL, rpm_delay: float = GEMINI_RPM_DELAY):
        try:
            from google import genai             # type: ignore
            from google.genai import types       # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install google-genai: pip install google-genai"
            ) from e

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GEMINI_API_KEY env var. Get free key: "
                "https://aistudio.google.com/apikey"
            )

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model
        self._rpm_delay = rpm_delay
        self._last_call = 0.0
        self._n_calls = 0
        self._n_failures = 0
        # Set to True by call_json when it detects a 429/quota error.
        # FallbackJudge reads this flag to decide whether to skip this
        # provider for the rest of the run.
        self._last_error_was_rate_limit = False

    def call_json(self, prompt: str, retry: int = 2) -> Optional[Dict[str, Any]]:
        """
        Call Gemini and parse JSON response. Returns None on parse failure
        after retries. Throttles to stay under free-tier RPM.
        """
        # Throttle
        elapsed = time.time() - self._last_call
        if elapsed < self._rpm_delay:
            time.sleep(self._rpm_delay - elapsed)

        self._last_error_was_rate_limit = False
        for attempt in range(retry + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    config=self._types.GenerateContentConfig(
                        temperature=0.1,        # judging needs to be stable
                        response_mime_type="application/json",
                    ),
                    contents=prompt,
                )
                self._last_call = time.time()
                self._n_calls += 1
                text = (response.text or "").strip()
                # Strip markdown fences if present
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, e)
                if attempt < retry:
                    time.sleep(2)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate_limit" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    self._last_error_was_rate_limit = True
                    self._n_failures += 1
                    logger.warning("Gemini RATE-LIMITED (TPD/RPD exhausted)")
                    return None  # fail fast — retry won't help
                logger.warning("Gemini call failed (attempt %d): %s", attempt + 1, e)
                if attempt < retry:
                    time.sleep(2)
        self._n_failures += 1
        return None

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def n_failures(self) -> int:
        return self._n_failures


# ════════════════════════════════════════════════════════════════
# GROQ CLIENT (alternative — higher free-tier quota)
# ════════════════════════════════════════════════════════════════

class GroqJudge:
    """
    Same interface as GeminiJudge, uses Groq Cloud (OpenAI-compatible API).

    Free-tier quotas (as of 2026, check console.groq.com for current):
      - llama-3.3-70b-versatile : ~1,000 RPD, 30 RPM
      - llama-3.1-8b-instant    : ~14,400 RPD, 30 RPM (much higher quota,
                                                       weaker as a judge)
      - qwen-qwq-32b            : reasoning-tuned Qwen, ~1,000 RPD
      - openai/gpt-oss-20b      : open-weight reasoning, ~1,000 RPD

    Get a free key at https://console.groq.com → set GROQ_API_KEY.
    """

    def __init__(self, model: str = GROQ_MODEL, rpm_delay: float = GROQ_RPM_DELAY):
        try:
            from groq import Groq                          # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install groq SDK: pip install groq"
            ) from e

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GROQ_API_KEY env var. Get free key: "
                "https://console.groq.com/keys"
            )

        self._client = Groq(api_key=api_key)
        self._model = model
        self._rpm_delay = rpm_delay
        self._last_call = 0.0
        self._n_calls = 0
        self._n_failures = 0
        self._last_error_was_rate_limit = False

    def call_json(self, prompt: str, retry: int = 2) -> Optional[Dict[str, Any]]:
        # Throttle
        elapsed = time.time() - self._last_call
        if elapsed < self._rpm_delay:
            time.sleep(self._rpm_delay - elapsed)

        self._last_error_was_rate_limit = False
        for attempt in range(retry + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
                self._last_call = time.time()
                self._n_calls += 1
                text = (response.choices[0].message.content or "").strip()
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
                return json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, e)
                if attempt < retry:
                    time.sleep(2)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate_limit" in err_str or "quota" in err_str:
                    self._last_error_was_rate_limit = True
                    self._n_failures += 1
                    logger.warning("Groq RATE-LIMITED on %s (TPD exhausted)", self._model)
                    return None  # fail fast
                logger.warning("Groq call failed (attempt %d): %s", attempt + 1, e)
                if attempt < retry:
                    time.sleep(2)
        self._n_failures += 1
        return None

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def n_failures(self) -> int:
        return self._n_failures


# ════════════════════════════════════════════════════════════════
# GENERIC OPENAI-COMPATIBLE CLIENT (covers most free providers)
# ════════════════════════════════════════════════════════════════
#
# Most modern LLM hosts expose an OpenAI-compatible /chat/completions
# endpoint. One adapter handles them all — only base URL, model name,
# and API-key env var differ. Uses `requests` (already in deps), no SDK
# install needed.

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        # Free Qwen 2.5 72B — high quality multilingual judge.
        # Free tier: 50 RPD without deposit, 1000 RPD with $10 deposit.
        "default_model": "qwen/qwen-2.5-72b-instruct:free",
        "default_rpm_delay": 4.0,
        "console": "https://openrouter.ai/keys",
        "notes": "50 RPD free / 1000 RPD with $10 lifetime credit deposit",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        # Pinned to a specific version (mistral-small-2506 has the best
        # free-tier limits as of 2026: 2.25M TPM, 5 RPS, no monthly cap).
        # The "-latest" alias works but rolls forward without notice.
        "default_model": "mistral-small-2506",
        "default_rpm_delay": 0.5,   # 5 RPS allowed, but be polite
        "console": "https://console.mistral.ai",
        "notes": "Free tier: 2.25M TPM, 5 RPS on mistral-small-2506",
    },
    "ollama": {
        # Local Ollama with /v1 OpenAI-compatibility endpoint.
        # Run: ollama pull qwen2.5:7b ; ollama serve
        "base_url": "http://localhost:11434/v1",
        "api_key_env": None,        # local, no key needed
        "default_model": "qwen2.5:7b",
        "default_rpm_delay": 0.0,   # no rate limit locally
        "console": "https://ollama.com",
        "notes": "Local — unlimited but limited by your hardware speed",
    },
}


class OpenAICompatibleJudge:
    """
    Generic judge for any OpenAI-compatible /chat/completions endpoint.

    Used by: openrouter, mistral, ollama. Anywhere that speaks
    {"model","messages","temperature","response_format"} returns
    choices[0].message.content. No SDK dependency — uses `requests`.
    """

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: Optional[str],
        model: str,
        rpm_delay: float = 2.0,
    ) -> None:
        try:
            import requests              # type: ignore
        except ImportError as e:
            raise ImportError("requests is required (pip install requests)") from e

        self._requests = requests
        self._provider_name = provider_name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._rpm_delay = rpm_delay
        self._last_call = 0.0
        self._n_calls = 0
        self._n_failures = 0
        self._supports_json_mode = True   # auto-disabled if first call 400s
        self._last_error_was_rate_limit = False

        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def call_json(self, prompt: str, retry: int = 2) -> Optional[Dict[str, Any]]:
        # Throttle
        elapsed = time.time() - self._last_call
        if elapsed < self._rpm_delay:
            time.sleep(self._rpm_delay - elapsed)

        url = f"{self._base_url}/chat/completions"

        self._last_error_was_rate_limit = False
        for attempt in range(retry + 1):
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            if self._supports_json_mode:
                payload["response_format"] = {"type": "json_object"}

            try:
                resp = self._session.post(url, json=payload, timeout=60)
                if resp.status_code == 429:
                    self._last_error_was_rate_limit = True
                    self._n_failures += 1
                    logger.warning(
                        "%s RATE-LIMITED on %s",
                        self._provider_name, self._model,
                    )
                    return None  # fail fast — fallback chain handles it
                if resp.status_code == 400 and self._supports_json_mode:
                    # Some providers reject response_format — retry without it
                    self._supports_json_mode = False
                    logger.debug(
                        "%s rejected response_format; retrying without it",
                        self._provider_name,
                    )
                    continue
                # Permanent client errors (model name wrong, auth bad, etc.)
                # Mark this provider as exhausted so the FallbackJudge chain
                # stops wasting time retrying it.
                if resp.status_code in (401, 403, 404, 410, 422):
                    self._last_error_was_rate_limit = True  # treated as "done"
                    self._n_failures += 1
                    logger.warning(
                        "%s permanent HTTP %d on %s — marking provider exhausted "
                        "(check model name / API key)",
                        self._provider_name, resp.status_code, self._model,
                    )
                    return None
                resp.raise_for_status()
                self._last_call = time.time()
                self._n_calls += 1

                data = resp.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                # Strip markdown fences if present
                text = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE,
                )
                return json.loads(text)

            except json.JSONDecodeError as e:
                logger.warning(
                    "%s: JSON parse failed (attempt %d): %s",
                    self._provider_name, attempt + 1, e,
                )
                if attempt < retry:
                    time.sleep(2)
            except Exception as e:
                logger.warning(
                    "%s call failed (attempt %d): %s",
                    self._provider_name, attempt + 1, e,
                )
                if attempt < retry:
                    time.sleep(2)
        self._n_failures += 1
        return None

    @property
    def n_calls(self) -> int:
        return self._n_calls

    @property
    def n_failures(self) -> int:
        return self._n_failures


# ════════════════════════════════════════════════════════════════
# JUDGE FACTORY
# ════════════════════════════════════════════════════════════════

def get_judge(provider: str, model: Optional[str] = None,
              rpm_delay: Optional[float] = None):
    """
    Factory: pick the right judge backend.

    Recommended (highest free quotas):
      - groq          : 14,400 RPD on 8B, 1,000 RPD on 70B. Already wired.
      - mistral       : European, generous on tokens (1B/month).
      - openrouter    : 50 RPD free (1k with $10 deposit) — gives Qwen 72B etc.
      - ollama        : local, unlimited if you have the hardware.

    Deprecated (kept for backward compat — quota too small for full eval):
      - gemini        : ~20 RPD on 2.5-flash free tier. Use only for tiny tests.
    """
    p = provider.lower().strip()

    if p == "gemini":
        logger.warning(
            "Gemini free tier is now ~20 RPD on 2.5-flash. "
            "Recommend switching to --judge-provider groq or mistral."
        )
        return GeminiJudge(
            model=model or GEMINI_MODEL,
            rpm_delay=rpm_delay if rpm_delay is not None else GEMINI_RPM_DELAY,
        )

    if p == "groq":
        return GroqJudge(
            model=model or GROQ_MODEL,
            rpm_delay=rpm_delay if rpm_delay is not None else GROQ_RPM_DELAY,
        )

    if p in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[p]
        api_key = os.getenv(preset["api_key_env"]) if preset["api_key_env"] else None
        if preset["api_key_env"] and not api_key:
            raise ValueError(
                f"Set {preset['api_key_env']} env var. "
                f"Get a key: {preset['console']}\n"
                f"Notes: {preset['notes']}"
            )
        return OpenAICompatibleJudge(
            provider_name=p,
            base_url=preset["base_url"],
            api_key=api_key,
            model=model or preset["default_model"],
            rpm_delay=rpm_delay
                if rpm_delay is not None
                else preset["default_rpm_delay"],
        )

    available = ["gemini", "groq", *PROVIDER_PRESETS.keys()]
    raise ValueError(
        f"Unknown judge provider: {provider!r}. "
        f"Available: {', '.join(available)}"
    )


def resolve_model_name(provider: str, model: Optional[str]) -> str:
    """Resolve effective model name for a provider (used for the JSON report)."""
    if model:
        return model
    p = provider.lower().strip()
    if p == "gemini":
        return GEMINI_MODEL
    if p == "groq":
        return GROQ_MODEL
    if p in PROVIDER_PRESETS:
        return PROVIDER_PRESETS[p]["default_model"]
    return "unknown"


# ════════════════════════════════════════════════════════════════
# FALLBACK CHAIN (auto-switch providers on rate-limit)
# ════════════════════════════════════════════════════════════════

class FallbackJudge:
    """
    Wraps a chain of judges. Calls them in order; on rate-limit (429 / TPD
    exhausted), marks that provider as exhausted for the rest of the run
    and falls through to the next one.

    Use case: free-tier token budgets are small. With Groq 8B + Cerebras +
    Mistral chained, you get ~34M+ tokens/day combined before any provider
    runs out. The script just keeps going.

    Construction:
        FallbackJudge.from_chain("groq,mistral", model=None, rpm_delay=None)

    Each provider in the chain uses its own default model unless overridden.
    """

    def __init__(self, judges: List[Tuple[str, Any]]):
        if not judges:
            raise ValueError("FallbackJudge requires at least one judge")
        self._judges = judges                    # [(provider_name, judge_instance), ...]
        self._exhausted: set = set()             # providers we've stopped using
        self._calls_per_provider: Dict[str, int] = {p: 0 for p, _ in judges}
        self._chain_failures: int = 0            # calls where the WHOLE chain failed

    @classmethod
    def from_chain(
        cls,
        chain: str,
        model: Optional[str] = None,
        rpm_delay: Optional[float] = None,
    ) -> "FallbackJudge":
        """Parse 'groq,mistral' into a FallbackJudge."""
        names = [n.strip().lower() for n in chain.split(",") if n.strip()]
        if not names:
            raise ValueError("Empty fallback chain")

        judges: List[Tuple[str, Any]] = []
        skipped: List[str] = []
        for name in names:
            try:
                j = get_judge(provider=name, model=model, rpm_delay=rpm_delay)
                judges.append((name, j))
                logger.info("Fallback judge initialized: %s", name)
            except (ValueError, ImportError) as e:
                skipped.append(f"{name} ({e})")
                logger.warning("Skipping %s in fallback chain: %s", name, e)

        if not judges:
            raise RuntimeError(
                "No fallback judges could be initialized. "
                f"All skipped: {', '.join(skipped)}"
            )
        if skipped:
            logger.info("Active chain: %s (skipped: %s)",
                        " → ".join(p for p, _ in judges),
                        ", ".join(skipped))
        else:
            print(f"[fallback] Chain: {' → '.join(p for p, _ in judges)}")
        return cls(judges)

    def call_json(self, prompt: str, retry: int = 2) -> Optional[Dict[str, Any]]:
        for provider_name, judge in self._judges:
            if provider_name in self._exhausted:
                continue

            result = judge.call_json(prompt, retry=retry)
            if result is not None:
                self._calls_per_provider[provider_name] += 1
                return result

            # Did this provider hit a rate limit? If so, mark it exhausted
            # so we don't waste time trying it again on subsequent calls.
            if getattr(judge, "_last_error_was_rate_limit", False):
                self._exhausted.add(provider_name)
                remaining = [p for p, _ in self._judges if p not in self._exhausted]
                if remaining:
                    print(
                        f"\n  ⚠️  {provider_name} EXHAUSTED. "
                        f"Falling through to: {' → '.join(remaining)}\n"
                    )
                else:
                    print(
                        f"\n  ❌ ALL PROVIDERS EXHAUSTED. "
                        f"Wait 24h or upgrade.\n"
                    )

        # All providers either exhausted or failed for other reasons —
        # this counts as an "eval-level failure": no metric could be computed
        self._chain_failures += 1
        return None

    @property
    def n_calls(self) -> int:
        return sum(j.n_calls for _, j in self._judges)

    @property
    def n_failures(self) -> int:
        # Count failures only from the LAST provider tried per call
        # (failures from earlier providers in the chain are recovered when
        # a later one succeeds, so they don't count as eval-level failures).
        # Approximate: sum all sub-judge failures, then subtract those that
        # were "recovered" by a later success.
        # Simpler accurate proxy: count calls where the chain returned None
        # (i.e. ALL judges failed). Tracked separately below.
        return self._chain_failures

    def calls_breakdown(self) -> Dict[str, int]:
        """Per-provider call count, useful for the report."""
        return dict(self._calls_per_provider)


# ════════════════════════════════════════════════════════════════
# QUERY REWRITER (matches production pipeline)
# ════════════════════════════════════════════════════════════════

QUERY_REWRITE_PROMPT = """You are a search query optimizer for Stanford ML / NLP \
course transcripts. The transcripts are in English; the user's question is in \
Vietnamese.

Convert the Vietnamese question into TWO English search queries:
- q1: direct, faithful translation
- q2: expanded retrieval query (add synonyms or related technical terms a \
lecturer might use; keep it natural)

Rules:
- Keep technical terms unchanged (transformer, attention, gradient descent, \
RLHF, BERT, etc.).
- Do NOT translate technical terms into Vietnamese.
- q2 should be DIFFERENT from q1 (different wording or expanded scope).

Vietnamese question:
{vi_query}

Return ONLY this JSON, nothing else:
{{"q1": "...", "q2": "..."}}"""


def _rewrite_cache_load(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _rewrite_cache_save(path: Optional[Path], cache: Dict[str, Dict[str, str]]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def rewrite_query(
    judge,
    vi_query: str,
    cache: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, str]:
    """
    Rewrite a Vietnamese query into English variants for retrieval.

    Returns: {"original": <vi>, "q1": <literal EN>, "q2": <expanded EN>}.
    Falls back to (original, original, original+" machine learning") if the
    judge returns malformed output — same fallback as the production
    query_writer.py.
    """
    # Cache hit?
    if cache is not None and vi_query in cache:
        cached = cache[vi_query]
        return {
            "original": vi_query,
            "q1": cached.get("q1", vi_query),
            "q2": cached.get("q2", vi_query),
        }

    result = judge.call_json(QUERY_REWRITE_PROMPT.format(vi_query=vi_query))
    q1 = (result or {}).get("q1")
    q2 = (result or {}).get("q2")
    if not isinstance(q1, str) or not isinstance(q2, str) or not q1 or not q2:
        # Same fallback as embed__data/query_writer.py:111-113
        q1 = vi_query
        q2 = vi_query + " machine learning"

    if cache is not None:
        cache[vi_query] = {"q1": q1, "q2": q2}

    return {"original": vi_query, "q1": q1, "q2": q2}


# ════════════════════════════════════════════════════════════════
# PHASE A: TEST QUERY GENERATION
# ════════════════════════════════════════════════════════════════

QUERY_GEN_PROMPT = """Bạn là người tạo bộ đánh giá cho hệ thống RAG học tập NLP.

Đoạn trích bên dưới được lấy từ một bài giảng Stanford (tiếng Anh, đã clean).
Hãy tạo MỘT cặp (câu hỏi, câu trả lời) BẰNG TIẾNG VIỆT, sao cho:
- Câu hỏi là điều một sinh viên Việt Nam có thể hỏi sau khi đọc đoạn trích.
- Câu trả lời dựa CHÍNH trên thông tin trong đoạn trích.
- Câu hỏi CỤ THỂ (không quá chung chung như "Bài giảng nói về gì?").
- Giữ nguyên thuật ngữ tiếng Anh (transformer, attention, gradient descent, ...).
- Câu trả lời 2-4 câu, không cần lặp lại nguyên văn đoạn trích.

LƯU Ý: Đoạn trích là một phần của bài giảng nên có thể tham chiếu tới nội dung
khác (ví dụ: "as I showed earlier"). Hãy tập trung vào THÔNG TIN CHÍNH trong
đoạn trích, không bận tâm tới các tham chiếu phụ.

ĐOẠN TRÍCH (English):
{chunk_text}

Trả về JSON đúng format này (không có gì thêm):
{{
  "question": "câu hỏi tiếng Việt cụ thể",
  "answer": "câu trả lời tiếng Việt 2-4 câu"
}}"""


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return chunks


def generate_queries(
    chunks: List[Dict[str, Any]],
    n: int,
    judge: GeminiJudge,
    out_path: Path,
    min_chunk_tokens: int = 100,
) -> List[TestQuery]:
    """
    Sample chunks, ask Gemini to generate (question, answer) pairs.
    Stratified by course so every course is represented. Saves to JSONL.
    """
    # Stratified sample: roughly equal across courses
    by_course: Dict[str, List[int]] = {}
    for i, c in enumerate(chunks):
        if c.get("token_count", 0) < min_chunk_tokens:
            continue
        by_course.setdefault(c.get("course", "Unknown"), []).append(i)

    courses = list(by_course.keys())
    per_course = max(1, n // len(courses))
    sampled_indices: List[int] = []
    rng = random.Random(42)
    for course in courses:
        pool = by_course[course]
        rng.shuffle(pool)
        sampled_indices.extend(pool[:per_course])
    rng.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:n]

    print(f"\n[generate] Sampling {len(sampled_indices)} chunks across {len(courses)} courses")
    print(f"[generate] Calling Gemini (~{GEMINI_RPM_DELAY:.0f}s between calls)...\n")

    queries: List[TestQuery] = []
    skipped_parse = 0
    skipped_short = 0

    for k, idx in enumerate(sampled_indices, 1):
        c = chunks[idx]
        chunk_text = c.get("chunk_text", "")
        course = c.get("course", "?")
        title = (c.get("title") or "")[:50]
        print(f"  [{k:>3}/{len(sampled_indices)}] {course} — {title}")

        result = judge.call_json(QUERY_GEN_PROMPT.format(chunk_text=chunk_text))
        if not result:
            skipped_parse += 1
            logger.debug("  [%d] parse failure or empty response", k)
            continue

        q_text = (result.get("question") or "").strip()
        a_text = (result.get("answer") or "").strip()
        if len(q_text) < 10 or len(a_text) < 20:
            skipped_short += 1
            logger.debug(
                "  [%d] too short: q=%d chars, a=%d chars",
                k, len(q_text), len(a_text),
            )
            continue

        chunk_id = c.get("chunk_id") or f"{c.get('video_id', '?')}:idx_{idx}"
        queries.append(TestQuery(
            query_id=f"q{k:03d}",
            question=q_text,
            ground_truth=a_text,
            source_chunk_id=chunk_id,
            source_video_id=c.get("video_id", ""),
            source_video_title=c.get("title", ""),
            source_start_time=float(c.get("start_time", 0.0)),
        ))

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    print(f"\n[generate] Saved {len(queries)} queries to {out_path}")
    if skipped_parse:
        print(f"[generate] Skipped {skipped_parse} chunks — LLM returned malformed JSON or hit a transient error")
    if skipped_short:
        print(f"[generate] Skipped {skipped_short} chunks — generated question/answer was too short")
    if not queries:
        print("[generate] WARNING: 0 queries saved. Re-run with -v / --verbose to see what the LLM returned.")
    return queries


def load_queries(path: Path) -> List[TestQuery]:
    out: List[TestQuery] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append(TestQuery(**d))
    return out


# ════════════════════════════════════════════════════════════════
# RAGAS METRIC IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════

# ─── Combined Precision + Relevancy (per-context, single LLM call) ──
# Token-saver: precision and relevancy both ask "is this context useful
# for this question?" — bundled into one judge call instead of two.
# Cuts per-context judge calls from 2 → 1, saving ~30-40% of tokens.
COMBINED_PRECISION_RELEVANCY_PROMPT = """Bạn là người chấm điểm hệ thống RAG.

CÂU HỎI: {question}
CÂU TRẢ LỜI ĐÚNG (ground truth): {ground_truth}

ĐOẠN NGỮ CẢNH (rank {rank}):
{context}

Đánh giá đoạn ngữ cảnh trên ở HAI khía cạnh:
1. "useful": Đoạn này có HỮU ÍCH để trả lời đúng câu hỏi không? (true/false)
2. Tỷ lệ câu trong đoạn liên quan TRỰC TIẾP đến câu hỏi:
   - "total_sentences": tổng số câu trong đoạn
   - "relevant_sentences": số câu liên quan trực tiếp

Trả về JSON đúng format:
{{
  "useful": true/false,
  "total_sentences": <int>,
  "relevant_sentences": <int>,
  "reason": "1 câu giải thích ngắn"
}}"""


def context_precision_and_relevancy(
    query: TestQuery,
    contexts: List[str],
    judge,
    judge_question: Optional[str] = None,
) -> Tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Combined judge: produces both precision and relevancy in ONE call per
    context, instead of two separate calls. Reduces token usage by ~30-40%.

    Args:
      judge_question: optional override for the question text sent to the
        judge LLM. When the eval pipeline runs query rewriting, pass the
        ENGLISH q2 here — it matches the language of the retrieved chunks
        so the judge isn't forced to translate while scoring relevancy.
        If None, falls back to the original (Vietnamese) query.question.

    Returns: (precision, relevancy, precision_judgments, relevancy_per_context)
    """
    q_for_judge = judge_question or query.question

    precision_judgments: List[Dict[str, Any]] = []
    relevancy_per_context: List[Dict[str, Any]] = []

    for rank, ctx in enumerate(contexts, 1):
        result = judge.call_json(COMBINED_PRECISION_RELEVANCY_PROMPT.format(
            question=q_for_judge,
            ground_truth=query.ground_truth,
            context=ctx,
            rank=rank,
        )) or {}

        useful = bool(result.get("useful", False))
        total = max(1, int(result.get("total_sentences", 1) or 1))
        relevant = max(0, int(result.get("relevant_sentences", 0) or 0))
        ratio = min(1.0, relevant / total)
        reason = (result.get("reason") or "")[:120]

        precision_judgments.append({"rank": rank, "useful": useful, "reason": reason})
        relevancy_per_context.append({
            "rank": rank, "total": total, "relevant": relevant, "ratio": ratio,
        })

    # MAP-style precision
    relevant_count = 0
    sum_precision = 0.0
    for k, j in enumerate(precision_judgments, 1):
        if j["useful"]:
            relevant_count += 1
            sum_precision += relevant_count / k
    precision = (sum_precision / relevant_count) if relevant_count else 0.0
    relevancy = mean(r["ratio"] for r in relevancy_per_context) if relevancy_per_context else 0.0

    return precision, relevancy, precision_judgments, relevancy_per_context


# ─── Context Recall ──────────────────────────────────────────────
RECALL_PROMPT = """Bạn là người chấm điểm hệ thống RAG.

CÂU TRẢ LỜI ĐÚNG (ground truth):
{ground_truth}

Hãy tách câu trả lời trên thành các "claim" (luận điểm) đơn lẻ.
Mỗi claim là MỘT thông tin atomic, không thể chia nhỏ nữa.

Sau đó, với mỗi claim, kiểm tra xem nó có được hỗ trợ (supported)
bởi BẤT KỲ đoạn nào trong CÁC ĐOẠN NGỮ CẢNH dưới đây không.

CÁC ĐOẠN NGỮ CẢNH:
{contexts_block}

Trả về JSON:
{{
  "claims": [
    {{"text": "claim 1", "supported": true/false}},
    {{"text": "claim 2", "supported": true/false}},
    ...
  ]
}}"""


def context_recall(
    query: TestQuery,
    contexts: List[str],
    judge: GeminiJudge,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Break ground-truth answer into atomic claims, check how many are
    supported by the retrieved contexts. Recall = supported / total.
    """
    contexts_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    result = judge.call_json(RECALL_PROMPT.format(
        ground_truth=query.ground_truth,
        contexts_block=contexts_block,
    ))
    claims = (result or {}).get("claims", [])
    if not claims:
        return 0.0, []
    supported = sum(1 for c in claims if c.get("supported"))
    return supported / len(claims), claims


# ════════════════════════════════════════════════════════════════
# RANK METRICS (deterministic — no LLM judge needed)
# ════════════════════════════════════════════════════════════════
#
# These metrics check whether the SOURCE chunk (the one that generated
# the test question) appears among retrieved chunks, and at what rank.
# Zero LLM calls — pure retrieval + position lookup.
#
# Hit@k = 1 if source chunk is in top-k retrieved chunks, else 0
#         (averaged across queries)
# MRR   = mean of (1 / source_rank), with source_rank=∞ → contribution 0
#
# Why these matter even though we have RAGAS:
#   - Deterministic: same input → same score, every run
#   - Cheap: zero API calls, runs in seconds
#   - Diagnoses retrieval/embedding quality independent of judge bias
#   - Catches cases where RAGAS scores look "OK" but the actual source
#     chunk wasn't even retrieved — meaning RAGAS judged different
#     chunks that happened to contain related info

def _doc_chunk_key(doc: Any) -> str:
    """
    Extract the deterministic chunk_id from a retrieved document.

    Prefer 'chunk_id' (full deterministic id, unique across videos) over
    'doc_id' (which is just chunk_index — collides across videos).
    """
    cid = doc.metadata.get("chunk_id")
    if cid:
        return str(cid)
    # Fallback: combine video_id + doc_id to disambiguate
    vid = doc.metadata.get("video_id", "?")
    did = doc.metadata.get("doc_id", "?")
    return f"{vid}:{did}"


def compute_rank_metrics(
    source_chunk_id: str,
    retriever_raw,
    query_text: str,
    rewrite_record: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run retrieval and find at what rank the source chunk appears.

    If rewrite_record is provided, runs retrieval for both q1 and q2,
    merges results by best rank across variants. Otherwise uses query_text
    directly.

    Returns dict with hit_at_{1,3,5,10,20}, reciprocal_rank, source_rank,
    n_retrieved, and the full retrieved chunk-id list (for debugging).
    """
    # ── Step 1: Get a single ranked list of retrieved chunk ids ──
    if rewrite_record is not None:
        # Best-rank-across-variants merge: for each chunk, take its best
        # rank across q1 and q2, then sort ascending by best rank.
        chunk_to_best_rank: Dict[str, int] = {}
        for variant in (rewrite_record["q1"], rewrite_record["q2"]):
            r = retriever_raw(variant)
            docs = r.get("documents", []) if isinstance(r, dict) else r
            for rank, d in enumerate(docs, 1):
                key = _doc_chunk_key(d)
                if key not in chunk_to_best_rank or rank < chunk_to_best_rank[key]:
                    chunk_to_best_rank[key] = rank
        retrieved_ids = [k for k, _ in sorted(
            chunk_to_best_rank.items(), key=lambda x: x[1]
        )]
    else:
        r = retriever_raw(query_text)
        docs = r.get("documents", []) if isinstance(r, dict) else r
        retrieved_ids = [_doc_chunk_key(d) for d in docs]

    # ── Step 2: Find rank of source chunk (1-indexed) ──
    source_rank: Optional[int] = None
    for i, cid in enumerate(retrieved_ids, 1):
        if cid == source_chunk_id:
            source_rank = i
            break

    # ── Step 3: Compute Hit@k, Precision@k, MRR, MAP, nDCG@k ──
    # All formulas assume binary relevance with ONE ground-truth doc per query.
    # In that setting:
    #   Recall@k = Hit@k          (you either retrieved the doc or not)
    #   Precision@k = Hit@k / k   (one relevant out of k)
    #   AP (average precision) = 1/source_rank if found, else 0
    #   MAP = mean(AP across queries) — equivalent to MRR for single-relevant
    #   nDCG@k = 1 / log2(rank+1) if rank ≤ k, else 0  (IDCG@k = 1.0)
    def _ndcg_k(k: int) -> float:
        if source_rank is None or source_rank > k:
            return 0.0
        return 1.0 / math.log2(source_rank + 1)

    def _precision_k(k: int) -> float:
        if source_rank is None or source_rank > k:
            return 0.0
        return 1.0 / k

    return {
        # Hit@K — found in top-k
        "hit_at_1":  int(source_rank == 1),
        "hit_at_3":  int(source_rank is not None and source_rank <= 3),
        "hit_at_5":  int(source_rank is not None and source_rank <= 5),
        "hit_at_6":  int(source_rank is not None and source_rank <= 6),
        "hit_at_10": int(source_rank is not None and source_rank <= 10),
        "hit_at_20": int(source_rank is not None and source_rank <= 20),
        # Precision@K — precision at fixed depth (different from Context Precision)
        "precision_at_1":  _precision_k(1),
        "precision_at_3":  _precision_k(3),
        "precision_at_5":  _precision_k(5),
        "precision_at_10": _precision_k(10),
        # MRR / MAP / AP
        "reciprocal_rank":      (1.0 / source_rank) if source_rank else 0.0,
        "reciprocal_rank_at_3": (1.0 / source_rank) if (source_rank is not None and source_rank <= 3) else 0.0,
        "average_precision":    (1.0 / source_rank) if source_rank else 0.0,
        # nDCG@K
        "ndcg_at_3":  _ndcg_k(3),
        "ndcg_at_5":  _ndcg_k(5),
        "ndcg_at_10": _ndcg_k(10),
        "ndcg_at_20": _ndcg_k(20),
        # Bookkeeping
        "source_rank": source_rank,
        "n_retrieved": len(retrieved_ids),
        "retrieved_ids": retrieved_ids[:20],
    }


# ════════════════════════════════════════════════════════════════
# SEMANTIC METRICS (deterministic, embedder-based — no LLM judge)
# ════════════════════════════════════════════════════════════════
#
# These complement RAGAS Relevancy. They use a sentence-transformer to
# measure how close retrieved contexts are to (a) the query and (b) the
# known source chunk. Zero LLM calls — runs in seconds per query.
#
# Why useful: RAGAS Relevancy depends on a stable LLM judge. When the
# judge is weak (8B) or fails (parse errors, 429s), Relevancy collapses.
# Semantic similarity is deterministic and immune to those issues.

DEFAULT_SEMANTIC_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_PRECISION_THRESHOLD = 0.5    # cosine ≥ this → context counts as "relevant"
SEMANTIC_RECALL_THRESHOLD = 0.7       # cosine ≥ this between context and source → recalled


_semantic_embedder = None  # lazy-loaded singleton

def _load_semantic_embedder(model_name: str = DEFAULT_SEMANTIC_EMBEDDER):
    global _semantic_embedder
    if _semantic_embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for semantic metrics. "
                "Install: pip install sentence-transformers"
            ) from e
        logger.info("Loading semantic-metric embedder: %s", model_name)
        _semantic_embedder = SentenceTransformer(model_name)
    return _semantic_embedder


def _cosine(a, b) -> float:
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_semantic_metrics(
    query_text: str,
    contexts: List[str],
    source_chunk_text: Optional[str],
    embedder,
) -> Dict[str, float]:
    """
    Embedder-based deterministic metrics.

    Returns:
      max_query_context_sim:  max cosine(query, ctx) — best match
      mean_query_context_sim: mean cosine(query, ctx) — overall closeness
      semantic_precision_at_k: fraction of contexts with cosine(q, c) ≥ τ_p
      semantic_recall:         1.0 if any context has cosine(c, source) ≥ τ_r,
                              else 0.0  (needs source_chunk_text; else None)
    """
    if not contexts:
        return {
            "max_query_context_sim": 0.0,
            "mean_query_context_sim": 0.0,
            "semantic_precision_at_k": 0.0,
            "semantic_recall": 0.0,
        }

    # Encode in one batch for speed
    texts = [query_text] + list(contexts)
    if source_chunk_text:
        texts.append(source_chunk_text)
    vecs = embedder.encode(texts, batch_size=32, show_progress_bar=False)

    q_vec = vecs[0]
    ctx_vecs = vecs[1:1 + len(contexts)]
    src_vec = vecs[-1] if source_chunk_text else None

    q_sims = [_cosine(q_vec, cv) for cv in ctx_vecs]

    sem_precision = (
        sum(1 for s in q_sims if s >= SEMANTIC_PRECISION_THRESHOLD) / len(q_sims)
    )

    sem_recall = 0.0
    if src_vec is not None:
        src_sims = [_cosine(cv, src_vec) for cv in ctx_vecs]
        sem_recall = 1.0 if any(s >= SEMANTIC_RECALL_THRESHOLD for s in src_sims) else 0.0

    return {
        "max_query_context_sim":   max(q_sims),
        "mean_query_context_sim":  sum(q_sims) / len(q_sims),
        "semantic_precision_at_k": sem_precision,
        "semantic_recall":         sem_recall,
    }


# ════════════════════════════════════════════════════════════════
# CONTEXT ENTITIES RECALL (LLM-judged)
# ════════════════════════════════════════════════════════════════

ENTITIES_PROMPT = """Hãy trích xuất các THỰC THỂ KỸ THUẬT cụ thể từ câu trả lời \
sau (tên thuật toán, mô hình, phương pháp, dataset, công thức, số liệu, thuật ngữ \
chuyên ngành...). Chỉ trả lại các từ/cụm từ cụ thể, không trả lại giải thích.

CÂU TRẢ LỜI:
{answer}

Trả về JSON đúng format này:
{{
  "entities": ["thực thể 1", "thực thể 2", ...]
}}

Nếu không có thực thể kỹ thuật cụ thể nào, trả về {{"entities": []}}."""


def context_entities_recall(
    query: TestQuery,
    contexts: List[str],
    judge,
) -> Tuple[float, Dict[str, Any]]:
    """
    Extract technical entities from the ground-truth answer (1 LLM call),
    then string-match each entity against the joined retrieved contexts.

    Returns: (score, details)
      score = entities found / total entities  (0.0 if no entities extracted)
      details = {entities_total, entities_found, entities: [{text, found}, ...]}
    """
    result = judge.call_json(ENTITIES_PROMPT.format(answer=query.ground_truth)) or {}
    entities = result.get("entities") or []
    if not isinstance(entities, list) or not entities:
        return 0.0, {"entities_total": 0, "entities_found": 0, "entities": []}

    joined = " ".join(contexts).lower()
    details = []
    for ent in entities:
        if not isinstance(ent, str):
            continue
        ent_clean = ent.strip()
        if not ent_clean:
            continue
        found = ent_clean.lower() in joined
        details.append({"text": ent_clean, "found": found})

    if not details:
        return 0.0, {"entities_total": 0, "entities_found": 0, "entities": []}

    found_count = sum(1 for d in details if d["found"])
    return found_count / len(details), {
        "entities_total": len(details),
        "entities_found": found_count,
        "entities": details,
    }


# ════════════════════════════════════════════════════════════════
# PHASE B: EVALUATION LOOP
# ════════════════════════════════════════════════════════════════

def evaluate(
    queries: List[TestQuery],
    retriever,
    judge,
    retriever_raw=None,
    limit: Optional[int] = None,
    rewrite: bool = False,
    rewrite_cache_path: Optional[Path] = None,
    k_top: int = 6,
    rank_only: bool = False,
    semantic_metrics: bool = True,
    chunk_text_lookup: Optional[Dict[str, str]] = None,
    entities_recall: bool = True,
) -> Tuple[List[QueryEval], Dict[str, Any]]:
    """
    Run rank metrics + RAGAS metrics on each query, return per-query + aggregate.

    Args:
      retriever:        production retriever (with optional rerank) for RAGAS.
      retriever_raw:    bare top-20 retriever (no rerank) for rank metrics.
                        If None, rank metrics are skipped.
      rewrite:          run VI→EN rewriting before retrieval (production path).
      rewrite_cache_path: cache file so reruns skip already-rewritten queries.
      k_top:            when rewriting, max merged-unique chunks for RAGAS.
      rank_only:        if True, skip RAGAS judges (no LLM calls). Useful as
                        a smoke test or when judge quota is exhausted.
    """
    if limit:
        queries = queries[:limit]

    rewrite_cache: Optional[Dict[str, Dict[str, str]]] = None
    if rewrite:
        rewrite_cache = _rewrite_cache_load(rewrite_cache_path)
        cached_n = len(rewrite_cache)
        print(
            f"[eval] Query rewriting ENABLED — matches production VI→EN pipeline."
            + (f" Cache: {cached_n} entries loaded." if cached_n else "")
        )
    else:
        print(
            "[eval] Query rewriting DISABLED — VI query goes directly to retriever.\n"
            "       NOTE: production uses VI→EN rewriting; without --rewrite-queries\n"
            "       these scores under-estimate real retrieval quality."
        )

    # Lazy-load semantic embedder if metrics enabled
    sem_embedder = None
    if semantic_metrics:
        try:
            sem_embedder = _load_semantic_embedder()
            print("[eval] Semantic metrics ENABLED (embedder loaded).")
        except ImportError as e:
            logger.warning("Semantic metrics disabled: %s", e)
            sem_embedder = None

    if entities_recall and rank_only:
        entities_recall = False  # rank_only ⇒ no LLM at all
    if entities_recall:
        print("[eval] Context Entities Recall ENABLED (+1 LLM call per query).")

    results: List[QueryEval] = []
    # Per-query LLM calls: P+Rel (6) + Recall (1) + Rewrite (1 if on) + Entities (1 if on)
    calls_per_query = 0 if rank_only else (
        7 + (1 if rewrite else 0) + (1 if entities_recall else 0)
    )
    mode_desc = "rank-only (no LLM)" if rank_only else f"~{calls_per_query} LLM calls/query"
    print(f"\n[eval] Running on {len(queries)} queries ({mode_desc})\n")
    t0 = time.time()

    for i, q in enumerate(queries, 1):
        print(f"  [{i:>3}/{len(queries)}] {q.query_id}: {q.question[:70]}")

        # 1a. Optional: rewrite VI → English variants (matches production)
        rewrite_record: Optional[Dict[str, str]] = None
        if rewrite:
            rewrite_record = rewrite_query(judge, q.question, cache=rewrite_cache)
            print(f"       q1: {rewrite_record['q1'][:70]}")
            print(f"       q2: {rewrite_record['q2'][:70]}")
            _rewrite_cache_save(rewrite_cache_path, rewrite_cache or {})

        # ── retrieval latency = sum of all retrieval-call wall time
        retrieval_latency_ms = 0.0

        # 1b. RANK METRICS (deterministic, no LLM) — done first so we have
        #     them even if RAGAS judges fail / quota out partway through
        rank_record: Dict[str, Any] = {}
        if retriever_raw is not None:
            t_rank = time.time()
            rank_record = compute_rank_metrics(
                source_chunk_id=q.source_chunk_id,
                retriever_raw=retriever_raw,
                query_text=q.question,
                rewrite_record=rewrite_record,
            )
            retrieval_latency_ms += (time.time() - t_rank) * 1000
            sr = rank_record["source_rank"]
            sr_text = f"#{sr}" if sr is not None else "MISS"
            print(
                f"       rank: source@{sr_text}  "
                f"H@3={rank_record['hit_at_3']} H@6={rank_record['hit_at_6']} "
                f"H@10={rank_record['hit_at_10']}  "
                f"MRR@3={rank_record['reciprocal_rank_at_3']:.3f} "
                f"nDCG@3={rank_record['ndcg_at_3']:.3f}"
            )

        # 1c. Retrieve contexts for RAGAS (one or two retrieval passes)
        t_ret = time.time()
        if rewrite_record is not None:
            seen: Dict[Any, Any] = {}
            for variant_text in (rewrite_record["q1"], rewrite_record["q2"]):
                r = retriever(variant_text)
                docs_v = r.get("documents", []) if isinstance(r, dict) else r
                for d in docs_v:
                    key = _doc_chunk_key(d)
                    if key not in seen:
                        seen[key] = d
            docs = list(seen.values())[:k_top]
        else:
            r = retriever(q.question)
            docs = r.get("documents", []) if isinstance(r, dict) else r
        retrieval_latency_ms += (time.time() - t_ret) * 1000

        contexts = [d.page_content for d in docs]
        chunk_ids = [_doc_chunk_key(d) for d in docs]

        # ─── SEMANTIC METRICS (deterministic, embedder-based) ───
        sem_record: Dict[str, float] = {}
        if sem_embedder is not None and contexts:
            # Use English q2 if rewriting (matches retrieval embedding space)
            # else fall back to original VI question
            sem_query = (rewrite_record or {}).get("q2", q.question)
            src_text = (chunk_text_lookup or {}).get(q.source_chunk_id)
            try:
                sem_record = compute_semantic_metrics(
                    query_text=sem_query,
                    contexts=contexts,
                    source_chunk_text=src_text,
                    embedder=sem_embedder,
                )
            except Exception as e:
                logger.warning("Semantic metrics failed for %s: %s", q.query_id, e)
                sem_record = {}

        # Skip RAGAS in rank_only mode OR when no contexts were retrieved
        if rank_only or not contexts:
            if not contexts and not rank_only:
                print("       (no contexts retrieved — RAGAS skipped)")
            results.append(QueryEval(
                query_id=q.query_id,
                question=q.question,
                n_contexts=len(contexts),
                context_precision=0.0,
                context_recall=0.0,
                context_relevancy=0.0,
                hit_at_1=rank_record.get("hit_at_1", 0),
                hit_at_3=rank_record.get("hit_at_3", 0),
                hit_at_5=rank_record.get("hit_at_5", 0),
                hit_at_6=rank_record.get("hit_at_6", 0),
                hit_at_10=rank_record.get("hit_at_10", 0),
                hit_at_20=rank_record.get("hit_at_20", 0),
                precision_at_1=rank_record.get("precision_at_1", 0.0),
                precision_at_3=rank_record.get("precision_at_3", 0.0),
                precision_at_5=rank_record.get("precision_at_5", 0.0),
                precision_at_10=rank_record.get("precision_at_10", 0.0),
                reciprocal_rank=rank_record.get("reciprocal_rank", 0.0),
                reciprocal_rank_at_3=rank_record.get("reciprocal_rank_at_3", 0.0),
                average_precision=rank_record.get("average_precision", 0.0),
                ndcg_at_3=rank_record.get("ndcg_at_3", 0.0),
                ndcg_at_5=rank_record.get("ndcg_at_5", 0.0),
                ndcg_at_10=rank_record.get("ndcg_at_10", 0.0),
                ndcg_at_20=rank_record.get("ndcg_at_20", 0.0),
                source_rank=rank_record.get("source_rank"),
                n_retrieved_for_rank=rank_record.get("n_retrieved", 0),
                max_query_context_sim=sem_record.get("max_query_context_sim", 0.0),
                mean_query_context_sim=sem_record.get("mean_query_context_sim", 0.0),
                semantic_precision_at_k=sem_record.get("semantic_precision_at_k", 0.0),
                semantic_recall=sem_record.get("semantic_recall", 0.0),
                retrieval_latency_ms=retrieval_latency_ms,
                retrieved_chunk_ids=chunk_ids,
                judge_notes={"rewrite": rewrite_record} if rewrite_record else {},
            ))
            continue

        # 2. RAGAS metrics — combined precision+relevancy saves ~30% tokens
        # Use English q2 for the judge prompt when available — matches the
        # language of the retrieved chunks so the judge isn't penalising
        # relevancy just because of VI question vs EN context language gap.
        judge_question = (
            rewrite_record.get("q2") if rewrite_record else None
        )
        print(f"       retrieved {len(contexts)} contexts. Judging...", end="", flush=True)
        prec, rel, prec_j, rel_j = context_precision_and_relevancy(
            q, contexts, judge, judge_question=judge_question,
        )
        print(" P+Rel", end="", flush=True)
        rec, rec_j = context_recall(q, contexts, judge)
        print(" R", end="", flush=True)

        # Context Entities Recall (one extra LLM call per query)
        ent_score = 0.0
        ent_record: Dict[str, Any] = {"entities_total": 0, "entities_found": 0, "entities": []}
        if entities_recall:
            ent_score, ent_record = context_entities_recall(q, contexts, judge)
            print(" E", end="", flush=True)
        print("", flush=True)

        print(
            f"       → precision={prec:.2f}  recall={rec:.2f}  "
            f"relevancy={rel:.2f}  entities={ent_score:.2f}"
        )

        notes: Dict[str, Any] = {
            "precision_judgments": prec_j,
            "recall_claims": rec_j,
            "relevancy_per_context": rel_j,
            "entities": ent_record,
        }
        if sem_record:
            notes["semantic"] = sem_record
        if rewrite_record is not None:
            notes["rewrite"] = rewrite_record
        if rank_record:
            notes["rank_top20_ids"] = rank_record.get("retrieved_ids", [])

        results.append(QueryEval(
            query_id=q.query_id,
            question=q.question,
            n_contexts=len(contexts),
            context_precision=prec,
            context_recall=rec,
            context_relevancy=rel,
            hit_at_1=rank_record.get("hit_at_1", 0),
            hit_at_3=rank_record.get("hit_at_3", 0),
            hit_at_5=rank_record.get("hit_at_5", 0),
            hit_at_6=rank_record.get("hit_at_6", 0),
            hit_at_10=rank_record.get("hit_at_10", 0),
            hit_at_20=rank_record.get("hit_at_20", 0),
            precision_at_1=rank_record.get("precision_at_1", 0.0),
            precision_at_3=rank_record.get("precision_at_3", 0.0),
            precision_at_5=rank_record.get("precision_at_5", 0.0),
            precision_at_10=rank_record.get("precision_at_10", 0.0),
            reciprocal_rank=rank_record.get("reciprocal_rank", 0.0),
            reciprocal_rank_at_3=rank_record.get("reciprocal_rank_at_3", 0.0),
            average_precision=rank_record.get("average_precision", 0.0),
            ndcg_at_3=rank_record.get("ndcg_at_3", 0.0),
            ndcg_at_5=rank_record.get("ndcg_at_5", 0.0),
            ndcg_at_10=rank_record.get("ndcg_at_10", 0.0),
            ndcg_at_20=rank_record.get("ndcg_at_20", 0.0),
            source_rank=rank_record.get("source_rank"),
            n_retrieved_for_rank=rank_record.get("n_retrieved", 0),
            max_query_context_sim=sem_record.get("max_query_context_sim", 0.0),
            mean_query_context_sim=sem_record.get("mean_query_context_sim", 0.0),
            semantic_precision_at_k=sem_record.get("semantic_precision_at_k", 0.0),
            semantic_recall=sem_record.get("semantic_recall", 0.0),
            context_entities_recall=ent_score,
            n_entities_extracted=ent_record.get("entities_total", 0),
            n_entities_found=ent_record.get("entities_found", 0),
            retrieval_latency_ms=retrieval_latency_ms,
            retrieved_chunk_ids=chunk_ids,
            judge_notes=notes,
        ))

    elapsed = time.time() - t0
    has_rank = retriever_raw is not None
    has_ragas = not rank_only and any(r.n_contexts > 0 for r in results)

    # ── Latency stats (per-query retrieval wall time) ──
    lats = [r.retrieval_latency_ms for r in results if r.retrieval_latency_ms > 0]
    sorted_lats = sorted(lats)
    def _pct(p):
        if not sorted_lats:
            return 0.0
        idx = max(0, min(len(sorted_lats) - 1, int(round(p * (len(sorted_lats) - 1)))))
        return sorted_lats[idx]

    aggregate = {
        "n_queries": len(results),
        "n_with_contexts": sum(1 for r in results if r.n_contexts > 0),
        "elapsed_seconds": round(elapsed, 1),
        "judge_calls": judge.n_calls,
        "judge_failures": getattr(judge, "n_failures", 0),
        # Rank metrics (deterministic, no LLM)
        "rank_metrics": {
            # ── Hit@K / Recall@K ────────────────────────────────────────
            "hit_at_1":    mean(r.hit_at_1  for r in results) if results and has_rank else None,
            "hit_at_3":    mean(r.hit_at_3  for r in results) if results and has_rank else None,
            "hit_at_5":    mean(r.hit_at_5  for r in results) if results and has_rank else None,
            "hit_at_6":    mean(r.hit_at_6  for r in results) if results and has_rank else None,
            "hit_at_10":   mean(r.hit_at_10 for r in results) if results and has_rank else None,
            "hit_at_20":   mean(r.hit_at_20 for r in results) if results and has_rank else None,
            "recall_at_6":  mean(r.hit_at_6  for r in results) if results and has_rank else None,
            "recall_at_20": mean(r.hit_at_20 for r in results) if results and has_rank else None,
            # ── Precision@K (rule-based, NOT context precision) ─────────
            "precision_at_1":  mean(r.precision_at_1  for r in results) if results and has_rank else None,
            "precision_at_3":  mean(r.precision_at_3  for r in results) if results and has_rank else None,
            "precision_at_5":  mean(r.precision_at_5  for r in results) if results and has_rank else None,
            "precision_at_10": mean(r.precision_at_10 for r in results) if results and has_rank else None,
            # ── MRR / MAP ───────────────────────────────────────────────
            "mrr":      mean(r.reciprocal_rank      for r in results) if results and has_rank else None,
            "mrr_at_3": mean(r.reciprocal_rank_at_3 for r in results) if results and has_rank else None,
            "map":      mean(r.average_precision    for r in results) if results and has_rank else None,
            # ── nDCG@K ──────────────────────────────────────────────────
            "ndcg_at_3":  mean(r.ndcg_at_3  for r in results) if results and has_rank else None,
            "ndcg_at_5":  mean(r.ndcg_at_5  for r in results) if results and has_rank else None,
            "ndcg_at_10": mean(r.ndcg_at_10 for r in results) if results and has_rank else None,
            "ndcg_at_20": mean(r.ndcg_at_20 for r in results) if results and has_rank else None,
            # ── Bookkeeping ─────────────────────────────────────────────
            "n_misses":   sum(1 for r in results if has_rank and r.source_rank is None),
        } if has_rank else None,
        # RAGAS metrics (LLM judge)
        "ragas_metrics": {
            "context_precision_mean": mean(r.context_precision for r in results) if results else 0.0,
            "context_recall_mean":    mean(r.context_recall    for r in results) if results else 0.0,
            "context_relevancy_mean": mean(r.context_relevancy for r in results) if results else 0.0,
            "context_entities_recall_mean": (
                mean(r.context_entities_recall for r in results) if results else 0.0
            ),
        } if has_ragas else None,
        # Semantic metrics (deterministic, embedder-based)
        "semantic_metrics": {
            "max_query_context_sim_mean":   mean(r.max_query_context_sim   for r in results) if results else 0.0,
            "mean_query_context_sim_mean":  mean(r.mean_query_context_sim  for r in results) if results else 0.0,
            "semantic_precision_at_k_mean": mean(r.semantic_precision_at_k for r in results) if results else 0.0,
            "semantic_recall_mean":         mean(r.semantic_recall         for r in results) if results else 0.0,
        } if any(r.max_query_context_sim > 0 or r.semantic_recall > 0 for r in results) else None,
        # Ops
        "latency_ms": {
            "mean":   round(mean(lats), 1) if lats else 0.0,
            "median": round(_pct(0.50), 1),
            "p90":    round(_pct(0.90), 1),
            "p95":    round(_pct(0.95), 1),
            "max":    round(max(lats), 1) if lats else 0.0,
        } if lats else None,
    }
    return results, aggregate


def render_summary(aggregate: Dict[str, Any], results: List[QueryEval]) -> str:
    lines = [
        "",
        "=" * 70,
        "RAGAS + RANK EVALUATION SUMMARY",
        "=" * 70,
        f"  Queries evaluated     : {aggregate['n_queries']}",
        f"  With contexts         : {aggregate['n_with_contexts']}",
        f"  Judge calls used      : {aggregate['judge_calls']}",
        f"  Judge failures        : {aggregate.get('judge_failures', 0)} "
        f"(silent None returns — affects RAGAS scores)",
        f"  Elapsed               : {aggregate['elapsed_seconds']}s",
        "",
    ]

    # ── RANK METRICS section ───────────────────────────────────
    rm = aggregate.get("rank_metrics")
    if rm is not None:
        lines += [
            "  RETRIEVAL STAGE (rule-based, deterministic, no LLM):",
            f"    Hit@K / Recall@K @ {{1,3,5,6,10,20}}",
            f"      Hit@1  : {rm['hit_at_1']:.3f}    Hit@3  : {rm['hit_at_3']:.3f}    Hit@5  : {rm['hit_at_5']:.3f}",
            f"      Hit@6  : {rm['hit_at_6']:.3f}    Hit@10 : {rm['hit_at_10']:.3f}    Hit@20 : {rm['hit_at_20']:.3f}",
            f"    Precision@K",
            f"      P@1    : {rm['precision_at_1']:.3f}    P@3    : {rm['precision_at_3']:.3f}    P@5    : {rm['precision_at_5']:.3f}    P@10   : {rm['precision_at_10']:.3f}",
            f"    Rank-aware",
            f"      MRR (top-20) : {rm['mrr']:.3f}    MAP : {rm['map']:.3f}",
            f"      nDCG@3 : {rm['ndcg_at_3']:.3f}    nDCG@5 : {rm['ndcg_at_5']:.3f}    nDCG@10 : {rm['ndcg_at_10']:.3f}    nDCG@20 : {rm['ndcg_at_20']:.3f}",
            "",
            "  RERANK STAGE (top-3 view):",
            f"    Hit@3 : {rm['hit_at_3']:.3f}    P@3 : {rm['precision_at_3']:.3f}    MRR@3 : {rm['mrr_at_3']:.3f}    nDCG@3 : {rm['ndcg_at_3']:.3f}",
            "",
            f"  Source-chunk misses (not in top-20): {rm['n_misses']} / {aggregate['n_queries']}",
            "",
            "  RANK INTERPRETATION:",
        ]
        h6 = rm["hit_at_6"]
        h20 = rm["hit_at_20"]
        ndcg = rm["ndcg_at_3"]
        if h6 < 0.40:
            lines.append("    - Hit@6 LOW: retriever often misses source chunk.")
            lines.append("      → Embedding/retrieval problem (try different weights, embedder).")
        elif h6 < 0.70:
            lines.append("    - Hit@6 OK. Source chunk usually retrieved in top-6.")
        else:
            lines.append("    - Hit@6 HEALTHY. Retriever consistently finds source chunk in top-6.")
        # Hit@20 vs Hit@6 gap reveals reranking-stage impact
        gap = h20 - h6
        if gap > 0.20:
            lines.append(f"    - Hit@20 - Hit@6 = {gap:.2f} : large recall gap.")
            lines.append("      → Source chunk is in retrieval pool but not in top-6 — reranker matters.")
        if ndcg < 0.40:
            lines.append("    - nDCG@3 LOW: even when found, source chunk ranked badly in top-3.")
            lines.append("      → Reranker / fusion weights problem.")
        elif ndcg < 0.70:
            lines.append("    - nDCG@3 OK. Source chunk reasonably ranked in top-3.")
        else:
            lines.append("    - nDCG@3 HEALTHY. Source chunk consistently at top of rerank.")
        lines.append("")

    # ── SEMANTIC METRICS section (deterministic, embedder-based) ──
    sm = aggregate.get("semantic_metrics")
    if sm is not None:
        lines += [
            "  SEMANTIC METRICS (deterministic, embedder-based — no LLM):",
            f"    Max  query↔context sim  : {sm['max_query_context_sim_mean']:.3f}",
            f"    Mean query↔context sim  : {sm['mean_query_context_sim_mean']:.3f}",
            f"    Semantic Precision@K    : {sm['semantic_precision_at_k_mean']:.3f}  (cosine ≥ {SEMANTIC_PRECISION_THRESHOLD:.2f})",
            f"    Semantic Recall         : {sm['semantic_recall_mean']:.3f}  (any ctx≈source ≥ {SEMANTIC_RECALL_THRESHOLD:.2f})",
            "",
        ]

    # ── RAGAS METRICS section ──────────────────────────────────
    ragas = aggregate.get("ragas_metrics")
    if ragas is not None:
        p = ragas["context_precision_mean"]
        r = ragas["context_recall_mean"]
        v = ragas["context_relevancy_mean"]
        e = ragas.get("context_entities_recall_mean", 0.0)
        lines += [
            "  RAGAS METRICS (LLM judge):",
            f"    Context Precision        : {p:.3f}",
            f"    Context Recall           : {r:.3f}",
            f"    Context Relevancy        : {v:.3f}",
            f"    Context Entities Recall  : {e:.3f}",
            "",
            "  RAGAS INTERPRETATION:",
        ]
        if p < 0.5:
            lines.append("    - Precision LOW: top retrieved chunks often NOT useful.")
            lines.append("      → Investigate retrieval ranking (BM25/dense weights, reranker).")
        elif p < 0.75:
            lines.append("    - Precision OK but improvable. Tune weights or rerank.")
        else:
            lines.append("    - Precision HEALTHY. Top chunks are usually useful.")
        if r < 0.5:
            lines.append("    - Recall LOW: retrieval misses key info from the answer.")
            lines.append("      → Try smaller chunks, increase k_retrieve, check boundaries.")
        elif r < 0.75:
            lines.append("    - Recall OK. Some answers partially supported by contexts.")
        else:
            lines.append("    - Recall HEALTHY. Most answer claims are findable in retrieval.")
        if v < 0.4:
            lines.append("    - Relevancy LOW: contexts have a lot of filler around the answer.")
            lines.append("      → Chunks too big? Try smaller MAX_TOKENS in chunker.")
        elif v < 0.65:
            lines.append("    - Relevancy OK. Reasonable signal-to-noise.")
        else:
            lines.append("    - Relevancy HIGH. Contexts tightly focused on the question.")
        lines.append("")

    # ── OPS / LATENCY section ──────────────────────────────────
    lat = aggregate.get("latency_ms")
    if lat is not None:
        lines += [
            "  OPS — RETRIEVAL LATENCY (ms per query, retrieval calls only):",
            f"    mean   : {lat['mean']:>7.1f} ms",
            f"    median : {lat['median']:>7.1f} ms",
            f"    p90    : {lat['p90']:>7.1f} ms",
            f"    p95    : {lat['p95']:>7.1f} ms",
            f"    max    : {lat['max']:>7.1f} ms",
            "",
        ]

    lines.append("  Use this scoreboard to compare chunking strategies.")
    lines.append("  Higher metrics = chunks more retrieval-friendly. Lower latency = better UX.")
    lines.append("=" * 70)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="RAGAS-style retrieval evaluation (context metrics, no LLM answer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                   help=f"Chunks JSONL (default: {DEFAULT_DATA_PATH})")
    p.add_argument("--faiss", type=Path, default=DEFAULT_FAISS_PATH,
                   help=f"FAISS index dir (default: {DEFAULT_FAISS_PATH})")
    p.add_argument("--bm25", type=Path, default=DEFAULT_BM25_PATH,
                   help=f"BM25 pickle (default: {DEFAULT_BM25_PATH})")
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH,
                   help=f"Test queries JSONL (default: {DEFAULT_QUERIES_PATH})")
    p.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH,
                   help=f"Eval report JSON (default: {DEFAULT_REPORT_PATH})")
    p.add_argument("--generate-queries", type=int, default=None, metavar="N",
                   help="Phase A: generate N test queries (then exit)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate first N queries (for quick test)")
    p.add_argument("--no-reranker", action="store_true",
                   help="Skip reranker for faster evaluation")
    p.add_argument("--k-rerank", type=int, default=3,
                   help="How many contexts to keep after reranker (default: 3)")
    p.add_argument("--rpm-delay", type=float, default=None,
                   help="Seconds between LLM calls (default: provider-specific)")
    # ── Env file (keep API keys out of shell history) ──────────
    p.add_argument(
        "--env-file", type=Path, default=None,
        help="Load API keys from a .env file. If omitted, looks for .env "
             "in the project root automatically.",
    )
    # ── LLM provider ────────────────────────────────────────────
    p.add_argument(
        "--judge-provider",
        choices=["groq", "openrouter", "mistral", "ollama", "gemini"],
        default="groq",
        help="Single LLM backend (used if --judge-fallback not set). "
             "RECOMMENDED: groq or mistral. AVOID gemini (~20 RPD).",
    )
    p.add_argument(
        "--judge-fallback", default=None,
        help="Comma-separated provider chain; auto-switches when one hits "
             "rate-limit. RECOMMENDED: 'groq,mistral' (combined plenty of "
             "free quota). Overrides --judge-provider.",
    )
    p.add_argument(
        "--judge-model", default=None,
        help="Override the model name. Defaults per provider:\n"
             f"  groq       = {GROQ_MODEL} (try llama-3.1-8b-instant for 14k RPD)\n"
             "  mistral    = mistral-small-2506 (2.25M TPM, 5 RPS)\n"
             "  openrouter = qwen/qwen-2.5-72b-instruct:free\n"
             "  ollama     = qwen2.5:7b (local)\n"
             f"  gemini     = {GEMINI_MODEL} (DEPRECATED — quota too small)"
    )
    # ── Query rewriting (matches production VI→EN pipeline) ────
    p.add_argument("--rewrite-queries", action="store_true",
                   help="Rewrite VI query to English variants before retrieval. "
                        "Matches production pipeline. STRONGLY RECOMMENDED.")
    p.add_argument("--rewrite-cache", type=Path,
                   default=_PROJECT_ROOT / "data" / "chunked" / "query_rewrite_cache.json",
                   help="JSON cache for rewritten queries (avoids repeat LLM calls)")
    p.add_argument("--k-top", type=int, default=6,
                   help="When rewriting, max merged-unique chunks to keep before "
                        "running RAGAS metrics (default: 6)")
    # ── Rank metrics ────────────────────────────────────────────
    p.add_argument("--rank-only", action="store_true",
                   help="Skip RAGAS LLM-judge calls. Compute only the "
                        "deterministic rank metrics (Hit@k, MRR). "
                        "Zero LLM cost, runs in seconds. Useful for fast "
                        "iteration or when judge quota is exhausted.")
    p.add_argument("--no-rank-metrics", action="store_true",
                   help="Skip rank metrics (Hit@k, MRR). RAGAS only.")
    p.add_argument("--no-semantic-metrics", action="store_true",
                   help="Skip semantic embedding metrics (max/mean cosine, "
                        "semantic precision/recall). Saves ~5-10s per query.")
    p.add_argument("--no-entities-recall", action="store_true",
                   help="Skip Context Entities Recall (saves ~1 LLM call per query).")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.data.exists():
        logger.error("Chunks file not found: %s", args.data)
        sys.exit(1)

    # ── Load .env if present (so API keys don't have to be in shell) ──
    env_file = args.env_file
    if env_file is None:
        # Auto-discover: project root, then current dir
        for candidate in (_PROJECT_ROOT / ".env", Path.cwd() / ".env"):
            if candidate.exists():
                env_file = candidate
                break
    if env_file is not None:
        if _DOTENV_AVAILABLE:
            _load_dotenv(env_file, override=False)
            print(f"[env] Loaded {env_file}")
        else:
            logger.warning(
                "Found %s but python-dotenv not installed. "
                "Run: pip install python-dotenv",
                env_file,
            )

    # ── Build judge: chain or single ──
    if args.judge_fallback:
        judge = FallbackJudge.from_chain(
            args.judge_fallback,
            model=args.judge_model,
            rpm_delay=args.rpm_delay,
        )
    else:
        judge = get_judge(
            provider=args.judge_provider,
            model=args.judge_model,
            rpm_delay=args.rpm_delay,
        )
        print(f"[eval] Judge: {args.judge_provider} ({args.judge_model or 'default model'})")

    # ─── PHASE A ───────────────────────────────────────────────
    if args.generate_queries:
        chunks = load_chunks(args.data)
        print(f"[generate] Loaded {len(chunks)} chunks from {args.data}")
        generate_queries(
            chunks=chunks,
            n=args.generate_queries,
            judge=judge,
            out_path=args.queries,
        )
        return

    # ─── PHASE B ───────────────────────────────────────────────
    if not args.queries.exists():
        logger.error(
            "Queries file not found: %s\n"
            "Run Phase A first: python embed__data/evaluate_ragas.py --generate-queries 50",
            args.queries,
        )
        sys.exit(1)

    queries = load_queries(args.queries)
    if not queries:
        logger.error("No queries loaded from %s", args.queries)
        sys.exit(1)
    print(f"[eval] Loaded {len(queries)} test queries from {args.queries}")

    if not args.faiss.exists() or not args.bm25.exists():
        logger.error(
            "Index files missing.\n  faiss: %s\n  bm25:  %s\n"
            "Build them first via embed__data/main.py or vectorstore_builder.py",
            args.faiss, args.bm25,
        )
        sys.exit(1)

    # Build retriever using the team's existing hybrid_improved code
    print("[eval] Building hybrid retriever (this may take a minute on cold start)...")
    from hybrid_improved import build_hybrid_retriever
    retriever = build_hybrid_retriever(
        data_path=str(args.data),
        faiss_index_path=str(args.faiss),
        bm25_path=str(args.bm25),
        k_retrieve=20,
        k_final=6,
        k_rerank=args.k_rerank,
        use_reranker=not args.no_reranker,
        weight_bm25=0.2,
        weight_dense=0.8,
        enable_tracing=False,
    )

    # Build a SECOND, "raw" retriever for rank metrics: top-20 fused, no rerank.
    # This is what we measure Hit@k / MRR against. It's deliberately separate
    # from the production retriever so rank metrics aren't capped by k_rerank.
    retriever_raw = None
    if not args.no_rank_metrics:
        print("[eval] Building raw top-20 retriever for rank metrics...")
        retriever_raw = build_hybrid_retriever(
            data_path=str(args.data),
            faiss_index_path=str(args.faiss),
            bm25_path=str(args.bm25),
            k_retrieve=30,
            k_final=20,
            k_rerank=20,           # no rerank applied (use_reranker=False)
            use_reranker=False,
            weight_bm25=0.2,
            weight_dense=0.8,
            enable_tracing=False,
        )

    # Build chunk_id → chunk_text lookup for Semantic Recall
    # (compares retrieved contexts to the original source chunk's text)
    chunk_text_lookup: Dict[str, str] = {}
    if not args.no_semantic_metrics:
        for c in load_chunks(args.data):
            cid = c.get("chunk_id")
            if cid:
                chunk_text_lookup[cid] = c.get("chunk_text", "")
        logger.info("Built chunk_text lookup with %d entries", len(chunk_text_lookup))

    results, aggregate = evaluate(
        queries, retriever, judge,
        retriever_raw=retriever_raw,
        limit=args.limit,
        rewrite=args.rewrite_queries,
        rewrite_cache_path=args.rewrite_cache if args.rewrite_queries else None,
        k_top=args.k_top,
        rank_only=args.rank_only,
        semantic_metrics=not args.no_semantic_metrics,
        chunk_text_lookup=chunk_text_lookup,
        entities_recall=not args.no_entities_recall,
    )

    # Render report
    summary = render_summary(aggregate, results)
    print(summary)

    # Persist full report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({
            "aggregate": aggregate,
            "config": {
                "data": str(args.data),
                "faiss": str(args.faiss),
                "bm25": str(args.bm25),
                "queries": str(args.queries),
                "use_reranker": not args.no_reranker,
                "k_rerank": args.k_rerank,
                "judge_provider": args.judge_provider,
                "judge_model": resolve_model_name(
                    args.judge_provider, args.judge_model,
                ),
                "rewrite_queries": args.rewrite_queries,
                "k_top": args.k_top,
                "rank_only": args.rank_only,
                "rank_metrics_enabled": not args.no_rank_metrics,
            },
            "per_query": [asdict(r) for r in results],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Full per-query report saved to {args.output}")


if __name__ == "__main__":
    main()
