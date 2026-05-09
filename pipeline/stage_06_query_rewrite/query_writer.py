import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_MODEL = "qwen2.5:3b"


def _get_llm(model: str) -> Any:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=0)


def _load_cache(cache_file: str) -> dict:
    path = Path(cache_file)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    if isinstance(data, dict):
        return data
    return {}


def _load_from_cache(vietnamese_query: str, cache_file: str) -> tuple[Optional[str], Optional[str]]:
    cached = _load_cache(cache_file).get(vietnamese_query)
    if not isinstance(cached, dict):
        return None, None

    q1 = cached.get("q1")
    q2 = cached.get("q2")
    if isinstance(q1, str) and isinstance(q2, str):
        return q1, q2
    return None, None


def _save_to_cache(vietnamese_query: str, q1: str, q2: str, cache_file: str) -> None:
    path = Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    cache = _load_cache(cache_file)
    cache[vietnamese_query] = {"q1": q1, "q2": q2}

    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)


def rewrite_vietnamese_query(
    vietnamese_query: str,
    model: str = DEFAULT_MODEL,
    cache_file: Optional[str] = None
) -> dict:

    if cache_file:
        cached_q1, cached_q2 = _load_from_cache(vietnamese_query, cache_file)
        if cached_q1 and cached_q2:
            return {
                "q1": cached_q1,
                "q2": cached_q2,
                "original": vietnamese_query,
                "cached": True
            }

    prompt = f"""
You are a search query optimizer for Stanford ML course transcripts.

Task:
Convert Vietnamese question into 2 English search queries.

Rules:
- Keep technical terms unchanged
- q1 = direct translation
- q2 = expanded semantic retrieval query
- Return ONLY valid JSON

Vietnamese:
{vietnamese_query}
"""

    llm = _get_llm(model)
    response = llm.invoke(prompt)
    text = response.content.strip()

    try:
        result = _extract_json(text)
        q1 = result["q1"]
        q2 = result["q2"]

        if not isinstance(q1, str) or not isinstance(q2, str):
            raise ValueError("q1 and q2 must be strings")

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        q1 = vietnamese_query
        q2 = vietnamese_query + " machine learning"

    final_result = {
        "q1": q1,
        "q2": q2,
        "original": vietnamese_query,
        "cached": False
    }

    if cache_file:
        _save_to_cache(vietnamese_query, q1, q2, cache_file)

    return final_result
