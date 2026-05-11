"""
pipeline/ask.py
═══════════════════════════════════════════════════════════════════════
Interactive RAG REPL — ask a Vietnamese question, watch all 9 stages.

Designed for pedagogical use: every stage is labelled, every output is
displayed. You can SEE the rewrite, retrieval, merge, and LLM answer.

USAGE
─────
    # Default: Groq for the LLM answer (fast, uses your existing key)
    python pipeline/ask.py

    # No Ollama? Skip the rewrite step (default if Ollama not running)
    python pipeline/ask.py --no-query-writer

    # Use Mistral instead of Groq for the LLM answer
    python pipeline/ask.py --llm-provider mistral

    # Use the reranker (slower but slightly better top-3 ranking)
    python pipeline/ask.py --rerank

    # Skip the LLM answer step (Stage 9 disabled — just see retrieval)
    python pipeline/ask.py --no-llm

WHAT EACH STAGE DOES
────────────────────
    Stage 6: VI question → q1 (literal EN) + q2 (expanded EN)   (optional)
    Stage 7: hybrid retrieval (BM25 + FAISS [+ reranker])
    Stage 8: group retrieved chunks by video, clean text
    Stage 9: LLM answer in Vietnamese with citations
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Make Windows console handle unicode (emojis, Vietnamese chars) ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# ── Make every stage subfolder importable ──
_PIPELINE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PIPELINE_DIR.parent
for _sub in (
    "stage_05_index", "stage_06_query_rewrite", "stage_07_retrieve",
    "stage_08_merge_context", "stage_09_llm_answer",
):
    _p = str(_PIPELINE_DIR / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load .env so GROQ_API_KEY / MISTRAL_API_KEY are available ──
try:
    from dotenv import load_dotenv  # type: ignore
    for _candidate in (_REPO_ROOT / ".env", Path.cwd() / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
            break
except ImportError:
    pass


# ════════════════════════════════════════════════════════════════
# DEFAULTS (project-relative)
# ════════════════════════════════════════════════════════════════

DEFAULT_DATA_PATH = str(_REPO_ROOT / "data" / "chunked" / "transcript_v3_t072.jsonl")
DEFAULT_FAISS_PATH = str(_REPO_ROOT / "indexes" / "faiss_index_072_n")
DEFAULT_BM25_PATH = str(_REPO_ROOT / "indexes" / "bm25_072.pkl")

GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
MISTRAL_DEFAULT_MODEL = "mistral-small-2506"


# ════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════

BAR = "═" * 70
THIN = "─" * 70


def banner(text: str) -> None:
    print()
    print(BAR)
    print(text)
    print(BAR)


def section(stage: str, label: str) -> None:
    print(f"\n[STAGE {stage}] {label}")


def kv(key: str, value: Any) -> None:
    print(f"   {key:<14} {value}")


# ════════════════════════════════════════════════════════════════
# STAGE 9: LLM ANSWER (via Groq or Mistral, OpenAI-compatible)
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_VI = """You are an expert AI teaching assistant for Stanford's NLP and Machine Learning courses (CS224N, CS224U, CS224V, CS124).

CORE PRINCIPLES:
1. ONLY answer based on the provided SOURCES. Never make up information.
2. If sources don't contain enough information, clearly state that.
3. Adapt your response language to match the user's question language (Vietnamese → Vietnamese, English → English).
4. Keep technical terms in English (transformer, attention, gradient descent, embedding, etc.).

RESPONSE STYLE:
- Explain concepts clearly at a student-friendly level (undergraduate/graduate).
- Use a conversational, friendly tone as if tutoring a student.
- Be concise but thorough - aim for 3-6 sentences unless deeper explanation is needed.
- When using information from sources, cite them as [1], [2], [3]... corresponding to the source numbers.
- If discussing code, explain what each part does.

STRUCTURE (when helpful):
- Start with direct answer
- Briefly explain the key concept
- Give 1-2 example if relevant
- Point to specific source for more detail

Never say "as an AI" or "I don't have" - just answer directly from the sources."""


def build_user_prompt(question: str, contexts: List[Dict[str, Any]], detected_lang: str = "unknown") -> str:
    """Build the user message with numbered sources."""
    blocks = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title") or "Unknown"
        url = ctx.get("url") or ""
        text = ctx.get("context") or ""
        blocks.append(f"[{i}] {title}\n     URL: {url}\n     Content: {text}")
    sources = "\n\n".join(blocks)
    
    # Language instruction based on detected language
    lang_instruction = ""
    if detected_lang == "vi":
        lang_instruction = "Trả lời bằng tiếng Việt. "
    elif detected_lang == "en":
        lang_instruction = "Answer in English. "
    
    return (
        f"SOURCES:\n{sources}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"{lang_instruction}"
        f"Provide your answer with source citations [n] where you use information from sources."
    )


def call_groq(question: str, contexts: List[Dict[str, Any]], model: str, detected_lang: str = "unknown") -> str:
    try:
        from groq import Groq  # type: ignore
    except ImportError as e:
        raise ImportError("Install: pip install groq") from e
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not in environment / .env")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_VI},
            {"role": "user", "content": build_user_prompt(question, contexts, detected_lang)},
        ],
        temperature=0.5,
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()


def call_mistral(question: str, contexts: List[Dict[str, Any]], model: str, detected_lang: str = "unknown") -> str:
    import requests  # type: ignore
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not in environment / .env")
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_VI},
                {"role": "user", "content": build_user_prompt(question, contexts, detected_lang)},
            ],
            "temperature": 0.5,
            "max_tokens": 1024,
        },
        timeout=60,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def call_llm(provider: str, model: str, question: str,
             contexts: List[Dict[str, Any]], detected_lang: str = "unknown") -> str:
    p = provider.lower().strip()
    if p == "groq":
        return call_groq(question, contexts, model, detected_lang)
    if p == "mistral":
        return call_mistral(question, contexts, model, detected_lang)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# ════════════════════════════════════════════════════════════════
# SETUP — build retriever + (optional) query rewriter once
# ════════════════════════════════════════════════════════════════

def setup_retriever(args) -> Any:
    print("\n[setup] Loading hybrid retriever (this takes ~30s on first run)...")
    t0 = time.time()
    from hybrid_retriever import build_hybrid_retriever
    retriever = build_hybrid_retriever(
        data_path=args.data_path,
        faiss_index_path=args.faiss,
        bm25_path=args.bm25,
        k_retrieve=20,
        k_final=args.k,
        k_rerank=args.k,
        use_reranker=args.rerank,
        weight_bm25=0.3,
        weight_dense=0.7,
        enable_tracing=False,
    )
    elapsed = time.time() - t0
    print(f"[setup] Retriever ready in {elapsed:.1f}s.")
    return retriever


def try_setup_query_writer() -> Optional[Any]:
    """Returns the rewrite function if Ollama is running; None otherwise."""
    try:
        from query_writer import rewrite_vietnamese_query
    except ImportError:
        return None
    # Quick connectivity test — try a tiny Ollama call to see if it's up.
    try:
        rewrite_vietnamese_query("test", model="qwen2.5:3b")
        return rewrite_vietnamese_query
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# MAIN REPL LOOP
# ════════════════════════════════════════════════════════════════

def process_one_question(
    question: str,
    retriever,
    rewriter: Optional[Any],
    args: argparse.Namespace,
) -> None:
    banner(f"NHẬP CÂU HỎI: {question}")

    # ── Stage 6: query rewriting (optional) ──
    queries = {"original": question}
    section("6/9", "Rewriting query...")
    if rewriter is not None and not args.no_query_writer:
        try:
            t0 = time.time()
            rewritten = rewriter(question)
            queries["q1_literal"] = rewritten.get("q1", question)
            queries["q2_expanded"] = rewritten.get("q2", question)
            kv("q1:", queries["q1_literal"])
            kv("q2:", queries["q2_expanded"])
            print(f"   ({time.time() - t0:.1f}s, via Ollama qwen2.5:3b)")
        except Exception as e:
            print(f"   (rewrite failed: {e}; using original VI only)")
    else:
        print("   (skipped — Ollama not running or --no-query-writer set)")
        kv("original:", question)

    # ── Stage 7: retrieval ──
    section("7/9", f"Retrieving top-{args.k} chunks...")
    t0 = time.time()
    seen: Dict[str, Any] = {}
    for variant_label, qtext in queries.items():
        result = retriever(qtext)
        docs = result.get("documents", []) if isinstance(result, dict) else result
        for d in docs:
            key = d.metadata.get("chunk_id") or d.metadata.get("doc_id") or id(d)
            if key not in seen:
                d.metadata.setdefault("retrieved_by", [])
                seen[key] = d
            seen[key].metadata["retrieved_by"].append(variant_label)
    docs = list(seen.values())[: args.k]
    elapsed = time.time() - t0
    if not docs:
        print("   No chunks retrieved. (Index empty? Index path wrong?)")
        return
    for i, d in enumerate(docs, 1):
        title = (d.metadata.get("title") or "?")[:60]
        chunk_id = d.metadata.get("chunk_id") or d.metadata.get("doc_id") or "?"
        retrieved_by = d.metadata.get("retrieved_by", ["?"])
        score = d.metadata.get("rerank_score")
        score_text = f" rerank={score:.2f}" if score else ""
        print(f"   {i}. [{chunk_id[:50]}]{score_text}")
        print(f"      {title}")
        print(f"      retrieved_by: {','.join(retrieved_by)}")
    print(f"   ({elapsed:.1f}s, {len(docs)} unique chunks)")

    # ── Stage 8: group by video, clean ──
    section("8/9", "Merging by video, cleaning text...")
    try:
        from context_to_jsonl import group_docs_by_video, merge_grouped_docs
        grouped = group_docs_by_video(docs)
        merged = merge_grouped_docs(grouped)
        print(f"   {len(grouped)} unique videos → {len(merged)} merged contexts:")
        for i, m in enumerate(merged, 1):
            title = (m.get("title") or "?")[:60]
            chunks_used = m.get("chunks_used", 0)
            preview = (m.get("context") or "")[:80].replace("\n", " ")
            print(f"   {i}. {title}  ({chunks_used} chunks)")
            print(f"      → {preview}...")
    except Exception as e:
        print(f"   (merge failed: {e})")
        return

    # ── Stage 9: LLM answer ──
    section("9/9", f"Generating answer ({args.llm_provider})...")
    if args.no_llm:
        print("   (skipped — --no-llm flag set)")
        banner("ANSWER (placeholder — Stage 9 not run)")
        if merged:
            print("Top context preview:")
            print((merged[0].get("context") or "")[:300] + "...")
        return

    try:
        t0 = time.time()
        answer = call_llm(args.llm_provider, args.llm_model, question, merged)
        elapsed = time.time() - t0
        banner(f"ANSWER  ({args.llm_provider} {args.llm_model}, {elapsed:.1f}s)")
        print(answer)

        # Show citation legend at the bottom
        print()
        print(THIN)
        print("CITATIONS:")
        for i, m in enumerate(merged, 1):
            url = m.get("url") or "(no url)"
            title = (m.get("title") or "?")[:60]
            print(f"  [{i}] {title}")
            print(f"      {url}")
    except Exception as e:
        banner("ANSWER FAILED")
        print(f"Error calling {args.llm_provider}: {e}")
        print()
        print("Falling back to context preview:")
        if merged:
            print((merged[0].get("context") or "")[:300] + "...")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive Vietnamese RAG REPL — ask, see all 9 stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                   help="Chunks JSONL path")
    p.add_argument("--faiss", default=DEFAULT_FAISS_PATH,
                   help="FAISS index folder path")
    p.add_argument("--bm25", default=DEFAULT_BM25_PATH,
                   help="BM25 pickle path")
    p.add_argument("--k", type=int, default=6,
                   help="Top-K chunks to retrieve (default: 6)")
    p.add_argument("--rerank", action="store_true",
                   help="Use the cross-encoder reranker (slower; slightly better top-K)")
    p.add_argument("--no-query-writer", action="store_true",
                   help="Skip Ollama query rewriting (auto-detects if Ollama is down)")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip Stage 9 (LLM answer). Just see retrieval.")
    p.add_argument("--llm-provider", choices=["groq", "mistral"], default="groq",
                   help="LLM backend for answer generation (default: groq)")
    p.add_argument("--llm-model", default=None,
                   help=f"Override LLM model. Defaults: groq={GROQ_DEFAULT_MODEL}, "
                        f"mistral={MISTRAL_DEFAULT_MODEL}")
    p.add_argument("question", nargs="?", default=None,
                   help="(optional) one-shot question; if omitted, runs interactive REPL")
    args = p.parse_args()

    if args.llm_model is None:
        args.llm_model = (
            GROQ_DEFAULT_MODEL if args.llm_provider == "groq" else MISTRAL_DEFAULT_MODEL
        )

    # Build retriever once (cold-start ~30s; subsequent queries ~5-15s)
    retriever = setup_retriever(args)

    # Optional: try to detect Ollama
    rewriter = None
    if not args.no_query_writer:
        print("[setup] Probing Ollama for query rewriting...")
        rewriter = try_setup_query_writer()
        if rewriter is None:
            print("[setup] Ollama not reachable. Falling back to original VI query only.")
        else:
            print("[setup] Ollama OK (qwen2.5:3b).")

    # One-shot mode
    if args.question:
        process_one_question(args.question, retriever, rewriter, args)
        return

    # REPL mode
    print()
    print(BAR)
    print("Interactive RAG REPL ready.")
    print("Gõ câu hỏi tiếng Việt, ấn Enter. Gõ 'exit' để thoát.")
    print(BAR)
    while True:
        try:
            question = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            return
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", ":q"):
            print("[exit]")
            return
        try:
            process_one_question(question, retriever, rewriter, args)
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
