"""
evaluation/answer_eval.py
═══════════════════════════════════════════════════════════════════════
Evaluate the LLM ANSWER step (Stage 9) on 8 standard metrics.

Pipeline per query:
  test_query (question + ground_truth)
        │
        ▼ retrieve top-K chunks (same as production)
        │
        ▼ LLM generates Vietnamese answer (same as production)
        │
        ▼ COMPARE ANSWER vs GROUND_TRUTH on 8 metrics
        │
        ▼ Aggregate, save report

METRICS
───────

  Rule-based (deterministic, no LLM):
    1. Exact Match           — answer.lower() == ground_truth.lower()
    2. String Presence       — fraction of GT content-words found in answer
    3. BLEU                  — n-gram precision against ground_truth
    4. ROUGE-1 / ROUGE-2 / ROUGE-L — n-gram recall variants

  LLM-judged (uses the Groq/Mistral judge chain — separate from answer LLM):
    5. Instruction Following — does the answer follow the prompt's rules?
                               (Vietnamese, cites [n], stays on topic)
    6. Response Relevancy    — does the answer actually address the question?
    7. Groundedness          — every claim supported by retrieved contexts?
                               (low = hallucination)
    8. Noise Sensitivity     — does adding an irrelevant context change the
                               answer significantly? (low robustness = bad)

USAGE
─────

    # Default: 20 queries from test_queries.jsonl, all 8 metrics, Groq judge
    python evaluation/answer_eval.py --limit 20

    # Skip LLM-judged metrics if quota is tight (rule-based only, ~free)
    python evaluation/answer_eval.py --limit 20 --rules-only

    # Use Mistral as the judge (different LLM than answer generator)
    python evaluation/answer_eval.py --limit 20 \\
        --answer-provider groq --judge-provider mistral

    # Save full per-query JSON report
    python evaluation/answer_eval.py --output reports/answer_eval.json

Note: this is for evaluating the ANSWER STEP, not retrieval. Retrieval-side
metrics live in `evaluation/ragas_eval.py` (RAGAS + rank metrics).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# ── Windows console: force UTF-8 ──
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# ── Make pipeline + sibling modules importable ──
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_PIPELINE_DIR = _REPO_ROOT / "pipeline"
for _sub in (
    str(_THIS_DIR),                                   # for ragas_eval (judges)
    str(_PIPELINE_DIR / "stage_05_index"),
    str(_PIPELINE_DIR / "stage_07_retrieve"),
    str(_PIPELINE_DIR / "stage_08_merge_context"),
    str(_PIPELINE_DIR),                               # for ask.py reuse
):
    if _sub not in sys.path:
        sys.path.insert(0, _sub)

# ── .env for API keys ──
try:
    from dotenv import load_dotenv  # type: ignore
    for _candidate in (_REPO_ROOT / ".env", Path.cwd() / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
            break
except ImportError:
    pass

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# DEFAULTS
# ════════════════════════════════════════════════════════════════

DEFAULT_QUERIES_PATH = _REPO_ROOT / "data" / "chunked" / "test_queries.jsonl"
DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "chunked" / "transcript_v3_t072.jsonl"
DEFAULT_FAISS_PATH = _REPO_ROOT / "indexes" / "faiss_index_072_n"
DEFAULT_BM25_PATH = _REPO_ROOT / "indexes" / "bm25_072.pkl"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "chunked" / "answer_eval_report.json"

ANSWER_PROVIDER_DEFAULT = "groq"
ANSWER_MODEL_DEFAULT = "llama-3.3-70b-versatile"


# ════════════════════════════════════════════════════════════════
# DATA MODEL
# ════════════════════════════════════════════════════════════════

@dataclass
class AnswerEval:
    """Per-query answer evaluation."""
    query_id: str
    question: str
    ground_truth: str
    answer: str
    n_contexts: int
    # Rule-based
    exact_match: float = 0.0
    string_presence: float = 0.0
    bleu: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    # LLM-judged
    instruction_following: float = 0.0
    response_relevancy: float = 0.0
    groundedness: float = 0.0
    noise_sensitivity: float = 0.0
    # Bookkeeping
    answer_latency_ms: float = 0.0
    judge_calls: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════
# RULE-BASED METRICS
# ════════════════════════════════════════════════════════════════

# Vietnamese + English stopwords for keyword matching — small list, just
# enough to filter the most uninformative tokens
_STOPWORDS = {
    # English
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of",
    "to", "in", "on", "for", "with", "by", "as", "at", "and", "or",
    "but", "if", "this", "that", "these", "those", "it", "its",
    # Vietnamese
    "là", "của", "và", "có", "được", "trong", "cho", "với", "này",
    "đó", "thì", "một", "các", "những", "khi", "để", "không", "ra",
    "vào", "đến", "từ", "tại", "về", "như", "đã", "sẽ", "rồi", "nữa",
}


def metric_exact_match(answer: str, ground_truth: str) -> float:
    """Binary: 1.0 if exact string match (case+whitespace insensitive)."""
    return 1.0 if answer.strip().lower() == ground_truth.strip().lower() else 0.0


def metric_string_presence(answer: str, ground_truth: str) -> float:
    """
    Fraction of content-words from ground_truth that appear (anywhere)
    in the answer. Range 0..1. Stopwords excluded.
    """
    gt_tokens = re.findall(r"[\wÀ-ỹ]+", ground_truth.lower())
    gt_content = [t for t in gt_tokens if t not in _STOPWORDS and len(t) > 1]
    if not gt_content:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for tok in set(gt_content) if tok in answer_lower)
    return found / len(set(gt_content))


def metric_bleu(answer: str, ground_truth: str) -> float:
    """
    Sentence-level BLEU (smoothed) on whitespace-tokenized text.
    Returns 0..1. Falls back to 0.0 if nltk unavailable.
    """
    try:
        from nltk.translate.bleu_score import (
            sentence_bleu, SmoothingFunction,
        )
    except ImportError:
        return 0.0

    ref = ground_truth.lower().split()
    hyp = answer.lower().split()
    if not ref or not hyp:
        return 0.0
    smooth = SmoothingFunction().method1
    return sentence_bleu([ref], hyp, smoothing_function=smooth)


def metric_rouge(answer: str, ground_truth: str) -> Dict[str, float]:
    """
    ROUGE-1 / ROUGE-2 / ROUGE-L F-scores. Range 0..1 each.
    Falls back to zeros if `rouge` package unavailable.
    """
    try:
        from rouge import Rouge  # type: ignore
    except ImportError:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    if not answer.strip() or not ground_truth.strip():
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    try:
        rouge = Rouge()
        scores = rouge.get_scores(answer, ground_truth)[0]
        return {
            "rouge_1": float(scores["rouge-1"]["f"]),
            "rouge_2": float(scores["rouge-2"]["f"]),
            "rouge_l": float(scores["rouge-l"]["f"]),
        }
    except Exception as e:
        logger.warning("ROUGE failed: %s", e)
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}


# ════════════════════════════════════════════════════════════════
# LLM-JUDGED METRICS
# ════════════════════════════════════════════════════════════════

INSTRUCTION_FOLLOWING_PROMPT = """Bạn là người chấm điểm hệ thống RAG.

CÂU TRẢ LỜI cần tuân thủ các yêu cầu:
1. Viết bằng TIẾNG VIỆT
2. Có trích dẫn dạng [1], [2], ... khi sử dụng nguồn
3. Tập trung vào câu hỏi, không lan man
4. Giữ thuật ngữ tiếng Anh chuyên ngành (transformer, attention, ...)

CÂU HỎI: {question}

CÂU TRẢ LỜI ĐƯỢC TẠO:
{answer}

Đánh giá mức độ tuân thủ các yêu cầu trên (0.0 = không tuân thủ, 1.0 = tuân thủ đầy đủ).

Trả về JSON: {{"score": <float 0..1>, "reason": "1 câu giải thích ngắn"}}"""


RESPONSE_RELEVANCY_PROMPT = """Bạn là người chấm điểm hệ thống RAG.

CÂU HỎI: {question}

CÂU TRẢ LỜI ĐƯỢC TẠO:
{answer}

Câu trả lời có thực sự trả lời câu hỏi không? Chấm điểm:
- 1.0 = câu trả lời đầy đủ và đúng trọng tâm
- 0.5 = trả lời một phần, hoặc lan man qua chủ đề khác
- 0.0 = không trả lời câu hỏi

Trả về JSON: {{"score": <float 0..1>, "reason": "1 câu giải thích ngắn"}}"""


GROUNDEDNESS_PROMPT = """Bạn là người chấm điểm hệ thống RAG (kiểm tra hallucination).

NGUỒN ĐƯỢC CUNG CẤP:
{contexts_block}

CÂU TRẢ LỜI ĐƯỢC TẠO:
{answer}

Hãy tách câu trả lời thành các tuyên bố/claim đơn lẻ. Với mỗi claim, kiểm tra:
- Có được hỗ trợ rõ ràng bởi NGUỒN không?
- Hay là claim được "bịa ra" (hallucination)?

Trả về JSON đúng format:
{{
  "claims": [
    {{"text": "claim 1", "supported": true/false}},
    {{"text": "claim 2", "supported": true/false}},
    ...
  ]
}}

Score cuối cùng = số claim được hỗ trợ / tổng số claim."""


NOISE_SENSITIVITY_PROMPT = """Bạn là người chấm điểm độ vững (robustness) của RAG.

CÂU HỎI: {question}

CÂU TRẢ LỜI GỐC (dựa trên nguồn liên quan):
{answer_clean}

CÂU TRẢ LỜI VỚI NHIỄU (có thêm nguồn không liên quan):
{answer_noisy}

Đánh giá độ ổn định:
- 1.0 = hai câu trả lời gần như giống nhau (mô hình bỏ qua nhiễu — TỐT)
- 0.5 = có khác biệt nhỏ
- 0.0 = câu trả lời với nhiễu hoàn toàn khác (mô hình bị nhiễu lừa — TỆ)

Trả về JSON: {{"score": <float 0..1>, "reason": "1 câu giải thích ngắn"}}"""


def _judge_score(judge, prompt: str, fallback: float = 0.0) -> Tuple[float, str]:
    """Helper: call judge, parse {score, reason}, default to fallback on error."""
    result = judge.call_json(prompt)
    if not result:
        return fallback, "(judge returned None)"
    try:
        score = float(result.get("score", fallback))
        score = max(0.0, min(1.0, score))
        reason = str(result.get("reason", ""))[:120]
        return score, reason
    except (TypeError, ValueError):
        return fallback, "(parse error)"


def metric_instruction_following(
    question: str, answer: str, judge,
) -> Tuple[float, str]:
    return _judge_score(judge, INSTRUCTION_FOLLOWING_PROMPT.format(
        question=question, answer=answer,
    ))


def metric_response_relevancy(
    question: str, answer: str, judge,
) -> Tuple[float, str]:
    return _judge_score(judge, RESPONSE_RELEVANCY_PROMPT.format(
        question=question, answer=answer,
    ))


def metric_groundedness(
    answer: str, contexts: List[str], judge,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Returns (score, per-claim-list)."""
    contexts_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    result = judge.call_json(GROUNDEDNESS_PROMPT.format(
        contexts_block=contexts_block,
        answer=answer,
    )) or {}
    claims = result.get("claims") or []
    if not claims or not isinstance(claims, list):
        return 0.0, []
    supported = sum(1 for c in claims if c.get("supported"))
    return supported / len(claims), claims


def metric_noise_sensitivity(
    question: str,
    clean_answer: str,
    noisy_answer: str,
    judge,
) -> Tuple[float, str]:
    return _judge_score(judge, NOISE_SENSITIVITY_PROMPT.format(
        question=question,
        answer_clean=clean_answer,
        answer_noisy=noisy_answer,
    ))


# ════════════════════════════════════════════════════════════════
# ANSWER GENERATION (reused from pipeline/ask.py)
# ════════════════════════════════════════════════════════════════

# Reusing the same prompt as production so we evaluate what users will see
SYSTEM_PROMPT_VI = """Bạn là trợ giảng AI chuyên về NLP và Machine Learning.

QUY TẮC:
- Chỉ trả lời dựa trên NGUỒN được cung cấp.
- Nếu các nguồn không đủ thông tin để trả lời, hãy nói rõ điều đó.
- Giải thích đơn giản, dễ hiểu cho sinh viên.
- Trích dẫn nguồn bằng [1], [2], [3]... khi dùng thông tin từ nó.
- Giữ nguyên thuật ngữ tiếng Anh chuyên ngành (transformer, attention, gradient descent...).

Trả lời bằng tiếng Việt, ngắn gọn (3-6 câu) trừ khi câu hỏi yêu cầu giải thích sâu."""


def _build_answer_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title") or "Unknown"
        text = ctx.get("context") or ctx.get("chunk_text") or ""
        blocks.append(f"[{i}] {title}\n     {text}")
    sources = "\n\n".join(blocks)
    return (
        f"NGUỒN:\n{sources}\n\n"
        f"CÂU HỎI:\n{question}\n\n"
        f"Trả lời bằng tiếng Việt, có trích dẫn [n] khi sử dụng nguồn."
    )


def call_answer_groq(question: str, contexts: List[Dict[str, Any]], model: str) -> str:
    from groq import Groq  # type: ignore
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY missing")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_VI},
            {"role": "user", "content": _build_answer_user_prompt(question, contexts)},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()


def call_answer_mistral(question: str, contexts: List[Dict[str, Any]], model: str) -> str:
    import requests  # type: ignore
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY missing")
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_VI},
                {"role": "user", "content": _build_answer_user_prompt(question, contexts)},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=60,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def generate_answer(provider: str, model: str,
                    question: str, contexts: List[Dict[str, Any]]) -> str:
    if provider == "groq":
        return call_answer_groq(question, contexts, model)
    if provider == "mistral":
        return call_answer_mistral(question, contexts, model)
    raise ValueError(f"Unknown answer provider: {provider}")


# ════════════════════════════════════════════════════════════════
# EVAL LOOP
# ════════════════════════════════════════════════════════════════

def load_test_queries(path: Path) -> List[Dict[str, Any]]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def build_retriever(args):
    """Reuse the same retriever pipeline/ask.py uses."""
    from hybrid_retriever import build_hybrid_retriever
    return build_hybrid_retriever(
        data_path=str(args.data_path),
        faiss_index_path=str(args.faiss),
        bm25_path=str(args.bm25),
        k_retrieve=20,
        k_final=args.k,
        k_rerank=args.k,
        use_reranker=False,           # speed for eval
        weight_bm25=0.3,
        weight_dense=0.7,
        enable_tracing=False,
    )


def evaluate_one(
    query: Dict[str, Any],
    retriever,
    judge,
    args: argparse.Namespace,
    noise_pool: List[str],
) -> AnswerEval:
    """Evaluate one query end-to-end."""
    q_id = query.get("query_id", "?")
    question = query.get("question", "")
    ground_truth = query.get("ground_truth", "")

    # ── Retrieve contexts ──
    r = retriever(question)
    docs = r.get("documents", []) if isinstance(r, dict) else r

    from context_to_jsonl import group_docs_by_video, merge_grouped_docs
    grouped = group_docs_by_video(docs)
    merged = merge_grouped_docs(grouped)
    contexts_text = [m.get("context", "") for m in merged]

    # ── Generate answer (clean) ──
    t0 = time.time()
    answer_error: Optional[str] = None
    try:
        answer = generate_answer(args.answer_provider, args.answer_model,
                                  question, merged)
    except Exception as e:
        # Loud failure: print to stdout (not just logger) so user sees it
        err_str = str(e)
        if "429" in err_str or "rate_limit" in err_str.lower() or "quota" in err_str.lower():
            answer_error = f"RATE-LIMITED: {args.answer_provider}/{args.answer_model}"
            print(f"\n   ⚠ ANSWER GEN RATE-LIMITED ({args.answer_provider}/{args.answer_model}).")
            print(f"     Switch with --answer-model llama-3.1-8b-instant "
                  f"(or --answer-provider mistral)")
        else:
            answer_error = f"ERROR: {type(e).__name__}: {err_str[:200]}"
            print(f"\n   ⚠ ANSWER GEN FAILED: {answer_error}")
        answer = ""
    answer_latency_ms = (time.time() - t0) * 1000

    # ── Rule-based metrics ──
    em = metric_exact_match(answer, ground_truth)
    sp = metric_string_presence(answer, ground_truth)
    bleu = metric_bleu(answer, ground_truth)
    rouge = metric_rouge(answer, ground_truth)

    # ── LLM-judged metrics ──
    if args.rules_only or not answer:
        if_score, if_reason = 0.0, "(rules-only or empty answer)"
        rr_score, rr_reason = 0.0, "(rules-only or empty answer)"
        gd_score, gd_claims = 0.0, []
        ns_score, ns_reason = 0.0, "(rules-only or empty answer)"
    else:
        if_score, if_reason = metric_instruction_following(question, answer, judge)
        rr_score, rr_reason = metric_response_relevancy(question, answer, judge)
        gd_score, gd_claims = metric_groundedness(answer, contexts_text, judge)

        # Noise Sensitivity: regenerate with one random irrelevant context added
        if noise_pool and merged:
            noise_text = random.choice(noise_pool)
            noisy_contexts = merged + [{"title": "(noise)", "context": noise_text}]
            try:
                noisy_answer = generate_answer(
                    args.answer_provider, args.answer_model, question, noisy_contexts,
                )
                ns_score, ns_reason = metric_noise_sensitivity(
                    question, answer, noisy_answer, judge,
                )
            except Exception as e:
                ns_score, ns_reason = 0.0, f"(noise gen failed: {e})"
        else:
            ns_score, ns_reason = 0.0, "(noise pool empty)"

    return AnswerEval(
        query_id=q_id,
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        n_contexts=len(merged),
        exact_match=em,
        string_presence=sp,
        bleu=bleu,
        rouge_1=rouge["rouge_1"],
        rouge_2=rouge["rouge_2"],
        rouge_l=rouge["rouge_l"],
        instruction_following=if_score,
        response_relevancy=rr_score,
        groundedness=gd_score,
        noise_sensitivity=ns_score,
        answer_latency_ms=answer_latency_ms,
        notes={
            "instruction_reason": if_reason,
            "relevancy_reason": rr_reason,
            "groundedness_claims": gd_claims,
            "noise_reason": ns_reason,
            "answer_error": answer_error,
        },
    )


# ════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════

def render_summary(results: List[AnswerEval], aggregate: Dict[str, Any]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("LLM ANSWER EVALUATION SUMMARY")
    lines.append("=" * 70)
    lines.append(f"  Queries evaluated   : {aggregate['n_queries']}")
    lines.append(f"  Answer model        : {aggregate['answer_model']}")
    lines.append(f"  Judge model         : {aggregate['judge_model']}")
    lines.append(f"  Total LLM calls     : {aggregate['total_llm_calls']}")
    lines.append(f"  Elapsed             : {aggregate['elapsed_seconds']}s")
    lines.append(f"  Mean answer latency : {aggregate['mean_answer_latency_ms']:.0f}ms")
    lines.append("")
    lines.append("  RULE-BASED METRICS (deterministic):")
    lines.append(f"    Exact Match       : {aggregate['exact_match']:.3f}")
    lines.append(f"    String Presence   : {aggregate['string_presence']:.3f}")
    lines.append(f"    BLEU              : {aggregate['bleu']:.3f}")
    lines.append(f"    ROUGE-1 / 2 / L   : {aggregate['rouge_1']:.3f}  /  "
                 f"{aggregate['rouge_2']:.3f}  /  {aggregate['rouge_l']:.3f}")

    if aggregate.get("instruction_following") is not None:
        lines.append("")
        lines.append("  LLM-JUDGED METRICS:")
        lines.append(f"    Instruction Following : {aggregate['instruction_following']:.3f}")
        lines.append(f"    Response Relevancy    : {aggregate['response_relevancy']:.3f}")
        lines.append(f"    Groundedness          : {aggregate['groundedness']:.3f}")
        lines.append(f"    Noise Sensitivity     : {aggregate['noise_sensitivity']:.3f}")

    lines.append("")
    lines.append("  INTERPRETATION:")
    em = aggregate["exact_match"]
    sp = aggregate["string_presence"]
    rl = aggregate["rouge_l"]
    if em > 0.5:
        lines.append("    - Exact Match HIGH: surprising. Check if test set has near-duplicate answers.")
    if sp < 0.4:
        lines.append("    - String Presence LOW: answers don't share GT vocabulary. May still be paraphrasing well.")
    if rl < 0.2:
        lines.append("    - ROUGE-L LOW: little surface overlap with GT. Check semantic metrics too.")

    grnd = aggregate.get("groundedness")
    if grnd is not None:
        if grnd < 0.7:
            lines.append("    - Groundedness LOW: hallucinations detected. LLM is making up info.")
        else:
            lines.append("    - Groundedness HEALTHY. Answers stay grounded in retrieved contexts.")

    nse = aggregate.get("noise_sensitivity")
    if nse is not None:
        if nse < 0.5:
            lines.append("    - Noise Sensitivity LOW: model gets confused when noise added (bad).")
        else:
            lines.append("    - Noise Sensitivity HEALTHY. Model robust to irrelevant contexts.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate the LLM ANSWER step on 8 metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--faiss", type=Path, default=DEFAULT_FAISS_PATH)
    p.add_argument("--bm25", type=Path, default=DEFAULT_BM25_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--limit", type=int, default=20,
                   help="Max queries to evaluate (default: 20; use 0 for all)")
    p.add_argument("--k", type=int, default=6,
                   help="Top-K chunks to retrieve per query")
    p.add_argument("--rules-only", action="store_true",
                   help="Skip LLM-judged metrics. Rule-based only (zero LLM calls).")

    p.add_argument("--answer-provider", choices=["groq", "mistral"],
                   default=ANSWER_PROVIDER_DEFAULT,
                   help="LLM that GENERATES the answer (default: groq)")
    p.add_argument("--answer-model", default=ANSWER_MODEL_DEFAULT,
                   help="Model name for the answer LLM")

    p.add_argument("--judge-provider",
                   choices=["groq", "mistral", "openrouter", "ollama", "gemini"],
                   default="groq",
                   help="LLM that JUDGES the answer (use a DIFFERENT one from "
                        "answer-provider to reduce bias). Default: groq.")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-fallback", default=None,
                   help="Comma-separated fallback chain, e.g. 'groq,mistral'")
    p.add_argument("--rpm-delay", type=float, default=None)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for noise sampling")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    random.seed(args.seed)

    if not args.queries.exists():
        logger.error("Queries file not found: %s", args.queries)
        sys.exit(1)
    queries = load_test_queries(args.queries)
    if args.limit and args.limit > 0:
        queries = queries[: args.limit]
    print(f"[eval] Loaded {len(queries)} queries from {args.queries}")

    # Build judge (skip if rules-only)
    judge = None
    if not args.rules_only:
        from ragas_eval import get_judge, FallbackJudge
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
        print(f"[eval] Judge: {args.judge_provider} ({args.judge_model or 'default'})")
    else:
        print("[eval] Rules-only mode: no LLM judge")

    # Build retriever
    print("[eval] Building retriever (cold-start ~30s)...")
    retriever = build_retriever(args)

    # Build a small noise pool from random chunks (for noise sensitivity metric)
    noise_pool: List[str] = []
    if not args.rules_only and args.data_path.exists():
        with args.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if random.random() < 0.01:
                    try:
                        c = json.loads(line)
                        text = c.get("chunk_text", "")
                        if 200 <= len(text) <= 800:
                            noise_pool.append(text)
                            if len(noise_pool) >= 50:
                                break
                    except Exception:
                        pass
        print(f"[eval] Noise pool: {len(noise_pool)} chunks")

    # Evaluate
    t0 = time.time()
    results: List[AnswerEval] = []
    print(f"\n[eval] Running {len(queries)} queries"
          f"{' (rules-only)' if args.rules_only else ' with LLM judging'}...\n")
    for i, q in enumerate(queries, 1):
        qid = q.get("query_id", "?")
        question = q.get("question", "")[:60]
        print(f"  [{i:>3}/{len(queries)}] {qid}: {question}")
        try:
            ev = evaluate_one(q, retriever, judge, args, noise_pool)
            results.append(ev)
            print(
                f"       EM={ev.exact_match:.2f} SP={ev.string_presence:.2f} "
                f"BLEU={ev.bleu:.2f} R-L={ev.rouge_l:.2f}"
                + ("" if args.rules_only else
                   f" | IF={ev.instruction_following:.2f} RR={ev.response_relevancy:.2f}"
                   f" GD={ev.groundedness:.2f} NS={ev.noise_sensitivity:.2f}")
            )
        except Exception as e:
            logger.warning("Query %s failed: %s", qid, e)

    elapsed = time.time() - t0

    # Aggregate
    aggregate: Dict[str, Any] = {
        "n_queries": len(results),
        "answer_model": f"{args.answer_provider}:{args.answer_model}",
        "judge_model": (
            f"{args.judge_provider}:{args.judge_model or 'default'}"
            if not args.rules_only else "(rules-only)"
        ),
        "total_llm_calls": getattr(judge, "n_calls", 0) if judge else 0,
        "elapsed_seconds": round(elapsed, 1),
        "mean_answer_latency_ms": (
            mean(r.answer_latency_ms for r in results) if results else 0.0
        ),
        "exact_match":     mean(r.exact_match     for r in results) if results else 0.0,
        "string_presence": mean(r.string_presence for r in results) if results else 0.0,
        "bleu":            mean(r.bleu            for r in results) if results else 0.0,
        "rouge_1":         mean(r.rouge_1         for r in results) if results else 0.0,
        "rouge_2":         mean(r.rouge_2         for r in results) if results else 0.0,
        "rouge_l":         mean(r.rouge_l         for r in results) if results else 0.0,
    }
    if not args.rules_only and results:
        aggregate["instruction_following"] = mean(r.instruction_following for r in results)
        aggregate["response_relevancy"]    = mean(r.response_relevancy    for r in results)
        aggregate["groundedness"]          = mean(r.groundedness          for r in results)
        aggregate["noise_sensitivity"]     = mean(r.noise_sensitivity     for r in results)
    else:
        aggregate["instruction_following"] = None
        aggregate["response_relevancy"]    = None
        aggregate["groundedness"]          = None
        aggregate["noise_sensitivity"]     = None

    # Render + save
    print(render_summary(results, aggregate))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({
            "aggregate": aggregate,
            "config": {
                "queries": str(args.queries),
                "answer_provider": args.answer_provider,
                "answer_model": args.answer_model,
                "judge_provider": args.judge_provider if not args.rules_only else None,
                "judge_model": args.judge_model,
                "rules_only": args.rules_only,
                "k": args.k,
            },
            "per_query": [asdict(r) for r in results],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] Full per-query report saved to {args.output}")


if __name__ == "__main__":
    main()
