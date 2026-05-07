"""
data/chunked/generate_queries.py
═══════════════════════════════════════════════════════════════════════
Generate Vietnamese test queries from chunked transcripts using Gemini.

This is Script 2 of the retrieval-eval pipeline:

    transcript_v3.jsonl ──► [THIS SCRIPT] ──► test_queries.jsonl

For each sampled chunk, calls Gemini Flash to generate a Vietnamese question
that can ONLY be answered from that chunk. The chunk's chunk_id becomes the
ground-truth label for retrieval evaluation.

This is a proxy for hand-written questions — it's biased (questions are
designed to be answerable) but cheap. Hand-written queries (~30, 2-3 hours
of student time) should replace this when available.

Prerequisites
─────────────
    export GEMINI_API_KEY=your_key_here
    # Get a free key at https://aistudio.google.com/apikey

Usage
─────
    # Generate 100 queries (default)
    python -m data.chunked.generate_queries \\
        --input data/chunked/transcript_v3_t072.jsonl \\
        --output data/chunked/test_queries.jsonl \\
        --n 100

    # Fewer queries for quick testing
    python -m data.chunked.generate_queries \\
        --input data/chunked/transcript_v3_t072.jsonl \\
        --output data/chunked/test_queries.jsonl \\
        --n 20

Run from the project root so module imports resolve.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_INPUT = _PROJECT_ROOT / "data" / "chunked" / "transcript_v3.jsonl"
DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "chunked" / "test_queries.jsonl"

# Gemini prompt for Vietnamese question generation
QUERY_GENERATION_PROMPT = """\
Đoạn trích bài giảng (tiếng Anh):
---
{chunk_text}
---

Hãy viết MỘT câu hỏi BẰNG TIẾNG VIỆT mà sinh viên Việt Nam có thể hỏi,
và câu hỏi đó CHỈ có thể trả lời được nhờ đoạn trích trên.

Yêu cầu:
- Câu hỏi phải cụ thể, KHÔNG quá chung chung
- Câu hỏi phải bằng tiếng Việt tự nhiên
- Câu hỏi phải cần thông tin từ đoạn trích để trả lời
- Có thể dùng thuật ngữ tiếng Anh (ví dụ: attention, transformer, gradient)

Trả về JSON: {{"query_vi": "...", "is_specific_enough": true/false}}
"""


# ════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    """Load chunked JSONL."""
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return chunks


def write_queries(queries: List[Dict[str, Any]], path: Path) -> int:
    """Write queries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    return len(queries)


# ════════════════════════════════════════════════════════════════
# STRATIFIED SAMPLING
# ════════════════════════════════════════════════════════════════

def stratified_sample(
    chunks: List[Dict[str, Any]],
    n: int,
    min_per_course: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample n chunks stratified by course.

    Each course gets at least min_per_course samples (if it has enough chunks).
    Remaining slots are distributed proportionally to corpus share.
    Only samples 'knowledge_unit' and 'semantic_unit' chunks (skips tail_short).

    Over-samples by 30% to compensate for Gemini query rejection downstream.
    """
    rng = random.Random(seed)

    # Filter to substantial chunks only
    eligible = [
        c for c in chunks
        if c.get("chunk_type") in ("knowledge_unit", "semantic_unit")
        and len(c.get("chunk_text", "").split()) >= 30
    ]

    # Group by course
    by_course: Dict[str, List[Dict]] = defaultdict(list)
    for c in eligible:
        course = c.get("course", "unknown")
        by_course[course].append(c)

    # Over-sample target to compensate for filtering
    target = int(n * 1.3)

    # Allocate: min_per_course guaranteed, rest proportional
    courses = sorted(by_course.keys())
    n_courses = len(courses)
    guaranteed = min(min_per_course, target // n_courses) if n_courses > 0 else 0
    remaining = target - guaranteed * n_courses

    sampled: List[Dict[str, Any]] = []
    for course in courses:
        pool = by_course[course]
        rng.shuffle(pool)

        # Guaranteed allocation
        course_n = guaranteed

        # Proportional extra
        if remaining > 0 and len(eligible) > 0:
            proportion = len(pool) / len(eligible)
            extra = max(0, int(round(remaining * proportion)))
            course_n += extra

        # Don't exceed pool size
        course_n = min(course_n, len(pool))
        sampled.extend(pool[:course_n])

    # Shuffle final sample and trim to target
    rng.shuffle(sampled)
    sampled = sampled[:target]

    logger.info(
        "Sampled %d chunks from %d courses (target: %d queries, over-sampled to %d)",
        len(sampled), len(courses), n, target,
    )
    for course in courses:
        count = sum(1 for s in sampled if s.get("course") == course)
        logger.info("  %s: %d samples", course, count)

    return sampled


# ════════════════════════════════════════════════════════════════
# GEMINI API CALL
# ════════════════════════════════════════════════════════════════

def init_gemini_client(api_key: str):
    """Initialize Gemini client using google-genai SDK."""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "Need google-genai. Install: pip install google-genai>=0.3.0"
        ) from e

    client = genai.Client(api_key=api_key)
    return client


def generate_query_from_chunk(
    client,
    chunk: Dict[str, Any],
    model: str = "gemini-2.5-flash",
    temperature: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    Call Gemini to generate one Vietnamese query from a chunk.

    Returns the parsed JSON response or None if the call fails / query is bad.
    """
    from google.genai import types

    prompt = QUERY_GENERATION_PROMPT.format(chunk_text=chunk["chunk_text"][:2000])

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )

        # Parse response
        text = response.text.strip()
        result = json.loads(text)

        query_vi = result.get("query_vi", "").strip()
        is_specific = result.get("is_specific_enough", False)

        # Validation
        if not is_specific:
            logger.debug("Query rejected (not specific enough): %s", query_vi[:60])
            return None

        if len(query_vi.split()) < 5:
            logger.debug("Query rejected (too short): %s", query_vi)
            return None

        return {"query_vi": query_vi}

    except json.JSONDecodeError as e:
        logger.warning("Bad JSON from Gemini: %s", e)
        return None
    except Exception as e:
        logger.warning("Gemini API error: %s", e)
        return None


# ════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════

def generate_all_queries(
    chunks: List[Dict[str, Any]],
    sampled: List[Dict[str, Any]],
    api_key: str,
    target_n: int,
    delay: float = 0.5,
    manual: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate Vietnamese queries for sampled chunks.
    If manual=True, prompts the user to type questions interactively.
    Otherwise, uses Gemini API.
    """
    client = init_gemini_client(api_key) if not manual else None

    queries: List[Dict[str, Any]] = []
    attempted = 0
    rejected = 0

    mode_str = "MANUAL MODE" if manual else "Gemini Flash"
    logger.info("Generating queries via %s (%d samples, target %d queries)...",
                mode_str, len(sampled), target_n)

    for i, chunk in enumerate(sampled):
        if len(queries) >= target_n:
            break

        attempted += 1
        
        if manual:
            print("\n" + "="*60)
            print(f"CHUNK {i+1}/{len(sampled)} (Course: {chunk.get('course', 'unknown')})")
            print("-" * 60)
            print(chunk.get("chunk_text", ""))
            print("-" * 60)
            print("Type a Vietnamese question that can be answered by this chunk.")
            print("(Or type 'skip' to skip this chunk, 'quit' to save and exit)")
            query_vi = input("Question: ").strip()
            
            if query_vi.lower() == 'quit':
                break
            if query_vi.lower() == 'skip' or not query_vi:
                rejected += 1
                continue
                
            result = {"query_vi": query_vi}
        else:
            result = generate_query_from_chunk(client, chunk)

            if result is None:
                rejected += 1
                time.sleep(delay)
                continue

        query_record = {
            "query_id": f"q{len(queries) + 1:03d}",
            "query_vi": result["query_vi"],
            "expected_chunk_id": chunk.get("chunk_id", ""),
            "expected_video_id": chunk.get("video_id", ""),
            "expected_start_time": chunk.get("start_time", 0.0),
            "expected_end_time": chunk.get("end_time", 0.0),
            "course": chunk.get("course", ""),
            "source_text": chunk.get("chunk_text", "")[:500],
        }
        queries.append(query_record)

        if not manual and (i + 1) % 10 == 0:
            logger.info(
                "Progress: %d/%d attempted, %d valid queries, %d rejected",
                attempted, len(sampled), len(queries), rejected,
            )

        # Rate limit if using API
        if not manual:
            time.sleep(delay)

    logger.info(
        "Query generation complete: %d valid queries from %d attempts (%d rejected)",
        len(queries), attempted, rejected,
    )
    return queries


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Vietnamese test queries from chunked transcripts using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m data.chunked.generate_queries \\\n"
            "      --input data/chunked/transcript_v3_t072.jsonl \\\n"
            "      --output data/chunked/test_queries.jsonl \\\n"
            "      --n 100\n"
        ),
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Chunked JSONL file to sample from (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL for generated queries (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of queries to generate (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API calls in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Enable manual mode to type queries yourself (skips API entirely)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    # Check API key (env var takes priority, falls back to hardcoded key)
    api_key = os.environ.get("GEMINI_API_KEY", "") or "API_KEY"

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("GENERATE TEST QUERIES")
    logger.info("  Input  : %s", args.input)
    logger.info("  Output : %s", args.output)
    logger.info("  Target : %d queries", args.n)
    logger.info("  Seed   : %d", args.seed)
    logger.info("=" * 60)

    # 1. Load chunks
    chunks = load_chunks(args.input)
    if not chunks:
        logger.error("No chunks loaded. Aborting.")
        sys.exit(1)

    # 2. Stratified sample
    sampled = stratified_sample(chunks, args.n, seed=args.seed)

    # 3. Generate queries
    queries = generate_all_queries(chunks, sampled, api_key, args.n, delay=args.delay, manual=args.manual)

    # 4. Write output
    written = write_queries(queries, args.output)

    # Summary
    courses_covered = set(q["course"] for q in queries)
    logger.info("")
    logger.info("=" * 60)
    logger.info("QUERY GENERATION COMPLETE")
    logger.info("  Queries written : %d", written)
    logger.info("  Output file     : %s", args.output)
    logger.info("  Courses covered : %s", ", ".join(sorted(courses_covered)))
    logger.info("=" * 60)

    # Per-course breakdown
    from collections import Counter
    course_counts = Counter(q["course"] for q in queries)
    for course, count in sorted(course_counts.items()):
        logger.info("  %s: %d queries", course, count)


if __name__ == "__main__":
    main()
