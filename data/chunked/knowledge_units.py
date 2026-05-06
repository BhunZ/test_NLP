"""
data/chunked/knowledge_units.py
═══════════════════════════════════════════════════════════════════════
Build "knowledge unit" chunks from cleaned sentence-level transcripts.

This is the TIME-ORDER-PRESERVING replacement for the topic-grouped
chunker. Topic grouping was destroying citation accuracy by merging
non-adjacent chunks across the lecture timeline. See plan §"Diagnosis
of knowledge_units.py" for full reasoning.

Two passes:

    Pass 1 — TIME CHUNKING
        Walk transcript segments in order. Pack into chunks bounded by:
          - target token count (180), max 200, min 150
          - max duration (90s), min duration (20s)
          - silence gaps > 3s
          - sentence boundaries (.!? when MIN_TOKENS reached)

    Pass 2 — EMBEDDING MERGE  ← the actual semantic step, FINALLY
        Walk Pass-1 chunks IN TIME ORDER (no topic grouping).
        Embed each chunk with bge-m3 (the same embedder used for retrieval).
        For each adjacent pair, compute cosine similarity.
        Merge if similarity ≥ THRESHOLD and combined size still within bounds.

Usage
─────
    # Full corpus, default threshold 0.65
    python -m data.chunked.knowledge_units

    # Quick test on first 5 videos
    python -m data.chunked.knowledge_units --limit 5

    # Try a different threshold without overwriting
    python -m data.chunked.knowledge_units --threshold 0.55 \\
                                           --output transcript_v3_t055.jsonl

    # Sweep multiple thresholds (writes one output per value)
    python -m data.chunked.knowledge_units --sweep 0.55 0.65 0.72 0.80

    # Skip Pass 2 entirely — see what the time chunker alone produces
    python -m data.chunked.knowledge_units --no-merge

Run from the project root so module imports resolve.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Project root on sys.path so `from core...` works when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.utils import format_timestamp, make_youtube_url, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

# Project-relative paths (no more hardcoded D:/NLP)
INPUT_PATH = _PROJECT_ROOT / "data" / "cleaned" / "transcripts_clean_sentence.jsonl"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "chunked"
DEFAULT_OUTPUT_NAME = "transcript_v3.jsonl"

# Token bounds — matches the original knowledge_units.py
MIN_TOKENS = 150
MAX_TOKENS = 200
MAX_TOKEN_BUFFER = 20    # allow +20 over MAX before forcing close

# Duration bounds (seconds)
MIN_DURATION = 20.0
MAX_DURATION = 90.0       # NOW ENFORCED (was unused in original)

# Silence-gap threshold for closing a chunk (seconds)
SILENCE_GAP_CLOSE = 3.0

# Embedding-merge bounds — slightly looser than the time-chunk bounds,
# because merged chunks can grow a bit while still being useful.
MERGE_MAX_TOKENS = MAX_TOKENS + 30      # 230
MERGE_MAX_DURATION = MAX_DURATION + 30  # 120 s

# Default cosine-similarity threshold for merging adjacent chunks.
# bge-m3 normalized cosines on adjacent lecture sentences typically cluster
# in the 0.55–0.85 range — 0.65 is a sensible starting point. Sweep with
# --sweep to tune against the eval harness once it exists.
DEFAULT_THRESHOLD = 0.65


# ASR corrections — common contractions stripped of apostrophes by YouTube ASR.
# CONSERVATIVE list: every entry's left side must NOT be a real English word.
# Do NOT add " were "→" we're " (corrupts past-tense "were"),
#         " ill "→" I'll " (corrupts "ill" = sick),
#         " hell "→" he'll " (corrupts "hell"), etc.
ASR_CORRECTIONS = {
    " dont ":   " don't ",
    " wont ":   " won't ",
    " cant ":   " can't ",
    " ive ":    " I've ",
    " im ":     " I'm ",
    " thats ":  " that's ",
    " id ":     " I'd ",
    " youd ":   " you'd ",
    " theyd ":  " they'd ",
    " weve ":   " we've ",
    " theres ": " there's ",
    " wheres ": " where's ",
    " whats ":  " what's ",
    " youre ":  " you're ",
    " theyre ": " they're ",
    " youll ":  " you'll ",
    " theyll ": " they'll ",
    # NOTE: " its " is intentionally NOT corrected — "its" (possessive)
    # is a real word, "it's" (it is) is grammatically distinct.
}

# Filler words / phrases removed during cleanup. Multi-word phrases
# must be processed BEFORE single words so "you know" doesn't leave
# orphaned "know" behind.
FILLER_PHRASES = (
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "so basically",
    "i think",
)
FILLER_WORDS = (
    "um",
    "uh",
    "ah",
    "er",
    "like",
    "basically",
    "actually",
    "literally",
    "honestly",
)


# ════════════════════════════════════════════════════════════════
# I/O HELPERS
# ════════════════════════════════════════════════════════════════

def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Bỏ qua dòng JSON lỗi: %s", e)


def write_jsonl(records: List[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ════════════════════════════════════════════════════════════════
# TEXT CLEANING (Bug 3 fix: filler removal now actually applied)
# ════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Clean ASR/transcript text:
      1. Apply contraction fixes (dont → don't)
      2. Remove filler phrases first, then filler words
      3. Strip non-essential characters (keep letters/digits/space/.,!?')
      4. Collapse whitespace, capitalize sentence start.
    """
    if not text:
        return ""

    # Pad with spaces so word-boundary regexes work at start/end
    padded = " " + text + " "

    # 1. ASR contraction fixes
    for wrong, correct in ASR_CORRECTIONS.items():
        padded = padded.replace(wrong, correct)
        padded = padded.replace(wrong.upper(), correct)
        padded = padded.replace(wrong.title(), correct.title())

    # 2. Filler removal — phrases first (case-insensitive, word-bounded)
    for phrase in FILLER_PHRASES:
        padded = re.sub(rf"\b{re.escape(phrase)}\b", " ", padded, flags=re.IGNORECASE)
    for word in FILLER_WORDS:
        padded = re.sub(rf"\b{re.escape(word)}\b", " ", padded, flags=re.IGNORECASE)

    # 3. Strip non-essential chars (keep letters/digits/space/.,!?')
    padded = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", padded)

    # 4. Collapse whitespace
    padded = re.sub(r"\s+", " ", padded).strip()

    # 5. Capitalize first letter if it's lowercase
    if padded and padded[0].islower():
        padded = padded[0].upper() + padded[1:]

    return padded


def token_count(text: str) -> int:
    """Estimate token count: ~1.3 tokens per English word."""
    return int(round(len(text.split()) * 1.3))


def is_complete_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text.strip()))


# ════════════════════════════════════════════════════════════════
# PASS 1: TIME-BASED CHUNKING
# ════════════════════════════════════════════════════════════════

def build_time_chunks(record: dict) -> List[dict]:
    """
    Walk a video's caption segments in time order, pack into chunks.

    Close conditions (any one triggers close):
      A. Current segment ends with .!? AND tokens >= MIN_TOKENS
         (sentence-aware close — preferred)
      B. tokens >= MAX_TOKENS + buffer (hard ceiling)
      C. duration >= MAX_DURATION (Bug 5 fix — was previously unenforced)
      D. tokens >= MIN_TOKENS, duration >= MIN_DURATION, AND silence gap > 3s
         (natural topic break)
      E. Last segment of video

    Bug 7 fix: instead of silently dropping tail chunks below MIN_TOKENS,
    we keep them and tag chunk_type="tail_short". They can be filtered
    downstream if undesired.
    """
    transcript = record.get("transcript", [])
    if not transcript:
        return []

    chunks: List[dict] = []
    current_segs: List[dict] = []
    current_tokens = 0
    chunk_start: Optional[float] = None
    current_end: Optional[float] = None

    for idx, seg in enumerate(transcript):
        raw_text = (seg.get("text") or "").strip()
        if not raw_text:
            continue

        text = clean_text(raw_text)
        if not text:
            continue

        seg_obj = {
            "text": text,
            "start": float(seg["start"]),
            "end": float(seg["start"]) + float(seg.get("duration", 0.0)),
            "tokens": token_count(text),
        }

        if chunk_start is None:
            chunk_start = seg_obj["start"]

        current_segs.append(seg_obj)
        current_tokens += seg_obj["tokens"]
        current_end = seg_obj["end"]
        duration = current_end - chunk_start

        next_seg = transcript[idx + 1] if idx + 1 < len(transcript) else None

        should_close = False

        # A. Sentence-aware close: prefer this when we have enough content
        if current_tokens >= MIN_TOKENS and is_complete_sentence(seg_obj["text"]):
            should_close = True
        # B. Hard token ceiling
        elif current_tokens >= MAX_TOKENS + MAX_TOKEN_BUFFER:
            should_close = True
        # C. Hard duration ceiling (Bug 5 fix)
        elif duration >= MAX_DURATION:
            should_close = True
        # D. Natural break: enough content + silence gap
        elif (
            current_tokens >= MIN_TOKENS
            and duration >= MIN_DURATION
            and next_seg is not None
        ):
            gap = float(next_seg["start"]) - current_end
            if gap > SILENCE_GAP_CLOSE:
                should_close = True
        # E. Last segment of video
        elif next_seg is None:
            should_close = True

        if should_close:
            chunk_text = clean_text(" ".join(s["text"] for s in current_segs))
            tokens_in_chunk = token_count(chunk_text)

            if chunk_text:
                # Bug 7: keep short tail chunks but flag them
                chunk_kind = (
                    "knowledge_unit" if tokens_in_chunk >= MIN_TOKENS else "tail_short"
                )

                chunks.append({
                    "video_id": record["video_id"],
                    "title": record.get("title", ""),
                    "course": record.get("course", ""),
                    "playlist_id": record.get("playlist_id", ""),
                    "published_at": record.get("published_at", ""),
                    "source": record.get("source", ""),
                    "chunk_type": chunk_kind,
                    "chunk_text": chunk_text,
                    "start_time": round(chunk_start, 3),
                    "end_time": round(current_end, 3),
                    "duration": round(duration, 3),
                    "token_count": tokens_in_chunk,
                })

            current_segs = []
            current_tokens = 0
            chunk_start = None
            current_end = None

    return chunks


# ════════════════════════════════════════════════════════════════
# PASS 2: EMBEDDING-BASED MERGE (the real "semantic" step)
# ════════════════════════════════════════════════════════════════

@dataclass
class MergeStats:
    """Diagnostic counters for one Pass-2 invocation."""
    input_chunks: int = 0
    output_chunks: int = 0
    merges_performed: int = 0
    skipped_size: int = 0       # would have merged but combined too big
    skipped_duration: int = 0   # would have merged but duration cap
    similarities: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.similarities is None:
            self.similarities = []

    def reduction_pct(self) -> float:
        if self.input_chunks == 0:
            return 0.0
        return (1.0 - self.output_chunks / self.input_chunks) * 100


def _can_merge(prev: dict, nxt: dict, similarity: float, threshold: float,
               stats: MergeStats) -> bool:
    """Gate function for merging two adjacent chunks."""
    if similarity < threshold:
        return False

    combined_tokens = prev["token_count"] + nxt["token_count"]
    if combined_tokens > MERGE_MAX_TOKENS:
        stats.skipped_size += 1
        return False

    combined_duration = nxt["end_time"] - prev["start_time"]
    if combined_duration > MERGE_MAX_DURATION:
        stats.skipped_duration += 1
        return False

    return True


def _merge_two(prev: dict, nxt: dict) -> dict:
    """Merge two adjacent (in time) chunks. Caller must ensure adjacency."""
    merged_text = clean_text(prev["chunk_text"] + " " + nxt["chunk_text"])
    return {
        **prev,
        "chunk_text": merged_text,
        "end_time": nxt["end_time"],
        "duration": round(nxt["end_time"] - prev["start_time"], 3),
        "token_count": token_count(merged_text),
        "chunk_type": "semantic_unit",
    }


def merge_by_embedding(
    chunks: List[dict],
    embedder,
    threshold: float = DEFAULT_THRESHOLD,
    stats: Optional[MergeStats] = None,
) -> List[dict]:
    """
    Pass 2 — the actual semantic merge.

    Walks chunks IN TIME ORDER (the input list is assumed to be in time order
    per video, which build_time_chunks guarantees). For each adjacent pair,
    computes cosine similarity from bge-m3 embeddings. Merges if similar
    enough AND combined size within bounds.

    No topic grouping. No cross-time merging. No reordering.

    The comparison anchor advances after each merge — i.e. after merging
    A+B, the next decision uses B's vector (most recent topic anchor) vs
    C's vector. This is the standard "moving window" approach and avoids
    drift from re-embedding the merged text repeatedly.
    """
    if stats is None:
        stats = MergeStats()
    stats.input_chunks = len(chunks)

    if len(chunks) <= 1:
        stats.output_chunks = len(chunks)
        return chunks

    # Embed in one batch — bge-m3 with normalize_embeddings=True returns
    # unit-norm vectors so dot product == cosine similarity.
    texts = [c["chunk_text"] for c in chunks]
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError("numpy is required for embedding merge") from e

    vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)

    merged: List[dict] = [chunks[0]]
    last_vec = vectors[0]

    for i in range(1, len(chunks)):
        sim = float(np.dot(last_vec, vectors[i]))
        stats.similarities.append(sim)

        if _can_merge(merged[-1], chunks[i], sim, threshold, stats):
            merged[-1] = _merge_two(merged[-1], chunks[i])
            stats.merges_performed += 1
            last_vec = vectors[i]    # advance comparison anchor
        else:
            merged.append(chunks[i])
            last_vec = vectors[i]

    stats.output_chunks = len(merged)
    return merged


# ════════════════════════════════════════════════════════════════
# CHUNK ID + URL ENRICHMENT (post-merge)
# ════════════════════════════════════════════════════════════════

def assign_ids_and_urls(chunks: List[dict], version: str = "v3") -> List[dict]:
    """
    Add chunk_id (deterministic), chunk_index, and url to each chunk.

    chunk_id format: {video_id}:{kind}:{version}:{index:04d}
    Deterministic across re-runs so FAISS docstore can be rebuilt without
    invalidating cached references downstream.

    Operates on a list of chunks for ONE video (assumes same video_id),
    assigns chunk_index by position in the list (which is time order
    after Pass 2).
    """
    out = []
    for i, c in enumerate(chunks):
        kind = c.get("chunk_type", "knowledge_unit")
        chunk_id = f"{c['video_id']}:{kind}:{version}:{i:04d}"
        url = make_youtube_url(c["video_id"], c.get("start_time", 0.0))
        out.append({
            "chunk_id": chunk_id,
            "chunk_index": i,
            **c,
            "url": url,
        })
    return out


# ════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════

def process_corpus(
    input_path: Path,
    output_path: Path,
    threshold: float,
    do_merge: bool,
    limit: Optional[int],
) -> Tuple[int, int, MergeStats]:
    """
    Run the full pipeline: time chunking → optional embedding merge → write.

    Returns: (videos_processed, chunks_written, total_merge_stats)
    """
    embedder = None
    if do_merge:
        logger.info("Đang tải mô hình AI nhẹ (all-MiniLM-L6-v2) để gộp chunk (Semantic Chunking) siêu tốc...")
        from sentence_transformers import SentenceTransformer
        
        # Dùng model nhẹ 80MB thay vì model nặng 2GB để tiết kiệm 6 tiếng đồng hồ!
        fast_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        
        class FastEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # Tắt thanh progress bar trong từng video vì nó quá nhanh
                embeddings = fast_model.encode(texts, batch_size=32, show_progress_bar=False)
                return embeddings.tolist()
                
        embedder = FastEmbedder()
        logger.info("Mô hình Semantic Chunking đã sẵn sàng!")

    total_stats = MergeStats()
    all_chunks: List[dict] = []
    videos_processed = 0
    t0 = time.time()

    for record in iter_jsonl(input_path):
        if limit is not None and videos_processed >= limit:
            break
        videos_processed += 1
        vid = record.get("video_id", "?")
        title = record.get("title", "")[:60]

        # Pass 1: time chunks
        time_chunks = build_time_chunks(record)
        if not time_chunks:
            logger.warning("Video %s không có chunks (rỗng?). Bỏ qua.", vid)
            continue

        # Pass 2: embedding merge (optional)
        if do_merge:
            video_stats = MergeStats()
            chunks = merge_by_embedding(time_chunks, embedder, threshold, video_stats)
            total_stats.input_chunks += video_stats.input_chunks
            total_stats.output_chunks += video_stats.output_chunks
            total_stats.merges_performed += video_stats.merges_performed
            total_stats.skipped_size += video_stats.skipped_size
            total_stats.skipped_duration += video_stats.skipped_duration
            total_stats.similarities.extend(video_stats.similarities)
            logger.info(
                "[%d] %s | %s — Pass1: %d → Pass2: %d (-%.1f%%, %d merges)",
                videos_processed, vid, title,
                video_stats.input_chunks, video_stats.output_chunks,
                video_stats.reduction_pct(), video_stats.merges_performed,
            )
        else:
            chunks = time_chunks
            total_stats.input_chunks += len(chunks)
            total_stats.output_chunks += len(chunks)
            logger.info(
                "[%d] %s | %s — %d time chunks (no merge)",
                videos_processed, vid, title, len(chunks),
            )

        # Add deterministic IDs + URLs
        chunks = assign_ids_and_urls(chunks)
        all_chunks.extend(chunks)

    written = write_jsonl(all_chunks, output_path)
    elapsed = time.time() - t0
    logger.info(
        "Hoàn tất. Videos: %d, Chunks: %d, Output: %s, Thời gian: %.1fs",
        videos_processed, written, output_path, elapsed,
    )
    return videos_processed, written, total_stats


def print_summary(stats: MergeStats, threshold: float) -> None:
    """Print Pass-2 diagnostic summary."""
    if stats.input_chunks == 0:
        print("\n(Không có chunks để báo cáo)")
        return

    print("\n" + "═" * 60)
    print(f"  EMBEDDING MERGE SUMMARY — threshold={threshold:.2f}")
    print("═" * 60)
    print(f"  Time chunks (Pass 1)   : {stats.input_chunks:>6}")
    print(f"  Semantic chunks (Pass 2): {stats.output_chunks:>6}")
    print(f"  Reduction              : {stats.reduction_pct():>5.1f}%")
    print(f"  Merges performed       : {stats.merges_performed:>6}")
    print(f"  Skipped (size cap)     : {stats.skipped_size:>6}")
    print(f"  Skipped (duration cap) : {stats.skipped_duration:>6}")

    if stats.similarities:
        import statistics
        sims = stats.similarities
        print()
        print(f"  Similarity distribution ({len(sims)} adjacent pairs):")
        print(f"    min    : {min(sims):.3f}")
        print(f"    p25    : {statistics.quantiles(sims, n=4)[0]:.3f}")
        print(f"    median : {statistics.median(sims):.3f}")
        print(f"    p75    : {statistics.quantiles(sims, n=4)[2]:.3f}")
        print(f"    max    : {max(sims):.3f}")
        print(f"    mean   : {statistics.mean(sims):.3f}")
        above = sum(1 for s in sims if s >= threshold)
        print(
            f"  Pairs with sim >= {threshold:.2f}: {above} "
            f"({above / len(sims) * 100:.1f}%)"
        )
    print("═" * 60 + "\n")


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build knowledge-unit chunks (time-chunk + embedding merge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=INPUT_PATH,
        help=f"Input cleaned-sentence JSONL (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=f"Output JSONL filename (default: {DEFAULT_OUTPUT_NAME} in {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for merging (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip Pass 2 (embedding merge). Useful for debugging Pass 1 alone.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N videos (for quick testing).",
    )
    parser.add_argument(
        "--sweep", type=float, nargs="+", default=None,
        help=(
            "Sweep multiple thresholds, writing one output file per value. "
            "Example: --sweep 0.55 0.65 0.72 0.80"
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    # Sweep mode: run once per threshold, output to suffixed filenames
    if args.sweep:
        for t in args.sweep:
            out_name = f"transcript_v3_t{int(t * 100):03d}.jsonl"
            out_path = OUTPUT_DIR / out_name
            print(f"\n{'#' * 60}\n# Threshold = {t:.2f} → {out_name}\n{'#' * 60}")
            _, _, stats = process_corpus(
                args.input, out_path, t, do_merge=True, limit=args.limit,
            )
            print_summary(stats, t)
        return

    # Single run
    out_path = args.output or (OUTPUT_DIR / DEFAULT_OUTPUT_NAME)
    _, _, stats = process_corpus(
        args.input, out_path, args.threshold,
        do_merge=not args.no_merge, limit=args.limit,
    )
    if not args.no_merge:
        print_summary(stats, args.threshold)


if __name__ == "__main__":
    main()
