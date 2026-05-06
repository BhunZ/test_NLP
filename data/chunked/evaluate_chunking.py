"""
data/chunked/evaluate_chunking.py
═══════════════════════════════════════════════════════════════════════
Evaluate chunking quality WITHOUT needing labeled queries.

This is the "smell test" stage — it tells you if your chunks look healthy
before you invest in building a Chroma-style labeled eval set.

Inspired by Chroma's chunking research (https://www.trychroma.com/research/
evaluating-chunking) — but their methodology requires (question, ground-truth
text span) pairs that don't exist for this corpus yet. So we use proxy metrics
that work on the chunked file alone.

THREE CATEGORIES OF CHECKS
──────────────────────────

A. STRUCTURAL HEALTH (fast, no embedder needed)
   - Token / duration / word distributions
   - % of chunks violating MIN/MAX bounds
   - Coverage: total tokens in vs out (did cleaning lose content?)
   - Per-video chunk count distribution

B. BOUNDARY QUALITY (needs embedder)
   - For each pair of consecutive chunks IN THE SAME VIDEO, compute cosine
     similarity using the embedder.
   - Distribution of these similarities.
   - HIGH sim (≥ 0.85): chunks should probably have been merged.
   - LOW sim (≤ 0.30): natural topic boundaries — that's fine.
   - SOMEWHERE-IN-THE-MIDDLE is healthiest.

C. INTRA-CHUNK COHERENCE (needs embedder)
   - For each chunk, split into two halves at the sentence-count midpoint.
   - Compute cosine similarity between halves.
   - LOW sim means the chunk spans unrelated content (either over-merged
     or chunked across a topic boundary). These are the chunks that will
     hurt retrieval most — they look like one chunk but contain multiple
     topics, so query embeddings get pulled to a "smeared" centroid.

OUTPUT
──────
- Console report with verdicts
- (optional) outliers.jsonl: chunks/pairs flagged for human inspection
- (optional) compare two output files side-by-side

USAGE
─────
    # Structural only (no embedder load — runs in seconds)
    python -m data.chunked.evaluate_chunking transcript_v3.jsonl --structural-only

    # Full eval with default embedder (all-MiniLM-L6-v2, ~80 MB)
    python -m data.chunked.evaluate_chunking transcript_v3.jsonl

    # Custom embedder
    python -m data.chunked.evaluate_chunking transcript_v3.jsonl \\
        --embedder sentence-transformers/all-MiniLM-L6-v2

    # Save flagged outliers for human review
    python -m data.chunked.evaluate_chunking transcript_v3.jsonl \\
        --outliers outliers_to_review.jsonl

    # Compare two chunking strategies (e.g. two thresholds) side by side
    python -m data.chunked.evaluate_chunking transcript_v3_t055.jsonl \\
        --compare transcript_v3_t072.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 on Windows console so block-drawing/check-mark chars render.
# Without this, default cp1252 console crashes on the report.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# Make `from ...` imports work when run as `python -m`
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# Healthy-range thresholds. Tune these as your team learns what "good" means.
TARGET_TOKENS_MIN = 100
TARGET_TOKENS_MAX = 250
TARGET_DURATION_MAX = 120.0    # seconds
HIGH_BOUNDARY_SIM = 0.85        # adjacent chunks this similar = under-merged
LOW_COHERENCE_SIM = 0.40        # chunk halves below this = over-merged
SOFT_HIGH_BOUNDARY = 0.75       # warning level


# ════════════════════════════════════════════════════════════════
# I/O
# ════════════════════════════════════════════════════════════════

def load_chunks(path: Path) -> List[dict]:
    chunks: List[dict] = []
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


def write_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_into_sentences(text: str) -> List[str]:
    """Naive sentence split — good enough for half-vs-half coherence."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def halve_chunk(text: str) -> Tuple[str, str]:
    """Split chunk text into two halves at sentence midpoint.

    Falls back to word midpoint for chunks with only one sentence.
    """
    sents = split_into_sentences(text)
    if len(sents) >= 2:
        mid = len(sents) // 2
        return " ".join(sents[:mid]), " ".join(sents[mid:])
    # Fallback: split by words
    words = text.split()
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]), " ".join(words[mid:])


# ════════════════════════════════════════════════════════════════
# STATS HELPERS
# ════════════════════════════════════════════════════════════════

@dataclass
class DistributionSummary:
    """Min/max/mean/median/quartiles for a numeric series."""
    n: int
    min_: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float
    p99: float
    max_: float
    mean: float

    @classmethod
    def of(cls, values: List[float]) -> Optional["DistributionSummary"]:
        if not values:
            return None
        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def pct(p: float) -> float:
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return sorted_vals[idx]

        return cls(
            n=n,
            min_=sorted_vals[0],
            p10=pct(0.10),
            p25=pct(0.25),
            median=statistics.median(sorted_vals),
            p75=pct(0.75),
            p90=pct(0.90),
            p99=pct(0.99),
            max_=sorted_vals[-1],
            mean=statistics.mean(sorted_vals),
        )

    def render(self, name: str, fmt: str = "{:.1f}") -> str:
        return (
            f"  {name:<22} n={self.n:>6}  "
            f"min={fmt.format(self.min_)}  "
            f"p10={fmt.format(self.p10)}  "
            f"p25={fmt.format(self.p25)}  "
            f"median={fmt.format(self.median)}  "
            f"p75={fmt.format(self.p75)}  "
            f"p90={fmt.format(self.p90)}  "
            f"max={fmt.format(self.max_)}  "
            f"mean={fmt.format(self.mean)}"
        )


def histogram(values: List[float], bins: int = 10, width: int = 40) -> str:
    """Tiny ASCII histogram for the console."""
    if not values:
        return "  (no data)"
    lo, hi = min(values), max(values)
    if hi == lo:
        return f"  all values = {lo:.3f}"
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(c / peak * width)
        lines.append(f"    [{edges[i]:.2f}, {edges[i+1]:.2f})  {bar} {c}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# A. STRUCTURAL HEALTH
# ════════════════════════════════════════════════════════════════

@dataclass
class StructuralReport:
    total_chunks: int
    total_videos: int
    chunks_per_video: DistributionSummary
    token_dist: DistributionSummary
    duration_dist: DistributionSummary
    word_dist: DistributionSummary
    chunk_type_counts: Dict[str, int]
    too_short: int            # count below TARGET_TOKENS_MIN
    too_long_tokens: int      # count above TARGET_TOKENS_MAX
    too_long_duration: int    # count above TARGET_DURATION_MAX
    chunks_missing_id: int
    chunks_missing_url: int
    duplicate_chunk_ids: int


def analyze_structure(chunks: List[dict]) -> StructuralReport:
    if not chunks:
        raise ValueError("Empty chunks list — nothing to analyze.")

    by_video: Dict[str, int] = defaultdict(int)
    chunk_types: Dict[str, int] = defaultdict(int)
    tokens: List[int] = []
    durations: List[float] = []
    words: List[int] = []
    seen_ids = set()
    dup_ids = 0
    missing_id = 0
    missing_url = 0
    too_short = too_long_t = too_long_d = 0

    for c in chunks:
        by_video[c.get("video_id", "?")] += 1
        chunk_types[c.get("chunk_type", "unknown")] += 1

        tk = int(c.get("token_count", 0))
        tokens.append(tk)
        if tk < TARGET_TOKENS_MIN:
            too_short += 1
        if tk > TARGET_TOKENS_MAX:
            too_long_t += 1

        d = float(c.get("duration", 0.0))
        durations.append(d)
        if d > TARGET_DURATION_MAX:
            too_long_d += 1

        words.append(len(c.get("chunk_text", "").split()))

        cid = c.get("chunk_id")
        if not cid:
            missing_id += 1
        else:
            if cid in seen_ids:
                dup_ids += 1
            seen_ids.add(cid)

        if not c.get("url"):
            missing_url += 1

    return StructuralReport(
        total_chunks=len(chunks),
        total_videos=len(by_video),
        chunks_per_video=DistributionSummary.of([float(v) for v in by_video.values()]),
        token_dist=DistributionSummary.of([float(t) for t in tokens]),
        duration_dist=DistributionSummary.of(durations),
        word_dist=DistributionSummary.of([float(w) for w in words]),
        chunk_type_counts=dict(chunk_types),
        too_short=too_short,
        too_long_tokens=too_long_t,
        too_long_duration=too_long_d,
        chunks_missing_id=missing_id,
        chunks_missing_url=missing_url,
        duplicate_chunk_ids=dup_ids,
    )


def render_structural(r: StructuralReport) -> List[str]:
    lines = []
    lines.append("─" * 70)
    lines.append("A. STRUCTURAL HEALTH")
    lines.append("─" * 70)
    lines.append(f"  Total chunks         : {r.total_chunks}")
    lines.append(f"  Total videos         : {r.total_videos}")
    lines.append(f"  Avg chunks/video     : {r.chunks_per_video.mean:.1f}")
    lines.append("")
    lines.append("  Chunk-type breakdown:")
    for k, v in sorted(r.chunk_type_counts.items(), key=lambda x: -x[1]):
        pct = v / r.total_chunks * 100
        lines.append(f"    {k:<22} {v:>6} ({pct:5.1f}%)")
    lines.append("")
    lines.append("  Distributions:")
    lines.append(r.token_dist.render("tokens"))
    lines.append(r.duration_dist.render("duration (s)"))
    lines.append(r.word_dist.render("words"))
    lines.append(r.chunks_per_video.render("chunks/video"))
    lines.append("")
    lines.append("  Bound violations:")
    pct_short = r.too_short / r.total_chunks * 100
    pct_long_t = r.too_long_tokens / r.total_chunks * 100
    pct_long_d = r.too_long_duration / r.total_chunks * 100
    lines.append(
        f"    < {TARGET_TOKENS_MIN} tokens     : {r.too_short:>5} ({pct_short:5.1f}%)"
        + ("  ⚠️  too many short chunks" if pct_short > 15 else "  ✓")
    )
    lines.append(
        f"    > {TARGET_TOKENS_MAX} tokens     : {r.too_long_tokens:>5} ({pct_long_t:5.1f}%)"
        + ("  ⚠️  oversized chunks" if pct_long_t > 5 else "  ✓")
    )
    lines.append(
        f"    > {TARGET_DURATION_MAX:.0f}s duration  : {r.too_long_duration:>5} ({pct_long_d:5.1f}%)"
        + ("  ⚠️  citation imprecise" if pct_long_d > 5 else "  ✓")
    )
    lines.append("")
    lines.append("  Metadata health:")
    lines.append(
        f"    missing chunk_id     : {r.chunks_missing_id:>5}"
        + ("  ⚠️" if r.chunks_missing_id else "  ✓")
    )
    lines.append(
        f"    duplicate chunk_id   : {r.duplicate_chunk_ids:>5}"
        + ("  ⚠️" if r.duplicate_chunk_ids else "  ✓")
    )
    lines.append(
        f"    missing url          : {r.chunks_missing_url:>5}"
        + ("  ⚠️" if r.chunks_missing_url else "  ✓")
    )
    return lines


# ════════════════════════════════════════════════════════════════
# B. BOUNDARY QUALITY (needs embedder)
# C. INTRA-CHUNK COHERENCE (needs embedder)
# ════════════════════════════════════════════════════════════════

def load_embedder(model_name: str):
    """Lazy import so structural-only mode doesn't pay the load cost."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Need sentence-transformers. Install: pip install sentence-transformers"
        ) from e
    logger.info("Loading embedder %s ...", model_name)
    return SentenceTransformer(model_name)


def cosine_sim(a, b) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def analyze_boundaries(
    chunks: List[dict], embedder
) -> Tuple[List[float], List[Tuple[float, dict, dict]]]:
    """
    For each pair of consecutive chunks IN THE SAME VIDEO, compute cosine sim.

    Returns: (list_of_sims, top_hot_pairs) where hot_pairs are pairs above
    HIGH_BOUNDARY_SIM (likely-should-have-been-merged candidates).
    """
    import numpy as np

    # Embed all chunk texts in batches
    texts = [c.get("chunk_text", "") for c in chunks]
    logger.info("Embedding %d chunks for boundary analysis...", len(texts))
    vecs = embedder.encode(
        texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )

    sims: List[float] = []
    hot_pairs: List[Tuple[float, dict, dict]] = []
    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        if a.get("video_id") != b.get("video_id"):
            continue
        sim = cosine_sim(vecs[i], vecs[i + 1])
        sims.append(sim)
        if sim >= HIGH_BOUNDARY_SIM:
            hot_pairs.append((sim, a, b))

    hot_pairs.sort(key=lambda x: -x[0])
    return sims, hot_pairs


def analyze_coherence(
    chunks: List[dict], embedder
) -> Tuple[List[float], List[Tuple[float, dict]]]:
    """
    For each chunk, split into halves and compute half-vs-half cosine sim.

    Returns: (list_of_sims, low_coherence_chunks) where low_coherence chunks
    are below LOW_COHERENCE_SIM (likely-over-merged candidates).
    """
    import numpy as np

    halves: List[str] = []
    for c in chunks:
        a, b = halve_chunk(c.get("chunk_text", ""))
        halves.append(a)
        halves.append(b)

    logger.info("Embedding %d halves for coherence analysis...", len(halves))
    vecs = embedder.encode(
        halves, batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )

    sims: List[float] = []
    bad: List[Tuple[float, dict]] = []
    for i, c in enumerate(chunks):
        # Skip chunks too short to split meaningfully
        if len(c.get("chunk_text", "").split()) < 30:
            continue
        sim = cosine_sim(vecs[2 * i], vecs[2 * i + 1])
        sims.append(sim)
        if sim < LOW_COHERENCE_SIM:
            bad.append((sim, c))

    bad.sort(key=lambda x: x[0])  # worst first
    return sims, bad


def render_semantic(
    boundary_sims: List[float],
    hot_pairs: List[Tuple[float, dict, dict]],
    coherence_sims: List[float],
    bad_coherence: List[Tuple[float, dict]],
) -> List[str]:
    lines = []
    lines.append("")
    lines.append("─" * 70)
    lines.append("B. BOUNDARY QUALITY (adjacent-chunk similarity)")
    lines.append("─" * 70)
    if not boundary_sims:
        lines.append("  No same-video adjacent pairs to evaluate.")
    else:
        bd = DistributionSummary.of(boundary_sims)
        lines.append(bd.render("adj. cosine sim", fmt="{:.3f}"))
        lines.append("")
        lines.append("  Distribution:")
        lines.append(histogram(boundary_sims, bins=10))
        lines.append("")

        n_high = sum(1 for s in boundary_sims if s >= HIGH_BOUNDARY_SIM)
        n_warn = sum(1 for s in boundary_sims if SOFT_HIGH_BOUNDARY <= s < HIGH_BOUNDARY_SIM)
        n_low = sum(1 for s in boundary_sims if s < 0.30)
        total = len(boundary_sims)
        lines.append(
            f"  Pairs sim ≥ {HIGH_BOUNDARY_SIM:.2f} (under-merged?)  : "
            f"{n_high:>5} ({n_high/total*100:5.1f}%)"
            + ("  ⚠️  many adjacent pairs almost identical" if n_high / total > 0.10 else "  ✓")
        )
        lines.append(
            f"  Pairs sim ≥ {SOFT_HIGH_BOUNDARY:.2f} (warning)        : "
            f"{n_warn:>5} ({n_warn/total*100:5.1f}%)"
        )
        lines.append(
            f"  Pairs sim < 0.30 (clean topic break)  : "
            f"{n_low:>5} ({n_low/total*100:5.1f}%)  — these are healthy"
        )
        lines.append("")
        if hot_pairs:
            lines.append(f"  Top-5 'should-have-merged' candidates (highest sim adjacent pairs):")
            for sim, a, b in hot_pairs[:5]:
                title = (a.get("title") or "?")[:50]
                lines.append(
                    f"    sim={sim:.3f}  {title}  "
                    f"[{a.get('start_time', 0):.0f}s] → [{b.get('start_time', 0):.0f}s]"
                )

    lines.append("")
    lines.append("─" * 70)
    lines.append("C. INTRA-CHUNK COHERENCE (chunk-half similarity)")
    lines.append("─" * 70)
    if not coherence_sims:
        lines.append("  Not enough chunks long enough to evaluate.")
    else:
        cd = DistributionSummary.of(coherence_sims)
        lines.append(cd.render("half-half sim", fmt="{:.3f}"))
        lines.append("")
        lines.append("  Distribution:")
        lines.append(histogram(coherence_sims, bins=10))
        lines.append("")

        n_bad = sum(1 for s in coherence_sims if s < LOW_COHERENCE_SIM)
        total = len(coherence_sims)
        lines.append(
            f"  Chunks half-sim < {LOW_COHERENCE_SIM:.2f} (over-merged?) : "
            f"{n_bad:>5} ({n_bad/total*100:5.1f}%)"
            + ("  ⚠️  chunks span unrelated content" if n_bad / total > 0.10 else "  ✓")
        )
        lines.append("")
        if bad_coherence:
            lines.append(f"  Top-5 'should-have-split' candidates (lowest half-sim chunks):")
            for sim, c in bad_coherence[:5]:
                title = (c.get("title") or "?")[:50]
                preview = c.get("chunk_text", "")[:80].replace("\n", " ")
                lines.append(
                    f"    sim={sim:.3f}  {title}  [{c.get('start_time', 0):.0f}s]"
                )
                lines.append(f"            \"{preview}...\"")
    return lines


# ════════════════════════════════════════════════════════════════
# OVERALL VERDICT
# ════════════════════════════════════════════════════════════════

def render_verdict(
    s: StructuralReport,
    boundary_sims: Optional[List[float]],
    coherence_sims: Optional[List[float]],
) -> List[str]:
    lines = ["", "═" * 70, "OVERALL VERDICT", "═" * 70]
    issues: List[str] = []

    if s.too_short / s.total_chunks > 0.15:
        issues.append(
            f"• {s.too_short / s.total_chunks * 100:.1f}% of chunks are below "
            f"{TARGET_TOKENS_MIN} tokens — chunker is producing fragments."
        )
    if s.too_long_duration / s.total_chunks > 0.05:
        issues.append(
            f"• {s.too_long_duration / s.total_chunks * 100:.1f}% of chunks "
            f"exceed {TARGET_DURATION_MAX}s — citation will point to a too-wide window."
        )
    if s.duplicate_chunk_ids:
        issues.append(
            f"• {s.duplicate_chunk_ids} duplicate chunk_ids — index will collide."
        )

    if boundary_sims:
        n_high = sum(1 for s_ in boundary_sims if s_ >= HIGH_BOUNDARY_SIM)
        if n_high / len(boundary_sims) > 0.10:
            issues.append(
                f"• {n_high / len(boundary_sims) * 100:.1f}% of adjacent chunks are "
                f"≥ {HIGH_BOUNDARY_SIM:.2f} similar — chunker is UNDER-merging. "
                f"Try a lower threshold."
            )

    if coherence_sims:
        n_bad = sum(1 for s_ in coherence_sims if s_ < LOW_COHERENCE_SIM)
        if n_bad / len(coherence_sims) > 0.10:
            issues.append(
                f"• {n_bad / len(coherence_sims) * 100:.1f}% of chunks have "
                f"half-similarity below {LOW_COHERENCE_SIM:.2f} — chunker is "
                f"OVER-merging or splitting at bad boundaries. Try a higher "
                f"threshold or shorter MAX_TOKENS."
            )

    if not issues:
        lines.append("  ✓ Chunks look healthy. Nothing structurally alarming.")
        lines.append("    Next step: build a labeled query set (Chroma-style)")
        lines.append("    so you can measure RETRIEVAL quality, not just chunk shape.")
    else:
        lines.append("  Issues detected:")
        lines.extend(f"    {x}" for x in issues)
        lines.append("")
        lines.append("  Suggested actions:")
        if any("UNDER-merging" in x for x in issues):
            lines.append("    - Re-run knowledge_units.py with a LOWER --threshold (e.g. 0.55)")
        if any("OVER-merging" in x for x in issues):
            lines.append("    - Re-run knowledge_units.py with a HIGHER --threshold (e.g. 0.72)")
        if any("fragments" in x for x in issues):
            lines.append("    - Review MIN_TOKENS / sentence-close logic in Pass 1")
        if any("too-wide window" in x for x in issues):
            lines.append("    - Review MAX_DURATION enforcement in Pass 1")
    return lines


# ════════════════════════════════════════════════════════════════
# OUTLIER FILE
# ════════════════════════════════════════════════════════════════

def write_outliers(
    path: Path,
    hot_pairs: List[Tuple[float, dict, dict]],
    bad_coherence: List[Tuple[float, dict]],
    n_each: int = 30,
) -> None:
    """Write top outliers to a JSONL file for human review."""
    out = []
    for sim, a, b in hot_pairs[:n_each]:
        out.append({
            "kind": "high_boundary_sim",
            "similarity": round(sim, 4),
            "video_title": a.get("title", ""),
            "chunk_a_id": a.get("chunk_id", ""),
            "chunk_a_time": a.get("start_time"),
            "chunk_a_text": a.get("chunk_text", "")[:200],
            "chunk_b_id": b.get("chunk_id", ""),
            "chunk_b_time": b.get("start_time"),
            "chunk_b_text": b.get("chunk_text", "")[:200],
            "verdict": "These adjacent chunks are very similar — consider merging them.",
        })
    for sim, c in bad_coherence[:n_each]:
        out.append({
            "kind": "low_intra_coherence",
            "half_similarity": round(sim, 4),
            "video_title": c.get("title", ""),
            "chunk_id": c.get("chunk_id", ""),
            "start_time": c.get("start_time"),
            "duration": c.get("duration"),
            "chunk_text": c.get("chunk_text", ""),
            "verdict": (
                "Two halves of this chunk look unrelated — "
                "consider splitting it (lower max-tokens or higher merge threshold)."
            ),
        })
    write_jsonl(out, path)
    logger.info("Wrote %d outliers to %s", len(out), path)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def evaluate_one(path: Path, embedder_name: Optional[str]) -> Dict[str, Any]:
    chunks = load_chunks(path)
    if not chunks:
        raise ValueError(f"No chunks loaded from {path}")
    structural = analyze_structure(chunks)

    boundary_sims = coherence_sims = None
    hot_pairs: List = []
    bad_coherence: List = []
    if embedder_name:
        embedder = load_embedder(embedder_name)
        boundary_sims, hot_pairs = analyze_boundaries(chunks, embedder)
        coherence_sims, bad_coherence = analyze_coherence(chunks, embedder)

    return {
        "path": path,
        "chunks": chunks,
        "structural": structural,
        "boundary_sims": boundary_sims,
        "hot_pairs": hot_pairs,
        "coherence_sims": coherence_sims,
        "bad_coherence": bad_coherence,
    }


def render_one(eval_result: Dict[str, Any]) -> str:
    lines = []
    lines.append("\n" + "═" * 70)
    lines.append(f"EVALUATION: {eval_result['path']}")
    lines.append("═" * 70)
    lines.extend(render_structural(eval_result["structural"]))
    if eval_result["boundary_sims"] is not None:
        lines.extend(
            render_semantic(
                eval_result["boundary_sims"],
                eval_result["hot_pairs"],
                eval_result["coherence_sims"],
                eval_result["bad_coherence"],
            )
        )
    lines.extend(
        render_verdict(
            eval_result["structural"],
            eval_result["boundary_sims"],
            eval_result["coherence_sims"],
        )
    )
    return "\n".join(lines)


def render_compare(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """Side-by-side comparison table for two chunking outputs."""
    sa, sb = a["structural"], b["structural"]

    def fmt_pct(n: int, total: int) -> str:
        return f"{n / total * 100:5.1f}%"

    rows = [
        ("Total chunks", str(sa.total_chunks), str(sb.total_chunks)),
        ("Chunks/video median", f"{sa.chunks_per_video.median:.0f}",
                                f"{sb.chunks_per_video.median:.0f}"),
        ("Median tokens", f"{sa.token_dist.median:.0f}", f"{sb.token_dist.median:.0f}"),
        ("Median duration", f"{sa.duration_dist.median:.1f}s",
                            f"{sb.duration_dist.median:.1f}s"),
        ("% < 100 tokens",
            fmt_pct(sa.too_short, sa.total_chunks),
            fmt_pct(sb.too_short, sb.total_chunks)),
        ("% > 250 tokens",
            fmt_pct(sa.too_long_tokens, sa.total_chunks),
            fmt_pct(sb.too_long_tokens, sb.total_chunks)),
        ("% > 120s duration",
            fmt_pct(sa.too_long_duration, sa.total_chunks),
            fmt_pct(sb.too_long_duration, sb.total_chunks)),
    ]
    if a["boundary_sims"] and b["boundary_sims"]:
        bsA = DistributionSummary.of(a["boundary_sims"])
        bsB = DistributionSummary.of(b["boundary_sims"])
        rows.append(("Median adj. sim", f"{bsA.median:.3f}", f"{bsB.median:.3f}"))
        rows.append(
            ("% pairs sim ≥ 0.85",
                fmt_pct(sum(1 for s in a["boundary_sims"] if s >= 0.85), len(a["boundary_sims"])),
                fmt_pct(sum(1 for s in b["boundary_sims"] if s >= 0.85), len(b["boundary_sims"])))
        )
    if a["coherence_sims"] and b["coherence_sims"]:
        csA = DistributionSummary.of(a["coherence_sims"])
        csB = DistributionSummary.of(b["coherence_sims"])
        rows.append(("Median half-half sim", f"{csA.median:.3f}", f"{csB.median:.3f}"))
        rows.append(
            ("% chunks half-sim < 0.40",
                fmt_pct(sum(1 for s in a["coherence_sims"] if s < 0.40), len(a["coherence_sims"])),
                fmt_pct(sum(1 for s in b["coherence_sims"] if s < 0.40), len(b["coherence_sims"])))
        )

    lines = ["\n" + "═" * 70, "SIDE-BY-SIDE COMPARISON", "═" * 70]
    name_a = a["path"].name
    name_b = b["path"].name
    lines.append(f"  {'metric':<30} {name_a:>18} {name_b:>18}")
    lines.append(f"  {'-' * 30} {'-' * 18} {'-' * 18}")
    for label, va, vb in rows:
        lines.append(f"  {label:<30} {va:>18} {vb:>18}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate chunking quality (no labeled queries needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Path to chunked JSONL file")
    p.add_argument("--compare", type=Path, default=None,
                   help="Second file to compare against side-by-side")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                   help=f"sentence-transformers model name (default: {DEFAULT_EMBEDDER})")
    p.add_argument("--structural-only", action="store_true",
                   help="Skip the semantic checks (no embedder load — fast)")
    p.add_argument("--outliers", type=Path, default=None,
                   help="Write top outliers to this JSONL for human review")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input.exists():
        logger.error("File not found: %s", args.input)
        sys.exit(1)

    embedder_name = None if args.structural_only else args.embedder

    result_a = evaluate_one(args.input, embedder_name)
    print(render_one(result_a))

    if args.compare:
        if not args.compare.exists():
            logger.error("Compare file not found: %s", args.compare)
            sys.exit(1)
        result_b = evaluate_one(args.compare, embedder_name)
        print(render_one(result_b))
        print(render_compare(result_a, result_b))

    if args.outliers and not args.structural_only:
        write_outliers(
            args.outliers, result_a["hot_pairs"], result_a["bad_coherence"]
        )


if __name__ == "__main__":
    main()
