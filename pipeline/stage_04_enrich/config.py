"""
Configuration for Pipeline 2 - Cross-lingual Knowledge Units.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pipeline2Config:
    root_dir: Path
    input_path: Path
    output_path: Path

    # transcript_v3 is already chunked; these values document the intended size band
    min_tokens: int = 120
    target_tokens: int = 160
    max_tokens: int = 190
    hard_max_tokens: int = 215
    max_duration: float = 85.0
    min_duration: float = 18.0

    # embedding_text controls
    max_keywords: int = 6
    min_keyword_length: int = 3
    enable_vi_hints: bool = True
    include_time_header: bool = True
    max_embedding_tokens: int = 240

    output_fields: tuple = (
        "chunk_id",
        "chunk_index",
        "video_id",
        "title",
        "course",
        "source",
        "chunk_type",
        "semantic_topic",
        "chunk_text",
        "embedding_text",
        "start_time",
        "end_time",
        "duration",
        "token_count",
    )


def build_pipeline2_config() -> Pipeline2Config:
    # Project-relative paths (was hardcoded D:/NLP → broken on any other machine).
    # pipeline/stage_04_enrich/config.py → repo root is parents[2]
    root = Path(__file__).resolve().parents[2]
    return Pipeline2Config(
        root_dir=root,
        # t072 = ngưỡng embedding-merge 0.72, bản cho retrieval tốt nhất trong sweep
        # (0.55 / 0.65 / 0.72 / 0.80). Cùng bản mà ask.py và orchestrator.py đang dùng.
        input_path=root / "data" / "chunked" / "transcript_v3_t072.jsonl",
        output_path=root / "data" / "chunked" / "transcript_v4.jsonl",
    )
