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
    return Pipeline2Config(
        root_dir=Path("D:/NLP"),
        input_path=Path("D:/NLP/data/chunked/transcript_v3.jsonl"),
        output_path=Path("D:/NLP/data/chunked/transcript_v4.jsonl"),
    )
