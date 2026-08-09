"""
Pipeline 2 - Cross-lingual Knowledge Units.

Buoc nay khong tao lai chunk tu transcript.
No chi enrich metadata cho cac chunk hien co va ghi ra transcript_v4 Chu:
  - Tao embedding_text voi Course, Lecture, Topic, Keywords, Keywords (VI)
  - Them chunk_id va chunk_index
  - Chuyen doi chunk_type

Input: transcript_v3.jsonl
Output: transcript_v4.jsonl
"""

from collections import defaultdict

from config import build_pipeline2_config
from utils import (
    detect_topic,
    extract_keywords,
    format_timestamp,
    get_vi_keywords,
    iter_jsonl,
    token_count,
    write_jsonl,
)

# Tao embedding_text cho RAG/similarity
# Format:
# Course: {course}
# Lecture: {title}
# Time: {start} - {end}
# Topic: {topic}
# Keywords: {keywords}
# Keywords (Vietnamese): {vi_keywords}
# Content: {chunk_text}
def build_crosslingual_embedding(record: dict, config) -> str:
    chunk_text = (record.get("chunk_text") or "").strip()
    course = record.get("course", "Unknown")
    title = record.get("title", "Unknown")

    # Lay topic tu transcript_v3 hoac phat hien neu chua co
    topic = record.get("semantic_topic") or detect_topic(chunk_text)

    # Trich xuat keywords tu chunk_text
    keywords = extract_keywords(
        chunk_text,
        max_keywords=config.max_keywords,
        min_length=config.min_keyword_length,
    )

    # Dich keywords sang tieng Viet
    vi_keywords = get_vi_keywords(keywords) if config.enable_vi_hints else []

    # Tao line Time neu can
    time_line = None
    if config.include_time_header:
        time_line = (
            f"Time: {format_timestamp(record.get('start_time', 0.0))}"
            f" - {format_timestamp(record.get('end_time', 0.0))}"
        )

    # Ham tao tung dong trong embedding_text
    def build_lines(
        active_keywords: list[str], active_vi_keywords: list[str]
    ) -> list[str]:
        lines = [
            f"Course: {course}",
            f"Lecture: {title}",
        ]
        # Them Time neu co
        if time_line:
            lines.append(time_line)
        # Them Topic
        lines.append(f"Topic: {topic}")
        # Them Keywords
        if active_keywords:
            lines.append(f"Keywords: {', '.join(active_keywords)}")
        # Them Keywords (Vietnamese)
        if active_vi_keywords:
            lines.append(f"Keywords (Vietnamese): {', '.join(active_vi_keywords)}")
        # Them Content
        lines.append("")
        lines.append("Content:")
        lines.append(chunk_text)
        return lines

    # Tao embedding_text
    active_keywords = list(keywords)
    active_vi_keywords = list(vi_keywords)
    lines = build_lines(active_keywords, active_vi_keywords)
    embedding_text = "\n".join(lines)

    # Neu vuot max_embedding_tokens, loai bo VI keywords truoc
    if token_count(embedding_text) > config.max_embedding_tokens and active_vi_keywords:
        active_vi_keywords = []
        lines = build_lines(active_keywords, active_vi_keywords)
        embedding_text = "\n".join(lines)

    # Con vuot thi loai bo keywords tu cuoi
    while (
        token_count(embedding_text) > config.max_embedding_tokens
        and len(active_keywords) > 3
    ):
        active_keywords = active_keywords[:-1]
        lines = build_lines(active_keywords, active_vi_keywords)
        embedding_text = "\n".join(lines)

    return embedding_text

# Ham chinh chay pipeline
def run_pipeline2() -> None:
    # Lay cau hinh
    config = build_pipeline2_config()

    print("=" * 50)
    print("PIPELINE 2 - CROSS-LINGUAL KNOWLEDGE UNITS")
    print("=" * 50)
    print(f"Input: {config.input_path}")
    print(f"Output: {config.output_path}")
    print(f"VI hints: {config.enable_vi_hints}")
    print()

    all_records = []
    video_chunk_count = defaultdict(int)

    # Doc tung record tu transcript_v3
    for record in iter_jsonl(config.input_path):
        # Tao embedding_text cho chunk
        embedding_text = build_crosslingual_embedding(record, config)
        record["embedding_text"] = embedding_text
        record["chunk_type"] = "crosslingual_knowledge"
        record["semantic_topic"] = record.get("semantic_topic") or detect_topic(
            record.get("chunk_text", "")
        )

        # Tao chunk_id va chunk_index
        video_id = record["video_id"]
        video_chunk_count[video_id] += 1
        chunk_index = video_chunk_count[video_id]

        record["chunk_id"] = f"{video_id}_{chunk_index:04d}"
        record["chunk_index"] = chunk_index

        all_records.append(record)

    # Ghi ra transcript_v4
    count = write_jsonl(all_records, config.output_path)

    print(f"Records: {count}")
    print(f"Videos: {len(video_chunk_count)}")
    print(f"Output: {config.output_path}")

if __name__ == "__main__":
    run_pipeline2()
