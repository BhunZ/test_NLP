import argparse

from chunk_config import build_default_config
from chunk_utils import (
    build_semantic_chunks_for_video,
    build_time_chunks_for_record,
    group_chunks_by_video,
    iter_jsonl,
    write_jsonl,
)


def run_time_pipeline(config) -> list[dict]:
    output_records = []
    total_videos = 0

    for record in iter_jsonl(config.input_path):
        total_videos += 1
        output_records.extend(build_time_chunks_for_record(record, config))

    count = write_jsonl(output_records, config.time_output_path)
    print(f"[time] processed videos: {total_videos}")
    print(f"[time] total chunks: {count}")
    print(f"[time] output: {config.time_output_path}")
    return output_records


def run_semantic_pipeline(config) -> list[dict]:
    time_chunks = run_time_pipeline(config)
    grouped = group_chunks_by_video(time_chunks)

    semantic_records = []
    for _, video_chunks in grouped.items():
        semantic_records.extend(build_semantic_chunks_for_video(video_chunks, config))

    count = write_jsonl(semantic_records, config.semantic_output_path)
    print(f"[semantic] processed videos: {len(grouped)}")
    print(f"[semantic] total chunks: {count}")
    print(f"[semantic] output: {config.semantic_output_path}")
    return semantic_records


def run_hybrid_pipeline(config) -> list[dict]:
    time_chunks = run_time_pipeline(config)
    grouped = group_chunks_by_video(time_chunks)

    hybrid_records = []
    for _, video_chunks in grouped.items():
        hybrid_records.extend(build_semantic_chunks_for_video(video_chunks, config))

    count = write_jsonl(hybrid_records, config.hybrid_output_path)
    print(f"[hybrid] processed videos: {len(grouped)}")
    print(f"[hybrid] total chunks: {count}")
    print(f"[hybrid] output: {config.hybrid_output_path}")
    return hybrid_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcript chunking pipeline")
    parser.add_argument(
        "--strategy",
        choices=["time", "semantic", "hybrid"],
        default="hybrid",
        help="Chunking strategy to run",
    )
    args = parser.parse_args()

    config = build_default_config(strategy=args.strategy)

    if config.strategy == "time":
        run_time_pipeline(config)
    elif config.strategy == "semantic":
        run_semantic_pipeline(config)
    else:
        run_hybrid_pipeline(config)


if __name__ == "__main__":
    main()
