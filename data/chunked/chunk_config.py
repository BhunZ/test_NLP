from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkConfig:
    # PATH CONFIG
    root_dir: Path

    #  IMPORTANT:
    # dùng file sentence-level (đã clean + split + merge tốt)
    input_path: Path

    # thư mục output chunk
    output_dir: Path

    # strategy: time | semantic | hybrid
    strategy: str = "hybrid"


    #TIME CHUNKING CONFIG

    # target token cho mỗi chunk (LLM-friendly)
    # ~150–200 là sweet spot cho embedding + retrieval
    target_tokens: int = 180

    # số từ tối thiểu để chunk có ý nghĩa
    min_chunk_words: int = 50

    # số từ tối đa (tránh chunk quá dài)
    max_chunk_words: int = 160

    # thời lượng tối đa của 1 chunk (giây)
    # giúp đảm bảo citation không quá rộng
    max_chunk_duration: float = 90.0

    # khoảng im lặng giữa 2 segment (giây)
    # nếu gap lớn → cắt chunk
    max_silence_gap: float = 8.0

    # số segment overlap giữa 2 chunk
    # giúp giữ context khi retrieve
    overlap_segments: int = 2


    # SEMANTIC CHUNKING CONFIG

    # số từ tối đa khi merge semantic
    max_semantic_words: int = 250

    # thời lượng tối đa khi merge semantic
    max_semantic_duration: float = 120.0

    # giữ lại lexical overlap (fallback)
    # nhưng giảm vai trò vì sẽ dùng embedding
    min_lexical_overlap: float = 0.05

    # NEW: semantic similarity (embedding-based)
    # dùng trong should_merge_semantic
    semantic_similarity_threshold: float = 0.8


    #  EMBEDDING / RETRIEVAL CONFIG

    # giới hạn token để tránh chunk quá dài cho embedding model
    max_embedding_tokens: int = 300

    # loại bỏ chunk quá ngắn (noise)
    min_chunk_chars: int = 80

    #  OUTPUT / VERSIONING

    # version để tracking dataset
    chunk_version: str = "v2"

    # có ghi intermediate time chunks không
    write_intermediate_time_chunks: bool = True


    # OUTPUT PATHS

    @property
    def time_output_path(self) -> Path:
        return self.output_dir / "transcripts_time_chunked.jsonl"

    @property
    def semantic_output_path(self) -> Path:
        return self.output_dir / "transcripts_semantic_chunked.jsonl"

    @property
    def hybrid_output_path(self) -> Path:
        return self.output_dir / "transcripts_hybrid_chunked.jsonl"


#  BUILD DEFAULT CONFIG
def build_default_config(strategy: str = "hybrid") -> ChunkConfig:
    root_dir = Path(__file__).resolve().parent.parent.parent

    return ChunkConfig(
        root_dir=root_dir,

        # CRITICAL FIX:
        # dùng file đã split sentence (chất lượng cao hơn nhiều)
        input_path=root_dir / "data" / "cleaned" / "transcripts_clean_sentence.jsonl",

        output_dir=root_dir / "data" / "chunked",

        strategy=strategy.lower().strip(),
    )