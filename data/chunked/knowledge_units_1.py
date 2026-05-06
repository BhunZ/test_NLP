"""
Knowledge Units Pipeline - Tao knowledge units tu transcript.

 принци:
  1. Time chunking: Chia transcript thanh cac chunk theo thoi gian
  2. Semantic merging: Merge cac chunk lien ke neu noidung tuong tu

 Muc tieu:
  - Giu nguyen transcript (khong biet mat)
  - Chi merge cac chunk lien ke (khong nhay lung tung)
  - Tranh mat tail (cac chunk ngan)
  - Chunk size phu hop cho embedding (120-190 tokens)

 Input: transcripts_clean_sentence.jsonl
 Output: transcript_v3.jsonl
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# cau hinh duong dan file
ROOT_DIR = Path("D:/NLP")
INPUT_PATH = ROOT_DIR / "data" / "cleaned" / "transcripts_clean_sentence.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "chunked" / "transcript_v3.jsonl"

# cau hinh chunk sizing
# Token: 1 token ~ 1.28 words (trung binh tieng Anh)
MIN_TOKENS = 120           # Token toi thieu trong chunk
TARGET_TOKENS = 160      # Token muc tieu
MAX_TOKENS = 190         # Token toi da
HARD_MAX_TOKENS = 215    # Token cuc dai (buffer)

MIN_DURATION = 18.0     # Thoi gian toi thieu (giay)
TARGET_DURATION = 65.0   # Thoi gian muc tieu
MAX_DURATION = 85.0     # Thoi gian toi da

SOFT_GAP_SECONDS = 0.9   # Khoang trong nho (de merge)
HARD_GAP_SECONDS = 1.8  # Khoang trong lon (khong merge)

MIN_SIMILARITY = 0.22       # Similarity toi thieu de merge
CONTINUATION_SIMILARITY = 0.12  # Similarity cho cac chunk tiep theo
MAX_MERGED_TOKENS = 215     # Token toi da sau khi merge

# filler words can loai bo khi clean text
LIGHT_FILLER_PATTERNS = [
    r"\bum\b", r"\buh\b", r"\bah\b", r"\ber\b",
    r"\byou know\b", r"\bi mean\b",
    r"\bsort of\b", r"\bkind of\b",
    r"\bso yeah\b", r"\bright so\b", r"\bokay so\b",
]

# loi ASR thuong gap va cach sua
ASR_CORRECTIONS = {
    "dont": "don't",
    "wont": "won't",
    "cant": "can't",
    "ive": "I've",
    "im": "I'm",
    "thats": "that's",
    "its": "it's",
    "id": "I'd",
    "youd": "you'd",
    "theyd": "they'd",
    "weve": "we've",
    "theres": "there's",
    "wheres": "where's",
    "whats": "what's",
}

# Cac tu dau dau hien thi tinh tiep theo (continuation)
# Khi text bat dau bang cac tu nay, co the la tiep theo cau truoc
CONTINUATION_STARTERS = (
    "and ", "or ", "but ", "so ", "because ",
    "which ", "that ", "then ", "also ",
    "this ", "these ", "those ", "it ", "they ",
    "he ", "she ", "we ", "i ", "you ",
    "for ", "to ", "of ", "in ", "on ", "with ",
)

# Topic keywords de phan loai semantic
# Su dung de gan nhan topic cho chunk (khong quyet dinh merge)
TOPIC_KEYWORDS = {
    "Neural Networks": [
        "neural", "network", "layer", "hidden", "weight", "bias", "activation",
    ],
    "Transformer": [
        "transformer", "attention", "self-attention", "multi-head", "positional",
        "query", "key", "value", "encoder", "decoder",
    ],
    "Training": [
        "training", "train", "learn", "optimizer", "gradient", "loss", "backprop",
        "epoch", "batch", "learning rate",
    ],
    "NLP": [
        "nlp", "language", "token", "bert", "gpt", "embedding", "vocabulary",
        "tokenizer",
    ],
    "Computer Vision": [
        "image", "pixel", "convolution", "cnn", "feature", "vision",
    ],
    "Reinforcement Learning": [
        "reward", "policy", "agent", "action", "environment",
    ],
    "Math/Statistics": [
        "probability", "statistics", "matrix", "vector", "derivative",
    ],
}

# DOC FILE JSONL
# Doc tung dong tu file JSONL, yield tung record
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

# GHI FILE JSONL
# Ghi danh sach dict ra file JSONL
# Tra ve so luong records da ghi
def write_jsonl(records: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count

# LAM SACH TEXT
# Buoc 1: Thay doi ky tu dac biet (en-dash, em-dash, apostrophe)
# Buoc 2: Sua loi ASR (dont -> don't)
# Buoc 3: Loai bo filler words
# Buoc 4: Loai khoang trang thua
# Buoc 5: Loai bo ky tu dat biet (chi giu chu cai, so, space, dau thong)
def clean_text(text: str) -> str:
    text = (
        (text or "")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2019", "'")
    )
    text = f" {text.strip()} "

    # Sua loi ASR
    for wrong, correct in ASR_CORRECTIONS.items():
        text = re.sub(rf"(?i)\b{re.escape(wrong)}\b", correct, text)

    # Loai bo filler words
    for pattern in LIGHT_FILLER_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Loai khoang trang thua
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Loai ky tu dat biet
    text = re.sub(r"[^a-zA-Z0-9\s.,!?':;/%$()\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# DEM SO TOKEN
# Uoc tinh so token tu so tu
# Quy tac: 1 token ~ 1.28 words
def token_count(text: str) -> int:
    words = re.findall(r"\S+", text or "")
    if not words:
        return 0
    return max(1, int(round(len(words) * 1.28)))

# DINH DANG THOI GIAN
# Chuyen doi giay sang dinh dang MM:SS hoac HH:MM:SS
def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

# KIEM TRA CO TINH TAP TU (continuation)
# Neu text bat dau bang cac tu trong CONTINUATION_STARTERS
# thi co the la tiep theo cau truoc (khong phai cau moi)
def looks_like_continuation(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(lowered.startswith(prefix) for prefix in CONTINUATION_STARTERS)

# PHAT HIEN SEMANTIC TOPIC
# Quet keyword trong text, tra ve topic co nhieu keyword nhat
def detect_semantic_topic(text: str) -> str:
    text_lower = text.lower()
    scores = defaultdict(int)

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[topic] += 1

    if scores:
        best_topic = max(scores, key=scores.get)
        if scores[best_topic] > 0:
            return best_topic

    return "General"

# TINH SIMILARITY GIUA 2 TEXT tính độ tương đồng
# Jaccard = intersection / union
#Overlap = 2 * intersection / (n1 + n2)
# Su dung overlap ratio: 2 * intersection / (n1 + n2)
# Tot hon Jaccard vi tinh ca do dai text
def compute_similarity(text1: str, text2: str) -> float:
    words1 = set(re.findall(r"[a-z0-9][a-z0-9\-']*", text1.lower()))
    words2 = set(re.findall(r"[a-z0-9][a-z0-9\-']*", text2.lower()))

    # Loai bo stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "for", "and", "or", "that", "this", "it", "we", "you", "they",
    }
    words1 -= stopwords
    words2 -= stopwords

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    total = len(words1) + len(words2)
    return (2 * intersection) / total if total > 0 else 0.0

# TAO EMBEDDING TEXT
# Format văn ban cho embedding:
# Course: {course}
# Title: {title}
# Time: {start} - {end}
# Topic: {topic}
#
# {content}
def build_embedding_text(chunk: dict) -> str:
    time_range = (
        f"{format_timestamp(chunk.get('start_time', 0.0))}"
        f" - {format_timestamp(chunk.get('end_time', 0.0))}"
    )
    return (
        f"Course: {chunk.get('course', 'Unknown')}\n"
        f"Title: {chunk.get('title', 'Unknown')}\n"
        f"Time: {time_range}\n"
        f"Topic: {chunk.get('semantic_topic', 'General')}\n\n"
        f"{chunk.get('chunk_text', '')}"
    )

# TAO CHUNK RECORD
# Tao chunk record tu danh sach segments
def build_chunk_record(record: dict, segments: list[dict], chunk_type: str) -> dict | None:
    if not segments:
        return None

    # Gop text tu tat ca segments
    chunk_text = clean_text(" ".join(segment["text"] for segment in segments))
    if not chunk_text:
        return None

    # Tinh thoi gian
    start_time = round(segments[0]["start"], 3)
    end_time = round(segments[-1]["end"], 3)

    # Phat hien topic
    semantic_topic = detect_semantic_topic(chunk_text)

    # Tao record
    chunk = {
        "video_id": record["video_id"],
        "title": record.get("title", ""),
        "course": record.get("course", ""),
        "source": record.get("source", ""),
        "chunk_type": chunk_type,
        "chunk_text": chunk_text,
        "start_time": start_time,
        "end_time": end_time,
        "duration": round(end_time - start_time, 3),
        "token_count": token_count(chunk_text),
        "semantic_topic": semantic_topic,
    }
    chunk["embedding_text"] = build_embedding_text(chunk)
    return chunk

# MERGE HAI CHUNK
# Gop 2 chunk thanh 1 (su dung cho semantic merge)
def merge_two_chunks(chunk1: dict, chunk2: dict, chunk_type: str = "semantic_knowledge") -> dict:
    merged_record = {
        "video_id": chunk1.get("video_id", ""),
        "title": chunk1.get("title", ""),
        "course": chunk1.get("course", ""),
        "source": chunk1.get("source", ""),
    }
    merged_segments = [
        {
            "text": chunk1.get("chunk_text", ""),
            "start": float(chunk1.get("start_time", 0.0)),
            "end": float(chunk1.get("end_time", 0.0)),
        },
        {
            "text": chunk2.get("chunk_text", ""),
            "start": float(chunk2.get("start_time", 0.0)),
            "end": float(chunk2.get("end_time", 0.0)),
        },
    ]
    return build_chunk_record(merged_record, merged_segments, chunk_type)

# SUA CAC CHUNK NHO
# Cac chunk qua nho (< MIN_TOKENS) thi merge voi chunk truoc hoac sau
# Tranh biet mat tail (cac chunk nho o cuoi video)
def repair_short_chunks(chunks: list[dict], chunk_type: str) -> list[dict]:
    if len(chunks) <= 1:
        return chunks

    repaired: list[dict] = []
    index = 0

    while index < len(chunks):
        chunk = chunks[index]

        # Neu chunk du lon, giu nguyen
        if chunk["token_count"] >= MIN_TOKENS:
            repaired.append(chunk)
            index += 1
            continue

        # Thu merge voi chunk truoc (neu con du room)
        if repaired and repaired[-1]["token_count"] + chunk["token_count"] <= MAX_MERGED_TOKENS:
            merged = merge_two_chunks(repaired[-1], chunk, chunk_type)
            if merged:
                repaired[-1] = merged
                index += 1
                continue

        # Thu merge voi chunk sau (neu con du room)
        if index + 1 < len(chunks) and (
            chunk["token_count"] + chunks[index + 1]["token_count"] <= MAX_MERGED_TOKENS
        ):
            merged = merge_two_chunks(chunk, chunks[index + 1], chunk_type)
            if merged:
                repaired.append(merged)
                index += 2
                continue

        # Khong merge duoc, giu nguyen
        repaired.append(chunk)
        index += 1

    return repaired

# CHUAN BI SEGMENTS
# Lam sach va dinh dang segment tu transcript
def prepare_segments(record: dict) -> list[dict]:
    prepared = []
    for segment in record.get("transcript", []):
        text = clean_text(segment.get("text", ""))
        if not text:
            continue

        start_time = float(segment["start"])
        end_time = start_time + float(segment["duration"])
        prepared.append(
            {
                "text": text,
                "start": start_time,
                "end": end_time,
                "tokens": token_count(text),
            }
        )
    return prepared

# TIME CHUNKING
# Chia transcript thanh cac chunk theo thoi gian va token
#
# Qua trinh:
#   1. Duyet qua tung segment
#   2. Them segment vao chunk hien tai
#   3. Kiem tra can dong chunk:
#      - Neu vuot HARD_MAX_TOKENS hoac MAX_DURATION -> dong
#      - Neu du TARGET_TOKENS va co khoang trong -> dong
#      - Neu du MIN_TOKENS va khoang trong lon -> dong
#      - Neu la segment cuoi -> dong
#   4. Sau khi chunk xong, goi repair_short_chunks de xu ly cac chunk nho
def build_time_chunks(record: dict) -> list[dict]:
    transcript = prepare_segments(record)
    if not transcript:
        return []

    chunks: list[dict] = []
    current_segments: list[dict] = []
    current_tokens = 0

    for index, segment in enumerate(transcript):
        # Kiem tra neu can dong chunk (vuot gioi han)
        if current_segments:
            candidate_tokens = current_tokens + segment["tokens"]
            candidate_duration = segment["end"] - current_segments[0]["start"]
            if (
                candidate_tokens > HARD_MAX_TOKENS or candidate_duration > MAX_DURATION
            ) and current_tokens >= MIN_TOKENS:
                chunk = build_chunk_record(record, current_segments, "time_knowledge")
                if chunk:
                    chunks.append(chunk)
                current_segments = []
                current_tokens = 0

        # Them segment hien tai vao chunk
        current_segments.append(segment)
        current_tokens += segment["tokens"]

        # Lay segment tiep theo
        next_segment = transcript[index + 1] if index + 1 < len(transcript) else None

        # Tinh khoang trong sau segment hien tai
        duration = current_segments[-1]["end"] - current_segments[0]["start"]
        gap_after = (
            float(next_segment["start"]) - float(current_segments[-1]["end"])
            if next_segment
            else None
        )

        # Cac dieu kien de dong chunk
        should_close = False
        if next_segment is None:
            # La segment cuoi cung
            should_close = True
        elif current_tokens >= HARD_MAX_TOKENS or duration >= MAX_DURATION:
            # Vuot gioi han cuc dai
            should_close = True
        elif current_tokens >= TARGET_TOKENS and (
            duration >= TARGET_DURATION or (gap_after is not None and gap_after >= SOFT_GAP_SECONDS)
        ):
            # Dat muc tieu hoac co khoang trong nho
            should_close = True
        elif current_tokens >= MIN_DURATION and duration >= MIN_DURATION and (
            gap_after is not None and gap_after >= HARD_GAP_SECONDS
        ):
            # Du token toi thieu va co khoang trong lon
            should_close = True

        # Dong chunk neu can
        if should_close:
            chunk = build_chunk_record(record, current_segments, "time_knowledge")
            if chunk:
                chunks.append(chunk)
            current_segments = []
            current_tokens = 0

    return repair_short_chunks(chunks, "time_knowledge")

# KIEM TRA CO NEN MERGE KHONG
# Quyet dinh co merge chunk hien tai voi chunk truoc hay khong
#
# Dieu kien merge:
#   1. Tong tokens khong vuot MAX_MERGED_TOKENS
#   2. Khoang trong khong qua HARD_GAP_SECONDS
#   3. Similarity >= MIN_SIMILARITY
#
# Hoac (relaxed cho continuation):
#   - Text co tinh tiep theo (looks_like_continuation)
#   - Cung topic + similarity >= CONTINUATION_SIMILARITY
def should_merge_adjacent(
    current_chunk: dict, prev_chunk: dict, min_similarity: float = MIN_SIMILARITY
) -> bool:
    current_text = current_chunk.get("chunk_text", "")
    prev_text = prev_chunk.get("chunk_text", "")

    current_tokens = current_chunk.get("token_count", token_count(current_text))
    prev_tokens = prev_chunk.get("token_count", token_count(prev_text))
    combined_tokens = current_tokens + prev_tokens

    # Tong tokens phai trong gioi han
    if combined_tokens > MAX_MERGED_TOKENS:
        return False

    # Khoang trong phai nho
    time_gap = float(current_chunk.get("start_time", 0.0)) - float(prev_chunk.get("end_time", 0.0))
    if time_gap > HARD_GAP_SECONDS:
        return False

    # Tinh similarity
    similarity = compute_similarity(current_text, prev_text)
    if similarity >= min_similarity:
        return True

    # Kiem tra topic
    current_topic = current_chunk.get("semantic_topic") or detect_semantic_topic(current_text)
    prev_topic = prev_chunk.get("semantic_topic") or detect_semantic_topic(prev_text)
    same_topic = current_topic == prev_topic and current_topic != "General"

    # Dieu kien relaxed cho continuation
    if (
        looks_like_continuation(current_text)
        and same_topic
        and similarity >= CONTINUATION_SIMILARITY
        and combined_tokens <= MAX_TOKENS + 20
    ):
        return True

    # Dieu kien relax cho chunk nho
    if (
        prev_tokens < TARGET_TOKENS
        and same_topic
        and time_gap <= SOFT_GAP_SECONDS
        and similarity >= CONTINUATION_SIMILARITY
    ):
        return True

    return False

# SEMANTIC MERGING
# Merge cac chunk lien ke neu noi dung tuong tu
#
# Qua trinh:
#   1. Duyet time_chunks theo thu tu (giai thiet da sorted)
#   2. So sanh voi chunk TRƯỚC DO (lien ke)
#   3. Merge neu should_merge_adjacent tra ve True
#   4. Sau khi merge xong, goi repair_short_chunks
#   5. Cap nhat embedding_text cho tat ca chunk
def build_semantic_chunks(
    time_chunks: list[dict], min_similarity: float = MIN_SIMILARITY
) -> list[dict]:
    if not time_chunks:
        return []

    semantic_chunks: list[dict] = []
    previous = None

    for chunk in time_chunks:
        if previous is None:
            # Chunk dau tien
            previous = dict(chunk)
            previous["chunk_type"] = "semantic_knowledge"
        elif should_merge_adjacent(chunk, previous, min_similarity):
            # Merge voi chunk truoc
            merged = merge_two_chunks(previous, chunk, "semantic_knowledge")
            previous = merged if merged else dict(chunk)
        else:
            # Khong merge, luu chunk hien tai va bat dau chunk moi
            semantic_chunks.append(previous)
            previous = dict(chunk)
            previous["chunk_type"] = "semantic_knowledge"

    # Luu chunk cuoi cung
    if previous:
        semantic_chunks.append(previous)

    # Sua cac chunk nho
    semantic_chunks = repair_short_chunks(semantic_chunks, "semantic_knowledge")

    # Cap nhat embedding_text
    for chunk in semantic_chunks:
        chunk["chunk_type"] = "semantic_knowledge"
        chunk["semantic_topic"] = detect_semantic_topic(chunk.get("chunk_text", ""))
        chunk["embedding_text"] = build_embedding_text(chunk)

    return semantic_chunks

# HAM CHINH
# Chay pipeline:
#   1. Doc tu input file
#   2. Time chunking
#   3. Semantic merge
#   4. Ghi ra output file
def main() -> None:
    print("=" * 50)
    print("KNOWLEDGE UNIT PIPELINE")
    print("=" * 50)
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    #print(f"Token range: {MIN_TOKENS}-{MAX_TOKENS} (hard max {HARD_MAX_TOKENS})")
    print()

    all_chunks = []
    total_videos = 0

    for record in iter_jsonl(INPUT_PATH):
        total_videos += 1
        time_chunks = build_time_chunks(record)
        semantic_chunks = build_semantic_chunks(time_chunks)
        all_chunks.extend(semantic_chunks)

    count = write_jsonl(all_chunks, OUTPUT_PATH)

    print(f"Videos: {total_videos}")
    print(f"Knowledge units: {count}")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()