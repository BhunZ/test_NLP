"""
Knowledge Units Pipeline - File chuan cho RAG
====================================
Tac dung:
  - Chia transcript thanh cac don vi tri thuc hoan chinh (knowledge units)
  - Dam bao moi chunk la mot y nghia hoan chinh, khong bi cat giua cau
  - Lam sach text: loai bo filler words va loi ASR
  - Tao embedding_text: Course + Title + Duration + Content
  - Semantic chunking: them pembed similarity de merge/split chunks

Cau hinh:
  - Token: 150-200 (hop ly cho embedding)
  - Duration: 20-90 giay
  - Chunk hoan chinh: cau phai ket thuc bang . ! ?
"""

import json
import re
from pathlib import Path
from typing import Iterable
from collections import defaultdict


# Project-relative paths (was hardcoded D:/NLP — now portable)
# pipeline/stage_03_chunk/knowledge_units.py → repo root is parents[2]
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data" / "cleaned" / "transcripts_clean_sentence.jsonl"
OUTPUT_PATH = ROOT_DIR / "data" / "chunked" / "transcript_v3.jsonl"


# CAU HINH CHUNKING
# So token toi thieu trong mot chunk
MIN_TOKENS = 150

# So token toi da trong mot chunk (200 hop ly cho context LLM)
MAX_TOKENS = 200

# Thoi gian toi da cua mot chunk (1.5 phut)
MAX_DURATION = 90.0

# Thoi gian toi thieu de chunk hop li (20 giay)
MIN_DURATION = 20.0


# DANH SACH FILLER WORDS
# Nhung tu khong mang nghia, thuong xuat hien trong bai noi
# Can loai bo khi embedding
FILLER_WORDS = {
    "um",  # /ʌm/
    "uh",  # /^/
    "ah",  # /a:/
    "er",  # /ɜ:/
    "like",  # nhu, ví dụ
    "you know",  # you know
    "sort of",  # sort of
    "kind of",  # kind of
    "basically",  # basically
    "actually",  # actually
    "literally",  # literally
    "honestly",  # honestly
    "so yeah",  # so yeah
    "okay so",  # okay so
    "right so",  # right so
    "like",
}


# TU SAI TRONG ASR CAN SUA
# ASR (Automatic Speech Recognition) thuong bi loi nhin tu
# Vi du: "dont" -> "don't", "im" -> "I'm"
ASR_CORRECTIONS = {
    " dont ": " don't ",
    " wont ": " won't ",
    " cant ": " can't ",
    " ive ": " I've ",
    " im ": " I'm ",
    " thats ": " that's ",
    " its ": " it's ",
    " id ": " I'd ",
    " youd ": " you'd ",
    " theyd ": " they'd ",
    " weve ": " we've ",
    " theres ": " there's ",
    " wheres ": " where's ",
    " whats ": " what's ",
}


# DOC FILE JSONL
# Doc tung dong tu file transcript_clean.jsonl
# Yield tung record mot luc de tiet kiem bo nho
def iter_jsonl(path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


# LAM SACH TEXT
# Buoc 1: Sua loi ASR (dont -> don't)
# Buoc 2: Loai bo filler words
# Buoc 3: Loai khoang trang thua
# Buoc 4: Loai bo ky tu dat biet (chi giu chu cai, so, khoang trang, dau)
def clean_text(text: str) -> str:
    # Buoc 1: Sua loi ASR
    for wrong, correct in ASR_CORRECTIONS.items():
        text = text.replace(wrong, correct)

    # Buoc 2: Loai bo filler words
    text_lower = text.lower()
    for filler in FILLER_WORDS:
        text_lower = text_lower.replace(filler, "")
    # Giu lai ky tu dau (bien viet hoa o vi tri dau)
    if text:
        text = text[0] + text[1:] if len(text) > 0 else text

    # Buoc 3: Loai khoang trang thua (nhieu space -> 1 space)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    # Buoc 4: Loai bo ky tu dat biet (chi giu a-z, A-Z, 0-9, space, .,!?' )
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)

    return text


# DEM SO TOKEN
# Uoc tinh so token tu so tu
# Quy tac: 1 token ~ 1.3 words (trung binh tieng Anh)
def token_count(text: str) -> int:
    words = text.split()
    return int(len(words) * 1.3)


# KIEM TRA CAU HOAN CHINH
# Tra ve True neu cau ket thuc bang dau . ! hoac ?
# Dung de dam bao khong cat giua giua 2 cau
def is_complete_sentence(text: str) -> bool:
    text = text.strip()
    # Tim dau ket thuc cau: . ! ?
    return bool(re.search(r"[.!?]$", text))


# CHIA TRANSCRIPT THANH CAC KNOWLEDGE UNITS
# Qua trinh:
#   1. Duyet qua tung segment trong transcript
#   2. Noi text tai vao chunk hien tai
#   3. Kiem tra can dong chunk:
#      - Neu vuot MAX_TOKENS -> dong
#      - Neu du MIN_TOKENS + cau hoan chinh -> dong
#      - Neu du MIN_TOKENS + khoang trong > 3s -> dong
#      - Neu la segment cuoi -> dong
#   4. Tao embedding_text cho RAG
def build_time_chunks(record: dict) -> list[dict]:
    transcript = record.get("transcript", [])
    if not transcript:
        return []

    chunks = []
    current_segments = []  # Cac segment trong chunk hien tai
    current_tokens = 0  # Tong token hien tai
    chunk_start = None  # Thoi diem bat dau chunk

    # Duyet qua tung segment
    for idx, seg in enumerate(transcript):
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Lam sach text
        text = clean_text(text)
        if not text:
            continue

        # Tao segment object
        seg_obj = {
            "text": text,
            "start": float(seg["start"]),
            "end": float(seg["start"]) + float(seg["duration"]),
            "tokens": token_count(text),
        }

        # Luu thoi diem bat dau chunk neu la segment dau tien
        if chunk_start is None:
            chunk_start = seg_obj["start"]

        # Them segment vao chunk hien tai
        current_segments.append(seg_obj)
        current_tokens += seg_obj["tokens"]
        current_end = seg_obj["end"]

        # Lay segment tiep theo (neu co)
        next_seg = transcript[idx + 1] if idx + 1 < len(transcript) else None

        # Tinh thoi gian chunk hien tai
        duration = current_end - chunk_start
        should_close = False

        # Dieu kien 1: Cau hoan chinh (UUU TIEN)
        # Neu cau ket thuc bang . ! ? -> dong chunk
        if is_complete_sentence(current_segments[-1]["text"]):
            should_close = True

        # Dieu kien 2: Vuot so token toi da (CHO PHEP QUA 1 TI)
        elif current_tokens >= MAX_TOKENS + 20:  # +20 buffer
            should_close = True

        # Dieu kien 3: Du token toi thieu VA khoang trong > 3s
        elif current_tokens >= MIN_TOKENS and duration >= MIN_DURATION:
            if next_seg:
                gap = float(next_seg["start"]) - current_end
                if gap > 3.0:
                    should_close = True

        # Dieu kien 4: La segment cuoi cung
        elif next_seg is None:
            should_close = True

        # Neu can dong chunk
        if should_close:
            # Noi tat ca segment thanh 1 text
            chunk_text = " ".join(seg["text"] for seg in current_segments)
            chunk_text = clean_text(chunk_text)

            # Chi tao chunk neu du token
            if chunk_text and token_count(chunk_text) >= MIN_TOKENS:
                # Tao embedding_text cho RAG
                # Format: Course + Title + Content (KHONG co Duration - chi can course/title/context)
                embedding_text = (
                    f"Course: {record.get('course', 'Unknown')}\n"
                    f"Title: {record.get('title', 'Unknown')}\n\n"
                    f"{chunk_text}"
                )

                chunks.append(
                    {
                        "video_id": record["video_id"],
                        "title": record.get("title", ""),
                        "course": record.get("course", ""),
                        "source": record.get("source", ""),
                        "chunk_type": "knowledge_unit",
                        "chunk_text": chunk_text,
                        "embedding_text": embedding_text,
                        "start_time": round(chunk_start, 3),
                        "end_time": round(current_end, 3),
                        "duration": round(duration, 3),
                        "token_count": token_count(chunk_text),
                    }
                )

            # Reset cho chunk tiep theo
            current_segments = []
            current_tokens = 0
            chunk_start = None

    return chunks


# ============================================================
# SEMANTIC CHUNKING - Nhom cac chunk co noi dung lien quan
# ============================================================
# Y tuong:
#   1. Sau khi tao time_chunks, tao semantic groups
#   2. Cac chunk cung topic duoc nhom lai voi nhau
#   3. Merge vao 1 chunk neu nho hon MAX_TOKENS
#   4. Split neu lon hon MAX_TOKENS

# Topic keywords de phan loai semantic
TOPIC_KEYWORDS = {
    "Neural Networks": [
        "neural",
        "network",
        "layer",
        "hidden",
        "weight",
        "bias",
        "activation",
    ],
    "Transformer": [
        "transformer",
        "attention",
        "self-attention",
        "multi-head",
        "positional",
    ],
    "Training": [
        "training",
        "train",
        "learn",
        "optimizer",
        "gradient",
        "loss",
        "backprop",
    ],
    "NLP": ["nlp", "language", "token", "bert", "gpt", "embedding", "vocabulary"],
    "Computer Vision": ["image", "pixel", "convolution", "cnn", "feature", "vision"],
    "Reinforcement Learning": ["reward", "policy", "agent", "action", "environment"],
    "Math/Statistics": [
        "probability",
        "statistics",
        "matrix",
        "vector",
        "derivative",
        "integral",
    ],
}


def detect_semantic_topic(text: str) -> str:
    """
    Phat hien semantic topic cua text

    Args:
        text: Van ban can phan loai

    Returns:
        Ten topic (hoac "General" neu khong khop)
    """
    text_lower = text.lower()
    scores = defaultdict(int)

    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[topic] += 1

    if scores:
        best_topic = max(scores, key=scores.get)
        if scores[best_topic] >= 2:
            return best_topic

    return "General"


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Tinh do similar giua 2 van ban
    Su dung keyword overlap (don gian nhung hieu qua)

    Args:
        text1: Van ban 1
        text2: Van ban 2

    Returns:
        Diem similar (0.0 - 1.0)
    """
    # Tach tu
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    # Loai bo stopwords nho
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "for",
        "and",
        "or",
    }
    words1 = words1 - stopwords
    words2 = words2 - stopwords

    if not words1 or not words2:
        return 0.0

    # Tinh Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def build_semantic_chunks(
    time_chunks: list[dict], min_similarity: float = 0.15
) -> list[dict]:
    """
    Semantic chunking: Gop cac chunk co noi dung lien quan

    Qua trinh:
      1. Phan loai semantic topic cho tung time_chunk
      2. Nhom cac chunk cung topic
      3. Merge neu tong token nho hon MAX_TOKENS
      4. Giu nguyen neu qua lon

    Args:
        time_chunks: Danh sach time_chunks tu build_time_chunks
        min_similarity: Ngưỡng similar toi thieu de merge (mac dinh 0.15)

    Returns:
        Danh sach semantic_chunks da merge
    """
    if not time_chunks:
        return []

    # Buoc 1: Phan loai topic cho tung chunk
    for chunk in time_chunks:
        topic = detect_semantic_topic(chunk["chunk_text"])
        chunk["semantic_topic"] = topic

    # Buoc 2: Nhom theo semantic topic
    topic_groups = defaultdict(list)
    for chunk in time_chunks:
        topic = chunk.get("semantic_topic", "General")
        topic_groups[topic].append(chunk)

    # Buoc 3: Merge trong tung nhom
    semantic_chunks = []

    for topic, chunks in topic_groups.items():
        if not chunks:
            continue

        # Neu chi co 1 chunk, giu nguyen
        if len(chunks) == 1:
            semantic_chunks.append(chunks[0])
            continue

        # Neu nhieu hon 1 chunk, kiem tra similar de merge
        current_group = []
        current_tokens = 0

        for chunk in chunks:
            chunk_tokens = chunk.get("token_count", token_count(chunk["chunk_text"]))

            # Them vao group hien tai neu duoc
            if not current_group:
                current_group.append(chunk)
                current_tokens = chunk_tokens
            else:
                # Kiem tra similar voi chunk cuoi cung
                last_chunk = current_group[-1]
                sim = compute_text_similarity(
                    last_chunk["chunk_text"], chunk["chunk_text"]
                )

                # Merge neu tong tokens con duoi MAX_TOKENS va similar duoi
                if (
                    current_tokens + chunk_tokens <= MAX_TOKENS + 30
                    and sim >= min_similarity
                ):
                    current_group.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    # Luu group hien tai va bat dau group moi
                    merged = merge_chunks(current_group)
                    if merged:
                        semantic_chunks.append(merged)
                    current_group = [chunk]
                    current_tokens = chunk_tokens

        # Luu group cuoi cung
        if current_group:
            merged = merge_chunks(current_group)
            if merged:
                semantic_chunks.append(merged)

    # Cap nhat chunk_type
    for chunk in semantic_chunks:
        chunk["chunk_type"] = "semantic_knowledge"

    return semantic_chunks


def merge_chunks(chunks: list[dict]) -> dict:
    """
    Merge nhieu chunk thanh 1 chunk

    Args:
        chunks: Danh sach chunk can merge

    Returns:
        Chunk da merge
    """
    if not chunks:
        return {}

    if len(chunks) == 1:
        return chunks[0]

    # Gop text
    chunk_texts = [c["chunk_text"] for c in chunks]
    merged_text = " ".join(chunk_texts)
    merged_text = clean_text(merged_text)

    # Tinh thoi gian
    start_time = min(c["start_time"] for c in chunks)
    end_time = max(c["end_time"] for c in chunks)
    duration = end_time - start_time

    # Lay thong tin tu chunk dau tien
    first = chunks[0]

    embedding_text = (
        f"Course: {first.get('course', 'Unknown')}\n"
        f"Title: {first.get('title', 'Unknown')}\n"
        f"Topic: {first.get('semantic_topic', 'General')}\n\n"
        f"{merged_text}"
    )

    return {
        "video_id": first["video_id"],
        "title": first.get("title", ""),
        "course": first.get("course", ""),
        "source": first.get("source", ""),
        "chunk_type": "semantic_knowledge",
        "chunk_text": merged_text,
        "embedding_text": embedding_text,
        "start_time": round(start_time, 3),
        "end_time": round(end_time, 3),
        "duration": round(duration, 3),
        "token_count": token_count(merged_text),
        "semantic_topic": first.get("semantic_topic", "General"),
    }


# GHI FILE JSONL
# Ghi danh sach dict ra file JSONL
# Tra ve so luong records da ghi
def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# HAM CHINH
def main():
    print("=" * 50)
    print("KNOWLEDGE UNIT PIPELINE")
    print("=" * 50)
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Token range: {MIN_TOKENS}-{MAX_TOKENS}")
    print()

    all_chunks = []
    total_videos = 0

    # Doc tung video va chia chunk
    for record in iter_jsonl(INPUT_PATH):
        total_videos += 1

        # Buoc 1: Time-based chunking
        time_chunks = build_time_chunks(record)

        # Buoc 2: Semantic chunking
        semantic_chunks = build_semantic_chunks(time_chunks)

        all_chunks.extend(semantic_chunks)

    # Ghi output
    count = write_jsonl(all_chunks, OUTPUT_PATH)

    print(f"Videos: {total_videos}")
    print(f"Knowledge units: {count}")
    print(f"Output: {OUTPUT_PATH}")


# CHAY: python knowledge_units.py
if __name__ == "__main__":
    main()
