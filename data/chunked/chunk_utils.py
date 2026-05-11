import json
import re
from typing import Iterable


# Cac tu pho bien trong tieng Anh, bo qua khi tinh lexical overlap
# Vi du: "the", "and", "is", "a", "an", "of", etc.
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "so",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "as",
    "at",
    "by",
    "from",
    "if",
    "then",
    "than",
    "into",
    "about",
    "also",
    "we",
    "you",
    "they",
    "he",
    "she",
    "i",
    "me",
    "our",
    "your",
    "their",
    "my",
    "them",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "have",
    "has",
    "had",
    "not",
    "just",
    "there",
    "here",
}


# Cac tien to bieu hien Y TIEP TUC cua noi dung
# Khi text bat dau bang cac tien to nay, noi dung se tiep tuc khong chuyen chu de
CONTINUATION_PREFIXES = (
    "and ",  # "And now..."
    "but ",  # "But also..."
    "so ",  # "So what..."
    "because ",  # "Because of..."
    "then ",  # "Then we..."
    "now ",  # "Now let's..."
    "for example",
    "for instance",
    "in other words",
    "that means",
    "this means",
    "which means",
    "however",
    "meanwhile",
)


# Cac tien to bieu hien CHUYEN CHU DE
# Khi text bat dau bang cac tien to nay, can bat dau chunk moi
TOPIC_SHIFT_PREFIXES = (
    "now let's",  # "Now let's move to..."
    "let's move",
    "moving on",
    "next ",  # "Next topic..."
    "in summary",
    "to summarize",
    "the key idea",
)


# Doc file JSONL, tra ve generator de tiet kiem bo nho
# Moi dong trong file la mot JSON object
def iter_jsonl(path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"SKIP line {line_no}: invalid JSON")


# Ghi danh sach dict ra file JSONL
# Tra ve so luong records da ghi
def write_jsonl(records: Iterable[dict], path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# Chuan hoa text: xu ly whitespace, dau cau, viet hoa dau cau
# - Thay \n, \r bang khoang trang
# - Nhieu khoang trang -> 1 khoang trang
# - Khoang trang truoc dau cau (,.;!?) -> xoa
# - Dau cau lap (!!??) -> 1 dau
# - Xoa dau - . , ; : ! ? o dau/cuoi
# - Viet hoa chu cai dau
def normalize_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"([,.;!?]){2,}", r"\1", text)
    text = text.strip(" -.,;:!?")
    if not text:
        return ""
    return text[0].upper() + text[1:]


# Dem so tu trong text
# Tach bang khoang trang
def word_count(text: str) -> int:
    return len(text.split())


# Uoc luong so token tu so tu
# Quy tac: 1 token ~ 1.3 words (trung binh trong tieng Anh)
def estimate_tokens(text: str) -> int:
    return max(1, int(round(word_count(text) * 1.3)))


# Tinh thoi diem ket thuc cua mot segment
# end = start + duration
def segment_end(seg: dict) -> float:
    return round(float(seg["start"]) + float(seg["duration"]), 3)


# Chuyen doi giay sang mili giay (so nguyen)
# Dung de tao chunk_id
def sec_to_ms_int(value: float) -> int:
    return int(round(float(value) * 1000))


# Tao chunk_id theo dinh dang chuan
# Format: video_id:chunk_type:version:index:start_ms:end_ms
# Vi du: phWxl0nkgKk:time:v1:0000:0000005880:0000045920
def build_chunk_id(
    video_id: str,
    chunk_type: str,
    chunk_index: int,
    start_time: float,
    end_time: float,
    version: str = "v1",
) -> str:
    start_ms = sec_to_ms_int(start_time)
    end_ms = sec_to_ms_int(end_time)
    return (
        f"{video_id}:{chunk_type}:{version}:"
        f"{chunk_index:04d}:{start_ms:010d}:{end_ms:010d}"
    )


# Trich xuat cac content words (từ có nghĩa) tu text
# Loai bo stopwords va cac tu ngan (<=2 ky tu)
# Tra ve set (de tinh overlap)
def tokenize_content_words(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2}


# Tinh lexical overlap giua 2 texts
# Su dung Jaccard Similarity: |A ∩ B| / |A ∪ B|
# Gia tri 0.0 = khong co tu chung, 1.0 = giong nhau hoan toan
# Vi du: A={"ai"," ML"}, B={"ai"," DL"} -> 1/3 = 0.33
def lexical_overlap_score(text_a: str, text_b: str) -> float:
    a = tokenize_content_words(text_a)
    b = tokenize_content_words(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Kiem tra text co bat dau bang tien to tiep tuc khong
# Neu co, noi dung duoc tiep tuc (KHONG chuyen chu de)
def starts_with_continuation(text: str) -> bool:
    lower = text.lower().strip()
    return any(lower.startswith(prefix) for prefix in CONTINUATION_PREFIXES)


# Kiem tra text co bat dau bang tien to chuyen chu de khong
# Neu co, can bat dau chunk moi
def starts_with_topic_shift(text: str) -> bool:
    lower = text.lower().strip()
    return any(lower.startswith(prefix) for prefix in TOPIC_SHIFT_PREFIXES)


# Quyet dinh co NEN DONG (ket thuc) chunk hien tai hay chua
# Su dung nhieu dieu kien OR (bat ky dieu kien nao dung thi dong)
# Cac dieu kien:
# 1. So tu hien tai >= max_chunk_words (qua lon)
# 2. So token hien tai >= target_tokens (du token)
# 3. Thoi gian >= max_duration VA so tu >= min (du lon va du thoi gian)
# 4. La segment cuoi cung (next_seg is None)
# 5. Khoang trong > max_silence_gap VA so tu >= min (im lang-lau)
# 6. Ket thuc bang dau cau (.!?) VA so tu >= min (cau hoan chinh)
def should_close_time_chunk(
    current_segments: list[dict],
    next_seg: dict | None,
    current_words: int,
    current_tokens: int,
    chunk_start: float,
    current_end: float,
    config,
) -> bool:
    duration = current_end - chunk_start

    if current_words >= config.max_chunk_words:
        return True

    if current_tokens >= config.target_tokens:
        return True

    if (
        duration >= config.max_chunk_duration
        and current_words >= config.min_chunk_words
    ):
        return True

    if next_seg is None:
        return True

    next_gap = float(next_seg["start"]) - current_end
    if next_gap > config.max_silence_gap and current_words >= config.min_chunk_words:
        return True

    current_text = current_segments[-1]["text"].strip()
    if current_words >= config.min_chunk_words and re.search(r"[.!?:]$", current_text):
        return True

    return False


# Chia transcript thanh cac time chunks
# Qua trinh:
# 1. Duyet qua tung segment trong transcript
# 2. Thu them segment vao chunk hien tai
# 3. Kiem tra co nen dong chunk khong (should_close_time_chunk)
# 4. Neu can dong -> tao chunk, reset cho chunk tiep theo
# 5. Xu ly overlap: giu lai mot so segment cuoi de tao context
def build_time_chunks_for_record(record: dict, config) -> list[dict]:
    transcript = record.get("transcript", [])
    if not transcript:
        return []

    chunks = []
    current_segments = []  # Cac segment trong chunk hien tai
    current_words = 0  # Tong so tu trong chunk
    current_tokens = 0  # Tong so token uoc tinh
    chunk_start = None  # Thoi diem bat dau chunk

    idx = 0
    while idx < len(transcript):
        seg = transcript[idx]
        text = normalize_text(seg.get("text", ""))
        if not text:
            idx += 1
            continue

        # Tao segment object voi cac thong tin can thiet
        seg_obj = {
            "text": text,
            "start": float(seg["start"]),
            "duration": float(seg["duration"]),
            "end": segment_end(seg),
            "segment_idx": idx,
            "word_count": word_count(text),
            "token_count": estimate_tokens(text),
        }

        # Neu la segment dau tien, luu thoi diem bat dau
        if chunk_start is None:
            chunk_start = seg_obj["start"]

        # Them segment vao chunk hien tai
        current_segments.append(seg_obj)
        current_words += seg_obj["word_count"]
        current_tokens += seg_obj["token_count"]
        current_end = seg_obj["end"]

        # Lay segment tiep theo (neu co)
        next_seg = transcript[idx + 1] if idx + 1 < len(transcript) else None

        # Kiem tra co nen dong chunk khong
        if should_close_time_chunk(
            current_segments=current_segments,
            next_seg=next_seg,
            current_words=current_words,
            current_tokens=current_tokens,
            chunk_start=chunk_start,
            current_end=current_end,
            config=config,
        ):
            # Noi tat ca segments thanh mot text
            chunk_text = normalize_text(
                " ".join(segment["text"] for segment in current_segments)
            )

            if chunk_text:
                chunk_index = len(chunks)
                start_time = round(chunk_start, 3)
                end_time = round(current_end, 3)
                chunk_record = {
                    "chunk_id": build_chunk_id(
                        record["video_id"],
                        "time",
                        chunk_index,
                        start_time,
                        end_time,
                        config.chunk_version,
                    ),
                    "video_id": record["video_id"],
                    "title": record.get("title", ""),
                    "course": record.get("course", ""),
                    "playlist_id": record.get("playlist_id", ""),
                    "published_at": record.get("published_at", ""),
                    "source": record.get("source", ""),
                    "chunk_type": "time",
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": round(end_time - start_time, 3),
                    "word_count": word_count(chunk_text),
                    "token_estimate": estimate_tokens(chunk_text),
                    "segment_count": len(current_segments),
                    "source_segment_start_idx": current_segments[0]["segment_idx"],
                    "source_segment_end_idx": current_segments[-1]["segment_idx"],
                }
                chunks.append(chunk_record)

            # Xu ly overlap: giu lai mot so segment cuoi
            # De tao context cho chunk tiep theo
            # Vi du: overlap=2 -> giu 2 segment cuoi
            overlap_count = min(config.overlap_segments, len(current_segments))
            if next_seg is None:
                break

            # Lay segment cuoi (overlap)
            overlap_segments = (
                current_segments[-overlap_count:] if overlap_count > 0 else []
            )
            # Reset voi cac segment overlap
            current_segments = overlap_segments[:]
            current_words = sum(item["word_count"] for item in current_segments)
            current_tokens = sum(item["token_count"] for item in current_segments)
            # Cap nhat lai thoi diem bat dau chunk moi
            chunk_start = current_segments[0]["start"] if current_segments else None
            # Cap nhat thoi diem ket thuc cua segment cuoi trong overlap
            current_end = current_segments[-1]["end"] if current_segments else None

        idx += 1

    return chunks


# Nhom cac chunks theo video_id
# Tra ve dict: {video_id: [chunks...]}
# Moi video se co mot list cac chunks theo thu tu chunk_index
def group_chunks_by_video(chunks: Iterable[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for row in chunks:
        grouped.setdefault(row["video_id"], []).append(row)

    for video_id in grouped:
        grouped[video_id].sort(key=lambda x: x["chunk_index"])
    return grouped


# Quyet dinh co NEN GOP hai time chunks thanh mot semantic chunk khong
# Su dung nhieuieu kien:
# RANGE BUOC (neu vi pham -> KHONG gop):
#   - Tong thoi gian qua max_semantic_duration
#   - Tong so tu qua max_semantic_words
#   - Co topic shift VA chunk cu > min (chuyen chu de thi khong gop)
# DIEU KIEN GOP:
#   - Chunk hien tai < min -> gop (chua du lon)
#   - Chunk tiep theo be < 20 -> gop (qua ngan)
#   - Tien to tiep tuc (continuation) -> gop
#   - Lexical overlap >= min -> gop (cung chu de)
#   - Text hien tai ket thuc ":" -> gop (dang danh sach)
def should_merge_semantic(current: dict, nxt: dict, config) -> bool:
    # Tinh tong neu gop
    merged_duration = float(nxt["end_time"]) - float(current["start_time"])
    merged_words = current["word_count"] + nxt["word_count"]

    # RANGE BUOC: Khong gop neu vuot gioi han
    if merged_duration > config.max_semantic_duration:
        return False

    if merged_words > config.max_semantic_words:
        return False

    # Neu la topic shift VA chunk cu du lon -> khong gop
    if (
        starts_with_topic_shift(nxt["chunk_text"])
        and current["word_count"] >= config.min_chunk_words
    ):
        return False

    # DIEU KIEN 1: Chunk cu qua ngan -> gop
    if current["word_count"] < config.min_chunk_words:
        return True

    # DIEU KIEN 2: Chunk moi qua ngan -> gop
    if nxt["word_count"] < max(20, config.min_chunk_words // 2):
        return True

    # DIEU KIEN 3: Co tien to tiep tuc -> gop
    if starts_with_continuation(nxt["chunk_text"]):
        return True

    # DIEU KIEN 4: Lexical overlap du lon -> gop
    if (
        lexical_overlap_score(current["chunk_text"], nxt["chunk_text"])
        >= config.min_lexical_overlap
    ):
        return True

    # DIEU KIEN 5: Dang danh sach -> gop
    if current["chunk_text"].rstrip().endswith(":"):
        return True

    return False


# Gop hai time chunks thanh mot semantic chunk
# Cap nhat:
#   - text: noi hai text
#   - start_time: giu nguyen (cua chunk cu)
#   - end_time: lay cua chunk moi
#   - parent_time_chunk_ids: luu reference den ca hai time chunks goc
def merge_semantic_pair(current: dict, nxt: dict) -> dict:
    merged_text = normalize_text(current["chunk_text"] + " " + nxt["chunk_text"])
    return {
        "video_id": current["video_id"],
        "title": current.get("title", ""),
        "course": current.get("course", ""),
        "playlist_id": current.get("playlist_id", ""),
        "published_at": current.get("published_at", ""),
        "source": current.get("source", ""),
        "chunk_type": "semantic",
        "chunk_text": merged_text,
        "start_time": current["start_time"],
        "end_time": nxt["end_time"],
        "duration": round(float(nxt["end_time"]) - float(current["start_time"]), 3),
        "word_count": word_count(merged_text),
        "token_estimate": estimate_tokens(merged_text),
        "parent_time_chunk_start": current["parent_time_chunk_start"],
        "parent_time_chunk_end": nxt["parent_time_chunk_end"],
        "parent_time_chunk_ids": current["parent_time_chunk_ids"]
        + nxt["parent_time_chunk_ids"],
        "time_chunk_count": current["time_chunk_count"] + nxt["time_chunk_count"],
    }


# Gop time chunks thanh semantic chunks cho mot video
# Qua trinh:
# 1. Bat dau voi time chunk dau tien
# 2. Duyet qua cac chunk tiep theo
# 3. Kiem tra co nen gop khong (should_merge_semantic)
# 4. Neu gop -> merge, tiep tuc
# 5. Neu khong -> luu chunk cu, bat dau chunk moi
# 6. Luu chunk cuoi cung
def build_semantic_chunks_for_video(time_chunks: list[dict], config) -> list[dict]:
    if not time_chunks:
        return []

    semantic_chunks = []
    # Khoi tao chunk dau tien
    current = {
        **time_chunks[0],
        "chunk_type": "semantic",
        "parent_time_chunk_start": time_chunks[0]["chunk_index"],
        "parent_time_chunk_end": time_chunks[0]["chunk_index"],
        "parent_time_chunk_ids": [time_chunks[0]["chunk_id"]],
        "time_chunk_count": 1,
    }

    # Duyet qua cac chunk tiep theo
    for nxt_raw in time_chunks[1:]:
        nxt = {
            **nxt_raw,
            "chunk_type": "semantic",
            "parent_time_chunk_start": nxt_raw["chunk_index"],
            "parent_time_chunk_end": nxt_raw["chunk_index"],
            "parent_time_chunk_ids": [nxt_raw["chunk_id"]],
            "time_chunk_count": 1,
        }

        # Kiem tra co nen gop khong
        if should_merge_semantic(current, nxt, config):
            current = merge_semantic_pair(current, nxt)
        else:
            # Luu chunk cu va bat dau chunk moi
            current["chunk_index"] = len(semantic_chunks)
            current["chunk_id"] = build_chunk_id(
                current["video_id"],
                "semantic",
                current["chunk_index"],
                current["start_time"],
                current["end_time"],
                config.chunk_version,
            )
            semantic_chunks.append(current)
            current = nxt

    # Luu chunk cuoi cung
    current["chunk_index"] = len(semantic_chunks)
    current["chunk_id"] = build_chunk_id(
        current["video_id"],
        "semantic",
        current["chunk_index"],
        current["start_time"],
        current["end_time"],
        config.chunk_version,
    )
    semantic_chunks.append(current)
    return semantic_chunks
