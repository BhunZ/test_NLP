import json
import re
from pathlib import Path

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
# ROOT_DIR: tự động lấy thư mục gốc của project
# → giúp code portable, không cần hardcode path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# File input: transcript thô (raw từ YouTube API / crawler)
INPUT_PATH = ROOT_DIR / "data" / "raw" / "transcripts.jsonl"

# File output: transcript đã clean + tách câu (sentence-level)
OUTPUT_PATH = ROOT_DIR / "data" / "cleaned" / "transcripts_clean_sentence.jsonl"


# ================= REGEX CLEAN =================
# Các pattern regex dùng để loại bỏ noise trong transcript

# Xóa các marker như [applause], [laughter], ...
BRACKET_NOISE_RE = re.compile(
    r"\[(?:inaudible|applause|laughter|music|silence|noise|crosstalk).*?\]",
    re.IGNORECASE,
)

# Xóa HTML tag như <font>, <c>, ...
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Xóa timestamp dạng "00:00 --> 00:01"
TIMESTAMP_RANGE_RE = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?\b"
)

# Xóa timestamp đơn lẻ "00:00"
STANDALONE_TIMESTAMP_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")

# Xóa filler words ở đầu câu (uh, um, well, so...)
LEADING_FILLER_RE = re.compile(
    r"^(?:(?:uh|um|erm|er|ah|hmm|mm|okay|ok|well|so|yeah)\b[\s,.-]*)+",
    re.IGNORECASE,
)

# Xóa filler ở cuối câu
TRAILING_FILLER_RE = re.compile(
    r"(?:[\s,.;!?-]*(?:uh|um|erm|er|ah|hmm|mm|you know|i mean))+$",
    re.IGNORECASE,
)

# Xóa filler nằm trong câu (giữ lại dấu câu xung quanh)
INLINE_FILLER_PATTERNS = [
    re.compile(r"(?i)(^|[\s,.;!?-])(?:uh|um|erm|er|ah|hmm|mm)(?=$|[\s,.;!?-])"),
    re.compile(r"(?i)(^|[\s,.;!?-])(?:you know|i mean)(?=$|[\s,.;!?-])"),
]


# ================= HÀM CLEAN TEXT =================
def clean_text(text: str) -> str:
    """
    Làm sạch text của 1 segment transcript.

    Mục tiêu:
    - loại bỏ noise (timestamp, filler, html)
    - chuẩn hóa text để split sentence chính xác hơn
    """

    if not text:
        return ""

    # 1. Xóa newline → tránh split sai câu
    text = text.replace("\r", " ").replace("\n", " ")

    # 2. Xóa timestamp
    text = TIMESTAMP_RANGE_RE.sub(" ", text)
    text = STANDALONE_TIMESTAMP_RE.sub(" ", text)

    # 3. Xóa noise và HTML
    text = BRACKET_NOISE_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)

    # 4. Chuẩn hóa whitespace (nhiều space → 1 space)
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Xóa filler ở đầu câu
    text = LEADING_FILLER_RE.sub("", text)

    # 6. Xóa filler trong câu
    for pattern in INLINE_FILLER_PATTERNS:
        text = pattern.sub(lambda m: m.group(1), text)

    # 7. Xóa filler cuối câu
    text = TRAILING_FILLER_RE.sub("", text)

    # 8. Chuẩn hóa dấu câu
    text = re.sub(r"\s+([,.;!?])", r"\1", text)   # bỏ space trước dấu
    text = re.sub(r"([,.;!?]){2,}", r"\1", text)  # !!! → !

    # 9. Xóa dấu câu dư ở đầu/cuối
    return text.strip(" -.,;:!?")


# ================= SPLIT SENTENCE =================
def split_sentences(text: str):
    """
    Tách text thành các câu.

    Logic:
    - split sau dấu . ! ?
    - giữ lại dấu câu
    """

    sentences = re.split(r'(?<=[.!?])\s+', text)

    # loại bỏ câu rỗng hoặc quá ngắn (noise)
    return [
        s.strip()
        for s in sentences
        if s.strip() and len(s.strip()) > 3
    ]


# ================= MERGE SENTENCE NGẮN =================
def merge_short_sentences(sentences, min_len=20):
    """
    Merge các câu quá ngắn để tránh:
    - noise (Okay, So, Yeah...)
    - embedding kém chất lượng

    Logic:
    - câu ngắn → merge với câu trước
    - câu trước ngắn → cũng merge
    - nếu câu trước kết thúc bằng dấu yếu (",", "and") → merge
    """

    merged = []

    for sent in sentences:
        if not merged:
            merged.append(sent)
            continue

        prev = merged[-1]

        # nếu câu trước kết thúc bằng dấu "yếu" → khả năng chưa kết thúc ý
        if prev.endswith((",", "and", "or", "but")):
            merged[-1] += " " + sent

        # nếu câu hiện tại quá ngắn → merge vào câu trước
        elif len(sent) < min_len:
            merged[-1] += " " + sent

        # nếu câu trước quá ngắn → merge tiếp
        elif len(prev) < min_len:
            merged[-1] += " " + sent

        else:
            merged.append(sent)

    return merged


# ================= SPLIT SEGMENT =================
def split_segment(seg):
    """
    Input:
        seg = {
            "text": "...",
            "start": ...,
            "duration": ...
        }

    Output:
        list các segment nhỏ (sentence-level)

    Ý tưởng:
    - clean text
    - split thành câu
    - merge câu ngắn
    - chia lại timestamp theo tỷ lệ độ dài câu
    """

    # 1. Clean text
    text = clean_text(seg.get("text", ""))
    if not text:
        return []

    # 2. Split thành câu
    sentences = split_sentences(text)

    # 3. Merge câu ngắn để giảm noise
    sentences = merge_short_sentences(sentences)

    # 4. Tính tổng độ dài (dùng cho chia timestamp)
    total_len = sum(len(s) for s in sentences)

    current_start = seg["start"]
    duration = seg["duration"]

    results = []

    for sent in sentences:
        # Tỷ lệ độ dài câu
        ratio = len(sent) / total_len if total_len > 0 else 0

        # Duration của câu (approximate)
        sent_duration = duration * ratio

        results.append({
            "text": sent,
            "start": round(current_start, 3),
            "duration": round(sent_duration, 3)
        })

        # cập nhật start cho câu tiếp theo
        current_start += sent_duration

    return results


# ================= MAIN PIPELINE =================
def process_file():
    """
    Pipeline chính:

    raw transcript
    → clean từng segment
    → split thành sentence-level
    → merge noise
    → ghi file mới
    """

    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)

            new_segments = []

            # duyệt từng segment trong transcript
            for seg in record.get("transcript", []):
                # split thành sentence-level segments
                new_segments.extend(split_segment(seg))

            # ghi đè transcript bằng version mới
            record["transcript"] = new_segments

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done:", OUTPUT_PATH)


# ================= ENTRY POINT =================
if __name__ == "__main__":
    process_file()