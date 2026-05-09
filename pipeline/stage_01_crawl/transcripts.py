import time  # Thư viện để xử lý thời gian, dùng cho sleep khi retry
from youtube_transcript_api import (
    YouTubeTranscriptApi,
)  # Thư viện để lấy transcript từ YouTube

# Tuple chứa các chuỗi để nhận diện khi bị YouTube block
BLOCK_HINTS = (
    "429",
    "too many requests",
    "request blocked",
    "unusual traffic",
    "ip blocked",
)


def classify_transcript_error(exc):
    """
    Phân loại lỗi khi lấy transcript để trả về mã lỗi phù hợp
    """
    msg = str(exc).lower()  # Chuyển thông báo lỗi thành chữ thường để so sánh
    # Kiểm tra nếu lỗi liên quan đến việc bị block (429, rate limit, v.v.)
    if any(hint in msg for hint in BLOCK_HINTS):
        return "blocked"
    # Kiểm tra nếu transcript bị tắt bởi người đăng video
    if "transcript is disabled" in msg:
        return "transcripts_disabled"
    # Kiểm tra nếu không có transcript cho video này
    if "no transcripts were found" in msg or "no transcript" in msg:
        return "no_transcript"
    # Các lỗi khác
    return "error"


def fetch_transcript(video_id, min_segments=50, retries=3, delay=5):
    """
    Lấy transcript của một video YouTube
    video_id: ID của video YouTube
    min_segments: Số đoạn transcript tối thiểu để chấp nhận (mặc định 50)
    retries: Số lần thử lại khi lỗi (mặc định 3)
    delay: Số giây chờ giữa các lần thử (mặc định 5)
    """
    ytt = YouTubeTranscriptApi()  # Khởi tạo đối tượng API để lấy transcript
    last_error = None  # Biến lưu lỗi cuối cùng

    # Thử lấy transcript nhiều lần nếu lỗi
    for attempt in range(1, retries + 1):
        try:
            # Lấy danh sách các bản transcript có sẵn cho video
            t_list = ytt.list(video_id)
            # Lọc chỉ lấy các transcript tiếng Anh
            en_tracks = [t for t in t_list if t.language_code.startswith("en")]

            # Nếu không có transcript tiếng Anh thì trả về lỗi
            if not en_tracks:
                return {"status": "no_en"}

            # Lọc các transcript được tạo thủ công (không phải auto-generated)
            manual = [t for t in en_tracks if not t.is_generated]
            # Ưu tiên transcript thủ công, nếu không có thì dùng auto-generated
            transcript = manual[0] if manual else en_tracks[0]
            # Đánh dấu có phải tự động tạo hay không
            is_auto = not bool(manual)

            # Lấy nội dung transcript
            raw_segments = transcript.fetch()
            # Chuyển đổi sang định dạng dict với text, start time, duration
            segments = [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in raw_segments
            ]

            # Nếu không có đoạn nào hoặc ít hơn số tối thiểu thì trả về lỗi
            if not segments or len(segments) < min_segments:
                return {"status": "too_short", "segment_count": len(segments)}

            # Trả về kết quả thành công với đầy đủ thông tin
            return {
                "status": "success",
                "language": transcript.language_code,  # Mã ngôn ngữ
                "is_auto_generated": is_auto,  # Có phải tự động tạo không
                "transcript": segments,  # Danh sách các đoạn transcript
            }

        except Exception as e:
            last_error = str(e)  # Lưu lỗi vào biến
            status = classify_transcript_error(e)  # Phân loại lỗi
            # Nếu là các lỗi không cần retry (bị block, không có transcript, vô hiệu hóa)
            if status in {"blocked", "transcripts_disabled", "no_transcript"}:
                return {"status": status, "error": last_error}
            # Nếu chưa hết số lần thử thì chờ rồi thử lại
            if attempt < retries:
                time.sleep(delay * attempt)

    # Nếu đã thử hết retries mà vẫn lỗi thì trả về lỗi chung
    return {"status": "error", "error": last_error or "unknown"}
