"""
core/utils.py – Utility functions cho RAG pipeline.
normalize - format timestamp - make youtube url
"""
import logging
import re
from typing import Optional
from urllib.parse import urlencode


def setup_logging(level: int = logging.INFO) -> None:
    """
    Cấu hình logging format thống nhất cho toàn bộ pipeline.

    Format:
        [2024-04-27 10:30:45] INFO:mymodule: Message here
    """
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize text: remove extra whitespace, newlines, tabs.

    Dùng trước khi embed để đảm bảo input sạch và consistent.

    Input:
        "  Hello\n  world  \t  "
    Output:
        "Hello world"
    """
    if not text:
        return ""

    # Chuyển newline/tab thành space
    text = re.sub(r"[\n\r\t]+", " ", text)
    # Xóa space dư
    text = re.sub(r"\s+", " ", text)
    # Strip đầu cuối
    return text.strip()


def format_timestamp(seconds: float) -> str:
    """
    Chuyển seconds thành định dạng HH:MM:SS.

    Input:
        125.5
    Output:
        "00:02:05"
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def make_youtube_url(video_id: str, start_time: float = 0) -> str:
    """
    Tạo YouTube URL với timestamp.

    Input:
        video_id="dQw4w9WgXcQ", start_time=125.5
    Output:
        "https://youtu.be/dQw4w9WgXcQ?t=125"

    Dùng để tạo link nhảy đến đúng vị trí trong video.
    """
    return f"https://youtu.be/{video_id}?t={int(start_time)}"
