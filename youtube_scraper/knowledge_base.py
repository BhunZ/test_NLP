import json  # Thư viện để xử lý JSON
import os  # Thư viện để thao tác với hệ thống file
from config import OUTPUT_FILE  # Import đường dẫn file output từ config


def append_jsonl(record):
    """
    Ghi một bản ghi vào file JSONL (JSON Lines)
    record: Dictionary chứa thông tin video và transcript
    """
    # Tạo thư mục chứa file output nếu chưa tồn tại
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Mở file ở chế độ append (thêm vào cuối), encoding UTF-8 để hỗ trợ tiếng Việt
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # Chuyển dictionary thành chuỗi JSON, đảm bảo không escape ký tự Unicode
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
