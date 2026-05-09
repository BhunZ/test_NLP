import os  # Thư viện để thao tác với hệ thống file


def load_processed_ids(path):
    """
    Đọc danh sách các ID video đã được xử lý từ file cache
    path: Đường dẫn đến file chứa danh sách ID
    """
    # Nếu file không tồn tại thì trả về set rỗng
    if not os.path.exists(path):
        return set()
    # Đọc file, mỗi dòng là một ID, bỏ các dòng trống
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def save_processed_id(path, vid):
    """
    Lưu ID video đã xử lý vào file cache
    path: Đường dẫn đến file cache
    vid: ID của video đã xử lý
    """
    # Mở file ở chế độ append (thêm vào cuối)
    with open(path, "a", encoding="utf-8") as f:
        # Ghi ID video, loại bỏ khoảng trắng thừa và thêm dòng mới
        f.write(vid.strip() + "\n")


def extract_playlist_id(url: str):
    """
    Trích xuất playlist ID từ URL YouTube
    url: URL của playlist YouTube
    """
    # Tách chuỗi theo "list=" và lấy phần sau, tách tiếp theo "&" để bỏ các tham số khác
    return url.split("list=")[-1].split("&")[0]
