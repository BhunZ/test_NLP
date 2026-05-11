import requests  # Thư viện để gọi HTTP request
from config import HTTP_TIMEOUT  # Import thời gian chờ tối đa cho HTTP request
from youtube_scraper.utils import (
    extract_playlist_id,
)  # Hàm trích xuất playlist ID từ URL


def get_playlist_videos(api_key, playlist_url):
    """
    Lấy danh sách tất cả video trong một playlist YouTube
    api_key: YouTube Data API key
    playlist_url: URL của playlist YouTube
    """
    # Trích xuất playlist ID từ URL
    playlist_id = extract_playlist_id(playlist_url)
    videos = []  # Danh sách video sẽ trả về
    next_page_token = None  # Token để lấy trang tiếp theo

    # Vòng lặp vô hạn để lấy tất cả video (sẽ break khi không còn trang tiếp theo)
    while True:
        # URL của YouTube Data API endpoint để lấy playlist items
        url = "https://www.googleapis.com/youtube/v3/playlistItems"
        # Tham số truy vấn
        params = {
            "part": "snippet",  # Yêu cầu lấy thông tin snippet (tiêu đề, mô tả, v.v.)
            "playlistId": playlist_id,  # ID của playlist
            "maxResults": 50,  # Số kết quả tối đa mỗi trang (tối đa là 50)
            "key": api_key,  # API key để xác thực
        }
        # Nếu có token của trang tiếp theo thì thêm vào params
        if next_page_token:
            params["pageToken"] = next_page_token

        # Gọi API GET với timeout
        res = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        res.raise_for_status()  # Nếu HTTP status là lỗi thì raise exception
        data = res.json()  # Chuyển đổi response sang JSON

        # Kiểm tra nếu API trả về lỗi
        if "error" in data:
            raise RuntimeError(data["error"])

        # Duyệt qua từng item trong kết quả
        for item in data.get("items", []):
            snippet = item["snippet"]
            # Bỏ qua nếu không phải video (có thể là kênh, playlist khác)
            if snippet["resourceId"]["kind"] != "youtube#video":
                continue

            # Thêm thông tin video vào danh sách
            videos.append(
                {
                    "video_id": snippet["resourceId"]["videoId"],  # ID của video
                    "title": snippet["title"],  # Tiêu đề video
                    "playlist_id": playlist_id,  # ID playlist chứa video
                    "published_at": snippet.get("publishedAt"),  # Ngày đăng video
                }
            )

        # Lấy token của trang tiếp theo
        next_page_token = data.get("nextPageToken")
        # Nếu không còn trang tiếp theo thì thoát vòng lặp
        if not next_page_token:
            break

    # Loại bỏ các video trùng lặp (dựa trên video_id) và trả về danh sách
    return list({v["video_id"]: v for v in videos}.values())
