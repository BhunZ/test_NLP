import logging  # Thư viện chuẩn Python để ghi log, theo dõi quá trình chạy
import os  # Thư viện để thao tác với hệ thống file, tạo thư mục
import random  # Thư viện để tạo số ngẫu nhiên, dùng cho sleep ngẫu nhiên
import time  # Thư viện để xử lý thời gian, delay

from config import (  # Import các biến cấu hình từ file config.py
    BLOCK_COOLDOWN_SECONDS,  # Số giây nghỉ khi bị block liên tục
    BLOCK_STREAK_THRESHOLD,  # Ngưỡng số lần block liên tục để dừng chương trình
    FAILED_CACHE_FILE,  # Đường dẫn file lưu các video đã thất bại
    LONG_BREAK_EVERY_N_SUCCESS,  # Số lần thành công để nghỉ dài
    LONG_BREAK_SECONDS,  # Số giây nghỉ dài
    MAX_RETRIES,  # Số lần thử lại tối đa khi lỗi
    MIN_SEGMENTS,  # Số đoạn transcript tối thiểu để chấp nhận
    PLAYLISTS,  # Danh sách playlist cần crawl
    RETRY_DELAY,  # Số giây chờ trước khi thử lại
    SLEEP_BETWEEN_CALLS_MAX,  # Thời gian chờ tối đa giữa các lần gọi API
    SLEEP_BETWEEN_CALLS_MIN,  # Thời gian chờ tối thiểu giữa các lần gọi API
    SUCCESS_CACHE_FILE,  # Đường dẫn file lưu các video đã thành công
    TARGET_MAX,  # Số lượng video tối đa cần xử lý
    YOUTUBE_API_KEY,  # API key của YouTube Data API
)
from youtube_scraper.knowledge_base import (
    append_jsonl,
)  # Hàm ghi dữ liệu vào file jsonl
from youtube_scraper.metadata import (
    get_playlist_videos,
)  # Hàm lấy danh sách video từ playlist
from youtube_scraper.transcripts import fetch_transcript  # Hàm lấy transcript của video
from youtube_scraper.utils import (
    load_processed_ids,
    save_processed_id,
)  # Hàm đọc/ghi cache video đã xử lý


logging.basicConfig(level=logging.INFO)  # Cấu hình logging hiển thị level INFO trở lên


def run():  # Hàm chính để chạy chương trình crawl transcript
    os.makedirs("assets", exist_ok=True)  # Tạo thư mục assets nếu chưa tồn tại

    # Đọc danh sách video đã xử lý thành công từ file cache
    success_ids = load_processed_ids(SUCCESS_CACHE_FILE)
    # Đọc danh sách video đã thất bại từ file cache
    failed_ids = load_processed_ids(FAILED_CACHE_FILE)

    # Đảm bảo thời gian sleep tối thiểu là 1 giây
    sleep_min = max(SLEEP_BETWEEN_CALLS_MIN, 1.0)
    # Đảm bảo thời gian sleep tối đa không nhỏ hơn tối thiểu
    sleep_max = max(SLEEP_BETWEEN_CALLS_MAX, sleep_min)

    total = 0  # Tổng số video đã xử lý thành công
    success = 0  # Số video lấy transcript thành công
    skipped = 0  # Số video bị bỏ qua (không có transcript tiếng Anh)
    failed = 0  # Số video thất bại (lỗi không xác định)
    consecutive_blocked = 0  # Số lần bị block liên tiếp

    # Duyệt qua từng playlist trong danh sách cấu hình
    for i, pl in enumerate(PLAYLISTS):
        name = pl["name"]  # Tên playlist
        url = pl["url"]  # URL của playlist

        logging.info(
            f"PLAYLIST {i + 1}/{len(PLAYLISTS)}: {name}"
        )  # Log tiến độ playlist hiện tại
        try:
            # Gọi API YouTube để lấy danh sách video trong playlist
            videos = get_playlist_videos(YOUTUBE_API_KEY, url)
        except Exception as e:
            # Nếu lỗi khi lấy playlist, log lỗi và chờ rồi chuyển playlist tiếp theo
            logging.error(f"FAILED TO LOAD PLAYLIST {name}: {e}")
            time.sleep(max(RETRY_DELAY, 10))
            continue

        # Duyệt qua từng video trong playlist
        for v in videos:
            vid = v["video_id"]  # Lấy ID video

            # Nếu video đã xử lý thành công trước đó thì bỏ qua
            if vid in success_ids:
                continue

            # Log thông tin video đang xử lý
            logging.info(f"[{total + skipped + failed}] Processing {vid}")

            try:
                # Gọi hàm fetch_transcript để lấy transcript của video
                result = fetch_transcript(
                    vid,
                    min_segments=MIN_SEGMENTS,  # Số đoạn tối thiểu
                    retries=MAX_RETRIES,  # Số lần thử lại
                    delay=RETRY_DELAY,  # Thời gian chờ giữa các lần thử
                )

                status = result.get("status", "error")  # Lấy trạng thái kết quả

                # Nếu lấy transcript thành công
                if status == "success":
                    # Tạo bản ghi chứa thông tin video và transcript
                    record = {
                        "video_id": vid,  # ID video
                        "title": v["title"],  # Tiêu đề video
                        "playlist_id": v["playlist_id"],  # ID playlist
                        "course": name,  # Tên khóa học/playlist
                        "source": "stanford_youtube",  # Nguồn dữ liệu
                        "language": result["language"],  # Ngôn ngữ transcript
                        "is_auto_generated": result[
                            "is_auto_generated"
                        ],  # Có phải tự động tạo không
                        "published_at": v["published_at"],  # Ngày đăng video
                        "transcript": result["transcript"],  # Nội dung transcript
                    }

                    # Ghi bản ghi vào file jsonl
                    append_jsonl(record)
                    # Lưu ID video vào cache thành công
                    save_processed_id(SUCCESS_CACHE_FILE, vid)
                    # Thêm vào set để kiểm tra nhanh hơn
                    success_ids.add(vid)

                    success += 1  # Tăng biến đếm thành công
                    total += 1  # Tăng tổng số đã xử lý
                    consecutive_blocked = 0  # Reset số lần block

                    # Nếu đủ số lần thành công thì nghỉ dài
                    if total % LONG_BREAK_EVERY_N_SUCCESS == 0:
                        logging.info("Taking a longer break...")
                        time.sleep(LONG_BREAK_SECONDS)

                    # Log thành công
                    logging.info(f"SUCCESS {vid} | total={total}")

                    # Nếu đã đạt số lượng mục tiêu thì dừng chương trình
                    if total >= TARGET_MAX:
                        logging.info("Reached TARGET_MAX")
                        return

                    # Chờ ngẫu nhiên trước khi xử lý video tiếp theo
                    time.sleep(random.uniform(sleep_min, sleep_max))
                    continue

                # Nếu video không có transcript tiếng Anh hoặc quá ngắn thì bỏ qua
                if status in {
                    "no_en",  # Không có transcript tiếng Anh
                    "too_short",  # Transcript quá ngắn
                    "no_transcript",  # Không có transcript
                    "transcripts_disabled",  # Transcript bị tắt
                }:
                    # Lưu vào cache thất bại nếu chưa có
                    if vid not in failed_ids:
                        save_processed_id(FAILED_CACHE_FILE, vid)
                        failed_ids.add(vid)

                    skipped += 1  # Tăng biến đếm bỏ qua
                    consecutive_blocked = 0  # Reset số lần block
                    logging.info(f"SKIP {vid} | reason={status}")  # Log lý do bỏ qua
                    time.sleep(
                        random.uniform(sleep_min, sleep_max)
                    )  # Chờ ngẫu nhiên rồi tiếp tục
                    continue

                # Nếu trạng thái là lỗi khác (không phải thành công hay bỏ qua)
                failed += 1  # Tăng biến đếm thất bại

                # Nếu bị block
                if status == "blocked":
                    consecutive_blocked += 1  # Tăng số lần block liên tiếp
                    logging.warning(
                        f"BLOCKED {vid} | consecutive_blocked={consecutive_blocked} | "
                        f"error={result.get('error', 'unknown')}"
                    )
                else:
                    # Lỗi khác không phải block
                    consecutive_blocked = 0
                    logging.error(
                        f"ERROR {vid} | status={status} | "
                        f"error={result.get('error', 'unknown')}"
                    )

                # Nếu số lần block vượt ngưỡng thì nghỉ dài và dừng chương trình
                if consecutive_blocked >= BLOCK_STREAK_THRESHOLD:
                    logging.warning(
                        "Detected too many consecutive blocked transcript requests. "
                        f"Cooling down for {BLOCK_COOLDOWN_SECONDS}s and stopping run."
                    )
                    time.sleep(BLOCK_COOLDOWN_SECONDS)  # Nghỉ theo cấu hình
                    return  # Kết thúc hàm

                # Chờ ngẫu nhiên rồi tiếp tục xử lý video tiếp theo
                time.sleep(random.uniform(sleep_min, sleep_max))

            except Exception as e:
                # Bắt lỗi không mong muốn khi xử lý video
                failed += 1
                consecutive_blocked = 0
                logging.error(f"FATAL {vid}: {e}")  # Log lỗi nghiêm trọng
                time.sleep(max(RETRY_DELAY, sleep_min))  # Chờ rồi tiếp tục
                continue

    # Khi hoàn thành tất cả playlist, log tổng kết
    logging.info(f"DONE | success={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    run()  # Chạy hàm main khi chạy file trực tiếp
