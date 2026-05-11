# Transcript Chunking Pipeline

Xử lý transcript YouTube thành các chunks tối ưu cho RAG system.

##  Cấu trúc

```
data/chunked/
├── pipeline.py                      # Entry point - chạy chunking
├── chunk_config.py                 # Cấu hình pipeline
├── chunk_utils.py                  # Hàm core
├── cleanup_for_embedding.py         # Clean + context injection
│
├── README.md                        # File này
├── EMBEDDING_READINESS_REPORT.md    # Báo cáo đánh giá
├── HYBRID_CHUNK_SCHEMA_INCIDENT.md  # Ghi nhận sự cố
│
├── transcripts_time_chunked.jsonl   # Output: time-based
├── transcripts_semantic_chunked.jsonl # Output: semantic
├── transcripts_hybrid_chunked.jsonl # Output: hybrid (gốc)
└── transcripts_enhanced.jsonl      # ✅ Output cuối - sẵn sàng embed
```

##  Cách chạy

### Bước 1: Chạy pipeline chunking

```bash
cd D:\youtube-rag-scraper\data\chunked

# Chạy hybrid (mặc định) - khuyến nghị
python pipeline.py --strategy hybrid

# Hoặc chạy từng loại
python pipeline.py --strategy time      # Chỉ time-based
python pipeline.py --strategy semantic  # Chỉ semantic
```

### Bước 2: Clean và chuẩn bị cho embedding

```bash
python cleanup_for_embedding.py
```

**Output:** `transcripts_enhanced.jsonl`

### Bước 3: Embed (sang folder embed)

```bash
# Copy sang data/ để embed script đọc được
copy transcripts_enhanced.jsonl ..\data\semantic_chunks.jsonl

# Hoặc chạy trực tiếp
python ..\embed\files\01_embed_and_index.py --input transcripts_enhanced.jsonl
```

##  Chiến lược Chunking

| Chiến lược | Mô tả | Output |
|------------|-------|--------|
| `time` | Chia theo thời gian (mỗi ~60s) | `transcripts_time_chunked.jsonl` |
| `semantic` | Gom nhóm theo nội dung liên quan | `transcripts_semantic_chunked.jsonl` |
| `hybrid` | Time → Semantic kết hợp | `transcripts_hybrid_chunked.jsonl` |

##  Input

- **File:** `data/cleaned/transcripts_clean.jsonl`
- **Format:**
```json
{
  "video_id": "xxx",
  "title": "Video Title",
  "transcript": "Full transcript...",
  "duration_seconds": 600
}
```

##  Output (transcripts_enhanced.jsonl)

```json
{
  "chunk_id": "phWxl0nkgKk:semantic:v1:0000:0000005880:0000045920",
  "video_id": "phWxl0nkgKk",
  "title": "Stanford CS25: V2 I Strategic Games",
  "course": "CS25_Transformers",
  "source": "stanford_youtube",
  "start_time": 5.88,
  "end_time": 45.92,
  "duration": 40.04,
  "chunk_text": "...",           // Transcript gốc (debug/citation)
  "embedding_text": "Course: CS25_Transformers\nVideo: ...\n\nThe bots were trained...", // Đã clean + context
  "word_count": 119,
  "token_estimate": 155
}
```

##  Cấu hình (chunk_config.py)

```python
target_tokens: 150          # Tokens mục tiêu/chunk
max_chunk_duration: 60.0    # Giây (time-based)
max_semantic_words: 220     # Từ (semantic)
overlap_segments: 2         # Overlap giữa các chunk
min_lexical_overlap: 0.08   # 8% overlap tối thiểu
```

##  Cleanup (cleanup_for_embedding.py)

Script xử lý:

1. **Context Injection** - Thêm header vào text:
   ```
   Course: CS25_Transformers
   Video: Stanford CS25: V2 I Strategic Games
   Source: stanford_youtube
   Time: 00:05 - 00:45
   ```

2. **Filler Word Removal** - Loại bỏ:
   - like, basically, you know, kind of
   - I think, I mean, well, yeah, okay...

3. **Text Cleanup**:
   - Viết hoa đầu câu
   - Thêm dấu chấm cuối
   - Xóa khoảng trắng thừa

##  Thống kê

- **Tổng chunks:** 18,395
- **Average words/chunk:** ~120
- **Estimated tokens/chunk:** ~150-180

##  Checklist trước khi embed

- [x] transcripts_hybrid_chunked.jsonl tồn tại
- [x] Chạy `python cleanup_for_embedding.py`
- [x] transcripts_enhanced.jsonl đã tạo
- [x] Kiểm tra filler words đã remove
- [x] Context header đúng format

##  Pipeline hoàn chỉnh

```
data/cleaned/transcripts_clean.jsonl
         ↓
   pipeline.py --strategy hybrid
         ↓
transcripts_hybrid_chunked.jsonl
         ↓
   cleanup_for_embedding.py
         ↓
transcripts_enhanced.jsonl  ← DÙNG ĐỂ EMBED
         ↓
   embed/files/01_embed_and_index.py
         ↓
   embed/files/index/ (FAISS + metadata)
```

##  Yêu cầu

- Python 3.10+
- Input: `data/cleaned/transcripts_clean.jsonl`
- Dependencies: xem `requirements.txt` root

---

**Ghi chú:** Giữ nguyên `chunk_text` cho debug/citation, dùng `embedding_text` để tạo vector.