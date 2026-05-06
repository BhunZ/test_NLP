# Chunked Data Pipeline

Xu ly transcript thanh cac knowledge units cho RAG.

## Pipeline Flow

```
transcripts_clean_sentence.jsonl (248 videos)
         |
         v
  knowledge_units_1.py        (Time chunking + Semantic merge)
         |
         v
   transcript_v3.jsonl          (13,561 chunks)
         |
         v
    pipeline2.py                 (Enrich metadata)
         |
         v
   transcript_v4.jsonl          (13,561 chunks + Keywords + VI hints)
```

## Cac File

### knowledge_units_1.py
Tao knowledge units tu transcript.

- Time chunking: Chia transcript thanh chunks theo thoi gian va token
- Semantic merge: Merge cac chunk lien ke neu similar (giu timeline)

### pipeline2.py
Enrich metadata cho cac chunk.

- Trich xuat keywords tieng Anh
- Dich keywords sang tieng Viet
- Tao embedding_text cho RAG
- Them chunk_id va chunk_index

### config1.py
Cau hinh cho pipeline2.

### utils1.py
Ham tien tro cho pipeline2.

## Chay Pipeline

Buoc 1: Time chunking + Semantic merge
```bash
python knowledge_units_1.py
```

Buoc 2: Enrich metadata
```bash
python pipeline2.py
```

## Token Config (knowledge_units_1.py)

| Variable | Gia tri | Mo ta |
|----------|--------|--------|
| MIN_TOKENS | 120 | Token toi thieu |
| TARGET_TOKENS | 160 | Token muc tieu |
| MAX_TOKENS | 190 | Token toi da |
| HARD_MAX_TOKENS | 215 | Token cuc dai |
| MAX_MERGED_TOKENS | 215 | Token toi da sau merge |

## Similarity Config

| Variable | Gia tri | Mo ta |
|----------|--------|--------|
| MIN_SIMILARITY | 0.22 | Toi thieu de merge |
| CONTINUATION_SIMILARITY | 0.12 | Cho tinh tiep theo |

## Gap Config

| Variable | Gia tri | Mo ta |
|----------|--------|--------|
| SOFT_GAP_SECONDS | 0.9 | Cho phep merge |
| HARD_GAP_SECONDS | 1.8 | Khong merge |

## Quy Doi

| | Gia tri |
|---------|--------|
| 1 segment | ~15-20 words |
| 1 chunk | 120-215 tokens |
| 1 token | ~1.28 words |

## Input/Output

| File | So luong | Mo ta |
|------|---------|-------|
| transcripts_clean_sentence.jsonl | 248 videos | Transcript da lam sach |
| transcript_v3.jsonl | 13,561 chunks | Time + Semantic chunks |
| transcript_v4.jsonl | 13,561 chunks | Enrich metadata |

## Chunk Metadata (transcript_v4.jsonl)

| Field | Mo ta |
|-------|-------|
| video_id | ID video goc |
| title | Ten bai giang |
| course | Ten khoa hoc |
| source | Nguon (stanford_youtube...) |
| chunk_type | crosslingual_knowledge |
| chunk_text | Noi dung chunk |
| embedding_text | Text cho vector hoa |
| start_time | Thoi diem bat dau (giay) |
| end_time | Thoi diem ket thuc (giay) |
| duration | Thoi luong chunk |
| token_count | So token |
| semantic_topic | Topic: Training, Transformer, NLP... |
| chunk_id | Unique ID: video_0001 |
| chunk_index | Thu tu trong video |

## Embedding Text Format

```
Course: {course}
Lecture: {title}
Time: {start} - {end}
Topic: {topic}
Keywords: {keywords}
Keywords (Vietnamese): {vi_keywords}

Content:
{chunk_text}
```

## Semantic Topic

Cac topic duoc phat hien:
- Neural Networks
- Transformer
- Training
- NLP
- Computer Vision
- Reinforcement Learning
- Math/Statistics
- General

## Notes

- Giu nguyen timeline (khong nhay lung tung)
- Semantic chi merge cac chunk lien ke
- Topic chi dung de annotate, khong quyet dinh merge
- chunk_text giu nguyen cho debug/citation
- embedding_text dung de tao vector