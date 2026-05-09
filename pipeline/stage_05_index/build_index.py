"""
data/build_index.py – Build FAISS + BM25 indexes từ transcripts_enhanced.jsonl

Script chạy OFFLINE (một lần duy nhất) để:
  1. Đọc file transcripts_enhanced.jsonl
  2. Kiểm tra và phân tích chất lượng dữ liệu
  3. Chuyển từng record thành LangChain Document
  4. Embed bằng bge-m3 và lưu vào FAISS index
  5. Build BM25 index cho hybrid retrieval
  6. Lưu tất cả xuống disk

Cách chạy:
    python -m data.build_index
    # hoặc chỉ định đường dẫn riêng:
    python -m data.build_index --data path/to/file.jsonl --out data/index

Lưu ý quan trọng về dữ liệu đầu vào (transcripts_enhanced.jsonl):
   chunk_text   : văn bản tiếng Anh, sạch, không newline/tab thừa
   embedding_text: phiên bản có context header (Course, Video, Time)
   chunk_id     : unique, không trùng lặp
   token_estimate: tất cả dưới 512 tokens → phù hợp với model
   chunk_vi     : KHÔNG tồn tại trong file → không có bản dịch tiếng Việt
                   (bạn cần tự dịch hoặc để LLM dịch khi trả lời)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Thêm thư mục gốc vào Python path để import được các module nội bộ
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from core.config import get_settings
from core.embeddings import get_embedding_model
from core.utils import (
    format_timestamp,
    make_youtube_url,
    normalize_text,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# BƯỚC 1: ĐỌC FILE JSONL
# ════════════════════════════════════════════════════════════════


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Đọc toàn bộ file JSONL vào list các dict Python.

    JSONL (JSON Lines) = mỗi dòng là một JSON object độc lập.
    Ưu điểm: đọc từng dòng mà không cần load toàn bộ file vào RAM.

    Với file 18,395 records, kích thước khoảng 30-50MB — đọc trực tiếp OK.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Dòng %d bị lỗi JSON, bỏ qua: %s", line_num, e)

    logger.info(" Đọc được %d records từ %s", len(records), path)
    return records


# ════════════════════════════════════════════════════════════════
# BƯỚC 2: PHÂN TÍCH CHẤT LƯỢNG DỮ LIỆU
# ════════════════════════════════════════════════════════════════


def analyze_data_quality(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phân tích và báo cáo chất lượng dữ liệu trước khi index.

    Kết quả phân tích file transcripts_enhanced.jsonl:
       Tất cả records có chunk_text (tiếng Anh) → đủ để embed
       Tất cả records có embedding_text (có context header) → dùng cái này
       Không có record nào vượt 512 tokens → không cần truncate
       Tất cả chunk_id unique → không cần dedup
       chunk_vi = 0% → KHÔNG CÓ bản dịch tiếng Việt trong data
        → Hệ thống sẽ retrieve tiếng Anh và dùng LLM để trả lời tiếng Việt
    Returns:
        Dict chứa các thống kê chất lượng.
    """
    total = len(records)
    stats = {
        "total": total,
        "has_chunk_text": sum(1 for r in records if r.get("chunk_text")),
        "has_embedding_text": sum(1 for r in records if r.get("embedding_text")),
        "has_vi_text": sum(1 for r in records if r.get("chunk_vi")),
        "over_512_tokens": sum(1 for r in records if r.get("token_estimate", 0) > 512),
        "unique_chunk_ids": len(set(r.get("chunk_id") for r in records)),
        "unique_videos": len(set(r.get("video_id") for r in records)),
        "unique_courses": list(set(r.get("course", "") for r in records if r.get("course"))),
    }

    # Báo cáo rõ ràng
    logger.info("─── PHÂN TÍCH CHẤT LƯỢNG DỮ LIỆU ───")
    logger.info("Tổng số records     : %d", stats["total"])
    logger.info("Có chunk_text (EN)  : %d / %d", stats["has_chunk_text"], total)
    logger.info("Có embedding_text   : %d / %d", stats["has_embedding_text"], total)
    logger.info(
        "Có chunk_vi (VI)    : %d / %d (%.1f%%)",
        stats["has_vi_text"],
        total,
        stats["has_vi_text"] / total * 100 if total > 0 else 0,
    )
    logger.info("Vượt 512 tokens     : %d / %d", stats["over_512_tokens"], total)
    logger.info("chunk_id unique     : %d / %d", stats["unique_chunk_ids"], total)
    logger.info("Số videos           : %d", stats["unique_videos"])
    logger.info("Các khóa học        : %s", ", ".join(stats["unique_courses"]))

    # Cảnh báo nếu có vấn đề
    if stats["has_vi_text"] == 0:
        logger.warning(
            "  KHÔNG có bản dịch tiếng Việt (chunk_vi) trong dữ liệu!\n"
            "   → Hệ thống sẽ retrieve văn bản tiếng Anh.\n"
            "   → LLM sẽ dịch và trả lời bằng tiếng Việt từ context tiếng Anh.\n"
            "   → Đây là cách tiếp cận đúng và được hỗ trợ bởi bge-m3."
        )

    return stats


# ════════════════════════════════════════════════════════════════
# BƯỚC 3: CHUYỂN ĐỔI THÀNH LANGCHAIN DOCUMENTS
# ════════════════════════════════════════════════════════════════


def records_to_langchain_documents(
    records: List[Dict[str, Any]],
) -> Tuple[List[Document], List[str]]:
    """
    Chuyển list dict (từ JSONL) thành list LangChain Document.

    LangChain Document có 2 phần:
      - page_content: văn bản sẽ được EMBED — dùng embedding_text
        (đã có context header: Course, Video, Time + nội dung chunk)
      - metadata: thông tin bổ sung — KHÔNG embed, dùng để hiển thị và filter

    Tại sao dùng embedding_text thay vì chunk_text để embed?
      embedding_text = header có context + chunk_text được paraphrase
      Ví dụ:
        "Course: CS25_Transformers
         Video: Stanford CS25: V2 I Strategic Games
         Source: stanford_youtube
         Time: 00:05 - 00:45

         The bots were trained for months..."

      → Khi query "Transformer training bots", vector sẽ match tốt hơn
        vì context header bổ sung thông tin course/video vào embedding.

    Tại sao dùng chunk_text trong metadata?
      chunk_text = transcript gốc, nguyên vẹn — dùng để hiển thị cho user
      và đưa vào prompt LLM (ngắn gọn, không lặp header).

    Returns:
        docs:  List[Document] — để đưa vào FAISS
        texts: List[str]      — để build BM25 (cần text thuần)
    """
    docs: List[Document] = []
    texts: List[str] = []  # dùng riêng cho BM25

    skipped = 0
    for r in records:
        # Lấy text để embed — ưu tiên embedding_text, fallback chunk_text
        embed_text = r.get("embedding_text") or r.get("chunk_text", "")
        embed_text = normalize_text(embed_text)

        # Bỏ qua nếu không có nội dung
        if not embed_text:
            skipped += 1
            continue

        # Tạo URL YouTube để nhảy đến đúng timestamp
        url = make_youtube_url(r.get("video_id", ""), r.get("start_time", 0))
        label = f"{r.get('title', 'Unknown')} [{format_timestamp(r.get('start_time', 0))}]"

        # Metadata — tất cả thông tin cần thiết để hiển thị nguồn
        metadata = {
            "chunk_id": r.get("chunk_id", ""),
            "video_id": r.get("video_id", ""),
            "title": r.get("title", ""),
            "course": r.get("course", ""),
            "source": r.get("source", ""),
            "start_time": r.get("start_time", 0.0),
            "end_time": r.get("end_time", 0.0),
            "url": url,
            "source_label": label,
            # chunk_text = nội dung gốc (không có header) → dùng cho LLM prompt
            "chunk_text": normalize_text(r.get("chunk_text", "")),
            # chunk_vi hiện không có trong dữ liệu nhưng giữ field để tương lai
            "vi_text": r.get("chunk_vi", ""),
        }

        docs.append(Document(page_content=embed_text, metadata=metadata))
        # BM25 dùng chunk_text (tiếng Anh thuần) để tính keyword score
        texts.append(normalize_text(r.get("chunk_text", "")))

    if skipped:
        logger.warning("Bỏ qua %d records không có nội dung", skipped)

    logger.info(" Tạo được %d LangChain Documents", len(docs))
    return docs, texts


# ════════════════════════════════════════════════════════════════
# BƯỚC 4: BUILD VÀ LƯU FAISS INDEX (dùng LangChain)
# ════════════════════════════════════════════════════════════════


def build_and_save_faiss(
    docs: List[Document],
    out_dir: Path,
    batch_size: int = 32,
) -> None:
    """
    Embed tất cả Document bằng bge-m3 và lưu FAISS index vào disk.

      IMPORTANT: LangChain API cho FAISS.from_documents() đã thay đổi.
    Phiên bản mới (langchain-community >= 0.0.10) gọi embedder một lần
    cho toàn bộ documents, không hỗ trợ batch trực tiếp.

    Do đó ta phải tự chia batch và thêm từng batch vào vectorstore.

    Tại sao dùng LangChain FAISS thay vì faiss-python trực tiếp?
      - FAISS.from_documents() xử lý toàn bộ pipeline: embed → index → lưu
      - FAISS.save_local() và load_local() tự động lưu cả index và docstore
        (docstore = nơi lưu metadata của từng vector)
      - Tương thích hoàn toàn với các retriever của LangChain

    Cấu trúc file sau khi lưu:
      out_dir/
        index.faiss  — binary file chứa các vector
        index.pkl    — pickle file chứa docstore (Document objects + metadata)

    Quá trình embedding:
      18,395 docs × ~140 words/doc × ~1.3 token/word ≈ 3.3M tokens
      Trên CPU: ~20-40 phút (tuỳ phần cứng)
      Trên GPU (T4): ~3-5 phút
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = get_embedding_model()

    logger.info(
        "Bắt đầu embed %d documents theo batch %d...",
        len(docs),
        batch_size,
    )

    # ─── CÁCH LÀM: Chia batch rồi thêm vào vectorstore ───
    # Lý do: LangChain không expose progress bar trực tiếp
    # và chúng ta muốn kiểm soát tiến trình + tránh OOM

    all_docs_batched: List[List[Document]] = [
        docs[i : i + batch_size] for i in range(0, len(docs), batch_size)
    ]

    vectorstore = None
    with tqdm(total=len(all_docs_batched), desc=" Building FAISS index") as pbar:
        for batch in all_docs_batched:
            if vectorstore is None:
                # Lần đầu: tạo mới vectorstore từ batch đầu tiên
                # Cách dùng ĐÚNG: FAISS.from_documents() không lấy batch_size param
                # Nó luôn embed toàn bộ documents được truyền vào
                vectorstore = FAISS.from_documents(
                    documents=batch,
                    embedding=embedder,
                )
            else:
                # Các lần sau: thêm vào vectorstore đã có
                #  add_documents() sẽ embed batch mới rồi thêm vào index
                vectorstore.add_documents(batch)
            pbar.update(1)

    if vectorstore is None:
        raise RuntimeError("Không có document nào để index!")

    # Lưu xuống disk — LangChain tự xử lý format
    #  save_local(folder_path) sẽ tạo 2 file:
    #    - index.faiss: vector index
    #    - index.pkl: docstore (metadata + document text)
    vectorstore.save_local(str(out_dir))

    logger.info(
        " FAISS index đã lưu tại %s (%d vectors)",
        out_dir,
        vectorstore.index.ntotal,
    )


# ════════════════════════════════════════════════════════════════
# BƯỚC 5: BUILD VÀ LƯU BM25 INDEX
# ════════════════════════════════════════════════════════════════


def build_and_save_bm25(texts: List[str], out_dir: Path) -> None:
    """
    Build BM25Okapi index và lưu bằng pickle.

    BM25 (Best Match 25) là thuật toán keyword search cổ điển nhưng rất hiệu quả.
    Dùng kết hợp với FAISS (hybrid retrieval) để cải thiện recall.

    Tại sao cần BM25 bên cạnh FAISS?
      - FAISS giỏi tìm văn bản có NGHĨA tương đồng (semantic similarity)
      - BM25 giỏi tìm văn bản có TỪ KHÓA khớp chính xác (exact keyword)
      - Ví dụ: query "GPT-4" → BM25 sẽ match chính xác chuỗi "GPT-4",
        trong khi FAISS có thể trả về docs về "large language models" không
        có chữ "GPT-4" nào.
      - Kết hợp cả hai (Hybrid) → tốt nhất cả hai mặt.

    Tokenisation đơn giản (split theo khoảng trắng) là đủ cho tiếng Anh.
    Với tiếng Việt sẽ cần tokeniser như pyvi hoặc underthesea.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Đang build BM25 index trên %d texts...", len(texts))

    # Tokenise: lowercase + split theo khoảng trắng
    # Đủ đơn giản và hiệu quả cho transcript tiếng Anh
    tokenised = [t.lower().split() for t in texts]

    bm25 = BM25Okapi(tokenised)

    bm25_path = out_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    logger.info(" BM25 index đã lưu tại %s", bm25_path)

    # Lưu thêm texts gốc để debug và để retriever dùng khi cần
    texts_path = out_dir / "chunk_texts.pkl"
    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)
    logger.info(" Chunk texts đã lưu tại %s", texts_path)


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════


def main(data_path: Path, out_dir: Path) -> None:
    """
    Pipeline đầy đủ để build index.

    Luồng thực hiện:
      1. Đọc JSONL
      2. Phân tích chất lượng dữ liệu
      3. Chuyển sang LangChain Documents
      4. Build + lưu FAISS index
      5. Build + lưu BM25 index
    """
    setup_logging()

    # Bước 1: Đọc dữ liệu
    logger.info("═══ BƯỚC 1: Đọc dữ liệu ═══")
    records = load_jsonl(data_path)

    # Bước 2: Phân tích chất lượng
    logger.info("═══ BƯỚC 2: Phân tích chất lượng dữ liệu ═══")
    analyze_data_quality(records)

    # Bước 3: Chuyển sang LangChain Documents
    logger.info("═══ BƯỚC 3: Tạo LangChain Documents ═══")
    docs, texts = records_to_langchain_documents(records)

    # Bước 4: Build FAISS
    logger.info("═══ BƯỚC 4: Build FAISS Index ═══")
    build_and_save_faiss(docs, out_dir)

    # Bước 5: Build BM25
    logger.info("═══ BƯỚC 5: Build BM25 Index ═══")
    build_and_save_bm25(texts, out_dir)

    logger.info("══════════════════════════════════════════")
    logger.info(" Build index hoàn tất! Output: %s", out_dir)
    logger.info("══════════════════════════════════════════")


if __name__ == "__main__":
    cfg = get_settings()

    parser = argparse.ArgumentParser(description="Build FAISS + BM25 index cho RAG")
    parser.add_argument(
        "--data",
        type=Path,
        default=cfg.raw_data_path,
        help=f"Đường dẫn file JSONL (mặc định: {cfg.raw_data_path})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.faiss_index_dir,
        help=f"Thư mục lưu index (mặc định: {cfg.faiss_index_dir})",
    )
    args = parser.parse_args()

    main(args.data, args.out)
