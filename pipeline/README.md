# Pipeline — full RAG flow, end-to-end

This folder contains the complete, runnable pipeline for the Vietnamese NLP
RAG chatbot. Code only — data files (chunked JSONL, FAISS / BM25 indexes)
stay under `data/` and `indexes/` at the repo root, where the live defaults
already point.

Each stage is a numbered subfolder. Stages 1–5 are **offline** (run once
when the corpus or chunking changes). Stages 6–9 run **per user query**.

```
User question (VI)
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 6  query_writer.py     │ VI → q1 (literal EN) + q2 (expanded) │
│ STAGE 7  hybrid_retriever.py │ BM25 + FAISS + reranker → top-K       │
│ STAGE 8  context_to_jsonl.py │ merge by video_id → final_context     │
│ STAGE 9  qwen_answer.py      │ Qwen 2.5 3B generates VN answer       │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
Vietnamese answer with citations
```

---

## The 9 stages

| # | Folder | Files | Runs |
|---|---|---|---|
| 1 | `stage_01_crawl/` | YouTube scraper (main, metadata, transcripts, search, knowledge_base, utils) | Once when corpus added |
| 2 | `stage_02_clean/` | `transcripts_clean.py` — strip fillers, sentence-split | Once after crawl |
| 3 | `stage_03_chunk/` | `knowledge_units.py` — time + topic chunking | Once after clean |
| 4 | `stage_04_enrich/` | `pipeline.py` + `config.py` + `utils.py` — keyword + embedding_text | Once after chunk |
| 5 | `stage_05_index/` | `build_index.py`, `vectorstore_builder.py`, `bm25_store.py` | Once after enrich |
| 6 | `stage_06_query_rewrite/` | `query_writer.py` — VI→EN expansion via Ollama Qwen 3B | Per query |
| 7 | `stage_07_retrieve/` | `hybrid_retriever.py`, `reranker.py` — hybrid BM25+FAISS+rerank | Per query |
| 8 | `stage_08_merge_context/` | `context_to_jsonl.py` — merge chunks by video, clean text | Per query |
| 9 | `stage_09_llm_answer/` | `qwen_answer.py` — Qwen 2.5 3B inference | Per query |

`orchestrator.py` (at the pipeline root) wires Stages 6–7 together and is
the entry point for query-time use.

---

## Quick reference

### Run a single query end-to-end

```bash
cd <repo-root>
python pipeline/orchestrator.py "Gradient là gì?"
```

This calls Stages 6 + 7 and prints retrieved chunks. To get a final
Vietnamese answer, pipe through Stages 8 and 9 (currently separate scripts;
will be unified later).

### Rebuild the corpus from scratch

```bash
# Stage 1: crawl
python pipeline/stage_01_crawl/main.py

# Stage 2: clean
python pipeline/stage_02_clean/transcripts_clean.py

# Stage 3: chunk
python pipeline/stage_03_chunk/knowledge_units.py

# Stage 4: enrich
python pipeline/stage_04_enrich/pipeline.py

# Stage 5: build indexes (~30-60 min on CPU)
python pipeline/stage_05_index/build_index.py
```

---

## Active data artifacts (referenced by defaults)

| Artifact | Path | Purpose |
|---|---|---|
| Cleaned transcripts | `data/cleaned/transcripts_clean_sentence.jsonl` | Stage 2 output → Stage 3 input |
| Chunked transcripts | `data/chunked/transcript_v3_t072.jsonl` | Stage 3 output → Stages 5+7 input |
| FAISS index | `indexes/faiss_index_072_n/` | Stage 5 output → Stage 7 dense retrieval |
| BM25 index | `indexes/bm25_072.pkl` | Stage 5 output → Stage 7 sparse retrieval |

---

## Evaluation lives next door, not here

`evaluation/` (sibling of this folder) holds eval scripts. They use the same
retrieval code from this folder but answer a different question ("how well
does retrieval work?") and shouldn't be in the production path.

```
evaluation/
├── ragas_eval.py        # RAGAS-style + rank metrics + semantic metrics + Context Entities Recall
└── metrics_eval.py      # Deterministic rank metrics only (Hit@K, MAP, MRR, nDCG)
```

---

## Notes on this reorganization

- **Originals are still in place** under `embed__data/`, `data/chunked/`,
  `youtube_scraper/`, etc. This folder is a clean copy with path fixes.
- **Three hardcoded paths** were fixed during the copy:
  1. `stage_03_chunk/knowledge_units.py` — was `D:/NLP`, now project-relative
  2. `orchestrator.py` — was `/Users/carwyn/Downloads/...`, now project-relative
  3. `evaluation/metrics_eval.py` — same fix + corrected FAISS index name
- **Two fixes** in `stage_09_llm_answer/qwen_answer.py`:
  - Removed duplicate `if __name__ == "__main__":` block
  - Changed `do_sample=False` → `True` so the temperature setting works
- **Two files** that weren't in the repo are now committed here:
  `context_to_jsonl.py` (Stage 8) and `qwen_answer.py` (Stage 9), both
  previously living only in `C:/Users/znigh/Downloads/`.

When the team is confident this folder is the source of truth, the original
locations under `embed__data/`, `core/`, root-level `clean_transcripts.py`
etc. can be removed.
