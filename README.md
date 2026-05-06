<p align="center">
  <h1 align="center">🧠 YouTube RAG Pipeline Demo</h1>
  <p align="center">
    <strong>End-to-end Retrieval-Augmented Generation pipeline using Vietnamese questions to search English Stanford Lectures.</strong>
  </p>
</p>

---

## 📌 Overview

This repository contains a standalone RAG pipeline demo. It takes a student's Vietnamese question, uses `bge-m3` to perform a cross-lingual search (FAISS + BM25 Reciprocal Rank Fusion) across English transcript chunks, and then uses Gemini 2.5 Flash to generate a final, easy-to-understand Vietnamese answer with citations.

---

## 🚀 How to Run the Demo

### 1. Setup Environment
Install the required dependencies:
```bash
pip install -r requirements.txt
```

*(Note: A temporary Gemini API key is configured in the demo script for easy testing. For production, you will need to set your own `GEMINI_API_KEY` in your environment).*

### 2. Run the Full Pipeline
You can immediately test the pipeline by asking a question in Vietnamese. 

```bash
python demo_pipeline.py --query "Theo đoạn trích, mục đích ban đầu của Geoffrey Hinton khi phát triển mạng neural là gì?"
```
**What happens when you run this?**
1. The script converts your question into a vector embedding.
2. It searches the pre-built `index/eval/t072/` database for the most relevant English chunks.
3. It passes the top retrieved chunks to Gemini, instructing it to act as an AI Teaching Assistant and answer your question in Vietnamese.

### 3. Evaluate the Chunking & Search Accuracy
If you want to test how well the retrieval system works *without* using the LLM to generate an answer, you can run the evaluation script against our ground-truth queries:

```bash
python -m data.chunked.eval_retrieval --queries data/chunked/test_queries.jsonl --index index/eval/t072 --show-queries
```
This script computes Hit@k and MRR metrics to prove that the correct chunk appears at the top of the search results.

---

## 📂 Project Structure

Here is a breakdown of the core files in this repository and what they do:

### The Demo Scripts
- `demo_pipeline.py` - **The main entry point.** Takes a user query, runs hybrid search, and outputs the Gemini AI answer.
- `data/chunked/eval_retrieval.py` - Evaluates the search accuracy using Hit@k metrics.
- `data/chunked/build_eval_index.py` - Builds the FAISS and BM25 search indexes from raw JSONL transcripts.
- `data/chunked/generate_queries.py` - Generates synthetic ground-truth test queries using Gemini for evaluation.

### Core Logic (Behind the scenes)
- `core/s02_hybrid_search.py` - Implements the actual search logic (Dense FAISS + Sparse BM25 merged using Reciprocal Rank Fusion).
- `core/llm.py` - The LLM client that talks to Gemini 2.5 Flash to generate the final Vietnamese answers based on the retrieved English text.
- `core/embeddings.py` - Loads the `bge-m3` cross-lingual embedding model to map Vietnamese and English text into the same vector space.
- `data/chunked/knowledge_units.py` - The chunking strategy that splits raw transcripts into logical `[start_time, end_time]` chunks.

### Data & Indexes
- `index/eval/` - Contains the pre-built FAISS and BM25 databases so you can test search instantly without waiting for the embedder.
- `data/chunked/transcript_v3_t072.jsonl` - The raw transcript data.
- `data/chunked/test_queries.jsonl` - A small set of ground-truth questions used for evaluation.
