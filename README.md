<p align="center">
  <h1 align="center">🧠 Stanford NLP Tutor</h1>
  <p align="center">
    <strong>AI-powered Q&A system for Stanford NLP courses using RAG pipeline.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/React-Frontend-blue?style=flat-square&logo=react" alt="Frontend">
    <img src="https://img.shields.io/badge/FastAPI-Backend-blue?style=flat-square&logo=fastapi" alt="Backend">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  </p>
</p>

---

## 📌 Overview

**Stanford NLP Tutor** is a Vietnamese-English bilingual Q&A system powered by:
- **Frontend**: React + TypeScript with beautiful UI
- **Backend**: FastAPI for API handling
- **RAG Pipeline**: 9-stage pipeline for retrieval and answer generation
- **Data**: 250+ Stanford CS224N, CS124, CS224U, CS224V lecture videos

```
User Query → Hybrid Retrieval (FAISS + BM25) → Reranking → LLM Answer (Groq/Mistral)
```

---

## 🚀 Quick Start

### Clone the Repository

```bash
# Clone the entire repository
git clone https://github.com/BhunZ/test_NLP.git
cd test_NLP

# Or clone a specific branch
git clone -b feature/pipeline-reorg https://github.com/BhunZ/test_NLP.git
cd test_NLP

# If you already cloned but want this specific branch
git checkout feature/pipeline-reorg
```

### Prerequisites

1. **Python 3.10+**
2. **Node.js 18+** (for frontend)
3. **API Keys** (see below)

### Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### API Keys Setup

Create `.env` file in project root:

```bash
# Required
GROQ_API_KEY=your_groq_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here

# Optional (for Query Rewrite - requires Ollama running locally)
# OLLAMA_MODEL=qwen2.5:3b
```

> Get free API keys from:
> - **Groq**: https://console.groq.com/
> - **Mistral**: https://console.mistral.ai/

### Run the Application

**Option 1: Start both backend and frontend**

```bash
# Terminal 1 - Backend (port 8001)
uvicorn backend.api:app --reload --port 8001

# Terminal 2 - Frontend
cd frontend
npm run dev
```

**Option 2: Backend only (for API testing)**

```bash
uvicorn backend.api:app --reload --port 8001
```

Then open http://localhost:5173 in your browser.

---

## 📦 Pre-built Models & Data Included

**✅ No need to rebuild** - The repository includes everything you need to start immediately:

| Folder | Contents | Size |
|--------|----------|------|
| `indexes/` | FAISS vector index + BM25 index | ~50MB |
| `data/chunked/transcript_v3_t072.jsonl` | 250+ Stanford lecture transcripts | ~10MB |
| `frontend/dist/` | Built frontend (ready to serve) | ~500KB |
| `lid.176.bin` | FastText language detection model | 125MB |

### What's included:
- ✅ Vector search index (FAISS) for semantic retrieval
- ✅ BM25 keyword search index
- ✅ 250+ video transcripts from CS224N, CS124, CS224U, CS224V
- ✅ Built frontend (no need to run `npm run build`)
- ✅ FastText model for language detection fallback

### Just run these commands:
```bash
pip install -r requirements.txt
python -m uvicorn backend.api:app --reload --port 8000
# Open http://localhost:8000 in browser
```

If you want to rebuild everything from scratch (optional), see [SETUP.md](SETUP.md).

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│   Pipeline  │
│  (React)    │     │ (FastAPI)   │     │  (Python)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                            │
                  ┌─────────────────────────┼─────────────────────────┐
                  │                         │                         │
            ┌─────▼─────┐            ┌─────▼─────┐            ┌─────▼─────┐
            │  Stage 7  │            │  Stage 8  │            │  Stage 9  │
            │ Retrieve  │            │   Merge   │            │ LLM Answer│
            └───────────┘            └───────────┘            └───────────┘
```

### Pipeline Stages

| Stage | Name | Description |
|-------|------|-------------|
| 01 | Crawl | (Already done - transcripts in `data/`) |
| 02 | Clean | Clean and normalize transcripts |
| 03 | Chunk | Split into semantic chunks (~500 chars) |
| 04 | Enrich | Add metadata (topic keywords, etc.) |
| 05 | Index | Build FAISS + BM25 indexes |
| 06 | Query Rewrite | Vietnamese → English expansion (Ollama) |
| 07 | Retrieve | Hybrid retrieval (FAISS + BM25) |
| 08 | Merge Context | Group chunks by video |
| 09 | LLM Answer | Generate answer via Groq/Mistral |

---

## 📁 Project Structure

```
youtube-rag-scraper/
├── backend/              # FastAPI backend (api.py, rag_service.py)
├── frontend/             # React + TypeScript frontend
├── pipeline/             # RAG pipeline stages
│   ├── stage_06_query_rewrite/   # Query rewriting (Ollama)
│   ├── stage_07_retrieve/        # Hybrid retriever
│   ├── stage_08_merge_context/   # Context merging
│   └── stage_09_llm_answer/      # LLM generation
├── data/                 # Processed data
│   ├── chunked/          # Chunked transcripts
│   └── cleaned/         # Cleaned transcripts
├── indexes/              # FAISS + BM25 indexes
├── reports/              # Benchmark evaluation results
├── embed__data/         # Evaluation scripts
├── evaluation/           # Additional evaluation code
├── tests/               # Test files
└── dead/                # Deprecated/legacy files (for review)
```

---

## 💻 Usage

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/ask` | POST | Non-streaming Q&A |
| `/api/v1/ask/stream` | POST | Streaming Q&A (SSE) |

### Example Request

```bash
curl -X POST http://localhost:8001/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Attention mechanism hoat dong nhu the nao?",
    "top_k": 6,
    "llm_provider": "groq",
    "rerank": false,
    "enable_rewrite": false
  }'
```

### Frontend Features

- **Bilingual**: Works with both Vietnamese and English queries
- **Settings**: Adjust LLM provider, Top-K, Rewrite, Course filter
- **Streaming**: Real-time answer streaming
- **Sources**: Clickable citations with YouTube video links
- **Confidence**: Shows answer confidence level

---

## ⚙️ Configuration

### Frontend Settings

| Setting | Options | Description |
|---------|---------|-------------|
| LLM Provider | Groq / Mistral | Choose LLM backend |
| Top-K | 1-15 | Number of chunks to retrieve |
| Rewrite | On/Off | Enable query rewriting (requires Ollama) |
| Rerank | On/Off | Enable cross-encoder reranking |
| Course Filter | All / CS224N / CS124 / CS224U / CS224V | Filter by course |

### Backend Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key |
| `MISTRAL_API_KEY` | Yes | Mistral API key |
| `OLLAMA_MODEL` | No | Ollama model for query rewrite (default: qwen2.5:3b) |

---

## 🧪 Running Tests

```bash
# Backend tests
cd backend
pytest

# Or run specific test
python -m pytest backend/tests/test_rag_utils.py -v
```

---

## 📊 Benchmark Reports

Evaluation results are stored in `reports/` folder:

- `ragas_report_FINAL.json` - RAGAS benchmark results
- `answer_eval_FINAL.json` - Answer quality evaluation

See `reports/README.md` for details.

---

## 🔧 Troubleshooting

### Backend not starting?

```bash
# Check if port is in use
netstat -ano | findstr 8001

# Try different port
uvicorn backend.api:app --port 8002
```

### Frontend not connecting to backend?

Check `frontend/vite.config.ts` - the proxy target should match your backend port (default: 8001).

### Query Rewrite not working?

1. Make sure Ollama is running: `ollama serve`
2. Pull the model: `ollama pull qwen2.5:3b`
3. Check the model is available: `ollama list`

---

## 📄 License

MIT License - See LICENSE file.

---

## 👤 Author

Built with ❤️ for Stanford NLP courses.