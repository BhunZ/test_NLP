# Backend (FastAPI)

## Run (dev)

From repo root:

```powershell
python -m uvicorn backend.api:app --reload --host 127.0.0.1 --port 8001
```

If `8001` is occupied, pick another free port and set frontend proxy via
`VITE_API_PROXY_TARGET`.

## Health check

```powershell
curl http://127.0.0.1:8001/api/v1/health
```

## Ask (sync)

```powershell
curl -X POST http://127.0.0.1:8001/api/v1/ask `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Co che attention la gi?\",\"top_k\":6,\"rerank\":false,\"llm_provider\":\"groq\"}"
```

## Ask (stream SSE)

```powershell
curl -N -X POST http://127.0.0.1:8001/api/v1/ask/stream `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Co che attention la gi?\",\"top_k\":6,\"rerank\":false,\"llm_provider\":\"groq\"}"
```

