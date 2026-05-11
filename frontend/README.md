# Frontend (Vite + React)

## Run (dev)

```powershell
cd frontend
npm install
npm run dev
```

## API proxy runtime

The frontend proxies `/api/*` requests to backend using `vite.config.ts`.

- Default target: `http://127.0.0.1:8001`
- Override target with env var:

```powershell
$env:VITE_API_PROXY_TARGET="http://127.0.0.1:8000"
npm run dev
```

You must restart Vite after changing `VITE_API_PROXY_TARGET`.

## Key UX features

- Streaming chat (`/api/v1/ask/stream`) with stage indicators
- Citation pills with hover preview + click-to-highlight source card
- Source modal with YouTube embed and fallback open-on-YouTube action
- Conversation history + RAG settings + theme preference persistence
