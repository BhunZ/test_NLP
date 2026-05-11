from __future__ import annotations

import json
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .models import AskRequest, AskResponse, HealthResponse
from .rag_service import RagService


app = FastAPI(title="Stanford NLP Vietnamese RAG API", version="0.1.0")

@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    print(f"DEBUG: 404 Not Found: {request.method} {request.url}")
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found", "method": request.method, "url": str(request.url)},
    )

# Local dev: Vite defaults to 5173; keep permissive but scoped.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_service: Optional[RagService] = None
_started = time.time()


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


@app.get("/")
def root():
    return {"message": "Stanford NLP Vietnamese RAG API is running", "endpoints": ["/api/v1/health", "/api/v1/ask", "/api/v1/ask/stream"]}


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = get_service()
    return HealthResponse(indexes_loaded=svc.indexes_loaded, uptime_s=int(time.time() - _started))


@app.post("/api/v1/ask/stream")
async def ask_stream(req: AskRequest):
    svc = get_service()

    async def event_generator():
        async for event in svc.ask_stream(
            query=req.query,
            top_k=req.top_k,
            rerank=req.rerank,
            llm_provider=req.llm_provider,
            enable_rewrite=req.enable_rewrite,
            course_filter=req.course_filter,
        ):
            # Yield event in SSE format
            yield {
                "event": event["event"],
                "data": json.dumps(event["data"])
            }

    return EventSourceResponse(event_generator())


@app.post("/api/v1/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    svc = get_service()
    payload = svc.ask(
        query=req.query,
        top_k=req.top_k,
        rerank=req.rerank,
        llm_provider=req.llm_provider,
        enable_rewrite=req.enable_rewrite,
        course_filter=req.course_filter,
    )
    return AskResponse.model_validate(payload)

