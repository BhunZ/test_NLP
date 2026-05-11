import os
import json
from typing import AsyncGenerator, Dict, List, Any, Optional

async def stream_groq(
    question: str, 
    contexts: List[Dict[str, Any]], 
    model: str,
    system_prompt: str,
    build_user_prompt_fn: Any,
    detected_lang: str = "unknown"
) -> AsyncGenerator[str, None]:
    try:
        from groq import Groq
    except ImportError:
        yield "Error: groq not installed"
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        yield "Error: GROQ_API_KEY not found"
        return

    client = Groq(api_key=api_key)
    
    # Use the same prompt building logic from ask.py
    user_prompt = build_user_prompt_fn(question, contexts, detected_lang)
    
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

async def stream_mistral(
    question: str, 
    contexts: List[Dict[str, Any]], 
    model: str,
    system_prompt: str,
    build_user_prompt_fn: Any,
    detected_lang: str = "unknown"
) -> AsyncGenerator[str, None]:
    # For mistral, we'll use the official SDK if possible, or fall back to requests with stream=True
    # Using a simplified version here.
    import httpx
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        yield "Error: MISTRAL_API_KEY not found"
        return

    user_prompt = build_user_prompt_fn(question, contexts, detected_lang)
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "stream": True,
                },
                timeout=60,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if content:
                        yield content
    except httpx.HTTPStatusError as exc:
        yield f"Error: mistral upstream status {exc.response.status_code}"
    except httpx.HTTPError as exc:
        yield f"Error: mistral upstream request failed ({exc})"
