import type { AskRequest, StreamEvent } from '../types/api';

export function askQuestionStream(
  req: AskRequest,
  onEvent: (event: StreamEvent) => void,
  onError: (error: Error) => void,
  onClose: () => void
) {
  const ctrl = new AbortController();
  
  fetch('/api/v1/ask/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ...req,
      top_k: req.top_k ?? 6,
      llm_provider: req.llm_provider ?? 'groq',
      rerank: req.rerank ?? false,
      enable_rewrite: req.enable_rewrite ?? false,
      course_filter: req.course_filter ?? null,
    }),
    signal: ctrl.signal,
  }).then(async (response) => {
    if (!response.ok) {
      throw new Error('Failed to start stream');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;

      // SSE frames may be separated by LF or CRLF depending on server/runtime.
      const parts = buffer.split(/\r?\n\r?\n/);
      buffer = parts.pop() || '';

      for (const part of parts) {
        if (!part.trim()) continue;

        let eventType = 'message';
        const dataLines: string[] = [];

        const lines = part.split(/\r?\n/);
        for (const line of lines) {
          const normalized = line.replace(/\r$/, '');
          if (normalized.startsWith('event:')) {
            eventType = normalized.substring(6).trim();
          } else if (normalized.startsWith('data:')) {
            dataLines.push(normalized.substring(5).trim());
          }
        }
        const dataStr = dataLines.join('\n');

        if (dataStr) {
          try {
            const data = JSON.parse(dataStr);
            console.log(`[SSE] ${eventType}:`, data);
            onEvent({ event: eventType as any, data });
          } catch (e) {
            onError(new Error(`[SSE] JSON parse error for event "${eventType}"`));
            return;
          }
        }
      }
    }
    onClose();
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      onError(err);
    }
  });

  return () => ctrl.abort();
}
