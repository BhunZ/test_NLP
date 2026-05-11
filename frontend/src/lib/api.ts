import type { AskRequest, AskResponse } from '../types/api';

export async function askQuestion(req: AskRequest): Promise<AskResponse> {
  const response = await fetch('/api/v1/ask', {
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
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Failed to ask question');
  }

  return response.json();
}
