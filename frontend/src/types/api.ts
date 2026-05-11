export type LLMProvider = 'groq' | 'mistral';

export interface AskRequest {
  query: string;
  top_k?: number;
  rerank?: boolean;
  llm_provider?: LLMProvider;
  course_filter?: string | null;
  enable_rewrite?: boolean;
}

export interface AskRewrites {
  q1?: string | null;
  q2?: string | null;
}

export interface AskAnswer {
  text_vi: string;
  citations: number[];
  confidence: 'high' | 'medium' | 'low';
}

export interface SourceChunk {
  chunk_id: string;
  rank: number;
  score?: number | null;
  video_id: string;
  video_title?: string | null;
  course?: string | null;
  youtube_url: string;
  start_seconds: number;
  end_seconds?: number | null;
  timestamp_display: string;
  thumbnail_url: string;
  text_en: string;
  retrieved_by: string[];
}

export interface AskMeta {
  total_chunks_retrieved: number;
  retrieval_latency_ms: number;
  answer_latency_ms: number;
  total_latency_ms: number;
  models: Record<string, any>;
  retrieval: Record<string, any>;
}

export interface AskResponse {
  query: string;
  detected_lang: 'vi' | 'en' | 'unknown';
  rewrites?: AskRewrites | null;
  answer: AskAnswer;
  sources: SourceChunk[];
  meta: AskMeta;
}

export interface HealthResponse {
  status: 'ok';
  indexes_loaded: boolean;
  uptime_s: number;
}

// Streaming Event Types
export type StreamEvent = 
  | { event: 'stage'; data: { id: string; label: string } }
  | { event: 'rewrites'; data: AskRewrites }
  | { event: 'sources'; data: SourceChunk[] }
  | { event: 'token'; data: { text: string } }
  | { event: 'done'; data: { answer: AskAnswer; meta: AskMeta } }
  | { event: 'error'; data: { message: string } };
