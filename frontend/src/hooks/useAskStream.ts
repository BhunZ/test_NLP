import { useState, useCallback } from 'react';
import type { AskRequest, SourceChunk, AskAnswer, AskMeta } from '../types/api';
import { askQuestionStream } from '../lib/streaming';
import type { Stage } from '../components/chat/StageIndicator';
import { askQuestion } from '../lib/api';

export function useAskStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<SourceChunk[]>([]);
  const [stages, setStages] = useState<Stage[]>([]);
  const [meta, setMeta] = useState<AskMeta | null>(null);
  const [answerData, setAnswerData] = useState<AskAnswer | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [rewrites, setRewrites] = useState<{q1?: string | null; q2?: string | null} | null>(null);

  const reset = useCallback(() => {
    setAnswer('');
    setSources([]);
    setStages([
      { id: 'init', label: 'Khởi tạo', status: 'pending' },
      { id: 'rewrite', label: 'Dịch câu hỏi', status: 'pending' },
      { id: 'retrieve', label: 'Tìm dữ liệu', status: 'pending' },
      { id: 'merge', label: 'Tổng hợp', status: 'pending' },
      { id: 'answer', label: 'Viết câu trả lời', status: 'pending' },
    ]);
    setMeta(null);
    setAnswerData(null);
    setErrorMessage(null);
    setRewrites(null);
  }, []);

  const ask = useCallback((req: AskRequest) => {
    reset();
    setIsStreaming(true);
    let fallbackStarted = false;

    const runFallback = async (reason?: string) => {
      if (fallbackStarted) return;
      fallbackStarted = true;
      try {
        const payload = await askQuestion(req);
        setSources(payload.sources ?? []);
        setAnswerData(payload.answer);
        setMeta(payload.meta);
        setAnswer(payload.answer.text_vi || '');
        setStages(prev => prev.map(s => ({ ...s, status: 'done' })));
        if (reason) setErrorMessage(`${reason} Đã chuyển sang chế độ thường.`);
      } catch (fallbackErr) {
        const msg = fallbackErr instanceof Error ? fallbackErr.message : 'Không thể nhận phản hồi từ máy chủ';
        setErrorMessage(msg);
      } finally {
        setIsStreaming(false);
      }
    };

    const abort = askQuestionStream(
      req,
      (event) => {
        switch (event.event) {
          case 'stage':
            setStages(prev => prev.map(s => {
              if (s.id === event.data.id) return { ...s, status: 'active', label: event.data.label };
              if (prev.findIndex(ps => ps.id === event.data.id) > prev.findIndex(ps => ps.id === s.id)) {
                return { ...s, status: 'done' };
              }
              return s;
            }));
            break;
          case 'sources':
            setSources(event.data);
            break;
          case 'rewrites':
            setRewrites(event.data);
            break;
          case 'token':
            setAnswer(prev => prev + event.data.text);
            break;
          case 'done':
            setAnswerData(event.data.answer);
            setMeta(event.data.meta);
            setAnswer(prev => prev || event.data.answer.text_vi || '');
            setStages(prev => prev.map(s => ({ ...s, status: 'done' })));
            break;
          case 'error':
            setErrorMessage(event.data.message);
            setAnswer(prev => prev || `Lỗi: ${event.data.message}`);
            setIsStreaming(false);
            break;
        }
      },
      (err) => {
        console.error(err);
        const msg = err instanceof Error ? err.message : 'Streaming bị gián đoạn';
        void runFallback(msg);
      },
      () => {
        if (!fallbackStarted) setIsStreaming(false);
      }
    );

    return abort;
  }, [reset]);

  return {
    ask,
    isStreaming,
    answer,
    sources,
    stages,
    meta,
    answerData,
    errorMessage,
    rewrites,
    reset
  };
}
