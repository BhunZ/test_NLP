import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, User, Layers, Clock, Copy, RefreshCw, Check } from 'lucide-react';
import { cn } from '../../utils/cn';
import type { SourceChunk } from '../../types/api';
import { CitationPill } from './CitationPill';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  text: string;
  isStreaming?: boolean;
  confidence?: 'high' | 'medium' | 'low';
  onCitationClick?: (rank: number) => void;
  citationLookup?: Record<number, SourceChunk>;
  meta?: {
    total_latency_ms?: number;
    total_chunks_retrieved?: number;
  } | null;
  elapsedTime?: number;
  currentModel?: string;
  onRegenerate?: () => void;
}

function renderCitationText(
  raw: string,
  onCitationClick?: (n: number) => void,
  citationLookup?: Record<number, SourceChunk>
) {
  const regex = /\[(\d+)\]/g;
  const result: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(raw)) !== null) {
    if (match.index > lastIndex) result.push(raw.substring(lastIndex, match.index));
    const n = Number.parseInt(match[1], 10);
    result.push(
      <CitationPill
        key={`${n}-${match.index}`}
        n={n}
        source={citationLookup?.[n]}
        onClick={onCitationClick}
      />
    );
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < raw.length) result.push(raw.substring(lastIndex));
  return result;
}

export function MessageBubble({ role, text, isStreaming, confidence, onCitationClick, citationLookup, meta, elapsedTime = 0, onRegenerate = () => {} }: MessageBubbleProps) {
  const isUser = role === 'user';
  const [copied, setCopied] = useState(false);

  const getConfidenceConfig = (conf: string | undefined) => {
    switch (conf) {
      case 'high':
        return { label: 'Cao', color: 'text-success bg-success/10', dots: '●●●' };
      case 'medium':
        return { label: 'Trung bình', color: 'text-warning bg-warning/10', dots: '●●○' };
      case 'low':
        return { label: 'Thấp', color: 'text-danger bg-danger/10', dots: '●○○' };
      default:
        return null;
    }
  };

  const handleCopy = async () => {
    try {
      const textWithoutCitations = text.replace(/\[\d+\]/g, '').trim();
      await navigator.clipboard.writeText(textWithoutCitations);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const confConfig = getConfidenceConfig(confidence);

  return (
    <div className={cn(
      "flex w-full gap-4 py-8 px-4 md:px-6 group",
      isUser ? "bg-transparent" : "bg-bg-elevated/30"
    )}>
      <div className="max-w-3xl mx-auto flex w-full gap-4 md:gap-6">
        <div className={cn(
          "w-8 h-8 rounded-sm shrink-0 flex items-center justify-center",
          isUser ? "bg-bg-elevated-2" : "bg-accent"
        )}>
          {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5 text-white" />}
        </div>
        
        <div className="flex-1 space-y-2 overflow-hidden">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-sm font-semibold text-text">
              {isUser ? 'Bạn' : 'AI Trợ Giảng'}
            </span>
            {!isUser && confConfig && (
              <span className={cn(
                "text-[10px] font-bold px-2 py-0.5 rounded-full flex items-center gap-1",
                confConfig.color
              )}>
                {confConfig.dots} {confConfig.label}
              </span>
            )}
            {isStreaming && elapsedTime > 0 && (
              <span className="flex items-center gap-1 text-[11px] text-accent animate-pulse">
                ⏱️ {elapsedTime}s
              </span>
            )}
            {meta && !isStreaming && (
              <div className="flex items-center gap-2 text-[11px] text-text-dim">
                {meta.total_latency_ms && (
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {(meta.total_latency_ms / 1000).toFixed(1)}s
                  </span>
                )}
                {meta.total_chunks_retrieved !== undefined && (
                  <span className="flex items-center gap-1">
                    <Layers className="w-3 h-3" />
                    {meta.total_chunks_retrieved} nguồn
                  </span>
                )}
              </div>
            )}
            
            {/* Action buttons - only show for assistant after streaming completes */}
            {!isUser && !isStreaming && (
              <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity ml-auto">
                <button
                  onClick={handleCopy}
                  className="p-1.5 rounded-md hover:bg-bg-elevated-2 text-text-dim hover:text-text transition-colors"
                  title="Copy answer"
                >
                  {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
            )}
          </div>

          <div className={cn(
            "prose prose-invert max-w-none break-words text-[15px] leading-relaxed font-sans",
            "[&_*]:font-sans [&_blockquote]:border-border [&_blockquote]:text-text-dim [&_strong]:font-semibold",
            isUser ? "text-text" : "text-text-dim"
          )}>
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({children}) => {
                  const parts = React.Children.toArray(children).flatMap((child) =>
                    typeof child === 'string'
                      ? renderCitationText(child, onCitationClick, citationLookup)
                      : child
                  );
                  return <p>{parts}</p>;
                },
              }}
            >
              {text}
            </ReactMarkdown>
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-accent ml-1 animate-pulse align-middle" />
            )}
          </div>

          {/* Regenerate button - show below answer after streaming completes */}
          {!isUser && !isStreaming && (
            <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border/30">
              <button
                type="button"
                onClick={() => onRegenerate && onRegenerate()}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs text-text-dim hover:text-text hover:bg-bg-elevated-2 transition-colors"
                title="Regenerate with different model"
              >
                <RefreshCw className="w-3 h-3" />
                Regenerate
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}