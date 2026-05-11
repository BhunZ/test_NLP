import React, { useState, useRef, useEffect } from 'react';
import { SendHorizonal, ChevronDown, ChevronUp, Zap, Layers, Settings2, Filter, Check, X } from 'lucide-react';
import { Button } from '../ui/button';
import type { Settings } from '../settings/SettingsDrawer';
import { cn } from '../../utils/cn';

interface InputBarProps {
  onSend: (query: string) => void;
  disabled?: boolean;
  placeholder?: string;
  inputRef?: React.RefObject<HTMLTextAreaElement | null>;
  settings?: Settings;
  onSettingsChange?: (settings: Settings) => void;
  meta?: {
    total_latency_ms?: number;
    total_chunks_retrieved?: number;
  } | null;
}

const COURSE_LABELS: Record<string, string> = {
  'CS224N': 'CS224N',
  'CS124': 'CS124',
  'CS224U': 'CS224U',
  'CS224V': 'CS224V',
};

const COURSE_OPTIONS = [
  { id: null, label: 'Tất cả' },
  { id: 'CS224N', label: 'CS224N: NLP' },
  { id: 'CS124', label: 'CS124: From Languages to Information' },
  { id: 'CS224U', label: 'CS224U: NLU' },
  { id: 'CS224V', label: 'CS224V: Conversational AI' },
];

export function InputBar({ onSend, disabled, placeholder = 'Hỏi gì đó về NLP...', inputRef, settings, onSettingsChange, meta }: InputBarProps) {
  const [query, setQuery] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [showStatusDetails, setShowStatusDetails] = useState(false);

  const setRefs = (el: HTMLTextAreaElement | null) => {
    textareaRef.current = el;
    if (inputRef) inputRef.current = el;
  };

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (query.trim() && !disabled) {
      onSend(query.trim());
      setQuery('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [query]);

  const updateSetting = (patch: Partial<Settings>) => {
    if (settings && onSettingsChange) {
      onSettingsChange({ ...settings, ...patch });
    }
  };

  const cycleLLM = () => {
    const newProvider = settings?.llmProvider === 'groq' ? 'mistral' : 'groq';
    updateSetting({ llmProvider: newProvider });
  };

  const toggleRewrite = () => {
    updateSetting({ enableRewrite: !settings?.enableRewrite });
  };

  const cycleCourse = () => {
    if (!settings) return;
    const currentIndex = COURSE_OPTIONS.findIndex(c => c.id === settings.courseFilter);
    const nextIndex = (currentIndex + 1) % COURSE_OPTIONS.length;
    updateSetting({ courseFilter: COURSE_OPTIONS[nextIndex].id });
  };

  const getStatusBadges = () => {
    if (!settings) return null;

    const badges: React.ReactNode[] = [];

    badges.push(
      <button
        key="llm"
        onClick={cycleLLM}
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-accent/10 text-accent font-medium hover:opacity-80 transition-opacity cursor-pointer"
        title="Click để đổi"
      >
        <Zap className="w-3 h-3" />
        {settings.llmProvider === 'groq' ? 'Groq' : 'Mistral'}
      </button>
    );

    badges.push(
      <span key="topk" className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] bg-bg-elevated-2 text-text-dim">
        <Layers className="w-3 h-3" />
        Top-K: {settings.topK}
      </span>
    );

    badges.push(
      <button
        key="rewrite"
        onClick={toggleRewrite}
        className={cn(
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] cursor-pointer hover:opacity-80 transition-opacity",
          settings.enableRewrite 
            ? "bg-accent/10 text-accent font-medium" 
            : "bg-bg-elevated-2 text-text-dim"
        )}
        title="Click để đổi"
      >
        <Settings2 className="w-3 h-3" />
        Rewrite: {settings.enableRewrite ? 'Bật' : 'Tắt'}
      </button>
    );

    const courseLabel = settings.courseFilter ? COURSE_LABELS[settings.courseFilter] || settings.courseFilter : 'Tất cả';
    badges.push(
      <button
        key="course"
        onClick={cycleCourse}
        className={cn(
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] cursor-pointer hover:opacity-80 transition-opacity",
          settings.courseFilter
            ? "bg-accent/10 text-accent font-medium"
            : "bg-bg-elevated-2 text-text-dim"
        )}
        title="Click để đổi"
      >
        <Filter className="w-3 h-3" />
        Lọc: {courseLabel}
      </button>
    );

    return badges;
  };

  const getDetailedInfo = () => {
    if (!settings) return null;

    return (
      <div className="mt-3 p-3 bg-bg-elevated-2/50 rounded-lg border border-border/50 space-y-3">
        {/* LLM Provider - Click to cycle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[11px] text-text-muted">
            <Zap className="w-3 h-3" />
            LLM Provider
          </div>
          <button
            onClick={cycleLLM}
            className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-accent/10 text-accent text-[11px] font-medium hover:opacity-80 transition-opacity"
          >
            {settings.llmProvider === 'groq' ? 'Groq (Llama 3)' : 'Mistral'}
            <span className="text-[9px] opacity-60">(click để đổi)</span>
          </button>
        </div>

        {/* Top-K Slider */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px]">
            <div className="flex items-center gap-2 text-text-muted">
              <Layers className="w-3 h-3" />
              Top-K: {settings.topK}
            </div>
          </div>
          <input
            type="range"
            min={1}
            max={15}
            value={settings.topK}
            onChange={(e) => updateSetting({ topK: parseInt(e.target.value) })}
            className="w-full h-1.5 bg-bg-elevated rounded-lg appearance-none cursor-pointer accent-accent"
          />
        </div>

        {/* Rewrite Toggle - Click to toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[11px] text-text-muted">
            <Settings2 className="w-3 h-3" />
            Rewrite
          </div>
          <button
            onClick={toggleRewrite}
            className={cn(
              "flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors",
              settings.enableRewrite 
                ? "bg-accent/10 text-accent" 
                : "bg-bg-elevated-2 text-text-dim"
            )}
          >
            {settings.enableRewrite ? (
              <><Check className="w-3 h-3" /> Bật</>
            ) : (
              <><X className="w-3 h-3" /> Tắt</>
            )}
            <span className="text-[9px] opacity-60">(click để đổi)</span>
          </button>
        </div>

        {/* Course Filter - Click to cycle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[11px] text-text-muted">
            <Filter className="w-3 h-3" />
            Lọc khóa học
          </div>
          <button
            onClick={cycleCourse}
            className={cn(
              "flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors",
              settings.courseFilter
                ? "bg-accent/10 text-accent"
                : "bg-bg-elevated-2 text-text-dim"
            )}
          >
            {settings.courseFilter 
              ? COURSE_LABELS[settings.courseFilter] || settings.courseFilter 
              : 'Tất cả'}
            <span className="text-[9px] opacity-60">(click để đổi)</span>
          </button>
        </div>

        {/* Meta info */}
        {meta && (meta.total_latency_ms || meta.total_chunks_retrieved !== undefined) && (
          <div className="pt-2 border-t border-border/50 space-y-1">
            {meta.total_latency_ms && (
              <div className="flex items-center gap-2 text-[11px] text-text-dim">
                <span className="text-text-muted">Thời gian xử lý:</span>
                {(meta.total_latency_ms / 1000).toFixed(1)}s
              </div>
            )}
            {meta.total_chunks_retrieved !== undefined && (
              <div className="flex items-center gap-2 text-[11px] text-text-dim">
                <span className="text-text-muted">Số chunks tìm được:</span>
                {meta.total_chunks_retrieved} chunks
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="sticky bottom-0 w-full p-4 bg-gradient-to-t from-bg via-bg/80 to-transparent">
      {/* Status Bar */}
      {settings && (
        <div className="max-w-3xl mx-auto mb-2">
          <div 
            className="flex flex-wrap items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => setShowStatusDetails(!showStatusDetails)}
          >
            {getStatusBadges()}
            {showStatusDetails ? (
              <ChevronUp className="w-3 h-3 text-text-dim" />
            ) : (
              <ChevronDown className="w-3 h-3 text-text-dim" />
            )}
          </div>
          
          {showStatusDetails && getDetailedInfo()}
          
          {/* Disclaimer */}
          <div className="mt-2 text-[11px] text-text-dim italic text-center">
            Mọi thông tin chỉ mang tính chất tham khảo
          </div>
        </div>
      )}

      <form 
        onSubmit={handleSubmit}
        className="max-w-3xl mx-auto relative flex items-end gap-2 p-2 rounded-md glass-card focus-within:ring-2 focus-within:ring-accent/50 transition-all"
      >
        <textarea
          ref={setRefs}
          rows={1}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className="flex-1 bg-transparent border-none focus:ring-0 text-text placeholder:text-text-muted resize-none py-2 px-3 min-h-[40px] max-h-[200px]"
        />
        <Button 
          type="submit" 
          size="icon" 
          disabled={!query.trim() || disabled}
          className="mb-1 rounded-sm h-10 w-10 shrink-0"
        >
          <SendHorizonal className="h-5 w-5" />
        </Button>
      </form>
    </div>
  );
}