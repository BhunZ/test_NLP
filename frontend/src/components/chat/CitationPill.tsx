import { useState } from 'react';
import type { SourceChunk } from '../../types/api';

interface CitationPillProps {
  n: number;
  source?: SourceChunk;
  onClick?: (n: number) => void;
}

export function CitationPill({ n, source, onClick }: CitationPillProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <span 
      className="relative inline-flex align-baseline mx-0.5"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <button
        onClick={() => onClick?.(n)}
        className="inline-flex items-center justify-center bg-accent-soft text-accent text-[11px] font-bold px-1.5 py-0 rounded-[4px] hover:bg-accent hover:text-white transition-colors cursor-pointer"
      >
        {n}
      </button>
      {source && showTooltip && (
        <span className="pointer-events-none absolute left-1/2 z-20 w-64 -translate-x-1/2 -translate-y-full rounded-md border border-border bg-bg-elevated p-2 text-left shadow-lg">
          <span className="block text-[10px] uppercase tracking-widest text-text-muted">
            Nguồn [{n}]
          </span>
          <span className="mt-1 block text-xs font-semibold text-text line-clamp-2">
            {source.video_title || 'Nguồn video'}
          </span>
          <span className="mt-1 block text-[11px] text-text-dim line-clamp-3">
            {source.text_en}
          </span>
        </span>
      )}
    </span>
  );
}