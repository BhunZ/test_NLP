import type { SourceChunk } from '../../types/api';
import { SourceCard } from './SourceCard';

interface SourceGridProps {
  sources: SourceChunk[];
  highlightedRank?: number | null;
  onOpenSource: (source: SourceChunk) => void;
  isLoading?: boolean;
}

export function SourceGrid({ sources, highlightedRank, onOpenSource, isLoading = false }: SourceGridProps) {
  if (!sources.length && !isLoading) return null;

  return (
    <div className="max-w-3xl mx-auto w-full px-4 md:px-6 py-8 space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <div className="h-px flex-1 bg-border" />
        <span className="text-[10px] font-bold uppercase tracking-widest text-text-muted">
          Nguồn trích dẫn {isLoading ? '(đang tải...)' : `(${sources.length})`}
        </span>
        <div className="h-px flex-1 bg-border" />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
        {isLoading && sources.length === 0
          ? Array.from({ length: 4 }).map((_, i) => (
              <div key={`sk-${i}`} className="h-48 rounded-md border border-border/60 bg-bg-elevated-2/40 animate-pulse" />
            ))
          : sources.map((source) => (
              <SourceCard
                key={source.chunk_id}
                source={source}
                rank={source.rank}
                isHighlighted={highlightedRank === source.rank}
                onOpen={onOpenSource}
              />
            ))}
      </div>
    </div>
  );
}
