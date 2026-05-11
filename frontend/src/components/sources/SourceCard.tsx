import { Play, ExternalLink } from 'lucide-react';
import type { SourceChunk } from '../../types/api';
import { cn } from '../../utils/cn';

interface SourceCardProps {
  source: SourceChunk;
  rank: number;
  isHighlighted?: boolean;
  onOpen: (source: SourceChunk) => void;
}

export function SourceCard({ source, rank, isHighlighted, onOpen }: SourceCardProps) {
  return (
    <div 
      data-source-rank={rank}
      onClick={() => onOpen(source)}
      className={cn(
        "group flex flex-col bg-bg-elevated border border-border rounded-md overflow-hidden hover:border-accent/50 hover:shadow-lg transition-all cursor-pointer",
        isHighlighted && "ring-2 ring-accent border-accent shadow-glow scale-[1.02]"
      )}
    >
      {/* Thumbnail Container */}
      <div className="relative aspect-video overflow-hidden bg-bg-elevated-2">
        <img 
          src={source.thumbnail_url} 
          alt={source.video_title || 'Video thumbnail'}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
        />
        <div className="absolute inset-0 bg-black/20 group-hover:bg-black/0 transition-colors" />
        
        {/* Rank Badge */}
        <div className="absolute top-2 left-2 bg-accent text-white text-[11px] font-bold px-2 py-0.5 rounded-sm shadow-sm">
          [{rank}]
        </div>

        {/* Timestamp Pill */}
        <div className="absolute bottom-2 right-2 bg-black/80 backdrop-blur-md text-white text-[11px] font-mono px-2 py-0.5 rounded-sm">
          {source.timestamp_display}
        </div>

        {/* Play Icon Overlay */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="w-10 h-10 bg-accent rounded-full flex items-center justify-center shadow-lg">
            <Play className="w-5 h-5 text-white fill-current" />
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-3 space-y-2 flex-1 flex flex-col">
        <div className="flex items-center gap-2">
          <div className="bg-accent/10 text-accent text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full">
            {source.course || 'Stanford NLP'}
          </div>
        </div>

        <h3 className="text-sm font-semibold text-text line-clamp-2 leading-tight group-hover:text-accent transition-colors">
          {source.video_title || 'Video bài giảng'}
        </h3>

        <p className="text-xs text-text-muted line-clamp-3 leading-relaxed flex-1 italic">
          "{source.text_en}"
        </p>

        <div className="pt-2 flex items-center justify-between border-t border-border/50">
          <div className="flex gap-1">
            {source.retrieved_by.map(variant => (
              <span key={variant} className="text-[9px] text-text-muted bg-bg-elevated-2 px-1 rounded-sm uppercase tracking-tighter">
                {variant}
              </span>
            ))}
          </div>
          <ExternalLink className="w-3 h-3 text-text-muted group-hover:text-accent transition-colors" />
        </div>
      </div>
    </div>
  );
}
