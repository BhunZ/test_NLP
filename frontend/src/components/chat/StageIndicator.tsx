import { Loader2, CheckCircle2, Circle } from 'lucide-react';
import { cn } from '../../utils/cn';

export interface Stage {
  id: string;
  label: string;
  status: 'pending' | 'active' | 'done';
}

interface StageIndicatorProps {
  stages: Stage[];
}

export function StageIndicator({ stages }: StageIndicatorProps) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 py-2">
      {stages.map((stage) => (
        <div 
          key={stage.id} 
          className={cn(
            "flex items-center gap-1.5 transition-all duration-300",
            stage.status === 'pending' ? "opacity-30 grayscale" : "opacity-100"
          )}
        >
          {stage.status === 'active' ? (
            <Loader2 className="w-3.5 h-3.5 text-accent animate-spin" />
          ) : stage.status === 'done' ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-success" />
          ) : (
            <Circle className="w-3.5 h-3.5 text-text-muted" />
          )}
          <span className={cn(
            "text-[11px] font-medium tracking-tight",
            stage.status === 'active' ? "text-accent" :
            stage.status === 'done' ? "text-text" : "text-text-muted"
          )}>
            {stage.label}
          </span>
        </div>
      ))}
    </div>
  );
}
