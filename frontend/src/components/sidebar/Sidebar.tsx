import { Plus, MessageSquare, Trash2, Clock } from 'lucide-react';
import { Button } from '../ui/button';
import type { Conversation } from '../../lib/storage';
import { cn } from '../../utils/cn';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  className?: string;
  compact?: boolean;
}

export function Sidebar({ conversations, activeId, onSelect, onNew, onDelete, className, compact = false }: SidebarProps) {
  return (
    <aside className={cn("w-64 flex-col bg-bg-elevated border-r border-border h-full", compact ? "flex" : "hidden md:flex", className)}>
      <div className="p-4 border-b border-border flex items-center justify-between">
        <span className="font-tight font-bold text-accent flex items-center gap-2">
          <span className="text-xl">📚</span> Stanford NLP
        </span>
        <Button 
          variant="outline" 
          size="icon" 
          onClick={onNew}
          className="h-8 w-8 hover:bg-accent hover:text-white transition-colors"
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-4">
        <div>
          <div className="text-[10px] font-bold text-text-muted px-2 py-4 uppercase tracking-widest flex items-center gap-2">
            <Clock className="w-3 h-3" />
            Lịch sử
          </div>
          
          <div className="space-y-1">
            {conversations.length === 0 ? (
              <div className="px-3 py-4 text-xs text-text-muted italic text-center">
                Chưa có hội thoại nào
              </div>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={cn(
                    "group relative flex items-center gap-3 px-3 py-2.5 rounded-md cursor-pointer transition-all",
                    activeId === conv.id 
                      ? "bg-accent/10 text-accent border border-accent/20" 
                      : "text-text-dim hover:bg-bg-elevated-2 hover:text-text border border-transparent"
                  )}
                  onClick={() => onSelect(conv.id)}
                >
                  <MessageSquare className={cn(
                    "w-4 h-4 shrink-0",
                    activeId === conv.id ? "text-accent" : "text-text-muted group-hover:text-text"
                  )} />
                  <span className="text-sm truncate pr-6 font-medium">
                    {conv.title}
                  </span>
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(conv.id);
                    }}
                    className="absolute right-2 opacity-100 md:opacity-0 md:group-hover:opacity-100 p-1 hover:text-danger transition-all"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-border bg-bg-elevated-2/30">
        <div className="text-[10px] text-text-muted font-mono uppercase tracking-tighter text-center">
          v0.1.0 • Built for Stanford CS224N
        </div>
      </div>
    </aside>
  );
}
