import { Lightbulb, GraduationCap, Keyboard, Sparkles } from 'lucide-react';

interface EmptyStateProps {
  onExampleClick: (query: string) => void;
}

const EXAMPLES = [
  "Cơ chế attention hoạt động ra sao?",
  "BERT vs GPT khác gì nhau?",
  "Tại sao cần positional encoding trong Transformer?",
  "RLHF là gì và tại sao nó quan trọng?"
];

const KEYBOARD_SHORTCUTS = [
  { keys: ['Ctrl', 'K'], action: 'Focus input' },
  { keys: ['Ctrl', 'L'], action: 'New chat' },
];

export function EmptyState({ onExampleClick }: EmptyStateProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-2xl mx-auto">
      <div className="w-16 h-16 bg-accent/20 rounded-lg flex items-center justify-center mb-6">
        <GraduationCap className="w-10 h-10 text-accent" />
      </div>
      
      <h1 className="text-3xl font-bold text-text mb-2 font-tight">
        Stanford NLP Tutor
      </h1>
      <p className="text-text-dim mb-8 max-w-md">
        Hỏi về NLP bằng tiếng Việt hoặc tiếng Anh. Trả lời được tổng hợp từ hơn 250 video bài giảng của Stanford.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full mb-8">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            onClick={() => onExampleClick(example)}
            className="flex items-start gap-3 p-4 rounded-md glass-card hover:bg-bg-elevated-2 transition-all text-left group"
          >
            <Lightbulb className="w-5 h-5 text-accent shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
            <span className="text-sm text-text-dim group-hover:text-text transition-colors">
              {example}
            </span>
          </button>
        ))}
      </div>

      {/* Keyboard Shortcuts */}
      <div className="flex items-center gap-6 text-[12px] text-text-dim">
        <div className="flex items-center gap-2">
          <Keyboard className="w-4 h-4" />
          <span className="font-medium">Phím tắt:</span>
        </div>
        <div className="flex gap-4">
          {KEYBOARD_SHORTCUTS.map((shortcut) => (
            <div key={shortcut.action} className="flex items-center gap-1.5">
              <div className="flex gap-0.5">
                {shortcut.keys.map((key, i) => (
                  <span key={i} className="px-1.5 py-0.5 bg-bg-elevated-2 rounded text-[10px] font-mono">
                    {key}
                  </span>
                ))}
              </div>
              <span className="text-text-dim">{shortcut.action}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tip */}
      <div className="mt-6 flex items-center gap-2 text-[11px] text-text-muted bg-bg-elevated-2/50 px-3 py-2 rounded-md">
        <Sparkles className="w-3 h-3 text-accent" />
        <span>Bạn có thể hỏi bằng tiếng Việt hoặc tiếng Anh - AI sẽ tự điều chỉnh ngôn ngữ phù hợp</span>
      </div>
    </div>
  );
}