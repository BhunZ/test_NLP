import { Sheet, SheetContent, SheetHeader, SheetTitle } from '../ui/sheet';
import { Button } from '../ui/button';
import type { LLMProvider } from '../../types/api';
import { cn } from '../../utils/cn';
import type { ThemeMode } from '../../lib/storage';

export interface Settings {
  llmProvider: LLMProvider;
  topK: number;
  rerank: boolean;
  enableRewrite: boolean;
  courseFilter: string[];
  theme: ThemeMode;
}

interface SettingsDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  settings: Settings;
  onSettingsChange: (settings: Settings) => void;
}

const COURSES = [
  { id: 'CS224N_NLP', label: 'CS224N: NLP' },
  { id: 'CS229_ML', label: 'CS229: Machine Learning' },
  { id: 'CS224U', label: 'CS224U: NLU' },
  { id: 'CS224R_RL', label: 'CS224R: RL' },
  { id: 'CS25_Transformers', label: 'CS25: Transformers' },
  { id: 'CME295_Transformers_LLMs', label: 'CME295: Transformers & LLMs' },
  { id: 'CME296_Diffusion', label: 'CME296: Diffusion Models' },
  { id: 'CS336_GPU_TPU', label: 'CS336: GPU/TPU Optimization' },
];

export function SettingsDrawer({ open, onOpenChange, settings, onSettingsChange }: SettingsDrawerProps) {
  const update = (patch: Partial<Settings>) => {
    onSettingsChange({ ...settings, ...patch });
  };

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-md flex flex-col">
        <SheetHeader className="mb-6 mt-2">
          <SheetTitle>Cấu hình RAG</SheetTitle>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto space-y-8 pr-2">
          {/* LLM Provider */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-text">LLM Provider</h3>
            <div className="grid grid-cols-2 gap-2">
              {(['groq', 'mistral'] as const).map((p) => (
                <Button
                  key={p}
                  variant={settings.llmProvider === p ? 'default' : 'outline'}
                  onClick={() => update({ llmProvider: p })}
                  className="capitalize"
                >
                  {p}
                </Button>
              ))}
            </div>
            <p className="text-[11px] text-text-muted">
              Groq nhanh hơn (Llama 3), Mistral thông minh hơn ở một số tác vụ.
            </p>
          </div>

          {/* Top-K Slider */}
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-semibold text-text">Số lượng nguồn (Top-K)</h3>
              <span className="text-sm font-mono text-accent">{settings.topK}</span>
            </div>
            <input
              type="range"
              min={1}
              max={15}
              value={settings.topK}
              onChange={(e) => update({ topK: parseInt(e.target.value) })}
              className="w-full h-1.5 bg-bg-elevated-2 rounded-lg appearance-none cursor-pointer accent-accent"
            />
            <p className="text-[11px] text-text-muted">
              Số lượng đoạn hội thoại trích dẫn từ video để trả lời.
            </p>
          </div>

          {/* Toggles */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-text">Tính năng nâng cao</h3>
            <div className="space-y-3">
              <label className={cn(
                "flex items-center justify-between p-3 rounded-md cursor-pointer transition-all",
                settings.rerank 
                  ? "bg-accent/10 border border-accent/30" 
                  : "bg-bg-elevated-2/50 hover:bg-bg-elevated-2 border border-transparent"
              )}>
                <div className="space-y-0.5">
                  <div className="text-sm font-medium">Reranking</div>
                  <div className="text-[11px] text-text-muted">Dùng Cross-Encoder để xếp hạng lại (chậm hơn)</div>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={settings.rerank}
                    onChange={(e) => update({ rerank: e.target.checked })}
                    className="sr-only"
                  />
                  <div className={cn(
                    "w-10 h-5 rounded-full transition-colors",
                    settings.rerank ? "bg-accent" : "bg-bg-elevated"
                  )}>
                    <div className={cn(
                      "absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform shadow",
                      settings.rerank ? "translate-x-5" : "translate-x-0.5"
                    )} />
                  </div>
                </div>
              </label>

              <label className={cn(
                "flex items-center justify-between p-3 rounded-md cursor-pointer transition-all",
                settings.enableRewrite 
                  ? "bg-accent/10 border border-accent/30" 
                  : "bg-bg-elevated-2/50 hover:bg-bg-elevated-2 border border-transparent"
              )}>
                <div className="space-y-0.5">
                  <div className="text-sm font-medium">Query Rewrite</div>
                  <div className="text-[11px] text-text-muted">Dịch và mở rộng câu hỏi (yêu cầu Ollama)</div>
                </div>
                <div className="relative">
                  <input
                    type="checkbox"
                    checked={settings.enableRewrite}
                    onChange={(e) => update({ enableRewrite: e.target.checked })}
                    className="sr-only"
                  />
                  <div className={cn(
                    "w-10 h-5 rounded-full transition-colors",
                    settings.enableRewrite ? "bg-accent" : "bg-bg-elevated"
                  )}>
                    <div className={cn(
                      "absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform shadow",
                      settings.enableRewrite ? "translate-x-5" : "translate-x-0.5"
                    )} />
                  </div>
                </div>
              </label>
            </div>
          </div>

          {/* Course Filter */}
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-text">Lọc theo khóa học</h3>
            <div className="space-y-2">
              {COURSES.map((c) => (
                <label
                  key={c.id}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 text-xs rounded-md cursor-pointer transition-colors",
                    settings.courseFilter.includes(c.id!)
                      ? "bg-accent/10 text-accent border border-accent/30" 
                      : "text-text-dim hover:bg-bg-elevated-2 border border-transparent"
                  )}
                >
                  <input
                    type="checkbox"
                    checked={settings.courseFilter.includes(c.id!)}
                    onChange={(e) => {
                      const newCourses = e.target.checked
                        ? [...settings.courseFilter, c.id!]
                        : settings.courseFilter.filter(id => id !== c.id);
                      update({ courseFilter: newCourses });
                    }}
                    className="w-4 h-4 rounded border-border accent-accent"
                  />
                  {c.label}
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-6 pt-4 border-t border-border">
          <Button 
            variant="secondary" 
            className="w-full" 
            onClick={() => onOpenChange(false)}
          >
            Đóng
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  );
}