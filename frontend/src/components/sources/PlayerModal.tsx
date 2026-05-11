import YouTube from 'react-youtube';
import { ExternalLink, Copy, Check } from 'lucide-react';
import { Dialog, DialogContent } from '../ui/dialog';
import { Button } from '../ui/button';
import type { SourceChunk } from '../../types/api';
import { useEffect, useState } from 'react';

interface PlayerModalProps {
  source: SourceChunk | null;
  onClose: () => void;
}

export function PlayerModal({ source, onClose }: PlayerModalProps) {
  const [copied, setCopied] = useState(false);
  /** YouTube blocks some videos from iframe embed (“playback disabled on other sites”). */
  const [embedBlocked, setEmbedBlocked] = useState(false);

  useEffect(() => {
    setEmbedBlocked(false);
  }, [source?.video_id, source?.youtube_url]);

  if (!source) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(source.text_en);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleYtError = () => setEmbedBlocked(true);

  return (
    <Dialog open={!!source} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl p-0 overflow-hidden bg-bg-elevated border-border shadow-2xl font-sans">
        <div className="aspect-video w-full bg-black">
          {embedBlocked ? (
            <div className="relative flex h-full min-h-[200px] w-full flex-col items-center justify-end bg-bg-elevated-2">
              <img
                src={source.thumbnail_url}
                alt=""
                className="absolute inset-0 h-full w-full object-cover opacity-40"
              />
              <div className="relative z-[1] w-full bg-gradient-to-t from-black/90 via-black/60 to-transparent px-6 py-8 text-center">
                <p className="text-sm font-medium text-text">
                  Video không được phép phát trên trang này
                </p>
                <p className="mt-2 text-xs text-text-muted">
                  Người tải lên có thể đã tắt nhúng; mở trực tiếp trên YouTube vẫn xem được.
                </p>
                <Button variant="default" size="sm" className="mt-4 gap-2" asChild>
                  <a href={source.youtube_url} target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="w-3.5 h-3.5" />
                    Xem trên YouTube
                  </a>
                </Button>
              </div>
            </div>
          ) : (
            <YouTube
              videoId={source.video_id}
              opts={{
                width: '100%',
                height: '100%',
              playerVars: {
                start: source.start_seconds,
                autoplay: 1,
                modestbranding: 1,
                rel: 0,
                ...(typeof window !== 'undefined'
                  ? { origin: window.location.origin }
                  : {}),
              },
              }}
              className="w-full h-full"
              onError={handleYtError}
            />
          )}
        </div>

        <div className="p-6 space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1 font-sans">
              <div className="flex items-center gap-2">
                <span className="bg-accent/10 text-accent text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full">
                  {source.course || 'Stanford NLP'}
                </span>
                <span className="text-text-muted text-xs font-mono">
                  t={source.timestamp_display}
                </span>
              </div>
              <h2 className="text-lg font-semibold text-text leading-snug tracking-tight">
                {source.video_title}
              </h2>
            </div>
            <div className="flex gap-2">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={handleCopy}
                className="h-8 gap-2"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-success" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? 'Đã sao chép' : 'Sao chép'}
              </Button>
              <Button 
                variant="default" 
                size="sm" 
                asChild
                className="h-8 gap-2"
              >
                <a href={source.youtube_url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="w-3.5 h-3.5" />
                  YouTube
                </a>
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-text-muted">
              Transcript Snippet
            </h3>
            <div className="rounded-md border border-border/50 bg-bg-elevated-2 p-4 text-sm leading-relaxed text-text-dim">
              {source.text_en}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
