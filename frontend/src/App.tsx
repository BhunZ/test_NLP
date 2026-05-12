import { useState, useRef, useEffect, useMemo } from 'react';
import { Settings as SettingsIcon, Menu, AlertCircle } from 'lucide-react';
import { Button } from './components/ui/button';
import { Sheet, SheetContent } from './components/ui/sheet';
import { InputBar } from './components/chat/InputBar';
import { MessageBubble } from './components/chat/MessageBubble';
import { EmptyState } from './components/chat/EmptyState';
import { StageIndicator } from './components/chat/StageIndicator';
import { SourceGrid } from './components/sources/SourceGrid';
import { PlayerModal } from './components/sources/PlayerModal';
import { Sidebar } from './components/sidebar/Sidebar';
import { SettingsDrawer } from './components/settings/SettingsDrawer';
import type { Settings } from './components/settings/SettingsDrawer';
import { useAskStream } from './hooks/useAskStream';
import type { SourceChunk } from './types/api';
import { loadConversations, saveConversations, generateTitle, loadUiPrefs, saveUiPrefs } from './lib/storage';
import type { Conversation, Message } from './lib/storage';

function App() {
  // --- PERSISTENCE STATE ---
  const [conversations, setConversations] = useState<Conversation[]>(() => loadConversations());
  const [activeId, setActiveId] = useState<string | null>(null);

  // --- UI STATE ---
  const [messages, setMessages] = useState<Message[]>([]);
  const [highlightedRank, setHighlightedRank] = useState<number | null>(null);
  const [selectedSource, setSelectedSource] = useState<SourceChunk | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'ok' | 'error'>('checking');
  const [uiToast, setUiToast] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const initialPrefs = useMemo(() => loadUiPrefs(), []);
  
  const [settings, setSettings] = useState<Settings>({
    llmProvider: 'groq',
    topK: 6,
    rerank: false,
    enableRewrite: false,
    courseFilter: [],
    theme: initialPrefs.theme,
  });

  const scrollRef = useRef<HTMLDivElement>(null);
  const { ask, isStreaming, answer, sources, stages, answerData, errorMessage, rewrites, reset, meta } = useAskStream();
  const [streamStartTime, setStreamStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState<number>(0);
  const [lastQuery, setLastQuery] = useState<string | null>(null);

  // Check backend connection
  useEffect(() => {
    fetch('/api/v1/health')
      .then(res => res.ok ? setBackendStatus('ok') : setBackendStatus('error'))
      .catch(() => setBackendStatus('error'));
  }, []);

  useEffect(() => {
    saveUiPrefs({ theme: settings.theme });
    const root = document.documentElement;
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldUseDark = settings.theme === 'dark' || (settings.theme === 'system' && prefersDark);
    root.classList.toggle('dark', shouldUseDark);
  }, [settings.theme]);

  useEffect(() => {
    if (!errorMessage) return;
    setUiToast(errorMessage);
    const t = setTimeout(() => setUiToast(null), 4000);
    return () => clearTimeout(t);
  }, [errorMessage]);

  // Timer for streaming elapsed time
  useEffect(() => {
    if (isStreaming && !streamStartTime) {
      setStreamStartTime(Date.now());
    } else if (!isStreaming && streamStartTime) {
      setStreamStartTime(null);
      setElapsedTime(0);
    }
  }, [isStreaming, streamStartTime]);

  useEffect(() => {
    if (!isStreaming || !streamStartTime) return;
    const interval = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - streamStartTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [isStreaming, streamStartTime]);

  // Sync UI messages when active conversation changes
  useEffect(() => {
    if (activeId) {
      const conv = conversations.find(c => c.id === activeId);
      if (conv) setMessages(conv.messages);
    } else {
      setMessages([]);
    }
  }, [activeId, conversations]);

  // Handle Sending
  const handleSend = (query: string) => {
    const now = Date.now();
    const userMsg: Message = {
      id: `u-${crypto.randomUUID()}`,
      role: 'user',
      text: query,
      timestamp: now,
    };

    const currentId = activeId ?? crypto.randomUUID();
    if (!activeId) setActiveId(currentId);

    // Persist user message immediately to avoid overwrite from conversation sync.
    setConversations(prev => {
      const existing = prev.find(c => c.id === currentId);
      let next: Conversation[];

      if (!existing) {
        const newConv: Conversation = {
          id: currentId,
          title: generateTitle(query),
          messages: [userMsg],
          createdAt: now,
          updatedAt: now,
        };
        next = [newConv, ...prev];
      } else {
        next = prev.map(c => {
          if (c.id !== currentId) return c;
          if (c.messages.some(m => m.id === userMsg.id)) return c;
          return { ...c, messages: [...c.messages, userMsg], updatedAt: now };
        });
      }

      saveConversations(next);
      return next;
    });

    // Keep local UI responsive.
    setMessages(prev => [...prev, userMsg]);

    setLastQuery(query);
    ask({
      query,
      top_k: settings.topK,
      llm_provider: settings.llmProvider,
      rerank: settings.rerank,
      enable_rewrite: settings.enableRewrite,
      course_filter: settings.courseFilter.length > 0 ? settings.courseFilter[0] : null,
    });
  };

  const handleRegenerate = () => {
    if (!lastQuery || !settings) return;
    const newModel = settings.llmProvider === 'groq' ? 'mistral' : 'groq';
    reset();
    setStreamStartTime(null);
    setElapsedTime(0);
    ask({
      query: lastQuery,
      top_k: settings.topK,
      llm_provider: newModel,
      rerank: settings.rerank,
      enable_rewrite: settings.enableRewrite,
      course_filter: settings.courseFilter.length > 0 ? settings.courseFilter[0] : null,
    });
  };

  // Commit to history when stream finishes
  useEffect(() => {
    const finalAnswerText = answer || answerData?.text_vi || '';
    if (!isStreaming && finalAnswerText && activeId) {
      const assistantMsg: Message = {
        id: `a-${crypto.randomUUID()}`,
        role: 'assistant',
        text: finalAnswerText,
        sources: sources,
        confidence: answerData?.confidence,
        timestamp: Date.now(),
      };

      setConversations(prev => {
        const next = prev.map(c => {
          if (c.id === activeId) {
            // Find the last user message in the current UI state to ensure we save the pair
            const lastUserMsg = messages.findLast(m => m.role === 'user');
            
            // Avoid duplicating messages if they are already in the conversation
            const existingIds = new Set(c.messages.map(m => m.id));
            const newMessages = [...c.messages];
            
            if (lastUserMsg && !existingIds.has(lastUserMsg.id)) {
              newMessages.push(lastUserMsg);
            }
            if (!existingIds.has(assistantMsg.id)) {
              newMessages.push(assistantMsg);
            }
            
            return { ...c, messages: newMessages, updatedAt: Date.now() };
          }
          return c;
        });
        saveConversations(next);
        return next;
      });
      
      reset();
    }
  }, [isStreaming, answer, answerData, activeId, messages, sources, reset]);

  const handleNewChat = () => {
    setActiveId(null);
    setMessages([]);
    reset();
    setIsSidebarOpen(false);
  };

  const handleDeleteChat = (id: string) => {
    setConversations(prev => {
      const next = prev.filter(c => c.id !== id);
      saveConversations(next);
      return next;
    });
    if (activeId === id) handleNewChat();
  };

  const handleCitationClick = (n: number) => {
    setHighlightedRank(n);
    const element = document.querySelector(`[data-source-rank="${n}"]`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    setTimeout(() => setHighlightedRank(null), 2400);
  };

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      if (e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key.toLowerCase() === 'l') {
        e.preventDefault();
        handleNewChat();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // Scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isStreaming, answer]);

  return (
    <div className="flex h-screen bg-bg text-text overflow-hidden font-sans">
      <Sidebar 
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={handleNewChat}
        onDelete={handleDeleteChat}
      />

      <Sheet open={isSidebarOpen} onOpenChange={setIsSidebarOpen}>
        <SheetContent side="left" className="p-0 w-72">
          <Sidebar
            conversations={conversations}
            activeId={activeId}
            onSelect={(id) => {
              setActiveId(id);
              setIsSidebarOpen(false);
            }}
            onNew={handleNewChat}
            onDelete={handleDeleteChat}
            compact
            className="w-full border-r-0"
          />
        </SheetContent>
      </Sheet>

      <main className="flex-1 flex flex-col relative overflow-hidden">
        <header className="h-14 border-b border-border glass flex items-center justify-between px-4 z-10">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsSidebarOpen(true)}>
              <Menu className="h-5 w-5" />
            </Button>
            <div className="flex flex-col">
              <h2 className="text-sm font-semibold truncate max-w-[200px] md:max-w-none">
                {activeId ? (conversations.find(c => c.id === activeId)?.title) : 'Stanford NLP Tutor'}
              </h2>
              {backendStatus === 'error' && (
                <span className="text-[10px] text-danger flex items-center gap-1">
                  <AlertCircle className="w-2.5 h-2.5" /> Lỗi kết nối server
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" onClick={() => setIsSettingsOpen(true)}>
              <SettingsIcon className="h-5 w-5" />
            </Button>
          </div>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto scroll-smooth no-scrollbar">
          {messages.length === 0 && !isStreaming ? (
            <EmptyState onExampleClick={handleSend} />
          ) : (
            <div className="pb-32">
              {/* Render History + Immediate User Message */}
              {messages.map((msg) => (
                <div key={msg.id}>
                  <MessageBubble
                    role={msg.role}
                    text={msg.text}
                    confidence={msg.confidence}
                    onCitationClick={handleCitationClick}
                    citationLookup={Object.fromEntries((msg.sources ?? []).map((s) => [s.rank, s]))}
                  />
                  {msg.sources && msg.sources.length > 0 && (
                    <SourceGrid 
                      sources={msg.sources} 
                      highlightedRank={highlightedRank}
                      onOpenSource={setSelectedSource} 
                    />
                  )}
                </div>
              ))}

              {/* Render Streaming Assistant Message */}
              {isStreaming && (
                <div className="animate-in fade-in slide-in-from-bottom-2 duration-300">
                  <MessageBubble
                    role="assistant"
                    text={answer || "Đang xử lý..."}
                    onRegenerate={lastQuery ? handleRegenerate : undefined}
                    confidence={answerData?.confidence}
                    isStreaming={true}
                    onCitationClick={handleCitationClick}
                    citationLookup={Object.fromEntries(sources.map((s) => [s.rank, s]))}
                    meta={meta}
                    elapsedTime={elapsedTime}
                  />
                  <div className="max-w-3xl mx-auto px-16 md:px-20 -mt-6 mb-4">
                    <StageIndicator stages={stages} />
                  </div>
                  {rewrites && (
                    <div className="max-w-3xl mx-auto px-16 md:px-20 mb-4">
                      <div className="text-xs text-text-dim bg-bg-elevated-2/50 rounded-md p-3 space-y-1">
                        <div className="font-medium text-text-muted">Query Rewrite:</div>
                        {rewrites.q1 && <div><span className="text-accent">Literal:</span> {rewrites.q1}</div>}
                        {rewrites.q2 && <div><span className="text-accent">Expanded:</span> {rewrites.q2}</div>}
                      </div>
                    </div>
                  )}
                  {sources.length > 0 && (
                    <SourceGrid 
                      sources={sources} 
                      highlightedRank={highlightedRank}
                      onOpenSource={setSelectedSource} 
                      isLoading={isStreaming && sources.length === 0}
                    />
                  )}
                  {sources.length === 0 && (
                    <SourceGrid
                      sources={[]}
                      highlightedRank={highlightedRank}
                      onOpenSource={setSelectedSource}
                      isLoading={isStreaming}
                    />
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <PlayerModal source={selectedSource} onClose={() => setSelectedSource(null)} />
        <SettingsDrawer open={isSettingsOpen} onOpenChange={setIsSettingsOpen} settings={settings} onSettingsChange={setSettings} />
        <InputBar onSend={handleSend} disabled={isStreaming || backendStatus === 'checking'} inputRef={inputRef} settings={settings} onSettingsChange={setSettings} meta={meta} isStreaming={isStreaming} elapsedTime={elapsedTime} />
        {uiToast && (
          <div className="absolute bottom-24 right-4 z-50 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger backdrop-blur">
            {uiToast}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
