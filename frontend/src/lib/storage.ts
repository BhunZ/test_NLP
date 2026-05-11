import type { SourceChunk } from '../types/api';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  sources?: SourceChunk[];
  confidence?: 'high' | 'medium' | 'low';
  timestamp: number;
  isStreaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'nlp_tutor_conversations';
const UI_PREFS_KEY = 'nlp_tutor_ui_prefs';

export type ThemeMode = 'dark' | 'light' | 'system';

export interface UiPrefs {
  theme: ThemeMode;
}

export function saveConversations(conversations: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

export function loadConversations(): Conversation[] {
  const data = localStorage.getItem(STORAGE_KEY);
  if (!data) return [];
  try {
    return JSON.parse(data);
  } catch (e) {
    console.error('Failed to parse conversations', e);
    return [];
  }
}

export function generateTitle(text: string): string {
  const words = text.split(' ');
  if (words.length <= 6) return text;
  return words.slice(0, 6).join(' ') + '...';
}

export function loadUiPrefs(): UiPrefs {
  const raw = localStorage.getItem(UI_PREFS_KEY);
  if (!raw) return { theme: 'system' };
  try {
    const parsed = JSON.parse(raw) as Partial<UiPrefs>;
    return { theme: parsed.theme ?? 'system' };
  } catch (e) {
    console.error('Failed to parse UI preferences', e);
    return { theme: 'system' };
  }
}

export function saveUiPrefs(prefs: UiPrefs) {
  localStorage.setItem(UI_PREFS_KEY, JSON.stringify(prefs));
}
