import React, { useState, useRef, useEffect } from 'react';
import {
  MessageSquare,
  Volume2,
  VolumeX,
  Mic,
  MicOff,
  Send,
  Sparkles,
  User,
  Bot,
  Loader2,
} from 'lucide-react';
import { AnswerSource, ChatMessage, FrameObservation } from '../types';

/**
 * How each answer path is labelled in the transcript.
 *
 * Every assistant turn carries one of these. The distinction it draws is the
 * whole point of the interface: a reader must be able to tell a dataset lookup
 * from a language model's prose without having to ask which they are reading,
 * because the two are equally fluent and only one of them is a fact.
 */
const SOURCE_BADGE: Record<AnswerSource, { label: string; title: string; className: string }> = {
  grounded: {
    label: 'recorded',
    title:
      "Assembled from the SurgVU release's own annotations and this machine's model output. No language model was involved.",
    className: 'text-emerald-300 bg-emerald-950/70 border-emerald-800/60',
  },
  extractive: {
    label: 'quoted',
    title:
      'Sentences ranked and quoted verbatim from the bundled knowledge base. Nothing here was generated.',
    className: 'text-sky-300 bg-sky-950/70 border-sky-800/60',
  },
  generative: {
    label: 'generated',
    title:
      'Written by a language model, using your own API key, grounded in the annotations above. Treat it as explanation, not as evidence.',
    className: 'text-amber-300 bg-amber-950/70 border-amber-800/60',
  },
  none: {
    label: 'no answer',
    title: 'Nothing in the annotations or the knowledge base answered this.',
    className: 'text-slate-400 bg-slate-900 border-slate-700',
  },
};

/**
 * Inline markdown: bold, italic, and code.
 *
 * The model replies in markdown and the thread printed it raw, so answers
 * arrived full of literal asterisks — `**Scene Summary:**` and `*Note: ...*`
 * rendered as punctuation rather than emphasis. Splitting on the delimiters
 * and returning React nodes keeps it safe: no HTML is ever constructed, so
 * model output cannot inject markup.
 *
 * `**` is matched before `*` in the alternation, otherwise the italic rule
 * would claim the first two asterisks of every bold span.
 */
const INLINE_MD = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`)/g;

function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  return text
    .split(INLINE_MD)
    .filter((part) => part !== '')
    .map((part, i) => {
      const key = `${keyPrefix}-${i}`;
      if (part.length > 4 && part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={key} className="font-semibold text-slate-100">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.length > 2 && part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={key} className="font-mono text-[11px] text-cyan-300 bg-black/40 px-1 rounded">
            {part.slice(1, -1)}
          </code>
        );
      }
      if (part.length > 2 && part.startsWith('*') && part.endsWith('*')) {
        return (
          <em key={key} className="italic text-slate-400">
            {part.slice(1, -1)}
          </em>
        );
      }
      return <React.Fragment key={key}>{part}</React.Fragment>;
    });
}

/** Block-level markdown: bullet lists, horizontal rules, paragraphs. */
function Markdown({ text }: { text: string }): React.ReactElement {
  const lines = text.split('\n');
  const blocks: React.ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (bullets.length === 0) return;
    const items = bullets;
    bullets = [];
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="list-disc pl-4 space-y-0.5 marker:text-slate-600">
        {items.map((item, i) => (
          <li key={i}>{renderInline(item, `li-${blocks.length}-${i}`)}</li>
        ))}
      </ul>
    );
  };

  lines.forEach((raw, index) => {
    const line = raw.trimEnd();
    const bullet = /^\s*[*-]\s+(.*)$/.exec(line);

    if (bullet) {
      bullets.push(bullet[1]);
      return;
    }
    flushBullets();

    if (/^\s*(\*\*\*|---|___)\s*$/.test(line)) {
      blocks.push(<hr key={`hr-${index}`} className="border-slate-800 my-1.5" />);
      return;
    }
    if (line.trim() === '') return;

    const heading = /^#{1,4}\s+(.*)$/.exec(line);
    blocks.push(
      <p key={`p-${index}`} className={heading ? 'font-semibold text-slate-100' : 'leading-relaxed'}>
        {renderInline(heading ? heading[1] : line, `p-${index}`)}
      </p>
    );
  });

  flushBullets();
  return <div className="space-y-1.5">{blocks}</div>;
}

interface ChatAssistantProps {
  currentObservation: FrameObservation | null;
  messages: ChatMessage[];
  isTtsEnabled: boolean;
  isMicListening: boolean;
  isMicStarting: boolean;
  /** An answer is in flight. Only ever true for a generative call. */
  isThinking?: boolean;
  onSendMessage: (text: string) => void;
  onToggleTts: () => void;
  onToggleMic: () => void;
}

export const ChatAssistant: React.FC<ChatAssistantProps> = ({
  messages,
  isTtsEnabled,
  isMicListening,
  isMicStarting,
  isThinking = false,
  onSendMessage,
  onToggleTts,
  onToggleMic,
}) => {
  const [inputText, setInputText] = useState('');
  const threadRef = useRef<HTMLDivElement | null>(null);

  /**
   * Scroll the transcript, not the document.
   *
   * `scrollIntoView` on a sentinel at the bottom of the thread walks up to
   * every scrollable ancestor — including the page — so each new answer
   * yanked the whole dashboard downwards and pushed the video off screen.
   * Setting `scrollTop` on the thread container moves only the container.
   */
  useEffect(() => {
    const thread = threadRef.current;
    if (thread) thread.scrollTop = thread.scrollHeight;
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  // Phrased so the answer has to distinguish recorded from predicted, which
  // is the distinction the whole interface exists to keep visible.
  const quickPrompts = [
    'What does the dataset record at this timestamp?',
    'Does the prediction match the recorded instruments?',
    'What anatomy is at risk during this task?',
    'Summarise what has happened in this case so far',
  ];

  // Height follows the viewport rather than a fixed 500px, so the transcript
  // is as tall as the window allows and a long answer can be read without
  // scrolling a small box inside a tall page. `sticky` keeps it in view while
  // the left column scrolls past.
  return (
    <div className="bg-[#0b1017] rounded-lg border border-slate-800/80 flex flex-col h-[calc(100vh-7rem)] min-h-[420px] shadow-lg sticky top-16">
      {/* Header with audio & mic toggles */}
      <div className="p-2.5 px-3 border-b border-slate-800/80 flex items-center justify-between bg-[#0e141f] rounded-t-lg flex-shrink-0">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Assistant
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {/* TTS Speaker Toggle */}
          <button
            onClick={onToggleTts}
            className={`p-1.5 rounded transition-colors ${
              isTtsEnabled
                ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
            title={isTtsEnabled ? 'Disable voice readback' : 'Enable voice readback'}
          >
            {isTtsEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
          </button>

          {/* Mic Speech-to-Text Toggle */}
          <button
            onClick={onToggleMic}
            className={`p-1.5 rounded transition-colors ${
              isMicListening
                ? 'bg-red-950 text-red-400 border border-red-700 animate-pulse'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
            disabled={isMicStarting}
            title={
              isMicStarting
                ? 'Waiting for microphone access…'
                : isMicListening
                  ? 'Recording — click again to transcribe and send'
                  : 'Push to talk: click, speak, click again'
            }
          >
            {isMicStarting ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : isMicListening ? (
              <Mic className="w-3.5 h-3.5 animate-pulse" />
            ) : (
              <MicOff className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Message Thread (Scrollable) */}
      <div ref={threadRef} className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2.5 text-xs">
        {messages.map((msg) => {
          const isUser = msg.sender === 'user';
          const isSystem = msg.sender === 'system';

          return (
            <div
              key={msg.id}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} space-y-0.5`}
            >
              <div
                className={`p-2.5 rounded-lg max-w-[95%] leading-relaxed ${
                  isUser
                    ? 'bg-cyan-950/80 text-cyan-100 border border-cyan-800/60'
                    : isSystem
                      ? 'bg-slate-900/90 text-slate-300 border border-slate-800 italic'
                      : 'bg-[#0e141f] text-slate-200 border border-slate-800/80'
                }`}
              >
                <Markdown text={msg.text} />
              </div>

              <span className="flex items-center gap-1.5 px-1">
                <span className="text-[9px] text-slate-500 font-mono">{msg.timestamp}</span>
                {msg.source && (
                  <span
                    title={SOURCE_BADGE[msg.source].title}
                    className={`text-[9px] px-1.5 py-px rounded border font-mono cursor-help ${SOURCE_BADGE[msg.source].className}`}
                  >
                    {SOURCE_BADGE[msg.source].label}
                  </span>
                )}
              </span>
            </div>
          );
        })}

        {isThinking && (
          <div className="flex items-start">
            <div className="p-2.5 rounded-lg bg-[#0e141f] border border-slate-800/80 flex items-center gap-2 text-slate-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span className="text-[11px]">Composing an answer…</span>
            </div>
          </div>
        )}
      </div>

      {/*
        Suggested prompts — all of them, wrapped rather than clipped.

        These were rendered two-at-a-time inside a horizontal scroller with
        `truncate max-w-[200px]`, so two suggestions were invisible and the two
        that showed were cut off mid-sentence with no hint they continued.
      */}
      <div className="px-2.5 py-2 border-t border-slate-800/60 flex flex-wrap gap-1.5 flex-shrink-0 bg-[#090d14]">
        {quickPrompts.map((prompt, i) => (
          <button
            key={i}
            onClick={() => onSendMessage(prompt)}
            className="text-[10px] leading-snug text-left px-2 py-1 rounded bg-slate-800/70 hover:bg-cyan-950/60 hover:text-cyan-300 text-slate-400 border border-slate-700/50 transition-colors"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Input Box */}
      <form onSubmit={handleSubmit} className="p-2.5 border-t border-slate-800/80 flex gap-2 flex-shrink-0 bg-[#0b1017] rounded-b-lg">
        <input
          id="input-assistant-chat"
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ask about this video…"
          className="flex-1 bg-[#131b26] border border-slate-700/80 rounded-md px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
        />

        <button
          type="submit"
          disabled={!inputText.trim()}
          className="p-1.5 px-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 text-black font-semibold rounded-md transition-colors flex items-center justify-center"
          title="Send message"
        >
          <Send className="w-3.5 h-3.5" />
        </button>
      </form>
    </div>
  );
};
