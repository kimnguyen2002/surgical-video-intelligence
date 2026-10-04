import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, GripHorizontal, Stethoscope, X } from 'lucide-react';

export type TourPhase = 'closed' | 'welcome' | 'steps';

interface TourStep {
  target: string;
  title: string;
  body: string;
}

const STEPS: TourStep[] = [
  {
    target: 'library',
    title: '1. Pick a surgical case',
    body: 'Open a case and choose a part marked with the scissors icon. That loads a short bundled clip, so nothing needs to be uploaded. You can also attach your own SurgVU part files from disk.',
  },
  {
    target: 'stage',
    title: '2. Watch and analyse',
    body: 'Press play, then switch on analysis. A detector running inside your browser draws boxes around instruments. No frame ever leaves your computer.',
  },
  {
    target: 'ribbon',
    title: '3. Recorded timeline',
    body: 'The ribbon shows what the dataset annotators recorded across the whole procedure. Click anywhere on it to jump to that moment.',
  },
  {
    target: 'facts',
    title: '4. Recorded vs predicted',
    body: 'Recorded labels come from the dataset. Predictions come from the on-device model and always show a confidence. The two are kept apart so you can see where they agree.',
  },
  {
    target: 'assistant',
    title: '5. Ask a question',
    body: 'Try "Which instruments are installed right now?" Answers come from the annotations first and are free. A badge on each reply shows whether it was recorded, quoted from a knowledge base, or generated. The mic button lets you speak.',
  },
  {
    target: 'settings',
    title: '6. Optional: your own Gemini key',
    body: 'Set your audience level here. If you want generated explanations, you can paste your own Gemini key. It stays in this browser and is never sent to anyone but Google.',
  },
  {
    target: 'about',
    title: '7. Learn how it works',
    body: 'The About page explains the dataset, the models, and their measured limits. Reopen this guide any time from the Guide button.',
  },
];

const SEEN_KEY = 'svi-tour-seen';

export function tourSeen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === '1';
  } catch {
    return false;
  }
}

export function markTourSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, '1');
  } catch {
    /* storage can be blocked; the tour simply reappears next visit */
  }
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

const CARD_W = 330;
const GAP = 22;
const MARGIN = 12;
const PAD = 6;

function findTarget(name: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(`[data-tour="${name}"]`);
}

function measure(el: HTMLElement): Box {
  const r = el.getBoundingClientRect();
  return { x: r.left - PAD, y: r.top - PAD, w: r.width + PAD * 2, h: r.height + PAD * 2 };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function defaultCardPosition(target: Box | null, cardH: number): { x: number; y: number } {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const cardW = Math.min(CARD_W, vw - MARGIN * 2);
  const centered = { x: (vw - cardW) / 2, y: Math.max(MARGIN, (vh - cardH) / 2) };
  if (!target) return centered;

  const fits = (x: number, y: number) =>
    x >= MARGIN && y >= MARGIN && x + cardW <= vw - MARGIN && y + cardH <= vh - MARGIN;
  const midY = target.y + target.h / 2 - cardH / 2;
  const midX = target.x + target.w / 2 - cardW / 2;
  const candidates = [
    { x: target.x + target.w + GAP, y: midY },
    { x: target.x - cardW - GAP, y: midY },
    { x: midX, y: target.y + target.h + GAP },
    { x: midX, y: target.y - cardH - GAP },
  ];
  for (const c of candidates) {
    const y = clamp(c.y, MARGIN, vh - cardH - MARGIN);
    const x = clamp(c.x, MARGIN, vw - cardW - MARGIN);
    if (fits(c.x, y) || (Math.abs(x - c.x) < 1 && Math.abs(y - c.y) < 1)) return { x, y };
  }
  // Nothing fits beside the target (a tall panel on a small screen): dock
  // to the bottom edge of the viewport, over the target if it must.
  return { x: clamp(midX, MARGIN, vw - cardW - MARGIN), y: vh - cardH - MARGIN };
}

function nearestPoint(box: Box, px: number, py: number): { x: number; y: number } {
  return { x: clamp(px, box.x, box.x + box.w), y: clamp(py, box.y, box.y + box.h) };
}

interface Props {
  phase: TourPhase;
  onPhaseChange: (phase: TourPhase) => void;
}

export function GuidedTour({ phase, onPhaseChange }: Props) {
  const [index, setIndex] = useState(0);
  const [target, setTarget] = useState<Box | null>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [cardH, setCardH] = useState(220);
  const cardRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ px: number; py: number; ox: number; oy: number } | null>(null);

  // Steps whose element is not on screen (the recorded ribbon exists only
  // once a part is loaded) are described in the middle of the screen rather
  // than skipped, so the guide never silently loses a step.
  const step = STEPS[index];

  const close = useCallback(() => {
    markTourSeen();
    onPhaseChange('closed');
    setIndex(0);
  }, [onPhaseChange]);

  const remeasure = useCallback(() => {
    const el = findTarget(STEPS[index].target);
    setTarget(el ? measure(el) : null);
  }, [index]);

  useEffect(() => {
    if (phase !== 'steps') return;
    setOffset({ x: 0, y: 0 });
    const el = findTarget(step.target);
    if (el) {
      const tall = el.getBoundingClientRect().height > window.innerHeight * 0.8;
      el.scrollIntoView({
        block: tall ? 'start' : 'center',
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      });
    }
    remeasure();
  }, [phase, index, step.target, remeasure]);

  useEffect(() => {
    if (phase !== 'steps') return;
    let frame = 0;
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(remeasure);
    };
    window.addEventListener('scroll', schedule, true);
    window.addEventListener('resize', schedule);
    // Layout can shift after a step (a clip loads, a panel grows) without a
    // scroll or resize event.
    const timer = window.setInterval(schedule, 600);
    return () => {
      cancelAnimationFrame(frame);
      window.clearInterval(timer);
      window.removeEventListener('scroll', schedule, true);
      window.removeEventListener('resize', schedule);
    };
  }, [phase, remeasure]);

  useLayoutEffect(() => {
    if (cardRef.current) setCardH(cardRef.current.offsetHeight);
  }, [phase, index, target === null]);

  useEffect(() => {
    if (phase === 'closed') return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
      if (phase !== 'steps') return;
      if (e.key === 'ArrowRight') setIndex((i) => Math.min(i + 1, STEPS.length - 1));
      if (e.key === 'ArrowLeft') setIndex((i) => Math.max(i - 1, 0));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [phase, close]);

  if (phase === 'closed') return null;

  if (phase === 'welcome') {
    return (
      <div
        className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-welcome-title"
      >
        <div className="w-full max-w-md rounded-2xl border border-line bg-card p-6 shadow-2xl">
          <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-xl border border-accent/40 bg-accent-soft text-accent">
            <Stethoscope className="h-6 w-6" />
          </div>
          <h2 id="tour-welcome-title" className="text-lg font-bold text-fg">
            New here? Take a 1-minute tour
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-fg2">
            This page analyses robotic surgery video entirely in your browser. A short guided
            walkthrough will point out where to start: loading a clip, reading the results, and
            asking questions.
          </p>
          <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
            <button
              onClick={close}
              className="rounded-lg border border-line px-4 py-2 text-sm font-medium text-fg2 transition-colors hover:bg-hover"
            >
              Skip, I'll explore
            </button>
            <button
              autoFocus
              onClick={() => {
                setIndex(0);
                onPhaseChange('steps');
              }}
              className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-on-accent transition-colors hover:bg-accent-hover"
            >
              Show me around
            </button>
          </div>
        </div>
      </div>
    );
  }

  const base = defaultCardPosition(target, cardH);
  const cardW = Math.min(CARD_W, window.innerWidth - MARGIN * 2);
  const card: Box = {
    x: clamp(base.x + offset.x, MARGIN, window.innerWidth - cardW - MARGIN),
    y: clamp(base.y + offset.y, MARGIN, window.innerHeight - cardH - MARGIN),
    w: cardW,
    h: cardH,
  };

  let arrow: { x1: number; y1: number; x2: number; y2: number } | null = null;
  if (target) {
    const tc = { x: target.x + target.w / 2, y: target.y + target.h / 2 };
    const cc = { x: card.x + card.w / 2, y: card.y + card.h / 2 };
    const from = nearestPoint(card, tc.x, tc.y);
    const to = nearestPoint(target, cc.x, cc.y);
    const overlap =
      card.x < target.x + target.w &&
      card.x + card.w > target.x &&
      card.y < target.y + target.h &&
      card.y + card.h > target.y;
    if (!overlap && Math.hypot(to.x - from.x, to.y - from.y) > 14) {
      arrow = { x1: from.x, y1: from.y, x2: to.x, y2: to.y };
    }
  }

  const onPointerDown = (e: React.PointerEvent) => {
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { px: e.clientX, py: e.clientY, ox: offset.x, oy: offset.y };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    setOffset({
      x: drag.current.ox + e.clientX - drag.current.px,
      y: drag.current.oy + e.clientY - drag.current.py,
    });
  };
  const onPointerUp = () => {
    drag.current = null;
  };

  const last = index === STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-[100]" role="dialog" aria-label="Guided tour">
      {/* Blocks clicks on the page underneath while the guide is open. */}
      <div className="absolute inset-0" onClick={(e) => e.stopPropagation()} />

      {target ? (
        <div
          className="pointer-events-none absolute rounded-xl border-2 border-accent/40 transition-all duration-300"
          style={{
            left: target.x,
            top: target.y,
            width: target.w,
            height: target.h,
            boxShadow: '0 0 0 9999px rgba(8, 14, 24, 0.6)',
          }}
        />
      ) : (
        <div className="pointer-events-none absolute inset-0 bg-[#080e18]/60" />
      )}

      {arrow && (
        <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true">
          <defs>
            <marker
              id="tour-arrowhead"
              markerWidth="10"
              markerHeight="10"
              refX="8"
              refY="5"
              orient="auto"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
            </marker>
          </defs>
          <line
            x1={arrow.x1}
            y1={arrow.y1}
            x2={arrow.x2}
            y2={arrow.y2}
            stroke="var(--accent)"
            strokeWidth="2.5"
            strokeDasharray="7 5"
            strokeLinecap="round"
            markerEnd="url(#tour-arrowhead)"
          />
        </svg>
      )}

      <div
        ref={cardRef}
        className="absolute rounded-2xl border border-line bg-card shadow-2xl"
        style={{ left: card.x, top: card.y, width: card.w }}
      >
        <div
          className="flex cursor-grab touch-none select-none items-center justify-between rounded-t-2xl border-b border-line px-3 py-1.5 active:cursor-grabbing"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
        >
          <span className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-fg3">
            <GripHorizontal className="h-3.5 w-3.5" />
            Drag to move
          </span>
          <button
            onPointerDown={(e) => e.stopPropagation()}
            onClick={close}
            aria-label="Close guide"
            className="rounded p-1 text-fg3 transition-colors hover:bg-hover hover:text-fg"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-4 pb-4 pt-3">
          <h3 className="text-sm font-bold text-fg">{step.title}</h3>
          <p className="mt-1.5 text-[13px] leading-relaxed text-fg2">{step.body}</p>
          {!target && (
            <p className="mt-2 text-[11px] text-accent">
              This part appears once a clip is loaded.
            </p>
          )}

          <div className="mt-4 flex items-center justify-between">
            <div className="flex gap-1.5" aria-label={`Step ${index + 1} of ${STEPS.length}`}>
              {STEPS.map((s, i) => (
                <span
                  key={s.target}
                  className={`h-1.5 rounded-full transition-all ${
                    i === index ? 'w-4 bg-accent' : 'w-1.5 bg-inset'
                  }`}
                />
              ))}
            </div>
            <div className="flex items-center gap-2">
              {index > 0 && (
                <button
                  onClick={() => setIndex(index - 1)}
                  className="flex items-center gap-1 rounded-md border border-line px-2.5 py-1.5 text-xs font-medium text-fg2 transition-colors hover:bg-hover"
                >
                  <ArrowLeft className="h-3.5 w-3.5" /> Back
                </button>
              )}
              <button
                onClick={() => (last ? close() : setIndex(index + 1))}
                className="flex items-center gap-1 rounded-md bg-accent px-3 py-1.5 text-xs font-semibold text-on-accent transition-colors hover:bg-accent-hover"
              >
                {last ? 'Finish' : 'Next'}
                {!last && <ArrowRight className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
