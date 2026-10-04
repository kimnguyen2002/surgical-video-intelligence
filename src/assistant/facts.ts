/**
 * The facts the assistant is allowed to assert, and where each one came from.
 *
 * This is the browser port of `VideoFacts` in the original Python backend. The
 * shape matters more than the code: an answer is assembled only from values
 * that appear here, so anything the assistant says can be traced to a field
 * with a known provenance.
 *
 * The three provenances are not interchangeable and the wording downstream
 * never treats them as such:
 *
 * - **recorded** — the SurgVU release's own annotations. A fact. No confidence
 *   attaches to it, because none applies.
 * - **predicted** — this machine's ONNX models, at runtime. An estimate, and
 *   always carries a percentage.
 * - **reference** — the specialty knowledge graph. True of the *operation* in
 *   general, not of this video, and never localised in the current frame.
 */

import type { FrameTruth } from '../services/groundTruth';
import type { LocalDetection } from '../services/localVision';
import type { AnatomicalRisk } from '../types';

/** One annotated span of the recording, instrument or task. */
export interface TimelineEvent {
  kind: 'tool' | 'task';
  label: string;
  display: string;
  start: number;
  end: number;
  duration: number;
  /** Robot arm, for instrument events only. */
  arm?: string;
}

/** A retrieved passage from the bundled knowledge base. */
export interface Passage {
  id: string;
  title: string;
  text: string;
  citation: string;
  score?: number;
}

export interface VideoFacts {
  /** What the release records for the current frame, when a library clip plays. */
  groundTruth: FrameTruth | null;

  /** On-device detector output for the current frame. Estimates, not facts. */
  detections: LocalDetection[];

  /** Presence classifier output for the current frame. */
  predictedTools: { label: string; confidence: number }[];

  /** Task classifier output for the current frame. */
  predictedTask: { label: string; confidence: number } | null;

  /** Every annotated span in the part being played. */
  timeline: TimelineEvent[];

  /** Knowledge-base passages retrieved for the question. */
  passages: Passage[];

  /** Reference anatomy for the current phase, from the knowledge graph. */
  risks: AnatomicalRisk[];

  /**
   * Playback position **in the coordinate system of the original part**.
   *
   * For a bundled excerpt this is the player's `currentTime` plus the clip's
   * `sourceStartSeconds`. Using the player's own clock would look right and be
   * wrong by up to four hours — every annotation lookup would land in the
   * opening minutes of a case while the viewer watches its middle.
   */
  timestamp: number | null;

  title: string | null;
  specialtyName: string;

  /**
   * True when the out-of-domain guard rejected the current frame. Suppresses
   * every predicted claim; recorded annotations are unaffected, since they do
   * not depend on the pixels.
   */
  outOfDomain: boolean;
}

export function emptyFacts(): VideoFacts {
  return {
    groundTruth: null,
    detections: [],
    predictedTools: [],
    predictedTask: null,
    timeline: [],
    passages: [],
    risks: [],
    timestamp: null,
    title: null,
    specialtyName: 'General Surgery',
    outOfDomain: false,
  };
}

/**
 * Whether there is anything at all to answer from.
 *
 * A recorded frame with no instrument installed still counts: "nothing was
 * mounted at this moment" is knowledge, not the absence of it. What does not
 * count is a part with no annotations bundled and no model output — there the
 * assistant has to decline rather than improvise.
 */
export function hasAnyFacts(facts: VideoFacts): boolean {
  return (
    (facts.groundTruth !== null && facts.groundTruth.provenance === 'recorded') ||
    facts.detections.length > 0 ||
    facts.predictedTools.length > 0 ||
    facts.predictedTask !== null ||
    facts.timeline.length > 0
  );
}

/** `1:23:45` for hours, `4:05` below an hour. */
export function hms(seconds: number | null | undefined): string {
  const total = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;
}

/** "a", "a and b", "a, b and c" — Oxford comma deliberately omitted. */
export function join(items: Iterable<string>): string {
  const list = [...items].filter(Boolean).map(String);
  if (list.length === 0) return '';
  if (list.length === 1) return list[0];
  return `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
}
