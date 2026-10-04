/**
 * Ground truth lookup for the bundled SurgVU cases.
 *
 * This is the browser counterpart of `surgical_main/ai/datasets/groundtruth.py`,
 * and it exists for the same reason: for a video that came from the SurgVU
 * release, the app does not have to *guess* what instrument is in use at
 * t = 1500 s. The release says so, from the robot's own installation log.
 *
 * So every assertion the interface makes about a library video is stamped with
 * where it came from. "The needle driver was installed at 00:39:45" and "a
 * model thinks it sees a needle driver" are different claims and must never
 * render identically.
 *
 * One property of the dataset shapes everything downstream: tool labels come
 * from installation logs, not from vision. A tool counts as present while it
 * is installed — including while it is off-screen or occluded. Recorded
 * presence is therefore *not* a bounding box and must not be drawn as one.
 */

import { SURGVU_CASES, SurgvuCase, SurgvuPart } from '../data/surgvuCases';
import {
  SURGVU_TASK_INTERVALS,
  SURGVU_TOOL_INTERVALS,
  SurgvuTaskInterval,
  SurgvuToolInterval,
} from '../data/surgvuLabels';
import { armColor, isTrainingClass, toolDisplay } from '../data/surgvuVocab';

export type Provenance = 'recorded' | 'predicted' | 'none';

/** A tool the dataset records as installed at the current timestamp. */
export interface RecordedTool {
  label: string;
  display: string;
  commercial: string;
  arm: string;
  start: number;
  end: number;
  /** False for labels outside the twelve trainable classes. */
  trainable: boolean;
}

export interface FrameTruth {
  caseId: string;
  part: number;
  timestamp: number;
  tools: RecordedTool[];
  task: SurgvuTaskInterval | null;
  provenance: Provenance;
}

export interface PartKey {
  caseId: string;
  part: number;
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

const partKey = (caseId: string, part: number) => `${caseId}/${part}`;

function groupByPart<T extends { case: string; part: number; start: number }>(
  rows: T[]
): Map<string, T[]> {
  const grouped = new Map<string, T[]>();
  for (const row of rows) {
    const key = partKey(row.case, row.part);
    const bucket = grouped.get(key);
    if (bucket) bucket.push(row);
    else grouped.set(key, [row]);
  }
  for (const bucket of grouped.values()) bucket.sort((a, b) => a.start - b.start);
  return grouped;
}

const TOOLS_BY_PART = groupByPart(SURGVU_TOOL_INTERVALS);
const TASKS_BY_PART = groupByPart(SURGVU_TASK_INTERVALS);

/**
 * Start times per part, so a timestamp lookup is a binary search rather than a
 * scan. Cases run for hours and the loop queries this on every rendered frame;
 * a linear scan over 250 intervals at 60 Hz is wasteful for no reason.
 */
const TOOL_STARTS = new Map<string, number[]>();
for (const [key, rows] of TOOLS_BY_PART) {
  TOOL_STARTS.set(key, rows.map((r) => r.start));
}

/** Index of the last interval whose start is <= timestamp. */
function upperBound(starts: number[], timestamp: number): number {
  let low = 0;
  let high = starts.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (starts[mid] <= timestamp) low = mid + 1;
    else high = mid;
  }
  return low - 1;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/**
 * Every tool the release records as installed at `timestamp`.
 *
 * Intervals overlap freely — four arms may each hold an instrument — so this
 * walks back from the binary-search position while intervals can still be
 * open. The walk is bounded by the longest interval in the part, which in
 * practice is a few dozen rows, not the whole list.
 */
export function toolsAt(caseId: string, part: number, timestamp: number): RecordedTool[] {
  const key = partKey(caseId, part);
  const rows = TOOLS_BY_PART.get(key);
  const starts = TOOL_STARTS.get(key);
  if (!rows || !starts) return [];

  const found: RecordedTool[] = [];
  for (let i = upperBound(starts, timestamp); i >= 0; i--) {
    const row = rows[i];
    if (row.start <= timestamp && timestamp < row.end) {
      found.push({
        label: row.label,
        display: row.display || toolDisplay(row.label),
        commercial: row.commercial,
        arm: row.arm,
        start: row.start,
        end: row.end,
        trainable: isTrainingClass(row.label),
      });
    }
  }

  // Arm order, then name, so the panel does not reshuffle between frames.
  found.sort((a, b) => a.arm.localeCompare(b.arm) || a.display.localeCompare(b.display));
  return found;
}

/** The task interval covering `timestamp`, if the release labelled one. */
export function taskAt(
  caseId: string,
  part: number,
  timestamp: number
): SurgvuTaskInterval | null {
  const rows = TASKS_BY_PART.get(partKey(caseId, part));
  if (!rows) return null;
  return rows.find((r) => r.start <= timestamp && timestamp < r.end) || null;
}

/** Everything known for certain about one moment of one video part. */
export function truthAt(caseId: string, part: number, timestamp: number): FrameTruth {
  const tools = toolsAt(caseId, part, timestamp);
  const task = taskAt(caseId, part, timestamp);
  const key = partKey(caseId, part);
  const hasLabels = TOOLS_BY_PART.has(key) || TASKS_BY_PART.has(key);

  return {
    caseId,
    part,
    timestamp,
    tools,
    task,
    // A part with labels but no tool installed right now is still *recorded*
    // knowledge: "nothing was installed" is a fact, not an absence of one.
    provenance: hasLabels ? 'recorded' : 'none',
  };
}

/** All tool intervals for a part, for the timeline ribbon. */
export function toolIntervalsFor(caseId: string, part: number): SurgvuToolInterval[] {
  return TOOLS_BY_PART.get(partKey(caseId, part)) || [];
}

/** All task intervals for a part, for the phase band. */
export function taskIntervalsFor(caseId: string, part: number): SurgvuTaskInterval[] {
  return TASKS_BY_PART.get(partKey(caseId, part)) || [];
}

/** Tool intervals for a part grouped into the arm tracks that carried them. */
export function armTracks(
  caseId: string,
  part: number
): { arm: string; color: ReturnType<typeof armColor>; intervals: SurgvuToolInterval[] }[] {
  const byArm = new Map<string, SurgvuToolInterval[]>();
  for (const interval of toolIntervalsFor(caseId, part)) {
    const arm = interval.arm || 'unknown';
    const bucket = byArm.get(arm);
    if (bucket) bucket.push(interval);
    else byArm.set(arm, [interval]);
  }
  return [...byArm.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([arm, intervals]) => ({ arm, color: armColor(arm), intervals }));
}

// ---------------------------------------------------------------------------
// Case / part helpers
// ---------------------------------------------------------------------------

export function findCase(caseId: string): SurgvuCase | undefined {
  return SURGVU_CASES.find((c) => c.caseId === caseId);
}

export function findPart(caseId: string, part: number): SurgvuPart | undefined {
  return findCase(caseId)?.parts.find((p) => p.part === part);
}

/**
 * Match a file the user picked to a known part, by filename.
 *
 * The release names files `case_002_video_part_001.mp4`, which encodes both
 * ids. Matching on the name rather than asking the user to say which part they
 * chose is what keeps a video from being played against another case's labels.
 */
export function matchFileToPart(filename: string): (PartKey & { part: number }) | null {
  const name = filename.trim().toLowerCase();
  for (const surgCase of SURGVU_CASES) {
    for (const part of surgCase.parts) {
      if (part.filename.toLowerCase() === name) {
        return { caseId: surgCase.caseId, part: part.part };
      }
    }
  }
  return null;
}

/** `1:23:45` for hours, `4:05` below an hour. */
export function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  if (hrs > 0) {
    return `${hrs}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
  return `${mins}:${String(secs).padStart(2, '0')}`;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.round(bytes / 1e3)} KB`;
}

/**
 * A one-line description of what the release records for a whole part —
 * used as the chat assistant's grounding when a library video is loaded.
 */
export function summariseFor(caseId: string, part: number): string {
  const tools = toolIntervalsFor(caseId, part);
  const tasks = taskIntervalsFor(caseId, part);
  if (tools.length === 0 && tasks.length === 0) {
    return `No SurgVU annotations are bundled for case ${caseId} part ${part}.`;
  }

  const toolCounts = new Map<string, number>();
  for (const t of tools) toolCounts.set(t.display, (toolCounts.get(t.display) || 0) + 1);
  const topTools = [...toolCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([name, n]) => `${name} (${n}×)`)
    .join(', ');

  const taskNames = [...new Set(tasks.map((t) => t.display))].join(', ');

  return [
    `Case ${caseId}, part ${part} — recorded in the SurgVU 2024 release.`,
    `Instrument installations: ${topTools || 'none'}.`,
    `Labelled tasks: ${taskNames || 'none'}.`,
  ].join(' ');
}
