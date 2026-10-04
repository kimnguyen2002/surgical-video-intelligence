/**
 * Deterministic answers from what the app actually knows.
 *
 * Browser port of `ai/llm/grounded.py`. This is the assistant when no language
 * model is connected — and it still runs *first* when one is, because for
 * factual questions about the video ("what instrument is in use?", "what is
 * happening?") the dataset's own annotations and the detector's output are
 * better evidence than anything a language model will produce from a prompt.
 *
 * Why this module exists
 * ----------------------
 * The original fallback ranked sentences from the whole system prompt against
 * the question. The system prompt contains the safety constraints, so asking
 * "what instrument is in use right now?" returned:
 *
 *     - Never use profanity, slurs, demeaning language, or crude humour…
 *     - Focus on operative workflow, the decision points within each phase…
 *
 * It was quoting its own instructions back as if they were findings. The fix
 * is not a better ranking function. It is to answer from **structured facts**
 * — recorded annotations, detector output, the procedure timeline — and to
 * fall back to text extraction only over *retrieved evidence*, never over the
 * prompt's own instruction layers.
 *
 * Provenance is carried into the wording. "The dataset records a needle
 * driver" and "the detector reports a needle driver at 71%" are different
 * claims, and the sentence the reader sees makes that obvious without them
 * having to check a badge.
 *
 * Every method returns `null` rather than guessing when it has nothing. A
 * `null` travels up to the caller, which says so plainly. Nothing here
 * fabricates.
 */

import { classifyIntent, type Intent } from './intents';
import { hasAnyFacts, hms, join, type TimelineEvent, type VideoFacts } from './facts';
import { toolDisplay } from '../data/surgvuVocab';

export interface GroundedAnswer {
  text: string;
  /** Which handler produced it, for the provenance badge in the UI. */
  intent: Intent;
  /** False throughout: nothing here is generated, it is assembled. */
  generative: false;
}

/**
 * Instruments the release records as installed at the current timestamp.
 *
 * Duplicates are collapsed with a count rather than repeated. The log is per
 * arm, so a case with a needle driver on USM1 and another on USM3 produces two
 * identical rows — and listing them verbatim gave "Monopolar curved scissors,
 * Needle driver, Cadiere forceps, Needle driver, Prograsp forceps and Tip-up
 * fenestrated grasper", which reads as a bug in the sentence rather than as
 * two arms genuinely holding the same instrument.
 *
 * The count is kept instead of silently de-duplicating, because "two needle
 * drivers are mounted" is a different fact from "a needle driver is mounted"
 * and the arm panel next to this will show both.
 */
function recordedTools(facts: VideoFacts): string[] {
  const gt = facts.groundTruth;
  if (!gt || gt.provenance !== 'recorded') return [];

  const counts = new Map<string, number>();
  for (const tool of gt.tools) {
    const name = tool.display || toolDisplay(tool.label);
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  return [...counts].map(([name, n]) => (n > 1 ? `${name} (\u00d7${n})` : name));
}

/** The number of installed instruments, counting each arm separately. */
function recordedToolCount(facts: VideoFacts): number {
  const gt = facts.groundTruth;
  return gt && gt.provenance === 'recorded' ? gt.tools.length : 0;
}

/**
 * "At 1:23:45" — or "Right now" when there is no clock to quote.
 *
 * The timestamp is always the *source* timestamp, so a bundled excerpt quotes
 * the time the moment carries in the full recording rather than the offset
 * into the excerpt. Quoting the excerpt's own clock would make every answer
 * disagree with the dataset it is citing.
 */
function at(facts: VideoFacts): string {
  const ts = facts.timestamp ?? facts.groundTruth?.timestamp ?? null;
  return ts !== null ? `At **${hms(ts)}**` : 'Right now';
}

/** Annotated spans starting within `window` seconds of now, nearest first. */
function nearbyEvents(facts: VideoFacts, window = 120): TimelineEvent[] {
  if (facts.timestamp === null || facts.timeline.length === 0) return [];
  const now = facts.timestamp;
  return facts.timeline
    .filter((e) => Math.abs(e.start - now) <= window || (e.start <= now && now <= e.end))
    .sort((a, b) => Math.abs(a.start - now) - Math.abs(b.start - now));
}

/**
 * How the predicted side of an answer is phrased.
 *
 * Confidence is never omitted. A prediction without a percentage is
 * indistinguishable in prose from a recorded fact, which is the one confusion
 * this whole interface exists to prevent.
 */
function describeDetections(facts: VideoFacts): string | null {
  if (facts.outOfDomain) return null;
  const parts: string[] = [];

  if (facts.detections.length > 0) {
    const listed = facts.detections
      .slice(0, 5)
      .map((d) => `${toolDisplay(d.label)} (${d.confidence}%)`)
      .join(', ');
    parts.push(`The on-device detector locates: ${listed}.`);
  }

  if (facts.predictedTools.length > 0) {
    const listed = facts.predictedTools
      .slice(0, 5)
      .map((t) => `${toolDisplay(t.label)} (${t.confidence}%)`)
      .join(', ');
    parts.push(`The presence classifier reports: ${listed}.`);
  }

  return parts.length > 0 ? parts.join(' ') : null;
}

/**
 * The caveat appended whenever predictions sit next to recorded annotations.
 *
 * The presence and detection checkpoints were fitted to five clips of the
 * public cat1 subset. Six of their fourteen classes never occur in that data
 * and cannot be predicted at all. Saying so next to the number is the
 * difference between a demonstration and a claim.
 */
const DETECTOR_CAVEAT =
  '*The detector and presence classifier were trained on five clips; where they ' +
  'disagree with the annotations above, trust the annotations.*';

// ---------------------------------------------------------------------------
// Intent handlers
// ---------------------------------------------------------------------------

function answerInstruments(facts: VideoFacts): string | null {
  const lines: string[] = [];
  const recorded = recordedTools(facts);

  if (recorded.length > 0) {
    lines.push(
      `${at(facts)}, the dataset records **${join(recorded)}** mounted on the robot arms.`
    );
    lines.push(
      'That comes from the installation log rather than from vision, so it means ' +
        '*installed* — an instrument stays on the list while it is off-screen or occluded.'
    );
  } else if (facts.groundTruth?.provenance === 'recorded') {
    lines.push(`${at(facts)}, the dataset records no instrument installed on any arm.`);
  }

  const predicted = describeDetections(facts);
  if (predicted) {
    lines.push(predicted);
    if (recorded.length > 0) lines.push(DETECTOR_CAVEAT);
  } else if (facts.outOfDomain) {
    lines.push(
      '*No prediction is shown: the out-of-domain guard rejected this frame, so the ' +
        'models are not being asked to judge it.*'
    );
  }

  return lines.length > 0 ? lines.join('\n\n') : null;
}

function answerTask(facts: VideoFacts): string | null {
  const gt = facts.groundTruth;
  const lines: string[] = [];
  const task = gt?.task?.display ?? null;

  if (task) {
    lines.push(`${at(facts)}, the annotated surgical task is **${task}**.`);
  } else if (gt?.provenance === 'recorded') {
    lines.push(
      `${at(facts)}, no surgical task is annotated — this falls between labelled segments.`
    );
  }

  if (!facts.outOfDomain && facts.predictedTask) {
    lines.push(
      `The task classifier predicts **${toolDisplay(facts.predictedTask.label)}** ` +
        `at ${facts.predictedTask.confidence}% confidence.`
    );
  }

  const tools = recordedTools(facts);
  if (tools.length > 0) lines.push(`Instruments in play: ${join(tools)}.`);

  const nearby = nearbyEvents(facts, 180);
  if (nearby.length > 0) {
    lines.push('Around this point:');
    lines.push(
      nearby
        .slice(0, 5)
        .map((e) => `- **${hms(e.start)}** — ${e.display}`)
        .join('\n')
    );
  }

  return lines.length > 0 ? lines.join('\n\n') : null;
}

function answerWhen(question: string, facts: VideoFacts): string | null {
  if (facts.timeline.length === 0) return null;

  const words: string[] = question.toLowerCase().match(/[a-z]+/g) || [];
  const terms = new Set(words.filter((w) => w.length > 3));
  const matches = facts.timeline.filter((e) => {
    const haystack = `${e.display} ${e.label}`.toLowerCase();
    return [...terms].some((t) => haystack.includes(t));
  });

  if (matches.length === 0) return null;
  matches.sort((a, b) => a.start - b.start);

  const head = matches[0];
  const lines = [
    `**${head.display}** first appears at **${hms(head.start)}** and runs to **${hms(head.end)}**.`,
  ];

  if (matches.length > 1) {
    lines.push(`It occurs ${matches.length} times in this recording:`);
    lines.push(
      matches
        .slice(0, 8)
        .map(
          (e) =>
            `- **${hms(e.start)} – ${hms(e.end)}** (${e.duration ? `${hms(e.duration)} long` : 'instant'})`
        )
        .join('\n')
    );
  }

  lines.push("*Times are from the dataset's own annotations.*");
  return lines.join('\n\n');
}

function answerCount(facts: VideoFacts): string | null {
  const tools = recordedTools(facts);
  // The count is of *arms carrying an instrument*, not of distinct instrument
  // names: two needle drivers on two arms is two instruments.
  const total = recordedToolCount(facts);
  if (total > 0) {
    return (
      `${at(facts)} the dataset records **${total}** ` +
      `instrument${total === 1 ? '' : 's'} installed: ${join(tools)}.`
    );
  }

  if (!facts.outOfDomain && facts.detections.length > 0) {
    const perClass = new Map<string, number>();
    for (const d of facts.detections) {
      const name = toolDisplay(d.label);
      perClass.set(name, (perClass.get(name) || 0) + 1);
    }
    const listed = [...perClass].map(([name, n]) => `${n}× ${name}`).join(', ');
    return (
      `${at(facts)} the detector locates **${facts.detections.length}** ` +
      `instrument${facts.detections.length === 1 ? '' : 's'} in frame: ${listed}. ` +
      '*These are predictions, not recorded annotations.*'
    );
  }

  return null;
}

function answerSummary(facts: VideoFacts): string | null {
  if (facts.timeline.length === 0) return answerTask(facts);

  const toolSeconds = new Map<string, number>();
  const taskSeconds = new Map<string, number>();
  for (const e of facts.timeline) {
    const bucket = e.kind === 'tool' ? toolSeconds : taskSeconds;
    bucket.set(e.display, (bucket.get(e.display) || 0) + e.duration);
  }

  const lines: string[] = [];
  if (facts.title) lines.push(`**${facts.title}**`);
  lines.push(`The recording carries **${facts.timeline.length}** annotated events.`);

  if (taskSeconds.size > 0) {
    lines.push('Surgical tasks by time:');
    lines.push(
      [...taskSeconds]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([name, secs]) => `- ${name} — ${hms(secs)}`)
        .join('\n')
    );
  }

  if (toolSeconds.size > 0) {
    lines.push('Instruments by time installed:');
    lines.push(
      [...toolSeconds]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
        .map(([name, secs]) => `- ${name} — ${hms(secs)}`)
        .join('\n')
    );
  }

  lines.push("*All figures read from the dataset's annotations.*");
  return lines.join('\n\n');
}

function answerRisk(facts: VideoFacts): string | null {
  if (facts.risks.length === 0) return null;
  const task = facts.groundTruth?.task?.display || 'this phase';

  // Name the specialty in the answer itself, not just in a badge.
  //
  // These entries come from the knowledge graph for whichever specialty is
  // selected, which defaults to General Surgery. That produced "common bile
  // duct" and "cystic artery" — cholecystectomy structures — for a SurgVU case
  // whose annotated tasks are uterine horn and rectal artery manipulation.
  // Anatomy from the wrong operation presented as "at risk here" is the most
  // dangerous thing this assistant could say, so the operation it belongs to
  // is stated up front where a reader cannot miss it.
  const lines = [
    `Structures conventionally at risk during **${task}**, for **${facts.specialtyName}** — ` +
      'the specialty currently selected:',
  ];

  lines.push(
    facts.risks
      .slice(0, 6)
      .map((r) => `- **${r.structure}** — ${r.mechanism}`)
      .join('\n')
  );

  // Emphasis is not nested here on purpose: the transcript renderer matches
  // `*italic*` with a character class that stops at the first `*`, so a
  // `**bold**` span inside an italic one leaves both sets of asterisks on
  // screen as literal punctuation.
  lines.push(
    '**Not detections.** Reference anatomy for this phase from the specialty knowledge ' +
      'graph, not derived from this video. Nothing here was localised in the current ' +
      'frame. If the list does not match what you are watching, change the specialty ' +
      'selector.'
  );
  return lines.join('\n\n');
}

// ---------------------------------------------------------------------------

/**
 * Compose a factual answer, or return `null` if there is not one to give.
 *
 * `null` is a real outcome and the caller must handle it, not paper over it.
 * It routes to the extractive ranker over retrieved passages, and failing
 * that to a plain statement that the app does not know — which is a better
 * answer than a fluent one with nothing behind it.
 */
export function answerFromFacts(question: string, facts: VideoFacts): GroundedAnswer | null {
  if (!hasAnyFacts(facts)) return null;

  const intent = classifyIntent(question);

  let text: string | null;
  switch (intent) {
    case 'instrument':
    case 'list':
      text = answerInstruments(facts);
      break;
    case 'task':
      text = answerTask(facts);
      break;
    case 'when':
      text = answerWhen(question, facts);
      break;
    case 'count':
      text = answerCount(facts);
      break;
    case 'summary':
      text = answerSummary(facts);
      break;
    case 'risk':
      text = answerRisk(facts);
      break;
    default:
      // An unclassified question still gets the current state, which is almost
      // always the context the asker means.
      text = answerTask(facts);
      break;
  }

  return text ? { text, intent, generative: false } : null;
}
