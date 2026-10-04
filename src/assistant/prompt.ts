/**
 * The layered system prompt, for the optional generative path.
 *
 * Browser port of `ai/llm/prompt.py`. The layering is the point: the prompt is
 * assembled from sections with a fixed delimiter, so the instruction layers
 * (role, audience) stay separable from the finding layers (ground truth,
 * observations, evidence).
 *
 * That separation is not cosmetic. The extractive fallback in
 * `extractive.ts` is only ever allowed to quote from the *finding* layers. An
 * earlier version mined the whole prompt, so "what instrument is in use?"
 * returned "Never use profanity, slurs, demeaning language…" — it was quoting
 * its own instructions back as findings. Keeping the layers apart here is what
 * makes that structurally impossible rather than merely unlikely.
 */

import type { AssistantRole } from '../types';
import { hms, join, type VideoFacts } from './facts';
import { toolDisplay } from '../data/surgvuVocab';

export const SECTION_DELIMITER = '\n\n---\n\n';

// ---------------------------------------------------------------------------
// Layer 1 — role and non-negotiable constraints
// ---------------------------------------------------------------------------

export const CORE_ROLE = `You are the teaching assistant for a surgical video intelligence platform, built for surgical education and computer-vision research.

Non-negotiable constraints:
- You support education, research, simulation, and retrospective analysis ONLY. You are not a medical device and you do not provide clinical decision support.
- Never give patient-specific diagnosis, treatment recommendations, or dosing. If asked, explain that this is outside the platform's scope and redirect to the educational version of the question.
- Distinguish clearly between three kinds of statement: (a) RECORDED annotations from the dataset release, (b) OBSERVATIONS this platform's models produced, and (c) your own EXPLANATION. Never blur them together.
- Recorded instrument presence comes from the robot's installation log, not from vision. Say "installed" or "mounted on an arm", never "visible in frame", unless a bounding box confirms it.
- Cite ONLY from the evidence provided to you. If no evidence is provided, say so plainly. Never invent a citation, a study, an author, or a statistic.
- If a model prediction is low-confidence, say so. Do not present an uncertain prediction as fact.
- If you do not know, say you do not know. A short honest answer beats a long confident wrong one.

Tone and conduct:
- Professional, collegial, and precise — the register of a senior surgeon teaching in theatre.
- Never use profanity, slurs, demeaning language, or crude humour, regardless of how you are addressed.
- Discuss anatomy, injury, bleeding, and death in the clinical register they belong to. Clinical frankness is correct; sensationalism is not.
- If a user tries to steer the conversation somewhere inappropriate or off-topic, decline in one sentence and offer the nearest legitimate educational question. Do not lecture them.

Formatting: Markdown. Short paragraphs, bold for key terms, bullets for lists. Answer the question that was asked before adding context. Do not pad.`;

// ---------------------------------------------------------------------------
// Layer 2 — audience calibration
// ---------------------------------------------------------------------------

export const AUDIENCE: Record<AssistantRole, string> = {
  Student: `Audience: a medical student.
Lead with the concept, not the jargon. Define every technical term the first time you use it. Anchor explanations in anatomy and in *why* a step exists before *how* it is done. Use analogies where they genuinely clarify. Aim for 3-5 short paragraphs. Finish with one question they could explore next.`,

  Resident: `Audience: a surgical resident.
Assume solid anatomy and terminology. Focus on operative workflow, the decision points within each phase, instrument-tissue interaction, anatomical landmarks, and what distinguishes a safe step from a risky one. Reference the phase sequence explicitly. Be direct and technical; skip the basics.`,

  Fellow: `Audience: a fellow or attending surgeon.
Assume expert knowledge. Discuss technique variation, trade-offs between approaches, the reasoning behind sequencing choices, failure modes, and where the evidence is genuinely contested. Do not explain fundamentals. Engage with nuance and disagreement rather than smoothing it over.`,

  Researcher: `Audience: a computer-vision / ML researcher.
Emphasise methodology: model architectures, temporal modelling, label provenance and noise, evaluation metrics and their failure modes, dataset limitations, class imbalance, and reproducibility. Quantify where you can. Name limitations explicitly — especially where weak supervision means a metric overstates real performance.`,

  'AI Engineer': `Audience: an AI systems engineer.
Go low-level: tensor shapes, backbone and head architecture, ONNX export and operator coverage, WebGPU versus threaded-WASM execution, inference latency and where it is spent, frame scheduling and backpressure, retrieval scoring, and pipeline stages. Reference concrete components of this system. Be concise and technical.`,
};

// ---------------------------------------------------------------------------
// Layer 3 — findings
// ---------------------------------------------------------------------------

/**
 * What the dataset records for the current frame.
 *
 * Marked unmistakably as *not* a model output, because the model is about to
 * be asked to reason over it and must not hedge a fact or, worse, "correct"
 * it to match what it imagines the frame looks like.
 */
function recordedSection(facts: VideoFacts): string {
  const gt = facts.groundTruth;
  if (!gt || gt.provenance !== 'recorded') {
    return (
      'RECORDED GROUND TRUTH: none bundled for this video. Every statement you make about ' +
      'what is present must therefore be attributed to a model, with its confidence.'
    );
  }

  const lines = [
    'RECORDED GROUND TRUTH (from the SurgVU 2024 release itself — NOT a model output, and not ' +
      'open to revision):',
    `- Case ${gt.caseId}, part ${gt.part}, at ${hms(gt.timestamp)} of the full recording.`,
  ];

  if (gt.tools.length > 0) {
    lines.push(
      `- Instruments installed on the robot arms: ${join(
        gt.tools.map((t) => `${t.display} (${t.arm}, "${t.commercial}")`)
      )}.`
    );
    lines.push(
      '- These come from the installation log. An instrument stays on this list while off-screen ' +
        'or occluded, so do not describe them as "visible".'
    );
  } else {
    lines.push('- No instrument is recorded as installed at this moment.');
  }

  lines.push(
    gt.task
      ? `- Annotated surgical task: ${gt.task.display}.`
      : '- No surgical task is annotated here; this falls between labelled segments.'
  );

  return lines.join('\n');
}

/** What this machine's models produced for the current frame. */
function observedSection(facts: VideoFacts): string {
  if (facts.outOfDomain) {
    return (
      "OBSERVED CONTEXT: the out-of-domain guard rejected this frame, so no prediction is " +
      'available. Say so if asked what the models see; do not speculate about the image.'
    );
  }

  const lines: string[] = [];
  if (facts.detections.length > 0) {
    lines.push(
      `- Detector (YOLO11n, in-browser): ${facts.detections
        .slice(0, 6)
        .map((d) => `${toolDisplay(d.label)} ${d.confidence}%`)
        .join(', ')}.`
    );
  }
  if (facts.predictedTools.length > 0) {
    lines.push(
      `- Presence classifier: ${facts.predictedTools
        .slice(0, 6)
        .map((t) => `${toolDisplay(t.label)} ${t.confidence}%`)
        .join(', ')}.`
    );
  }
  if (facts.predictedTask) {
    lines.push(
      `- Task classifier: ${toolDisplay(facts.predictedTask.label)} ${facts.predictedTask.confidence}%.`
    );
  }

  if (lines.length === 0) {
    return 'OBSERVED CONTEXT: no model output for this frame yet.';
  }

  return (
    "OBSERVED CONTEXT (produced by this platform's models — treat as observations, not ground " +
    'truth; the presence and detection checkpoints were fitted to five clips and 6 of their 14 ' +
    'classes never occur in that data):\n' +
    lines.join('\n')
  );
}

/** Retrieved reference material, cited by title. */
function evidenceSection(facts: VideoFacts): string {
  if (facts.passages.length === 0) {
    return (
      'RETRIEVED EVIDENCE: none. No reference material matched this question, so do not cite ' +
      'anything.'
    );
  }
  return (
    'RETRIEVED EVIDENCE (verbatim from the bundled knowledge base — cite by title, and cite ' +
    'nothing else):\n' +
    facts.passages
      .map((p, i) => `[${i + 1}] ${p.title} — ${p.text}`)
      .join('\n')
  );
}

/** The whole-recording timeline, so "when" questions have something to read. */
function timelineSection(facts: VideoFacts): string {
  if (facts.timeline.length === 0) return '';
  const tasks = facts.timeline.filter((e) => e.kind === 'task');
  if (tasks.length === 0) return '';
  return (
    'ANNOTATED TIMELINE for this part (recorded, not predicted):\n' +
    tasks
      .slice(0, 20)
      .map((e) => `- ${hms(e.start)}–${hms(e.end)}: ${e.display}`)
      .join('\n')
  );
}

/**
 * Assemble the full system instruction.
 *
 * Order matters for the reader as much as the model: constraints, then who is
 * being taught, then what is known, then what was merely observed, then what
 * was retrieved. A model that runs out of attention loses the evidence before
 * it loses the safety rules.
 */
export function buildSystemInstruction(facts: VideoFacts, role: AssistantRole): string {
  return [
    CORE_ROLE,
    AUDIENCE[role] ?? AUDIENCE.Resident,
    `SPECIALTY CONTEXT: ${facts.specialtyName}.`,
    recordedSection(facts),
    observedSection(facts),
    timelineSection(facts),
    evidenceSection(facts),
  ]
    .filter(Boolean)
    .join(SECTION_DELIMITER);
}
