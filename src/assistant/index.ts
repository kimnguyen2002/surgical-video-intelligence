/**
 * The assistant: which answer path runs, and in what order.
 *
 * Three paths, tried in this order, and the order is the design:
 *
 * 1. **Grounded** (`grounded.ts`) — structured facts. Free, instant, offline,
 *    and for factual questions about the video it is simply *better*: the
 *    dataset's installation log knows which instrument is mounted, and no
 *    language model reasoning over a prompt will beat a lookup.
 * 2. **Extractive** (`extractive.ts`) — sentences ranked out of the retrieved
 *    knowledge base. Free, and traceable to indexed material.
 * 3. **Generative** (`../services/byokGemini.ts`) — only when the visitor has
 *    supplied their own key, and only ever *on top of* the grounding from the
 *    first two, never instead of it.
 *
 * Why generative is last rather than first
 * ----------------------------------------
 * A demo that calls a model for "what instrument is in use?" spends a request
 * to paraphrase a value it already had, and risks the paraphrase being wrong.
 * Running the lookup first means the expensive path is reserved for questions
 * that actually need prose — *why* a step is sequenced this way, what the
 * trade-off is — which is also what keeps a visitor's quota spend proportional
 * to curiosity rather than to playback time.
 */

import { answerFromFacts, type GroundedAnswer } from './grounded';
import { cannotAnswer, composeFromPassages } from './extractive';
import { classifyIntent } from './intents';
import { hasAnyFacts, type VideoFacts } from './facts';
import { retrieve } from './retrieval';
import { buildSystemInstruction } from './prompt';
import { apiKey, generate, GeminiError, type ChatTurn } from '../services/byokGemini';
import type { AssistantRole, SpecialtyId } from '../types';

export type AnswerSource = 'grounded' | 'extractive' | 'generative' | 'none';

export interface Answer {
  text: string;
  source: AnswerSource;
  /** True only for text a language model wrote. Drives the UI's badge. */
  generative: boolean;
  /** Present when the generative path was attempted and failed. */
  warning?: string;
  model?: string;
}

export interface AskOptions {
  facts: VideoFacts;
  role: AssistantRole;
  specialty: SpecialtyId | null;
  history: ChatTurn[];
  /** Visitor's explicit choice; generation never happens without it. */
  useGenerative: boolean;
  signal?: AbortSignal;
}

/**
 * Intents whose answer is a lookup, not an explanation.
 *
 * For these the grounded answer is returned directly even when a key is
 * configured. Paraphrasing "the dataset records a needle driver and a cadiere
 * forceps" through a language model cannot make it more correct, can make it
 * less, and costs a request.
 */
const FACTUAL_INTENTS = new Set(['instrument', 'count', 'when', 'list']);

export async function ask(question: string, options: AskOptions): Promise<Answer> {
  const { facts, role, specialty, history, useGenerative, signal } = options;

  // Retrieval feeds both the extractive path and the generative prompt, so it
  // runs once, up front, regardless of which path is taken.
  const passages = retrieve(question, specialty);
  const grounded: GroundedAnswer | null = answerFromFacts(question, facts);
  const enriched: VideoFacts = { ...facts, passages };

  const intent = classifyIntent(question);
  const wantsLookup = FACTUAL_INTENTS.has(intent);

  // --- 1. Grounded, when it is the right kind of answer --------------------
  // Either the question is a lookup, or there is no generative path available
  // to prefer over it.
  const generativeAvailable = useGenerative && apiKey.configured;
  if (grounded && (wantsLookup || !generativeAvailable)) {
    return { text: grounded.text, source: 'grounded', generative: false };
  }

  // --- 3. Generative, grounded in what the app already established ---------
  if (generativeAvailable) {
    try {
      const { text, model } = await generate(
        [...history, { role: 'user', text: question }],
        buildSystemInstruction(enriched, role),
        signal
      );
      return { text, source: 'generative', generative: true, model };
    } catch (err) {
      // A failed generation must never silently become a fabricated answer,
      // and must never lose the free answer that was already available. The
      // grounded text is returned with the failure attached so the reader sees
      // both what went wrong and what is still known.
      const warning =
        err instanceof GeminiError
          ? `${err.failure.status}: ${err.failure.message}${err.failure.hint ? ` — ${err.failure.hint}` : ''}`
          : err instanceof Error
            ? err.message
            : 'The language model call failed.';

      if (grounded) {
        return { text: grounded.text, source: 'grounded', generative: false, warning };
      }
      const extractive = composeFromPassages(question, passages);
      if (extractive) {
        return { text: extractive.text, source: 'extractive', generative: false, warning };
      }
      return {
        text: cannotAnswer(apiKey.configured),
        source: 'none',
        generative: false,
        warning,
      };
    }
  }

  // --- 1b. Grounded, for anything left ------------------------------------
  if (grounded) return { text: grounded.text, source: 'grounded', generative: false };

  // --- 2. Extractive over retrieved evidence -------------------------------
  const extractive = composeFromPassages(question, passages);
  if (extractive) return { text: extractive.text, source: 'extractive', generative: false };

  // --- Nothing. Say so. ----------------------------------------------------
  return { text: cannotAnswer(apiKey.configured), source: 'none', generative: false };
}

export { hasAnyFacts };
export type { VideoFacts };
export * from './facts';
export { retrieve, allPassages } from './retrieval';
export { classifyIntent } from './intents';
