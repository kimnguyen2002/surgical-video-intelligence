/**
 * Extractive composition over retrieved evidence.
 *
 * Browser port of `ai/llm/knowledge.py`. When the structured facts in
 * `grounded.ts` cannot answer a question, this ranks the sentences already
 * present in the retrieved knowledge-base passages against the question and
 * assembles the best-matching ones.
 *
 * It composes; it does not generate. Answers vary with the corpus and the
 * video rather than replaying a fixed script, and every sentence is traceable
 * to indexed material. The UI labels it as retrieved rather than generated,
 * because the one thing worse than a thin answer is a fluent one with nothing
 * behind it.
 *
 * What it is *not* allowed to read
 * --------------------------------
 * Only retrieved passages. Never the system prompt. An earlier version ranked
 * sentences from the whole prompt, which contains the safety constraints, so
 * "what instrument is in use?" came back with "Never use profanity, slurs,
 * demeaning language…". The instruction layers are not findings and are not
 * in scope here — this function only ever sees `Passage[]`.
 */

import type { Passage } from './facts';

const WORD = /[a-z0-9]+/g;

const STOP = new Set([
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'are', 'was', 'for',
  'on', 'with', 'as', 'by', 'at', 'from', 'that', 'this', 'it', 'be', 'what',
  'how', 'why', 'when', 'where', 'which', 'who', 'do', 'does', 'did', 'can',
  'you', 'i', 'me', 'my', 'we', 'please', 'tell', 'show', 'explain',
]);

/**
 * Strip a plural suffix.
 *
 * Deliberately minimal: it exists to stop "instruments" and "instrument" being
 * treated as unrelated terms, which is the commonest reason a lexical match
 * misses a passage that plainly answers the question.
 *
 * It does **not** attempt verb forms. Undoubling a final consonant would be
 * needed for "clipping" → "clip", and that same rule turns "press" into "pres"
 * while "pressing" also becomes "pres" — matching by accident in one direction
 * and failing in the other. Aggressive stemming also collides distinct
 * surgical terms, which is worse than missing a match.
 */
function stem(word: string): string {
  if (word.length < 4 || !word.endsWith('s')) return word;

  // "process", "status", "analysis" — the trailing s is part of the word.
  if (word.endsWith('ss') || word.endsWith('us') || word.endsWith('is')) return word;

  // "arteries" -> "artery"
  if (word.endsWith('ies') && word.length > 4) return `${word.slice(0, -3)}y`;

  // "es" comes off only after a sibilant, where it is a real plural marker:
  // "boxes" -> "box", "arches" -> "arch". Applying it everywhere turns
  // "tissues" into "tissu" while "tissue" stays whole, so the two stop
  // matching — the exact failure this function is meant to prevent.
  if (word.endsWith('es')) {
    const base = word.slice(0, -2);
    if (/(s|x|z|ch|sh)$/.test(base)) return base;
    return word.slice(0, -1);
  }

  return word.slice(0, -1);
}

function terms(text: string): Set<string> {
  const found: string[] = text.toLowerCase().match(WORD) || [];
  return new Set(found.filter((w) => w.length > 2 && !STOP.has(w)).map(stem));
}

/** Split into sentences, discarding fragments too short to carry a claim. */
function sentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 25);
}

export interface ExtractiveAnswer {
  text: string;
  citations: string[];
  generative: false;
}

/**
 * Rank and assemble sentences from `passages` that bear on `question`.
 *
 * Returns `null` when nothing in the corpus overlaps the question, so the
 * caller can say it does not know rather than present an empty list as an
 * answer.
 */
export function composeFromPassages(
  question: string,
  passages: Passage[]
): ExtractiveAnswer | null {
  const queryTerms = terms(question);
  if (queryTerms.size === 0 || passages.length === 0) return null;

  const scored: { score: number; sentence: string; citation: string }[] = [];
  for (const passage of passages) {
    for (const sentence of sentences(passage.text)) {
      const sentenceTerms = terms(sentence);
      if (sentenceTerms.size === 0) continue;

      let overlap = 0;
      for (const t of queryTerms) if (sentenceTerms.has(t)) overlap++;
      if (overlap === 0) continue;

      // Jaccard-ish, mildly favouring information-dense sentences: the union
      // under a square root penalises long rambling sentences less harshly
      // than plain Jaccard, which otherwise buries every detailed answer.
      const union = new Set([...queryTerms, ...sentenceTerms]).size;
      scored.push({
        score: overlap / Math.sqrt(union),
        sentence,
        citation: passage.citation,
      });
    }
  }

  scored.sort((a, b) => b.score - a.score);

  // De-duplicate near-identical sentences from overlapping chunks.
  const selected: { sentence: string; citation: string }[] = [];
  const seen: Set<string>[] = [];
  for (const candidate of scored) {
    const candidateTerms = terms(candidate.sentence);
    const duplicate = seen.some((prior) => {
      let shared = 0;
      for (const t of candidateTerms) if (prior.has(t)) shared++;
      return shared / Math.max(candidateTerms.size, 1) > 0.7;
    });
    if (duplicate) continue;

    selected.push({ sentence: candidate.sentence, citation: candidate.citation });
    seen.push(candidateTerms);
    if (selected.length >= 5) break;
  }

  if (selected.length === 0) return null;

  const body = selected.map((s) => `- ${s.sentence} *(${s.citation})*`).join('\n');
  const citations = [...new Set(selected.map((s) => s.citation))];

  return {
    text:
      'Here is what the reference material says about that:\n\n' +
      `${body}\n\n` +
      '*These sentences are quoted from the bundled surgical knowledge base. No ' +
      'language model produced them — nothing above is a generated explanation.*',
    citations,
    generative: false,
  };
}
