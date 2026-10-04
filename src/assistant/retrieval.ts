/**
 * Retrieval over the bundled surgical knowledge base.
 *
 * The original backend indexed uploaded PDFs and slides into a vector store
 * and queried it with sentence-transformers embeddings. Neither the store nor
 * a 90 MB embedding model belongs in a static page, so this does the same job
 * the way a browser can afford: the knowledge base is flattened once into
 * passages, and a question is matched against them lexically.
 *
 * The honest trade is that this finds passages sharing *words* with the
 * question, not passages sharing *meaning*. For a corpus this size — a few
 * hundred passages of controlled surgical vocabulary, where "cystic duct" is
 * written the same way everywhere — lexical matching is close to as good, and
 * it costs nothing to ship. The limitation is stated in the About page rather
 * than hidden.
 */

import { SURGICAL_KNOWLEDGE_BASE } from '../data/knowledge';
import type { SpecialtyId } from '../types';
import type { Passage } from './facts';

const WORD = /[a-z0-9]+/g;

const STOP = new Set([
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'is', 'are', 'was', 'for',
  'on', 'with', 'as', 'by', 'at', 'from', 'that', 'this', 'it', 'be', 'what',
  'how', 'why', 'when', 'where', 'which', 'who', 'do', 'does', 'did', 'can',
  'you', 'i', 'me', 'my', 'we', 'please', 'tell', 'show', 'explain',
]);

function tokens(text: string): string[] {
  const found: string[] = text.toLowerCase().match(WORD) || [];
  return found.filter((w) => w.length > 2 && !STOP.has(w));
}

/**
 * Flatten the knowledge base into retrievable passages, once at module load.
 *
 * Each passage is prose rather than a field dump, because the extractive
 * ranker downstream selects *sentences*: a passage built as "structure:
 * mechanism: protect:" yields fragments that read as broken when quoted back.
 */
function buildPassages(): Passage[] {
  const passages: Passage[] = [];

  for (const specialty of Object.values(SURGICAL_KNOWLEDGE_BASE)) {
    passages.push({
      id: `${specialty.id}:overview`,
      title: `${specialty.name} — overview`,
      text:
        `${specialty.name} covers ${specialty.category.toLowerCase()}. ${specialty.description} ` +
        `The operative workflow is conventionally divided into these phases: ` +
        `${specialty.phases.join('; ')}. ` +
        `Instruments in routine use include ${specialty.instruments.join(', ')}.`,
      citation: `${specialty.name} knowledge graph`,
    });

    for (const risk of specialty.anatomicalRisks) {
      passages.push({
        id: `${specialty.id}:risk:${risk.id}`,
        title: `${risk.structure} (${specialty.name})`,
        text:
          `In ${specialty.name.toLowerCase()}, the ${risk.structure} is a ${risk.level.toLowerCase()} ` +
          `structure. ${risk.mechanism} ${risk.protect}`,
        citation: `${specialty.name} knowledge graph — ${risk.structure}`,
      });
    }

    for (const sample of specialty.sampleCases) {
      passages.push({
        id: `${specialty.id}:case:${sample.id}`,
        title: sample.title,
        text: `${sample.title}. ${sample.procedure}, running ${sample.durationFormatted}. ${sample.description}`,
        citation: `${specialty.name} reference case — ${sample.procedure}`,
      });
    }
  }

  return passages;
}

const PASSAGES = buildPassages();

/**
 * Inverse document frequency over the passage corpus.
 *
 * Without it, "surgical" and "instrument" — which appear in nearly every
 * passage — dominate every query, and a question about the cystic duct
 * retrieves the same generic overviews as a question about trocar placement.
 */
const DOCUMENT_FREQUENCY = (() => {
  const df = new Map<string, number>();
  for (const passage of PASSAGES) {
    for (const term of new Set(tokens(`${passage.title} ${passage.text}`))) {
      df.set(term, (df.get(term) || 0) + 1);
    }
  }
  return df;
})();

function idf(term: string): number {
  const df = DOCUMENT_FREQUENCY.get(term) || 0;
  return Math.log((PASSAGES.length + 1) / (df + 1)) + 1;
}

const PASSAGE_TERMS = new Map<string, Set<string>>(
  PASSAGES.map((p) => [p.id, new Set(tokens(`${p.title} ${p.text}`))])
);

/**
 * The passages most relevant to `question`.
 *
 * When a specialty is selected its passages are boosted rather than filtered
 * to. Filtering would be wrong: a question about the cystic duct asked while
 * a colorectal case is loaded still has a correct answer, and refusing it
 * because the selector says "colorectal" would be unhelpful and confusing.
 * Boosting keeps the selected specialty first without hiding the rest.
 */
export function retrieve(
  question: string,
  specialty: SpecialtyId | null,
  limit = 6
): Passage[] {
  const queryTerms = new Set(tokens(question));
  if (queryTerms.size === 0) return [];

  const scored: { passage: Passage; score: number }[] = [];
  for (const passage of PASSAGES) {
    const passageTerms = PASSAGE_TERMS.get(passage.id)!;

    let score = 0;
    for (const term of queryTerms) {
      if (passageTerms.has(term)) score += idf(term);
    }
    if (score === 0) continue;

    // Normalise by passage length so a long overview does not outrank a short,
    // precisely-matching risk entry purely by having more words in it.
    score /= Math.sqrt(passageTerms.size);
    if (specialty && passage.id.startsWith(`${specialty}:`)) score *= 1.35;

    scored.push({ passage, score });
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit).map((s) => ({ ...s.passage, score: s.score }));
}

/** Every passage, for the knowledge panel's browse view. */
export function allPassages(): Passage[] {
  return PASSAGES;
}
