/**
 * What a question is asking for.
 *
 * Ported from `ml/llm/grounded.py` in the original Python backend. The
 * patterns and — crucially — their **order** are carried over unchanged,
 * because the order is load-bearing and both of the orderings below were wrong
 * on the first pass:
 *
 *   * `summary` must precede `task`, because "summarise this procedure"
 *     contains "procedure", which the task pattern matches.
 *   * `count` must precede `instrument`, because "how many instruments" names
 *     an instrument.
 *
 * Stems are matched with `\w*` rather than a closing `\b`: `\b(summar)\b` does
 * not match "summarise", since the boundary assertion fails mid-word.
 */

export type Intent =
  | 'when'
  | 'summary'
  | 'count'
  | 'risk'
  | 'instrument'
  | 'task'
  | 'list'
  | 'general';

/** First match wins. Do not reorder without reading the note above. */
const INTENT_PATTERNS: [Intent, RegExp][] = [
  ['when', /\b(when|at what (?:time|point)|how long|timestamps?|what time)\b/i],
  [
    'summary',
    /\b(summar\w*|overview|recap|walk me through|describe (?:the|this)\s+(?:whole|entire|full|video|procedure|operation|case))\b/i,
  ],
  ['count', /\b(how many|how much|count|number of)\b/i],
  ['risk', /\b(risks?|danger\w*|at risk|avoid|careful|complications?|injur\w*)\b/i],
  // "What is on screen?" names no instrument, but it is one of the commonest
  // ways the question is asked — and when boxes exist they are the best
  // possible answer to it.
  [
    'instrument',
    /\b(instruments?|tools?|devices?|forceps|scissors|drivers?|staplers?|graspers?|sealers?|cautery|clip applier|retractors?|on screen|on-screen|visible|can you see|in (?:the )?frame)\b/i,
  ],
  [
    'task',
    /\b(tasks?|steps?|phase|stage|procedure|doing|happening|going on|what.*(?:now|currently))\b/i,
  ],
  ['list', /\b(list|which|what)\b.*\b(used|present|appear\w*|seen)\b/i],
];

export function classifyIntent(question: string): Intent {
  for (const [name, pattern] of INTENT_PATTERNS) {
    if (pattern.test(question || '')) return name;
  }
  return 'general';
}
