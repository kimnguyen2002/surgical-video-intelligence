/**
 * Tests for the assistant's answer paths.
 *
 * These pin the behaviours that are easy to break and hard to notice: the
 * intent ordering, the recorded/predicted wording, and — most importantly —
 * the refusal to invent an answer when there is nothing to say. A regression
 * in any of them produces output that still reads fluently.
 */

import { describe, expect, it } from 'vitest';

import { classifyIntent } from './intents';
import { answerFromFacts } from './grounded';
import { composeFromPassages } from './extractive';
import { emptyFacts, hasAnyFacts, hms, join, type VideoFacts } from './facts';
import { retrieve } from './retrieval';
import type { FrameTruth } from '../services/groundTruth';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function recordedTruth(overrides: Partial<FrameTruth> = {}): FrameTruth {
  return {
    caseId: '001',
    part: 1,
    timestamp: 4512,
    provenance: 'recorded',
    task: {
      case: '001',
      part: 1,
      start: 4365,
      end: 5407,
      label: 'suturing',
      display: 'Suturing',
    },
    tools: [
      {
        label: 'needle_driver',
        display: 'Needle driver',
        commercial: 'Large SutureCut Needle Driver',
        arm: 'USM1',
        start: 4400,
        end: 4800,
        trainable: true,
      },
      {
        label: 'needle_driver',
        display: 'Needle driver',
        commercial: 'Large SutureCut Needle Driver',
        arm: 'USM3',
        start: 4400,
        end: 4800,
        trainable: true,
      },
      {
        label: 'cadiere_forceps',
        display: 'Cadiere forceps',
        commercial: 'Cadiere Forceps',
        arm: 'USM2',
        start: 4400,
        end: 4800,
        trainable: true,
      },
    ],
    ...overrides,
  };
}

function factsWithTruth(overrides: Partial<VideoFacts> = {}): VideoFacts {
  return {
    ...emptyFacts(),
    groundTruth: recordedTruth(),
    timestamp: 4512,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------

describe('intent classification', () => {
  it('routes the obvious cases', () => {
    expect(classifyIntent('what instruments are on screen?')).toBe('instrument');
    expect(classifyIntent('when does suturing happen?')).toBe('when');
    expect(classifyIntent('what structures are at risk here?')).toBe('risk');
  });

  // Both of these orderings were wrong on the first pass and the failure is
  // silent: the question gets a confident answer to a different question.
  it('prefers summary over task, because "procedure" appears in both', () => {
    expect(classifyIntent('summarise this procedure')).toBe('summary');
  });

  it('prefers count over instrument, because the question names an instrument', () => {
    expect(classifyIntent('how many instruments are installed?')).toBe('count');
  });

  it('matches stems rather than whole words', () => {
    // `\b(summar)\b` fails mid-word; the pattern uses \w* for this reason.
    expect(classifyIntent('summarise the case')).toBe('summary');
    expect(classifyIntent('give me a summary')).toBe('summary');
  });
});

describe('grounded answers', () => {
  it('returns null rather than improvising when nothing is known', () => {
    const facts = emptyFacts();
    expect(hasAnyFacts(facts)).toBe(false);
    expect(answerFromFacts('what is happening?', facts)).toBeNull();
  });

  it('collapses duplicate instruments with a count instead of repeating them', () => {
    const answer = answerFromFacts('what instruments are installed?', factsWithTruth());
    expect(answer).not.toBeNull();
    // Two arms carry a needle driver; it must appear once, with the count.
    expect(answer!.text).toContain('Needle driver (×2)');
    expect(answer!.text.match(/Needle driver/g)).toHaveLength(1);
  });

  it('counts arms, not distinct instrument names', () => {
    const answer = answerFromFacts('how many instruments are installed?', factsWithTruth());
    expect(answer!.text).toContain('**3**');
  });

  it('says "installed", never "visible", for log-derived presence', () => {
    const answer = answerFromFacts('what instruments are installed?', factsWithTruth());
    expect(answer!.text).toMatch(/mounted on the robot arms/);
    expect(answer!.text).toMatch(/installed/);
    expect(answer!.text).not.toMatch(/visible in frame/i);
  });

  it('quotes the source timestamp it was given', () => {
    const answer = answerFromFacts('what is happening?', factsWithTruth());
    expect(answer!.text).toContain(hms(4512)); // 1:15:12
  });

  it('never marks a grounded answer as generative', () => {
    const answer = answerFromFacts('what instruments are installed?', factsWithTruth());
    expect(answer!.generative).toBe(false);
  });

  it('attaches a confidence to every predicted claim', () => {
    const facts = factsWithTruth({
      detections: [
        { label: 'needle_driver', confidence: 71, box: { ymin: 0, xmin: 0, ymax: 10, xmax: 10 } },
      ],
    });
    const answer = answerFromFacts('what instruments are on screen?', facts);
    expect(answer!.text).toContain('71%');
    // A prediction shown next to recorded facts must carry its caveat.
    expect(answer!.text).toMatch(/trained on five clips/);
  });

  it('withholds predicted claims when the frame was refused', () => {
    const facts = factsWithTruth({ outOfDomain: true, detections: [] });
    const answer = answerFromFacts('what instruments are on screen?', facts);
    expect(answer!.text).toMatch(/out-of-domain guard/);
    // Recorded facts survive a refusal: they do not depend on the pixels.
    expect(answer!.text).toContain('Cadiere forceps');
  });

  it('reports a recorded absence as knowledge, not as a gap', () => {
    const facts = factsWithTruth({ groundTruth: recordedTruth({ tools: [] }) });
    const answer = answerFromFacts('what instruments are installed?', facts);
    expect(answer!.text).toMatch(/no instrument installed/i);
  });
});

describe('timeline questions', () => {
  const facts = factsWithTruth({
    timeline: [
      { kind: 'task', label: 'suturing', display: 'Suturing', start: 4365, end: 5407, duration: 1042 },
      { kind: 'task', label: 'suturing', display: 'Suturing', start: 7000, end: 7400, duration: 400 },
    ],
  });

  it('answers "when" from the annotated timeline', () => {
    const answer = answerFromFacts('when does suturing happen?', facts);
    expect(answer!.intent).toBe('when');
    expect(answer!.text).toContain(hms(4365));
    expect(answer!.text).toContain('occurs 2 times');
  });

  it('returns null for a term the timeline does not contain', () => {
    const answer = answerFromFacts('when does cholecystectomy happen?', facts);
    expect(answer).toBeNull();
  });
});

describe('extractive composition', () => {
  const passages = [
    {
      id: 'p1',
      title: 'Cystic duct',
      text:
        'The cystic duct is divided between clips during cholecystectomy. ' +
        'Dissection stays within the hepatocystic triangle to protect the common bile duct.',
      citation: 'General Surgery knowledge graph',
    },
  ];

  it('quotes matching sentences with a citation', () => {
    const answer = composeFromPassages('what about the cystic duct?', passages);
    expect(answer).not.toBeNull();
    expect(answer!.text).toContain('cystic duct');
    expect(answer!.citations).toContain('General Surgery knowledge graph');
    expect(answer!.generative).toBe(false);
  });

  it('returns null when nothing in the corpus overlaps the question', () => {
    expect(composeFromPassages('what is the weather in Oslo?', passages)).toBeNull();
  });

  it('stems plurals so "instruments" matches "instrument"', () => {
    const stemmed = composeFromPassages('tell me about clips', [
      {
        id: 'p2',
        title: 'Clips',
        text: 'A clip applier places a clip across the duct before it is divided sharply.',
        citation: 'test',
      },
    ]);
    expect(stemmed).not.toBeNull();
  });

  it('does not over-stem words ending in a sibilant plus s', () => {
    // "tissues" -> "tissue" must still match "tissue"; the naive rule produced
    // "tissu" from one and left the other whole, so they stopped matching.
    const answer = composeFromPassages('how are tissues handled?', [
      {
        id: 'p3',
        title: 'Tissue',
        text: 'Delicate tissue is grasped broadly so that retraction does not tear it.',
        citation: 'test',
      },
    ]);
    expect(answer).not.toBeNull();
  });
});

describe('retrieval', () => {
  it('finds the passage about a named structure', () => {
    const results = retrieve('what is the risk to the common bile duct?', 'general');
    expect(results.length).toBeGreaterThan(0);
    expect(results.some((p) => /bile duct/i.test(`${p.title} ${p.text}`))).toBe(true);
  });

  it('returns nothing for a question with no content words', () => {
    expect(retrieve('a the of', null)).toHaveLength(0);
  });

  it('still answers about another specialty when one is selected', () => {
    // Boosting, not filtering: a correct answer must not be hidden because the
    // selector says something else.
    const results = retrieve('common bile duct', 'colorectal');
    expect(results.some((p) => /bile duct/i.test(`${p.title} ${p.text}`))).toBe(true);
  });
});

describe('formatting helpers', () => {
  it('formats hours only when there are hours', () => {
    expect(hms(75)).toBe('1:15');
    expect(hms(4512)).toBe('1:15:12');
    expect(hms(0)).toBe('0:00');
  });

  it('joins lists without an Oxford comma', () => {
    expect(join(['a'])).toBe('a');
    expect(join(['a', 'b'])).toBe('a and b');
    expect(join(['a', 'b', 'c'])).toBe('a, b and c');
    expect(join([])).toBe('');
  });
});
