/**
 * Tests for annotation lookup and the excerpt time offset.
 *
 * The offset is the most dangerous thing in this codebase, because getting it
 * wrong does not throw and does not render blank. It produces *real*
 * annotations from the *wrong moment* — confident, precisely timed, and
 * entirely wrong — which is indistinguishable from the app working unless you
 * check it against the video. So it is pinned here rather than trusted.
 */

import { describe, expect, it } from 'vitest';

import {
  findPart,
  formatClock,
  matchFileToPart,
  taskAt,
  taskIntervalsFor,
  toolIntervalsFor,
  toolsAt,
  truthAt,
} from './groundTruth';
import { SURGVU_CLIPS, clipFor } from '../data/surgvuClips';

describe('bundled excerpts', () => {
  it('ships an excerpt for every case it claims', () => {
    expect(SURGVU_CLIPS.length).toBeGreaterThan(0);
    for (const clip of SURGVU_CLIPS) {
      expect(clip.file).toMatch(/^clips\/.+\.mp4$/);
      expect(clip.durationSeconds).toBeGreaterThan(0);
      expect(clip.sizeBytes).toBeGreaterThan(0);
    }
  });

  it('cuts every excerpt from a part that actually exists', () => {
    for (const clip of SURGVU_CLIPS) {
      const part = findPart(clip.caseId, clip.part);
      expect(part, `case ${clip.caseId} part ${clip.part}`).toBeDefined();
      expect(part!.filename).toBe(clip.sourceFilename);
    }
  });

  it('keeps every excerpt inside its source part', () => {
    for (const clip of SURGVU_CLIPS) {
      const part = findPart(clip.caseId, clip.part)!;
      expect(clip.sourceStartSeconds).toBeGreaterThanOrEqual(0);
      expect(clip.sourceStartSeconds + clip.durationSeconds).toBeLessThanOrEqual(
        part.durationSeconds
      );
    }
  });

  /**
   * The point of the selection algorithm: every excerpt must land somewhere
   * the release actually annotated. An excerpt over unlabelled footage shows
   * an empty ground-truth panel, which reads as a broken app.
   */
  it('lands every excerpt on annotated footage', () => {
    for (const clip of SURGVU_CLIPS) {
      const midpoint = clip.sourceStartSeconds + clip.durationSeconds / 2;
      const truth = truthAt(clip.caseId, clip.part, midpoint);

      expect(truth.provenance, `case ${clip.caseId}`).toBe('recorded');
      expect(truth.task, `case ${clip.caseId} has a labelled task`).not.toBeNull();
      expect(truth.tools.length, `case ${clip.caseId} has instruments`).toBeGreaterThan(0);
    }
  });

  it('agrees with the manifest the builder recorded', () => {
    for (const clip of SURGVU_CLIPS) {
      const midpoint = clip.sourceStartSeconds + clip.durationSeconds / 2;
      const task = taskAt(clip.caseId, clip.part, midpoint);
      expect(clip.tasks).toContain(task!.display);
    }
  });

  /**
   * The offset bug, stated as a test.
   *
   * Reading the player's own clock instead of the source clock is the failure
   * this guards: at excerpt-time 37 s the app must look up source-time
   * `start + 37`, not 37.
   */
  it('reads different annotations at excerpt-time and source-time', () => {
    const clip = SURGVU_CLIPS.find((c) => c.sourceStartSeconds > 600);
    expect(clip, 'at least one excerpt is cut from deep into a part').toBeDefined();

    const playerTime = 37;
    const sourceTime = clip!.sourceStartSeconds + playerTime;

    const correct = truthAt(clip!.caseId, clip!.part, sourceTime);
    const naive = truthAt(clip!.caseId, clip!.part, playerTime);

    expect(correct.task).not.toBeNull();
    // If these ever agree, the test has stopped proving anything — the whole
    // risk is that the naive lookup returns something plausible.
    expect(correct.task?.display).not.toBe(naive.task?.display);
  });
});

describe('annotation lookup', () => {
  const clip = SURGVU_CLIPS[0];

  it('reports overlapping instruments across arms', () => {
    const midpoint = clip.sourceStartSeconds + clip.durationSeconds / 2;
    const tools = toolsAt(clip.caseId, clip.part, midpoint);
    // The arms are genuinely concurrent; a lookup returning only one would
    // mean the backward walk stopped at the first match.
    const arms = new Set(tools.map((t) => t.arm));
    expect(arms.size).toBe(tools.length);
  });

  it('returns nothing outside every interval', () => {
    const truth = truthAt(clip.caseId, clip.part, 1e9);
    expect(truth.tools).toHaveLength(0);
    expect(truth.task).toBeNull();
    // Still "recorded": the part has labels, they just do not cover this time.
    expect(truth.provenance).toBe('recorded');
  });

  it('reports no provenance for a part with no bundled labels', () => {
    expect(truthAt('999', 1, 10).provenance).toBe('none');
  });

  it('orders intervals by start time', () => {
    const intervals = toolIntervalsFor(clip.caseId, clip.part);
    for (let i = 1; i < intervals.length; i++) {
      expect(intervals[i].start).toBeGreaterThanOrEqual(intervals[i - 1].start);
    }
  });

  it('never ends an interval before it starts', () => {
    for (const row of [
      ...toolIntervalsFor(clip.caseId, clip.part),
      ...taskIntervalsFor(clip.caseId, clip.part),
    ]) {
      expect(row.end).toBeGreaterThan(row.start);
    }
  });
});

describe('file matching', () => {
  it('matches a known part by filename', () => {
    const clip = SURGVU_CLIPS[0];
    expect(matchFileToPart(clip.sourceFilename)).toEqual({
      caseId: clip.caseId,
      part: clip.part,
    });
  });

  it('is case-insensitive and tolerates surrounding whitespace', () => {
    const clip = SURGVU_CLIPS[0];
    expect(matchFileToPart(`  ${clip.sourceFilename.toUpperCase()}  `)).not.toBeNull();
  });

  /**
   * Rejecting rather than guessing is the whole point: playing case 003's
   * video against case 002's labels would produce confident, precisely-timed,
   * entirely wrong ground truth.
   */
  it('rejects a filename it does not recognise', () => {
    expect(matchFileToPart('holiday_video.mp4')).toBeNull();
    expect(matchFileToPart('case_999_video_part_001.mp4')).toBeNull();
  });
});

describe('formatClock', () => {
  it('adds an hours field only above an hour', () => {
    expect(formatClock(59)).toBe('0:59');
    expect(formatClock(600)).toBe('10:00');
    expect(formatClock(3661)).toBe('1:01:01');
  });

  it('survives junk rather than rendering NaN', () => {
    expect(formatClock(Number.NaN)).toBe('0:00');
    expect(formatClock(-5)).toBe('0:00');
  });
});
