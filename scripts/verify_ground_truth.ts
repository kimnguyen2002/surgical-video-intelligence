/**
 * Checks the bundled SurgVU ground truth against the release it came from.
 *
 * The interval index in services/groundTruth.ts is a binary search with a
 * backward walk, which is the kind of code that is either exactly right or
 * quietly off by one interval — and being off by one interval here means the
 * app confidently reports the wrong instrument at the wrong minute of an
 * operation. So the index is cross-checked against a brute-force scan at every
 * boundary of every interval in the corpus.
 *
 * Run with:  npm run verify:data
 */

import { toolsAt, taskAt, matchFileToPart, armTracks, summariseFor } from '../src/services/groundTruth';
import { SURGVU_TOOL_INTERVALS } from '../src/data/surgvuLabels';

let failures = 0;
const check = (name: string, cond: boolean, extra = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${extra ? ' — ' + extra : ''}`);
  if (!cond) failures++;
};

// case_000: stapler on USM4 recorded 444.8 -> 954.5
const mid = toolsAt('000', 1, 500);
check('stapler present mid-interval', mid.some(t => t.label === 'stapler' && t.arm === 'USM4'),
  mid.map(t => `${t.display}/${t.arm}`).join(', '));
check('stapler absent before interval', !toolsAt('000', 1, 400).some(t => t.label === 'stapler'));
check('stapler absent after interval', !toolsAt('000', 1, 1000).some(t => t.label === 'stapler'));
check('interval is half-open at start', toolsAt('000', 1, 444.8).some(t => t.label === 'stapler'));
check('interval is half-open at end', !toolsAt('000', 1, 954.5).some(t => t.label === 'stapler'));

// Multi-arm overlap is the whole reason presence is multi-label.
let maxConcurrent = 0, at = 0;
for (const iv of SURGVU_TOOL_INTERVALS.filter(i => i.case === '003' && i.part === 1)) {
  const n = toolsAt('003', 1, iv.start + 0.5).length;
  if (n > maxConcurrent) { maxConcurrent = n; at = iv.start; }
}
check('multiple arms carry tools at once', maxConcurrent >= 3, `${maxConcurrent} concurrent at t=${at.toFixed(1)}s`);

// Exhaustive cross-check: brute-force scan must equal the indexed lookup.
let mismatches = 0, samples = 0;
for (const c of ['000','001','002','003','004','005']) {
  for (const p of [1,2]) {
    const rows = SURGVU_TOOL_INTERVALS.filter(i => i.case === c && i.part === p);
    if (!rows.length) continue;
    for (const iv of rows) {
      for (const t of [iv.start, (iv.start+iv.end)/2, iv.end - 0.01]) {
        samples++;
        const brute = rows.filter(r => r.start <= t && t < r.end).length;
        if (toolsAt(c, p, t).length !== brute) mismatches++;
      }
    }
  }
}
check('indexed lookup matches brute force', mismatches === 0, `${samples} samples, ${mismatches} mismatches`);

// Task labels
check('case_000 task at 1800s', taskAt('000', 1, 1800)?.label === 'skills_application',
  taskAt('000', 1, 1800)?.display || 'none');
check('case_000 no task at 100s', taskAt('000', 1, 100) === null);

// Filename binding — the guard against playing one case against another's labels
check('filename binds to correct part',
  JSON.stringify(matchFileToPart('case_003_video_part_002.mp4')) === '{"caseId":"003","part":2}');
check('unknown filename rejected', matchFileToPart('some_random_video.mp4') === null);
check('case-insensitive match', matchFileToPart('CASE_005_VIDEO_PART_001.MP4')?.caseId === '005');

// Arm tracks
const tracks = armTracks('001', 1);
check('arm tracks built', tracks.length >= 3, tracks.map(t => `${t.arm}:${t.intervals.length}`).join(' '));

console.log('\n' + summariseFor('001', 1));
console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
