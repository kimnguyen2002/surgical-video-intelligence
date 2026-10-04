/**
 * GENERATED FILE - do not edit by hand.
 *
 * Produced by scripts/build_clips.py. Regenerate with: npm run build:clips
 *
 * One short excerpt per SurgVU case, bundled with the app so the player is not
 * empty for a visitor who does not have the 8.9 GB release on disk. Each window
 * was chosen by how much *labelled* activity it contains rather than taken from
 * the start of the recording - see the module docstring in build_clips.py.
 */

export interface SurgvuClip {
  caseId: string;
  /** The part of the original recording this excerpt was cut from. */
  part: number;
  /** Path under `public/`, served at the site root. */
  file: string;
  sourceFilename: string;
  /**
   * Offset of the excerpt within the original part, in seconds.
   *
   * Every ground-truth lookup adds this to the player's `currentTime`. Without
   * it the app would read annotations from the opening minutes of the case
   * while the viewer watches its middle - confident, precisely-timed and
   * entirely wrong.
   */
  sourceStartSeconds: number;
  durationSeconds: number;
  sizeBytes: number;
  /** Instruments the release records as installed during this window. */
  tools: string[];
  /** Tasks the release labels during this window. */
  tasks: string[];
}

export const SURGVU_CLIPS: SurgvuClip[] = [
  {
    caseId: "000",
    part: 1,
    file: "clips/case_000_excerpt.mp4",
    sourceFilename: "case_000_video_part_001.mp4",
    sourceStartSeconds: 2015.0,
    durationSeconds: 75,
    sizeBytes: 7132060,
    tools: ["Clip applier", "Force bipolar", "Permanent cautery hook/spatula", "Prograsp forceps"],
    tasks: ["General skills application"],
  },
  {
    caseId: "001",
    part: 1,
    file: "clips/case_001_excerpt.mp4",
    sourceFilename: "case_001_video_part_001.mp4",
    sourceStartSeconds: 4490.0,
    durationSeconds: 75,
    sizeBytes: 3527590,
    tools: ["Cadiere forceps", "Force bipolar", "Monopolar curved scissors", "Needle driver", "Prograsp forceps", "Stapler", "Tip-up fenestrated grasper"],
    tasks: ["Suturing"],
  },
  {
    caseId: "002",
    part: 2,
    file: "clips/case_002_excerpt.mp4",
    sourceFilename: "case_002_video_part_002.mp4",
    sourceStartSeconds: 590.0,
    durationSeconds: 75,
    sizeBytes: 4844714,
    tools: ["Clip applier", "Force bipolar", "Monopolar curved scissors", "Needle driver", "Prograsp forceps"],
    tasks: ["General skills application"],
  },
  {
    caseId: "003",
    part: 1,
    file: "clips/case_003_excerpt.mp4",
    sourceFilename: "case_003_video_part_001.mp4",
    sourceStartSeconds: 6875.0,
    durationSeconds: 75,
    sizeBytes: 4746454,
    tools: ["Bipolar forceps", "Cadiere forceps", "Monopolar curved scissors", "Needle driver"],
    tasks: ["Suturing"],
  },
  {
    caseId: "004",
    part: 1,
    file: "clips/case_004_excerpt.mp4",
    sourceFilename: "case_004_video_part_001.mp4",
    sourceStartSeconds: 15920.0,
    durationSeconds: 75,
    sizeBytes: 7037185,
    tools: ["Bipolar forceps", "Cadiere forceps", "Clip applier", "Monopolar curved scissors"],
    tasks: ["Rectal artery/vein manipulation"],
  },
  {
    caseId: "005",
    part: 1,
    file: "clips/case_005_excerpt.mp4",
    sourceFilename: "case_005_video_part_001.mp4",
    sourceStartSeconds: 12500.0,
    durationSeconds: 75,
    sizeBytes: 6269712,
    tools: ["Bipolar forceps", "Clip applier", "Grasping retractor", "Monopolar curved scissors", "Needle driver"],
    tasks: ["Rectal artery/vein manipulation"],
  },
];

export function clipFor(caseId: string): SurgvuClip | undefined {
  return SURGVU_CLIPS.find((c) => c.caseId === caseId);
}

/** Total bundled video, so the About page can state the real figure. */
export const SURGVU_CLIPS_TOTAL_BYTES = SURGVU_CLIPS.reduce(
  (sum, c) => sum + c.sizeBytes,
  0
);
