/**
 * GENERATED FILE - do not edit by hand.
 *
 * Produced by scripts/build_clips.py. Regenerate with: npm run build:clips
 *
 * One short excerpt per SurgVU part, bundled with the app so the player is not
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
    file: "clips/case_000_part_001_excerpt.mp4",
    sourceFilename: "case_000_video_part_001.mp4",
    sourceStartSeconds: 1970.0,
    durationSeconds: 120,
    sizeBytes: 11746814,
    tools: ["Clip applier", "Force bipolar", "Permanent cautery hook/spatula", "Prograsp forceps"],
    tasks: ["General skills application"],
  },
  {
    caseId: "001",
    part: 1,
    file: "clips/case_001_part_001_excerpt.mp4",
    sourceFilename: "case_001_video_part_001.mp4",
    sourceStartSeconds: 4445.0,
    durationSeconds: 120,
    sizeBytes: 6727408,
    tools: ["Cadiere forceps", "Force bipolar", "Monopolar curved scissors", "Needle driver", "Prograsp forceps", "Stapler", "Tip-up fenestrated grasper"],
    tasks: ["Suturing"],
  },
  {
    caseId: "002",
    part: 1,
    file: "clips/case_002_part_001_excerpt.mp4",
    sourceFilename: "case_002_video_part_001.mp4",
    sourceStartSeconds: 15380.0,
    durationSeconds: 120,
    sizeBytes: 7196342,
    tools: ["Bipolar forceps", "Cadiere forceps", "Clip applier", "Monopolar curved scissors", "Needle driver"],
    tasks: ["Rectal artery/vein manipulation"],
  },
  {
    caseId: "002",
    part: 2,
    file: "clips/case_002_part_002_excerpt.mp4",
    sourceFilename: "case_002_video_part_002.mp4",
    sourceStartSeconds: 545.0,
    durationSeconds: 120,
    sizeBytes: 7183609,
    tools: ["Clip applier", "Force bipolar", "Monopolar curved scissors", "Needle driver", "Prograsp forceps"],
    tasks: ["General skills application"],
  },
  {
    caseId: "003",
    part: 1,
    file: "clips/case_003_part_001_excerpt.mp4",
    sourceFilename: "case_003_video_part_001.mp4",
    sourceStartSeconds: 6830.0,
    durationSeconds: 120,
    sizeBytes: 7642197,
    tools: ["Bipolar forceps", "Cadiere forceps", "Monopolar curved scissors", "Needle driver"],
    tasks: ["Suturing"],
  },
  {
    caseId: "003",
    part: 2,
    file: "clips/case_003_part_002_excerpt.mp4",
    sourceFilename: "case_003_video_part_002.mp4",
    sourceStartSeconds: 515.0,
    durationSeconds: 120,
    sizeBytes: 6614862,
    tools: ["Bipolar forceps", "Monopolar curved scissors", "Needle driver", "Stapler"],
    tasks: [],
  },
  {
    caseId: "004",
    part: 1,
    file: "clips/case_004_part_001_excerpt.mp4",
    sourceFilename: "case_004_video_part_001.mp4",
    sourceStartSeconds: 15875.0,
    durationSeconds: 120,
    sizeBytes: 11937265,
    tools: ["Bipolar forceps", "Cadiere forceps", "Clip applier", "Monopolar curved scissors"],
    tasks: ["Rectal artery/vein manipulation"],
  },
  {
    caseId: "004",
    part: 2,
    file: "clips/case_004_part_002_excerpt.mp4",
    sourceFilename: "case_004_video_part_002.mp4",
    sourceStartSeconds: 890.0,
    durationSeconds: 120,
    sizeBytes: 9070880,
    tools: ["Bipolar forceps", "Cadiere forceps", "Stapler", "Vessel sealer"],
    tasks: [],
  },
  {
    caseId: "005",
    part: 1,
    file: "clips/case_005_part_001_excerpt.mp4",
    sourceFilename: "case_005_video_part_001.mp4",
    sourceStartSeconds: 12455.0,
    durationSeconds: 120,
    sizeBytes: 9981188,
    tools: ["Bipolar forceps", "Clip applier", "Grasping retractor", "Monopolar curved scissors", "Needle driver"],
    tasks: ["Rectal artery/vein manipulation"],
  },
];

export function clipFor(caseId: string, part: number): SurgvuClip | undefined {
  return SURGVU_CLIPS.find((c) => c.caseId === caseId && c.part === part);
}

/** Total bundled video, so the About page can state the real figure. */
export const SURGVU_CLIPS_TOTAL_BYTES = SURGVU_CLIPS.reduce(
  (sum, c) => sum + c.sizeBytes,
  0
);
