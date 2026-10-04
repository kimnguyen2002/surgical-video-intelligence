/**
 * GENERATED FILE — do not edit by hand.
 *
 * Produced by scripts/build_surgvu_data.py from the SurgVU 2024 release:
 *   videos  surgvu24_videos_only/
 *   labels  surgvu24_labels_updated_v2/labels/
 *
 * Regenerate with:  python3 scripts/build_surgvu_data.py
 */

export interface SurgvuPart {
  /** Part index as the label CSVs number it — 1-based, not 0-based. */
  part: number;
  /** Exact filename on disk; the file picker matches against this. */
  filename: string;
  /** True container duration in seconds, read from the video itself. */
  durationSeconds: number;
  fps: number;
  width: number;
  height: number;
  sizeBytes: number;
}

export interface SurgvuCase {
  caseId: string;
  dirName: string;
  parts: SurgvuPart[];
  totalDurationSeconds: number;
  /** Normalised tool classes this case's labels actually contain. */
  toolClasses: string[];
  taskClasses: string[];
  toolIntervalCount: number;
  taskIntervalCount: number;
}

/** Directory the videos live in, relative to the surg workspace root. */
export const SURGVU_VIDEO_ROOT = 'surgvu24_videos_only';

/**
 * Optional object-store root for streaming the **full** video parts.
 *
 * Empty by default, and that default is the point. An earlier build hardcoded
 * a Google Cloud Storage bucket here, so every visitor who opened a case
 * streamed hundreds of megabytes out of it and the author paid egress on all
 * of it — an open-ended bill attached to a public link, growing with exactly
 * the attention the project was published to attract.
 *
 * The three ways to watch a case now, in the order the app tries them:
 *
 * 1. **Bundled excerpt** (`src/data/surgvuClips.ts`) — 75 seconds per case,
 *    33 MB in total, served as static files. Costs nothing and always works.
 * 2. **Attach a local folder** — the File System Access API binds the real
 *    files off the visitor's own disk. Nothing is uploaded and a five-hour
 *    part seeks as fast as the disk can serve it.
 * 3. **This URL**, if someone deliberately sets `VITE_SURGVU_STORAGE_URL` for
 *    a private deployment where they are happy to pay for the bandwidth.
 *
 * Note that the SurgVU release carries its own redistribution terms. Anyone
 * setting this is responsible for honouring them.
 */
export const SURGVU_STORAGE_URL =
  (import.meta.env.VITE_SURGVU_STORAGE_URL as string | undefined)?.trim() || '';

/** Streaming URL for a full part, or `null` when no store is configured. */
export function getRemoteVideoUrl(caseId: string, partNumber: number): string | null {
  if (!SURGVU_STORAGE_URL) return null;
  const surgCase = SURGVU_CASES.find((c) => c.caseId === caseId);
  if (!surgCase) return null;
  const part = surgCase.parts.find((p) => p.part === partNumber);
  if (!part) return null;
  return `${SURGVU_STORAGE_URL.replace(/\/+$/, '')}/${surgCase.dirName}/${part.filename}`;
}

export const SURGVU_CASES: SurgvuCase[] = [
  {
    "caseId": "000",
    "dirName": "case_000",
    "parts": [
      {
        "part": 1,
        "filename": "case_000_video_part_001.mp4",
        "durationSeconds": 2518.883,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 306962159
      }
    ],
    "toolClasses": [
      "cadiere_forceps",
      "clip_applier",
      "force_bipolar",
      "permanent_cautery_hook_spatula",
      "prograsp_forceps",
      "stapler",
      "suction_irrigator",
      "vessel_sealer"
    ],
    "taskClasses": [
      "skills_application"
    ],
    "totalDurationSeconds": 2518.883,
    "toolIntervalCount": 13,
    "taskIntervalCount": 1
  },
  {
    "caseId": "001",
    "dirName": "case_001",
    "parts": [
      {
        "part": 1,
        "filename": "case_001_video_part_001.mp4",
        "durationSeconds": 7459.683,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 992079892
      }
    ],
    "toolClasses": [
      "bipolar_forceps",
      "cadiere_forceps",
      "clip_applier",
      "force_bipolar",
      "grasping_retractor",
      "monopolar_curved_scissors",
      "needle_driver",
      "permanent_cautery_hook_spatula",
      "prograsp_forceps",
      "stapler",
      "tip_up_fenestrated_grasper",
      "vessel_sealer"
    ],
    "taskClasses": [
      "rectal_artery_vein",
      "skills_application",
      "suspensory_ligaments",
      "suturing",
      "uterine_horn"
    ],
    "totalDurationSeconds": 7459.683,
    "toolIntervalCount": 44,
    "taskIntervalCount": 5
  },
  {
    "caseId": "002",
    "dirName": "case_002",
    "parts": [
      {
        "part": 1,
        "filename": "case_002_video_part_001.mp4",
        "durationSeconds": 19338.35,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 1935459741
      },
      {
        "part": 2,
        "filename": "case_002_video_part_002.mp4",
        "durationSeconds": 1342.583,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 236153075
      }
    ],
    "toolClasses": [
      "bipolar_forceps",
      "cadiere_forceps",
      "clip_applier",
      "force_bipolar",
      "grasping_retractor",
      "monopolar_curved_scissors",
      "needle_driver",
      "permanent_cautery_hook_spatula",
      "prograsp_forceps",
      "stapler",
      "vessel_sealer"
    ],
    "taskClasses": [
      "range_of_motion",
      "rectal_artery_vein",
      "retraction_collision_avoidance",
      "skills_application",
      "suspensory_ligaments",
      "suturing",
      "uterine_horn"
    ],
    "totalDurationSeconds": 20680.933,
    "toolIntervalCount": 58,
    "taskIntervalCount": 13
  },
  {
    "caseId": "003",
    "dirName": "case_003",
    "parts": [
      {
        "part": 1,
        "filename": "case_003_video_part_001.mp4",
        "durationSeconds": 17462.933,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 1866819525
      },
      {
        "part": 2,
        "filename": "case_003_video_part_002.mp4",
        "durationSeconds": 2008.767,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 331444406
      }
    ],
    "toolClasses": [
      "bipolar_forceps",
      "cadiere_forceps",
      "clip_applier",
      "grasping_retractor",
      "monopolar_curved_scissors",
      "needle_driver",
      "permanent_cautery_hook_spatula",
      "prograsp_forceps",
      "stapler",
      "vessel_sealer"
    ],
    "taskClasses": [
      "rectal_artery_vein",
      "suspensory_ligaments",
      "suturing",
      "uterine_horn"
    ],
    "totalDurationSeconds": 19471.7,
    "toolIntervalCount": 57,
    "taskIntervalCount": 6
  },
  {
    "caseId": "004",
    "dirName": "case_004",
    "parts": [
      {
        "part": 1,
        "filename": "case_004_video_part_001.mp4",
        "durationSeconds": 16322.883,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 1854377338
      },
      {
        "part": 2,
        "filename": "case_004_video_part_002.mp4",
        "durationSeconds": 3578.65,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 512235399
      }
    ],
    "toolClasses": [
      "bipolar_forceps",
      "cadiere_forceps",
      "clip_applier",
      "grasping_retractor",
      "monopolar_curved_scissors",
      "needle_driver",
      "permanent_cautery_hook_spatula",
      "prograsp_forceps",
      "stapler",
      "vessel_sealer"
    ],
    "taskClasses": [
      "range_of_motion",
      "rectal_artery_vein",
      "retraction_collision_avoidance",
      "skills_application",
      "suspensory_ligaments",
      "suturing",
      "uterine_horn"
    ],
    "totalDurationSeconds": 19901.533,
    "toolIntervalCount": 41,
    "taskIntervalCount": 10
  },
  {
    "caseId": "005",
    "dirName": "case_005",
    "parts": [
      {
        "part": 1,
        "filename": "case_005_video_part_001.mp4",
        "durationSeconds": 17982.15,
        "fps": 60.0,
        "width": 1280,
        "height": 720,
        "sizeBytes": 1552628497
      }
    ],
    "toolClasses": [
      "bipolar_forceps",
      "cadiere_forceps",
      "clip_applier",
      "grasping_retractor",
      "monopolar_curved_scissors",
      "needle_driver",
      "prograsp_forceps",
      "stapler",
      "vessel_sealer"
    ],
    "taskClasses": [
      "rectal_artery_vein",
      "retraction_collision_avoidance",
      "suspensory_ligaments",
      "suturing",
      "uterine_horn"
    ],
    "totalDurationSeconds": 17982.15,
    "toolIntervalCount": 46,
    "taskIntervalCount": 8
  }
];

/** Every part across every case, flattened, in playback order. */
export const SURGVU_PARTS = SURGVU_CASES.flatMap((c) =>
  c.parts.map((p) => ({ ...p, caseId: c.caseId, dirName: c.dirName }))
);
