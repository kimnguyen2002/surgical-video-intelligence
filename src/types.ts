export type SpecialtyId =
  | 'general'
  | 'bariatric'
  | 'colorectal'
  | 'gynecology'
  | 'urology'
  | 'cardiothoracic'
  | 'orthopedic'
  | 'neurosurgery'
  | 'ent'
  | 'plastics'
  | 'pediatric'
  | 'vascular';

/*
 * `GeminiModelId` is gone. It was a union of hardcoded model ids that had to be
 * edited every time Google retired one, and a stale entry failed as a 404 from
 * inside a dropdown that claimed to offer it. The app now lists what the
 * visitor's own key can actually call; `AppState.modelId` is a plain string.
 *
 * Its `'ground-truth-only'` member is gone for a better reason: it was a mode
 * you could switch *out of*. Showing only what the release records is no longer
 * a setting, it is the default and the floor — every answer starts there, and a
 * language model can only ever be added on top.
 */

export type AssistantRole =
  | 'Student'
  | 'Resident'
  | 'Fellow'
  | 'Researcher'
  | 'AI Engineer';

export type RiskLevel = 'CRITICAL' | 'CAUTION' | 'WARNING' | 'INFO';

export interface AnatomicalRisk {
  id: string;
  structure: string;
  level: RiskLevel;
  mechanism: string;
  protect: string;
  specialtyId: SpecialtyId;
  relevantPhases?: string[];
}

export interface BoundingBox {
  ymin: number; // 0 - 1000 normalized
  xmin: number; // 0 - 1000 normalized
  ymax: number; // 0 - 1000 normalized
  xmax: number; // 0 - 1000 normalized
}

export type DetectionType = 'predicted' | 'recorded' | 'refused';

/** Which video is on the stage, and therefore what can be said about it. */
export type VideoSourceType = 'none' | 'library' | 'file' | 'camera' | 'url' | 'youtube';

export interface DetectedInstrument {
  id: string;
  label: string;
  confidence?: number; // 0 - 100 (undefined for recorded ground truth)
  box: BoundingBox;
  type: DetectionType;
  description?: string;
  color?: string;
}

export interface SurgicalPhaseInfo {
  phase: string;
  progress: number; // 0 - 100
  timeInPhase?: string;
  criticalChecklist?: string[];
}

/**
 * What the SurgVU release records for the frame that was analysed.
 *
 * Kept separate from `instruments` on purpose. These come from the robot's
 * installation log, carry no bounding box and no confidence, and are true
 * whether or not a model ran. Merging them into the prediction list would
 * erase the distinction the whole interface is built to preserve.
 */
export interface RecordedContext {
  caseId: string;
  part: number;
  tools: {
    label: string;
    display: string;
    commercial: string;
    arm: string;
    trainable: boolean;
  }[];
  taskDisplay: string | null;
}

export interface FrameObservation {
  id: string;
  timestampSeconds: number;
  timestampFormatted: string;
  thumbnailUrl: string;
  phase: string;
  summary: string;
  instruments: DetectedInstrument[];
  anatomyIdentified: string[];
  risksPresent: AnatomicalRisk[];
  criticalViewOfSafetyScore?: number; // 0 to 6
  isOutOfDomain?: boolean;
  /** Dataset facts for this timestamp, when a library video is loaded. */
  recorded?: RecordedContext | null;
  /** Which Gemini model produced `instruments`, if any did. */
  modelUsed?: string;
}

/** A structured failure from the Gemini proxy, shown rather than swallowed. */
export interface ApiFailure {
  code: number;
  status: string;
  message: string;
  hint?: string;
}

/** Result of the server's key/reachability probe. */
export interface ApiHealth {
  ok: boolean;
  keyPresent: boolean;
  reason?: string;
  hint?: string;
  model?: string;
  quotaExceeded?: boolean;
  suggestedModel?: string;
}

/**
 * Which answer path produced a message.
 *
 * Rendered as a badge on every assistant turn. A reader must be able to tell,
 * without asking, whether they are looking at a dataset lookup, a quotation
 * from the knowledge base, or text a language model wrote.
 */
export type AnswerSource = 'builtin' | 'grounded' | 'extractive' | 'generative' | 'none';

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant' | 'system';
  text: string;
  timestamp: string;
  /** Set on assistant turns; absent on user and system messages. */
  source?: AnswerSource;
  frameContextId?: string;
  frameTimestamp?: string;
}

export interface SpecialtyConfig {
  id: SpecialtyId;
  name: string;
  category: string;
  description: string;
  instruments: string[];
  anatomicalRisks: AnatomicalRisk[];
  phases: string[];
  sampleCases: SampleCase[];
}

/**
 * A short written description of a representative case for a specialty.
 *
 * These are illustrative background for the knowledge base — they are not the
 * video library, and they carry no detections. Real, playable footage comes
 * from `src/data/surgvuCases.ts`, which is generated from the SurgVU release.
 */
export interface SampleCase {
  id: string;
  title: string;
  filename: string;
  procedure: string;
  durationFormatted: string;
  durationSeconds: number;
  thumbnailUrl?: string;
  videoUrl?: string;
  description: string;
}
