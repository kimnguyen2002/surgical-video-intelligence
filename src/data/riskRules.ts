/**
 * What is at risk *right now*, derived from the video.
 *
 * The Anatomy at Risk panel used to print one fixed list — the twelve
 * cholecystectomy structures of the "General Surgery" catalogue — regardless
 * of which case was loaded, which task was underway, or whether any video was
 * loaded at all. It was labelled "reference", which was honest, but it meant
 * the panel never once responded to the operation on screen.
 *
 * It is now driven by the two live signals the app already has, and each entry
 * carries which one put it there:
 *
 * **Recorded** — the SurgVU task annotated at the playhead. A fact from the
 * release, so the structures a trainee is working near during "Rectal
 * artery/vein manipulation" are known, not guessed.
 *
 * **Predicted** — the instruments the on-device detector currently sees. An
 * energy device on screen means thermal injury is live in a way it is not when
 * only graspers are in the field.
 *
 * These remain educational references keyed to a training task, not
 * observations of a particular patient's anatomy. The panel says so, and every
 * card states why it appeared.
 */

import { AnatomicalRisk } from '../types';
import { SURGICAL_KNOWLEDGE_BASE } from './knowledge';
import { toolDisplay } from './surgvuVocab';

export interface ActiveRisk extends AnatomicalRisk {
  /** Plain-language explanation of why this card is on screen. */
  reason: string;
  /** Which signal raised it — governs how it is styled. */
  source: 'recorded' | 'predicted';
}

/** Pull an existing catalogue entry by id so wording stays in one place. */
function fromCatalogue(specialty: keyof typeof SURGICAL_KNOWLEDGE_BASE, id: string): AnatomicalRisk {
  const found = SURGICAL_KNOWLEDGE_BASE[specialty].anatomicalRisks.find((r) => r.id === id);
  if (!found) throw new Error(`Unknown risk ${specialty}/${id}`);
  return found;
}

// ---------------------------------------------------------------------------
// Task → anatomy
// ---------------------------------------------------------------------------

/**
 * Structures relevant to each SurgVU task.
 *
 * SurgVU is robotic training footage on porcine models, so these are the
 * structures the named exercise is performed around. Tasks with no anatomical
 * target — range of motion, collision avoidance — carry technique risks
 * instead, because "nothing is at risk" is false for a live instrument in a
 * body cavity.
 */
const TASK_RISKS: Record<string, AnatomicalRisk[]> = {
  uterine_horn: [
    fromCatalogue('gynecology', 'ureter_uterine'),
    fromCatalogue('gynecology', 'bladder_dome'),
  ],
  suspensory_ligaments: [
    fromCatalogue('gynecology', 'ureter_uterine'),
    {
      id: 'gonadal_pedicle',
      structure: 'gonadal vessels in the suspensory ligament',
      level: 'CRITICAL',
      mechanism:
        'The ovarian/gonadal pedicle runs inside the ligament being divided; taken without control it retracts and bleeds out of view.',
      protect: 'Skeletonise and seal the pedicle before division; never divide the ligament blind.',
      specialtyId: 'gynecology',
    },
  ],
  rectal_artery_vein: [
    fromCatalogue('colorectal', 'ureter_left'),
    fromCatalogue('colorectal', 'hypogastric_plexus'),
  ],
  suturing: [
    {
      id: 'needle_adjacent',
      structure: 'structures behind the suture bite',
      level: 'CAUTION',
      mechanism:
        'A needle driven deeper than the tissue in front of it catches bowel or vessel lying behind, and the injury is not seen at the time.',
      protect: 'Take bites under direct vision with the needle tip visible throughout its arc.',
      specialtyId: 'general',
    },
    {
      id: 'needle_loss',
      structure: 'the needle itself',
      level: 'WARNING',
      mechanism: 'A needle released outside the field is a retained foreign body until proven otherwise.',
      protect: 'Keep the needle in view or in the driver; account for it before closing.',
      specialtyId: 'general',
    },
  ],
  retraction_collision_avoidance: [
    {
      id: 'offscreen_instrument',
      structure: 'tissue outside the camera view',
      level: 'CRITICAL',
      mechanism:
        'A robotic arm moved while off-screen exerts full force on whatever it meets; the endoscope shows none of it.',
      protect: 'Move only instruments you can see; recentre the camera before repositioning an arm.',
      specialtyId: 'general',
    },
    {
      id: 'traction_tear',
      structure: 'serosa and capsule under traction',
      level: 'CAUTION',
      mechanism: 'Robotic graspers have no force feedback, so retraction tears before it feels tight.',
      protect: 'Retract with broad grasps and watch tissue blanching rather than relying on feel.',
      specialtyId: 'general',
    },
  ],
  range_of_motion: [
    {
      id: 'offscreen_instrument',
      structure: 'tissue outside the camera view',
      level: 'CRITICAL',
      mechanism:
        'Range-of-motion practice sweeps instruments through space the endoscope is not looking at.',
      protect: 'Keep every moving tip inside the field of view.',
      specialtyId: 'general',
    },
  ],
  skills_application: [
    {
      id: 'traction_tear',
      structure: 'serosa and capsule under traction',
      level: 'CAUTION',
      mechanism: 'Robotic graspers have no force feedback, so retraction tears before it feels tight.',
      protect: 'Retract with broad grasps and watch tissue blanching rather than relying on feel.',
      specialtyId: 'general',
    },
  ],
  other: [],
};

// ---------------------------------------------------------------------------
// Instrument → hazard
// ---------------------------------------------------------------------------

const THERMAL: AnatomicalRisk = {
  id: 'thermal_spread',
  structure: 'tissue adjacent to the active tip',
  level: 'CRITICAL',
  mechanism:
    'Energy spreads several millimetres beyond the jaws and conducts along ducts and vessels; the resulting necrosis perforates days later, having looked fine at the time.',
  protect: 'Keep the active tip off unseen tissue, use short bursts, and stay clear of ducts and bowel.',
  specialtyId: 'general',
};

const TRACTION: AnatomicalRisk = {
  id: 'traction_tear',
  structure: 'serosa and capsule under traction',
  level: 'CAUTION',
  mechanism: 'Robotic graspers have no force feedback, so retraction tears before it feels tight.',
  protect: 'Retract with broad grasps and watch tissue blanching rather than relying on feel.',
  specialtyId: 'general',
};

const INSTRUMENT_RISKS: Record<string, AnatomicalRisk[]> = {
  monopolar_curved_scissors: [THERMAL],
  permanent_cautery_hook_spatula: [THERMAL],
  vessel_sealer: [THERMAL],
  force_bipolar: [THERMAL],
  bipolar_forceps: [THERMAL],
  bipolar_dissector: [THERMAL],
  stapler: [
    {
      id: 'staple_line',
      structure: 'tissue inside the stapler jaws',
      level: 'CRITICAL',
      mechanism:
        'Whatever is between the jaws is divided when the stapler fires, including a structure lying behind the target.',
      protect: 'Confirm both jaws are clear before firing; check the posterior jaw, not only the anterior.',
      specialtyId: 'general',
    },
  ],
  clip_applier: [
    {
      id: 'clip_wrong_structure',
      structure: 'the structure inside the clip',
      level: 'CRITICAL',
      mechanism:
        'A clip is placed on whatever is identified as the target; a misidentified duct or ureter is occluded permanently.',
      protect: 'Identify the structure along its length before clipping, not at one point.',
      specialtyId: 'general',
    },
  ],
  needle_driver: [
    {
      id: 'needle_adjacent',
      structure: 'structures behind the suture bite',
      level: 'CAUTION',
      mechanism:
        'A needle driven deeper than the tissue in front of it catches bowel or vessel lying behind.',
      protect: 'Take bites under direct vision with the needle tip visible throughout its arc.',
      specialtyId: 'general',
    },
  ],
  grasping_retractor: [TRACTION],
  prograsp_forceps: [TRACTION],
  cadiere_forceps: [TRACTION],
  tip_up_fenestrated_grasper: [TRACTION],
};

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

export interface RiskInputs {
  /** The SurgVU task recorded at the playhead, if any. */
  recordedTaskLabel: string | null;
  recordedTaskDisplay: string | null;
  /** snake_case classes the detector currently sees. */
  detectedToolLabels: string[];
}

/**
 * The risks that apply to this moment.
 *
 * Recorded-task risks come first: a fact about the exercise outranks an
 * inference from a box. A structure raised by both keeps the recorded reason,
 * since that is the stronger claim.
 */
export function risksFor({
  recordedTaskLabel,
  recordedTaskDisplay,
  detectedToolLabels,
}: RiskInputs): ActiveRisk[] {
  const byId = new Map<string, ActiveRisk>();

  if (recordedTaskLabel) {
    for (const risk of TASK_RISKS[recordedTaskLabel] || []) {
      byId.set(risk.id, {
        ...risk,
        source: 'recorded',
        reason: `the dataset records the ${recordedTaskDisplay || recordedTaskLabel} task here`,
      });
    }
  }

  // Deduplicate the instruments first: three needle drivers on screen raise
  // the same hazard once, not three times.
  for (const label of [...new Set(detectedToolLabels)]) {
    for (const risk of INSTRUMENT_RISKS[label] || []) {
      if (byId.has(risk.id)) continue;
      byId.set(risk.id, {
        ...risk,
        source: 'predicted',
        reason: `a ${toolDisplay(label).toLowerCase()} is visible in this frame`,
      });
    }
  }

  const order = { CRITICAL: 0, CAUTION: 1, WARNING: 2, INFO: 3 };
  return [...byId.values()].sort(
    (a, b) =>
      (a.source === b.source ? 0 : a.source === 'recorded' ? -1 : 1) ||
      order[a.level] - order[b.level]
  );
}
