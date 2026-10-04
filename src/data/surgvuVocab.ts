/**
 * SurgVU class vocabularies.
 *
 * Ported verbatim from `surgical_main/ai/training/config.py` so the browser
 * app and the training pipeline name the same instrument the same way. The
 * snake_case ids are the dataset's own, normalised; the display strings are
 * what the SurgVU paper prints (arXiv:2501.09209).
 *
 * Keep this in sync with config.py by hand — it is short, and a generated file
 * here would be one more thing to regenerate for twenty lines of constants.
 */

/** The twelve instrument classes the detection/presence models are trained on. */
export const TOOL_CLASSES: string[] = [
  'needle_driver',
  'cadiere_forceps',
  'prograsp_forceps',
  'monopolar_curved_scissors',
  'bipolar_forceps',
  'stapler',
  'force_bipolar',
  'vessel_sealer',
  'permanent_cautery_hook_spatula',
  'clip_applier',
  'tip_up_fenestrated_grasper',
  'grasping_retractor',
];

export const TOOL_DISPLAY: Record<string, string> = {
  needle_driver: 'Needle driver',
  cadiere_forceps: 'Cadiere forceps',
  prograsp_forceps: 'Prograsp forceps',
  monopolar_curved_scissors: 'Monopolar curved scissors',
  bipolar_forceps: 'Bipolar forceps',
  stapler: 'Stapler',
  force_bipolar: 'Force bipolar',
  vessel_sealer: 'Vessel sealer',
  permanent_cautery_hook_spatula: 'Permanent cautery hook/spatula',
  clip_applier: 'Clip applier',
  tip_up_fenestrated_grasper: 'Tip-up fenestrated grasper',
  grasping_retractor: 'Grasping retractor',
};

export const TASK_CLASSES: string[] = [
  'suturing',
  'uterine_horn',
  'rectal_artery_vein',
  'suspensory_ligaments',
  'skills_application',
  'range_of_motion',
  'retraction_collision_avoidance',
  'other',
];

export const TASK_DISPLAY: Record<string, string> = {
  suturing: 'Suturing',
  uterine_horn: 'Uterine horn',
  rectal_artery_vein: 'Rectal artery/vein manipulation',
  suspensory_ligaments: 'Suspensory ligaments',
  skills_application: 'General skills application',
  range_of_motion: 'Range of motion',
  retraction_collision_avoidance: 'Retraction and collision avoidance',
  other: 'Other',
};

/**
 * Whether a recorded label is one a model could ever predict.
 *
 * The released labels are not confined to the twelve training classes —
 * case_000 records a `suction_irrigator`, which no checkpoint has a head for.
 * The interface shows the recorded fact either way, but marks the ones no
 * model can be scored against, so an empty prediction is not read as a miss.
 */
export function isTrainingClass(label: string): boolean {
  return TOOL_CLASSES.includes(label);
}

export function toolDisplay(label: string): string {
  return TOOL_DISPLAY[label] || label.replace(/_/g, ' ');
}

export function taskDisplay(label: string): string {
  return TASK_DISPLAY[label] || label.replace(/_/g, ' ');
}

/**
 * Robot arm identifiers used by the tool labels.
 *
 * Tool presence is recorded per arm (USM = universal setup module), which is
 * why presence is genuinely multi-label: up to four arms carry an instrument
 * at once. Each arm gets a stable colour so the ribbon reads as four tracks.
 */
export const ARM_ORDER = ['USM1', 'USM2', 'USM3', 'USM4'] as const;

export const ARM_COLORS: Record<string, { bar: string; text: string; dot: string }> = {
  USM1: { bar: 'bg-emerald-500/80', text: 'text-emerald-300', dot: '#10b981' },
  USM2: { bar: 'bg-sky-500/80', text: 'text-sky-300', dot: '#0ea5e9' },
  USM3: { bar: 'bg-violet-500/80', text: 'text-violet-300', dot: '#8b5cf6' },
  USM4: { bar: 'bg-orange-500/80', text: 'text-orange-300', dot: '#f97316' },
};

export function armColor(arm: string) {
  return ARM_COLORS[arm] || { bar: 'bg-slate-500/80', text: 'text-slate-300', dot: '#64748b' };
}
