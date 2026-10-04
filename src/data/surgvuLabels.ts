/**
 * GENERATED FILE — do not edit by hand.
 *
 * Produced by scripts/build_surgvu_data.py from the SurgVU 2024 release:
 *   videos  surgvu24_videos_only/
 *   labels  surgvu24_labels_updated_v2/labels/
 *
 * Regenerate with:  python3 scripts/build_surgvu_data.py
 */

export interface SurgvuToolInterval {
  case: string;
  part: number;
  start: number;
  end: number;
  /** snake_case class id, shared with the training pipeline. */
  label: string;
  display: string;
  /** Robot arm the tool was installed on: USM1..USM4. */
  arm: string;
  /** Manufacturer name from the release, e.g. "SureForm Stapler 60". */
  commercial: string;
}

export interface SurgvuTaskInterval {
  case: string;
  part: number;
  start: number;
  end: number;
  label: string;
  display: string;
}

export const SURGVU_TOOL_INTERVALS: SurgvuToolInterval[] = [
 {
  "case": "000",
  "part": 1,
  "start": 444.8,
  "end": 954.5,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "000",
  "part": 1,
  "start": 477.98,
  "end": 514.53,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM3",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "000",
  "part": 1,
  "start": 524.73,
  "end": 540.6,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM3",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "000",
  "part": 1,
  "start": 1038.66,
  "end": 1067.06,
  "label": "vessel_sealer",
  "display": "Vessel sealer",
  "arm": "USM4",
  "commercial": "Vessel Sealer Extend"
 },
 {
  "case": "000",
  "part": 1,
  "start": 1494.38,
  "end": 2518.0,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM2",
  "commercial": "Force Bipolar"
 },
 {
  "case": "000",
  "part": 1,
  "start": 1496.58,
  "end": 1599.26,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM1",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "000",
  "part": 1,
  "start": 1501.36,
  "end": 2029.53,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM4",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "000",
  "part": 1,
  "start": 1612.96,
  "end": 1630.38,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM1",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "000",
  "part": 1,
  "start": 1675.05,
  "end": 2518.0,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM1",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "000",
  "part": 1,
  "start": 2048.41,
  "end": 2072.0,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "000",
  "part": 1,
  "start": 2082.83,
  "end": 2100.78,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "000",
  "part": 1,
  "start": 2122.66,
  "end": 2344.66,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM4",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "000",
  "part": 1,
  "start": 2358.95,
  "end": 2518.0,
  "label": "suction_irrigator",
  "display": "suction irrigator",
  "arm": "USM4",
  "commercial": "EndoWrist Suction Irrigator"
 },
 {
  "case": "001",
  "part": 1,
  "start": 0.0,
  "end": 954.81,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Fenestrated Bipolar Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 0.0,
  "end": 7459.0,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "001",
  "part": 1,
  "start": 0.0,
  "end": 962.61,
  "label": "vessel_sealer",
  "display": "Vessel sealer",
  "arm": "USM3",
  "commercial": "Vessel Sealer Extend"
 },
 {
  "case": "001",
  "part": 1,
  "start": 0.0,
  "end": 7459.0,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "001",
  "part": 1,
  "start": 0.0,
  "end": 7459.0,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 0.0,
  "end": 1000.81,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 973.51,
  "end": 1961.31,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM1",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "001",
  "part": 1,
  "start": 986.01,
  "end": 1965.21,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM3",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 1021.41,
  "end": 3340.11,
  "label": "grasping_retractor",
  "display": "Grasping retractor",
  "arm": "USM4",
  "commercial": "Small Grasping Retractor"
 },
 {
  "case": "001",
  "part": 1,
  "start": 1975.71,
  "end": 2193.61,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega Needle Driver"
 },
 {
  "case": "001",
  "part": 1,
  "start": 1992.71,
  "end": 2182.01,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "001",
  "part": 1,
  "start": 2197.61,
  "end": 3885.51,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM3",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 2221.51,
  "end": 2967.01,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM1",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "001",
  "part": 1,
  "start": 2974.81,
  "end": 3009.81,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM1",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3018.81,
  "end": 3041.71,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM1",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3053.41,
  "end": 3084.21,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM1",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3094.91,
  "end": 3103.31,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM1",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3105.41,
  "end": 3109.81,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM1",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3134.51,
  "end": 3169.81,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM1",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3179.11,
  "end": 3240.71,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM1",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3251.01,
  "end": 3930.31,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM1",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3442.51,
  "end": 3664.81,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3933.91,
  "end": 5512.31,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM3",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3939.41,
  "end": 4021.71,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM1",
  "commercial": "Force Bipolar"
 },
 {
  "case": "001",
  "part": 1,
  "start": 3957.31,
  "end": 4522.41,
  "label": "tip_up_fenestrated_grasper",
  "display": "Tip-up fenestrated grasper",
  "arm": "USM4",
  "commercial": "Tip-Up Fenestrated Grasper"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4029.11,
  "end": 4339.81,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4347.11,
  "end": 4516.91,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM1",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4524.31,
  "end": 4876.01,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM1",
  "commercial": "Force Bipolar"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4551.41,
  "end": 4561.21,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 45"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4565.31,
  "end": 4572.31,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 45"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4599.51,
  "end": 4606.61,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 45"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4628.11,
  "end": 4842.41,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 45"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4883.21,
  "end": 5502.31,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4944.21,
  "end": 5520.11,
  "label": "tip_up_fenestrated_grasper",
  "display": "Tip-up fenestrated grasper",
  "arm": "USM4",
  "commercial": "Tip-Up Fenestrated Grasper"
 },
 {
  "case": "001",
  "part": 1,
  "start": 5989.01,
  "end": 7459.0,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6000.21,
  "end": 7459.0,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6004.71,
  "end": 6545.71,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM3",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6562.81,
  "end": 6585.71,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6597.51,
  "end": 6618.91,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6632.81,
  "end": 6638.91,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6652.71,
  "end": 6679.31,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM3",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6689.71,
  "end": 6706.71,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6718.51,
  "end": 6733.41,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6743.41,
  "end": 7459.0,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM3",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "002",
  "part": 1,
  "start": 0.0,
  "end": 303.23,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM1",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 0.0,
  "end": 19338.35,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM2",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 0.0,
  "end": 304.43,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM2",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 0.0,
  "end": 307.83,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM4",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 2609.68,
  "end": 3578.18,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 2616.98,
  "end": 4989.48,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 2624.08,
  "end": 4139.98,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 3811.78,
  "end": 7963.68,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 4151.08,
  "end": 4252.88,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 4263.38,
  "end": 5017.58,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 5002.38,
  "end": 5081.88,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM1",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 5027.98,
  "end": 6396.78,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 5126.18,
  "end": 7960.58,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM1",
  "commercial": "Force Bipolar"
 },
 {
  "case": "002",
  "part": 1,
  "start": 6446.18,
  "end": 7115.38,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 7132.48,
  "end": 7140.48,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 7142.78,
  "end": 7961.28,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 11443.08,
  "end": 12563.38,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 11514.78,
  "end": 12464.88,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 11548.38,
  "end": 12421.78,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 12452.28,
  "end": 13849.58,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 12548.28,
  "end": 13848.48,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 12580.38,
  "end": 13851.18,
  "label": "grasping_retractor",
  "display": "Grasping retractor",
  "arm": "USM4",
  "commercial": "Small Grasping Retractor"
 },
 {
  "case": "002",
  "part": 1,
  "start": 13881.38,
  "end": 14038.98,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 13887.18,
  "end": 14764.08,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 13894.58,
  "end": 14759.18,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 14056.68,
  "end": 14131.78,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 14144.08,
  "end": 14812.58,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 14775.28,
  "end": 14796.08,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM4",
  "commercial": "Mega Needle Driver"
 },
 {
  "case": "002",
  "part": 1,
  "start": 14784.78,
  "end": 17221.48,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM1",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 14805.98,
  "end": 15418.98,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 14851.78,
  "end": 17229.08,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM2",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "002",
  "part": 1,
  "start": 15442.48,
  "end": 15461.98,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 1,
  "start": 15498.18,
  "end": 15706.98,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 1,
  "start": 15748.38,
  "end": 15767.48,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 1,
  "start": 15786.68,
  "end": 15903.38,
  "label": "vessel_sealer",
  "display": "Vessel sealer",
  "arm": "USM4",
  "commercial": "Vessel Sealer Extend"
 },
 {
  "case": "002",
  "part": 1,
  "start": 16183.28,
  "end": 16521.38,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 16542.78,
  "end": 16553.58,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "002",
  "part": 1,
  "start": 16604.18,
  "end": 16612.08,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "002",
  "part": 1,
  "start": 16638.88,
  "end": 16813.28,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "002",
  "part": 1,
  "start": 17017.98,
  "end": 17153.28,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "002",
  "part": 1,
  "start": 17209.88,
  "end": 18058.48,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 1,
  "start": 17276.48,
  "end": 19338.35,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM2",
  "commercial": "Force Bipolar"
 },
 {
  "case": "002",
  "part": 1,
  "start": 18088.38,
  "end": 19338.35,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM4",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "002",
  "part": 2,
  "start": 0.0,
  "end": 95.83,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM2",
  "commercial": ""
 },
 {
  "case": "002",
  "part": 2,
  "start": 0.0,
  "end": 1342.0,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM2",
  "commercial": ""
 },
 {
  "case": "002",
  "part": 2,
  "start": 0.0,
  "end": 70.83,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM4",
  "commercial": ""
 },
 {
  "case": "002",
  "part": 2,
  "start": 228.03,
  "end": 623.03,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 2,
  "start": 254.73,
  "end": 357.33,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM2",
  "commercial": "Force Bipolar"
 },
 {
  "case": "002",
  "part": 2,
  "start": 281.63,
  "end": 1342.0,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM1",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "002",
  "part": 2,
  "start": 417.73,
  "end": 1342.0,
  "label": "force_bipolar",
  "display": "Force bipolar",
  "arm": "USM2",
  "commercial": "Force Bipolar"
 },
 {
  "case": "002",
  "part": 2,
  "start": 639.83,
  "end": 654.23,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 2,
  "start": 662.23,
  "end": 697.43,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 2,
  "start": 710.23,
  "end": 737.73,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 2,
  "start": 745.83,
  "end": 765.23,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "002",
  "part": 2,
  "start": 784.43,
  "end": 805.23,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 2,
  "start": 834.93,
  "end": 849.73,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 2,
  "start": 904.63,
  "end": 918.33,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "002",
  "part": 2,
  "start": 931.13,
  "end": 1342.0,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM4",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "003",
  "part": 1,
  "start": 0.0,
  "end": 44.28,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 0.0,
  "end": 535.03,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 0.0,
  "end": 100.88,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 53.28,
  "end": 530.85,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 117.92,
  "end": 155.42,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 162.77,
  "end": 552.42,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 2585.65,
  "end": 3998.42,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 2589.38,
  "end": 4995.0,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 2598.03,
  "end": 4420.53,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 4005.6,
  "end": 7717.2,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 4429.02,
  "end": 4512.43,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 1,
  "start": 4521.73,
  "end": 4984.25,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 5003.93,
  "end": 6210.65,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 1,
  "start": 5016.07,
  "end": 6197.87,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Fenestrated Bipolar Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 6205.97,
  "end": 6932.53,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 6219.5,
  "end": 6932.32,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 6944.25,
  "end": 8760.1,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 6948.95,
  "end": 8762.5,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 1,
  "start": 7726.6,
  "end": 9072.9,
  "label": "grasping_retractor",
  "display": "Grasping retractor",
  "arm": "USM4",
  "commercial": "Small Grasping Retractor"
 },
 {
  "case": "003",
  "part": 1,
  "start": 8785.6,
  "end": 9043.2,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 8791.7,
  "end": 9045.6,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 9060.7,
  "end": 9558.7,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 9065.8,
  "end": 9127.0,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 9081.2,
  "end": 9564.5,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 9135.3,
  "end": 9209.4,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 1,
  "start": 9214.4,
  "end": 9556.8,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 1,
  "start": 12081.8,
  "end": 15527.7,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 12084.1,
  "end": 13385.0,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 1,
  "start": 12088.0,
  "end": 15520.8,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 13394.5,
  "end": 13424.6,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 13431.4,
  "end": 13461.3,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 13478.9,
  "end": 13517.6,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 13528.0,
  "end": 13549.7,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 13556.4,
  "end": 13585.3,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 13639.7,
  "end": 14112.1,
  "label": "vessel_sealer",
  "display": "Vessel sealer",
  "arm": "USM3",
  "commercial": "Vessel Sealer Extend"
 },
 {
  "case": "003",
  "part": 1,
  "start": 14417.7,
  "end": 14655.7,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "003",
  "part": 1,
  "start": 14698.0,
  "end": 14949.7,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "003",
  "part": 1,
  "start": 15122.7,
  "end": 15514.0,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "003",
  "part": 1,
  "start": 15905.5,
  "end": 16612.4,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM4",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "003",
  "part": 1,
  "start": 15915.1,
  "end": 17462.93,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM2",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "003",
  "part": 1,
  "start": 16629.9,
  "end": 16656.1,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 16661.8,
  "end": 16680.9,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 16688.2,
  "end": 16708.0,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 16724.9,
  "end": 16748.4,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "003",
  "part": 1,
  "start": 16761.4,
  "end": 16828.2,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 1,
  "start": 16843.6,
  "end": 17462.93,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM4",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "003",
  "part": 2,
  "start": 0.0,
  "end": 210.06,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM2",
  "commercial": ""
 },
 {
  "case": "003",
  "part": 2,
  "start": 0.0,
  "end": 204.16,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM4",
  "commercial": ""
 },
 {
  "case": "003",
  "part": 2,
  "start": 214.86,
  "end": 445.56,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "003",
  "part": 2,
  "start": 251.06,
  "end": 611.26,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM2",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "003",
  "part": 2,
  "start": 459.26,
  "end": 584.96,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 2,
  "start": 596.56,
  "end": 1471.06,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM4",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "003",
  "part": 2,
  "start": 622.76,
  "end": 878.56,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM2",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "003",
  "part": 2,
  "start": 906.56,
  "end": 1491.76,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM2",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "003",
  "part": 2,
  "start": 1476.66,
  "end": 1760.16,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "003",
  "part": 2,
  "start": 1501.66,
  "end": 1506.66,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM2",
  "commercial": "Fenestrated Bipolar Forceps"
 },
 {
  "case": "003",
  "part": 2,
  "start": 1509.16,
  "end": 1767.96,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM2",
  "commercial": "Fenestrated Bipolar Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 0.0,
  "end": 211.29,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 0.0,
  "end": 206.85,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 0.0,
  "end": 203.67,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 2499.12,
  "end": 5087.12,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 2561.72,
  "end": 3641.82,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 2594.52,
  "end": 4347.52,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 3657.32,
  "end": 8334.52,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 4362.82,
  "end": 4403.92,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "004",
  "part": 1,
  "start": 4414.22,
  "end": 5109.22,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 5095.32,
  "end": 6681.22,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Fenestrated Bipolar Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 5124.52,
  "end": 6670.82,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "004",
  "part": 1,
  "start": 6677.42,
  "end": 7736.12,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 6702.92,
  "end": 8297.12,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 7746.02,
  "end": 7783.02,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "004",
  "part": 1,
  "start": 7792.62,
  "end": 8316.72,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 8309.22,
  "end": 9105.42,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 8332.92,
  "end": 9111.02,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "004",
  "part": 1,
  "start": 8345.82,
  "end": 9120.12,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 11576.12,
  "end": 12645.92,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 11598.92,
  "end": 12623.42,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "004",
  "part": 1,
  "start": 11602.62,
  "end": 12834.32,
  "label": "grasping_retractor",
  "display": "Grasping retractor",
  "arm": "USM4",
  "commercial": "Small Grasping Retractor"
 },
 {
  "case": "004",
  "part": 1,
  "start": 12633.62,
  "end": 12815.32,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 12658.02,
  "end": 12810.92,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 12829.82,
  "end": 13831.32,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 12832.82,
  "end": 13834.02,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "004",
  "part": 1,
  "start": 12850.02,
  "end": 16322.88,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 13845.62,
  "end": 16322.88,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "004",
  "part": 1,
  "start": 13852.32,
  "end": 15940.62,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "004",
  "part": 1,
  "start": 15950.52,
  "end": 15976.52,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "004",
  "part": 1,
  "start": 15988.02,
  "end": 16033.62,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "004",
  "part": 1,
  "start": 16045.32,
  "end": 16078.42,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "004",
  "part": 1,
  "start": 16084.42,
  "end": 16153.72,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "004",
  "part": 1,
  "start": 16163.42,
  "end": 16202.82,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM3",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "004",
  "part": 2,
  "start": 0.0,
  "end": 1593.54,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": ""
 },
 {
  "case": "004",
  "part": 2,
  "start": 0.0,
  "end": 924.64,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": ""
 },
 {
  "case": "004",
  "part": 2,
  "start": 17.84,
  "end": 927.74,
  "label": "vessel_sealer",
  "display": "Vessel sealer",
  "arm": "USM3",
  "commercial": "Vessel Sealer Extend"
 },
 {
  "case": "004",
  "part": 2,
  "start": 934.94,
  "end": 1036.54,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM3",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "004",
  "part": 2,
  "start": 1002.84,
  "end": 1327.94,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "004",
  "part": 2,
  "start": 1430.64,
  "end": 1517.74,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM4",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "004",
  "part": 2,
  "start": 2124.54,
  "end": 2182.14,
  "label": "permanent_cautery_hook_spatula",
  "display": "Permanent cautery hook/spatula",
  "arm": "USM4",
  "commercial": "Permanent Cautery Hook"
 },
 {
  "case": "004",
  "part": 2,
  "start": 2137.14,
  "end": 2316.04,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM1",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 0.0,
  "end": 472.09,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 0.0,
  "end": 17982.0,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 0.0,
  "end": 191.59,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 0.0,
  "end": 32.66,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 37.54,
  "end": 471.89,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 194.84,
  "end": 474.61,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 2174.71,
  "end": 4648.77,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 2176.96,
  "end": 3349.56,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 2184.39,
  "end": 3362.54,
  "label": "prograsp_forceps",
  "display": "Prograsp forceps",
  "arm": "USM4",
  "commercial": "ProGrasp Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 3355.31,
  "end": 3546.91,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 3366.19,
  "end": 7050.99,
  "label": "cadiere_forceps",
  "display": "Cadiere forceps",
  "arm": "USM4",
  "commercial": "Cadiere Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 3552.04,
  "end": 4416.01,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 4422.81,
  "end": 4633.84,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 4647.27,
  "end": 6268.79,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 4673.01,
  "end": 6227.09,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Fenestrated Bipolar Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 6234.79,
  "end": 7048.89,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 6276.09,
  "end": 7048.49,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 9416.69,
  "end": 10992.39,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 9418.19,
  "end": 11891.59,
  "label": "grasping_retractor",
  "display": "Grasping retractor",
  "arm": "USM4",
  "commercial": "Small Grasping Retractor"
 },
 {
  "case": "005",
  "part": 1,
  "start": 9422.99,
  "end": 11000.19,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 10999.39,
  "end": 11244.59,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Mega SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11023.19,
  "end": 11782.59,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM1",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11248.29,
  "end": 11287.29,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11290.29,
  "end": 11782.89,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11799.59,
  "end": 11908.19,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11803.69,
  "end": 11893.59,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM1",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11901.89,
  "end": 13563.89,
  "label": "grasping_retractor",
  "display": "Grasping retractor",
  "arm": "USM1",
  "commercial": "Small Grasping Retractor"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11914.79,
  "end": 12514.79,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11950.89,
  "end": 13580.59,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM2",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 12528.09,
  "end": 12555.79,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "005",
  "part": 1,
  "start": 12562.69,
  "end": 12581.79,
  "label": "clip_applier",
  "display": "Clip applier",
  "arm": "USM4",
  "commercial": "Large Clip Applier"
 },
 {
  "case": "005",
  "part": 1,
  "start": 12589.59,
  "end": 12679.29,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM4",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 12774.89,
  "end": 13561.29,
  "label": "vessel_sealer",
  "display": "Vessel sealer",
  "arm": "USM4",
  "commercial": "Vessel Sealer Extend"
 },
 {
  "case": "005",
  "part": 1,
  "start": 13614.09,
  "end": 16331.29,
  "label": "grasping_retractor",
  "display": "Grasping retractor",
  "arm": "USM4",
  "commercial": "Small Grasping Retractor"
 },
 {
  "case": "005",
  "part": 1,
  "start": 14353.99,
  "end": 14590.09,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "Stapler 45"
 },
 {
  "case": "005",
  "part": 1,
  "start": 14613.19,
  "end": 14945.89,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 14954.39,
  "end": 15311.09,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 15340.89,
  "end": 15348.59,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "005",
  "part": 1,
  "start": 15352.59,
  "end": 15362.69,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "005",
  "part": 1,
  "start": 15396.29,
  "end": 15404.09,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "005",
  "part": 1,
  "start": 15454.79,
  "end": 15635.19,
  "label": "stapler",
  "display": "Stapler",
  "arm": "USM3",
  "commercial": "SureForm Stapler 60"
 },
 {
  "case": "005",
  "part": 1,
  "start": 15754.79,
  "end": 16343.19,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large SutureCut Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 16349.69,
  "end": 16750.69,
  "label": "bipolar_forceps",
  "display": "Bipolar forceps",
  "arm": "USM3",
  "commercial": "Maryland Bipolar Forceps"
 },
 {
  "case": "005",
  "part": 1,
  "start": 16759.09,
  "end": 16922.79,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 },
 {
  "case": "005",
  "part": 1,
  "start": 16926.19,
  "end": 17078.89,
  "label": "needle_driver",
  "display": "Needle driver",
  "arm": "USM3",
  "commercial": "Large Needle Driver"
 },
 {
  "case": "005",
  "part": 1,
  "start": 17090.49,
  "end": 17437.39,
  "label": "monopolar_curved_scissors",
  "display": "Monopolar curved scissors",
  "arm": "USM3",
  "commercial": "Monopolar Curved Scissors"
 }
];

export const SURGVU_TASK_INTERVALS: SurgvuTaskInterval[] = [
 {
  "case": "000",
  "part": 1,
  "start": 1773.3,
  "end": 2337.17,
  "label": "skills_application",
  "display": "General skills application"
 },
 {
  "case": "001",
  "part": 1,
  "start": 226.19,
  "end": 851.42,
  "label": "uterine_horn",
  "display": "Uterine horn"
 },
 {
  "case": "001",
  "part": 1,
  "start": 1237.15,
  "end": 1748.61,
  "label": "suspensory_ligaments",
  "display": "Suspensory ligaments"
 },
 {
  "case": "001",
  "part": 1,
  "start": 2348.32,
  "end": 3280.3,
  "label": "rectal_artery_vein",
  "display": "Rectal artery/vein manipulation"
 },
 {
  "case": "001",
  "part": 1,
  "start": 4369.08,
  "end": 5407.21,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "001",
  "part": 1,
  "start": 6362.58,
  "end": 7375.07,
  "label": "skills_application",
  "display": "General skills application"
 },
 {
  "case": "002",
  "part": 1,
  "start": 30.2,
  "end": 162.3,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "002",
  "part": 1,
  "start": 3226.25,
  "end": 3389.32,
  "label": "retraction_collision_avoidance",
  "display": "Retraction and collision avoidance"
 },
 {
  "case": "002",
  "part": 1,
  "start": 4287.68,
  "end": 4837.73,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "002",
  "part": 1,
  "start": 5661.94,
  "end": 6884.33,
  "label": "uterine_horn",
  "display": "Uterine horn"
 },
 {
  "case": "002",
  "part": 1,
  "start": 7380.65,
  "end": 7837.91,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "002",
  "part": 1,
  "start": 11915.29,
  "end": 12397.29,
  "label": "range_of_motion",
  "display": "Range of motion"
 },
 {
  "case": "002",
  "part": 1,
  "start": 12700.99,
  "end": 13350.19,
  "label": "suspensory_ligaments",
  "display": "Suspensory ligaments"
 },
 {
  "case": "002",
  "part": 1,
  "start": 14198.29,
  "end": 14612.39,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "002",
  "part": 1,
  "start": 15210.89,
  "end": 15839.79,
  "label": "rectal_artery_vein",
  "display": "Rectal artery/vein manipulation"
 },
 {
  "case": "002",
  "part": 1,
  "start": 16192.29,
  "end": 16443.29,
  "label": "rectal_artery_vein",
  "display": "Rectal artery/vein manipulation"
 },
 {
  "case": "002",
  "part": 1,
  "start": 17354.09,
  "end": 19338.35,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "002",
  "part": 2,
  "start": 0.0,
  "end": 4.16,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "002",
  "part": 2,
  "start": 409.48,
  "end": 1287.65,
  "label": "skills_application",
  "display": "General skills application"
 },
 {
  "case": "003",
  "part": 1,
  "start": 4533.28,
  "end": 4849.9,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "003",
  "part": 1,
  "start": 5413.05,
  "end": 6087.3,
  "label": "uterine_horn",
  "display": "Uterine horn"
 },
 {
  "case": "003",
  "part": 1,
  "start": 6451.35,
  "end": 7031.61,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "003",
  "part": 1,
  "start": 8267.91,
  "end": 8547.68,
  "label": "suspensory_ligaments",
  "display": "Suspensory ligaments"
 },
 {
  "case": "003",
  "part": 1,
  "start": 9173.86,
  "end": 9441.29,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "003",
  "part": 1,
  "start": 12677.46,
  "end": 13666.08,
  "label": "rectal_artery_vein",
  "display": "Rectal artery/vein manipulation"
 },
 {
  "case": "004",
  "part": 1,
  "start": 22.41,
  "end": 194.14,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "004",
  "part": 1,
  "start": 3368.21,
  "end": 3499.92,
  "label": "retraction_collision_avoidance",
  "display": "Retraction and collision avoidance"
 },
 {
  "case": "004",
  "part": 1,
  "start": 4436.54,
  "end": 5053.84,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "004",
  "part": 1,
  "start": 5574.59,
  "end": 6632.51,
  "label": "uterine_horn",
  "display": "Uterine horn"
 },
 {
  "case": "004",
  "part": 1,
  "start": 7799.68,
  "end": 8260.02,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "004",
  "part": 1,
  "start": 8865.37,
  "end": 9098.08,
  "label": "range_of_motion",
  "display": "Range of motion"
 },
 {
  "case": "004",
  "part": 1,
  "start": 11979.15,
  "end": 12582.45,
  "label": "suspensory_ligaments",
  "display": "Suspensory ligaments"
 },
 {
  "case": "004",
  "part": 1,
  "start": 13486.25,
  "end": 13814.45,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "004",
  "part": 1,
  "start": 14678.25,
  "end": 16322.65,
  "label": "rectal_artery_vein",
  "display": "Rectal artery/vein manipulation"
 },
 {
  "case": "004",
  "part": 2,
  "start": 2457.82,
  "end": 3151.11,
  "label": "skills_application",
  "display": "General skills application"
 },
 {
  "case": "005",
  "part": 1,
  "start": 340.33,
  "end": 469.88,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "005",
  "part": 1,
  "start": 3060.28,
  "end": 3196.68,
  "label": "retraction_collision_avoidance",
  "display": "Retraction and collision avoidance"
 },
 {
  "case": "005",
  "part": 1,
  "start": 4112.19,
  "end": 4585.09,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "005",
  "part": 1,
  "start": 5293.66,
  "end": 6157.35,
  "label": "uterine_horn",
  "display": "Uterine horn"
 },
 {
  "case": "005",
  "part": 1,
  "start": 6500.27,
  "end": 6916.05,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "005",
  "part": 1,
  "start": 10250.08,
  "end": 10681.38,
  "label": "suspensory_ligaments",
  "display": "Suspensory ligaments"
 },
 {
  "case": "005",
  "part": 1,
  "start": 11323.28,
  "end": 11770.88,
  "label": "suturing",
  "display": "Suturing"
 },
 {
  "case": "005",
  "part": 1,
  "start": 12189.18,
  "end": 12604.58,
  "label": "rectal_artery_vein",
  "display": "Rectal artery/vein manipulation"
 }
];
