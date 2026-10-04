import { AssistantRole, SpecialtyId } from './types';

/*
 * Two lists used to live here and are gone:
 *
 * `GEMINI_MODELS` hardcoded model ids for a dropdown. The app now reads the
 * catalogue from the visitor's own key at runtime, so the list cannot go stale
 * and a retired id cannot turn into a 404 at the worst moment.
 *
 * `ANALYSIS_INTERVALS` rationed how often a frame was analysed, because every
 * tick was a billed request. Inference runs on-device now, so there is nothing
 * left to ration and the control would only slow the app down on purpose.
 */

export const SPECIALTIES_LIST: { id: SpecialtyId; name: string }[] = [
  { id: 'general', name: 'General Surgery' },
  { id: 'bariatric', name: 'Bariatric Surgery' },
  { id: 'colorectal', name: 'Colorectal Surgery' },
  { id: 'gynecology', name: 'Gynecologic Surgery' },
  { id: 'urology', name: 'Urologic Surgery' },
  { id: 'cardiothoracic', name: 'Cardiothoracic Surgery' },
  { id: 'orthopedic', name: 'Orthopedic Surgery' },
  { id: 'neurosurgery', name: 'Neurosurgery' },
  { id: 'ent', name: 'ENT / Otolaryngology' },
  { id: 'plastics', name: 'Plastic Surgery' },
  { id: 'pediatric', name: 'Pediatric Surgery' },
  { id: 'vascular', name: 'Vascular Surgery' },
];

export const ASSISTANT_ROLES: AssistantRole[] = [
  'Student',
  'Resident',
  'Fellow',
  'Researcher',
  'AI Engineer',
];

export const RISK_COLOR_MAP = {
  CRITICAL: {
    badgeBg: 'bg-red-950/80',
    badgeText: 'text-red-400',
    border: 'border-red-900/60',
    text: 'text-red-300',
    accent: '#ef4444',
  },
  CAUTION: {
    badgeBg: 'bg-amber-950/80',
    badgeText: 'text-amber-400',
    border: 'border-amber-900/60',
    text: 'text-amber-300',
    accent: '#f59e0b',
  },
  WARNING: {
    badgeBg: 'bg-yellow-950/80',
    badgeText: 'text-yellow-400',
    border: 'border-yellow-900/60',
    text: 'text-yellow-300',
    accent: '#eab308',
  },
  INFO: {
    badgeBg: 'bg-blue-950/80',
    badgeText: 'text-blue-400',
    border: 'border-blue-900/60',
    text: 'text-blue-300',
    accent: '#3b82f6',
  },
};
