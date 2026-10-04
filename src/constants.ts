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
    badgeBg: 'bg-risk-soft',
    badgeText: 'text-risk',
    border: 'border-risk-line',
    text: 'text-risk',
    accent: '#ef4444',
  },
  CAUTION: {
    badgeBg: 'bg-warn-soft',
    badgeText: 'text-warn',
    border: 'border-warn-line',
    text: 'text-warn',
    accent: '#f59e0b',
  },
  WARNING: {
    badgeBg: 'bg-warn-soft',
    badgeText: 'text-warn',
    border: 'border-warn-line',
    text: 'text-warn',
    accent: '#eab308',
  },
  INFO: {
    badgeBg: 'bg-accent-soft',
    badgeText: 'text-accent',
    border: 'border-accent/40',
    text: 'text-accent',
    accent: '#3b82f6',
  },
};
