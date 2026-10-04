import React from 'react';
import { AlertTriangle, CheckCircle2, Loader2, RefreshCw, X, Sparkles } from 'lucide-react';
import { ApiFailure, ApiHealth } from '../types';

interface ApiStatusBannerProps {
  health: ApiHealth | null;
  lastFailure: ApiFailure | null;
  isChecking: boolean;
  onRecheck: () => void;
  onDismiss: () => void;
  onSelectModel?: (modelId: string) => void;
}

/**
 * Say out loud when the vision API is not working or quota has been reached.
 */
export const ApiStatusBanner: React.FC<ApiStatusBannerProps> = ({
  health,
  lastFailure,
  isChecking,
  onRecheck,
  onDismiss,
  onSelectModel,
}) => {
  const DEVICE_FAILURES: Record<string, string> = {
    MICROPHONE: 'Microphone unavailable',
    CAMERA: 'Camera unavailable',
    TRANSCRIPTION: 'Could not transcribe that recording',
    NO_VIDEO: 'No video loaded',
    VIDEO_URL: 'That URL could not be played',
  };

  const isQuota =
    lastFailure?.code === 429 ||
    lastFailure?.status === 'RESOURCE_EXHAUSTED' ||
    Boolean(health?.quotaExceeded);

  // A live request failure outranks a stale health probe.
  const problem: { title: string; detail: string; hint?: string } | null = lastFailure
    ? {
        title:
          DEVICE_FAILURES[lastFailure.status] ||
          (isQuota
            ? 'Gemini Model Quota / Rate Limit Reached (429)'
            : `Gemini call failed — ${lastFailure.status} ${lastFailure.code || ''}`.trim()),
        detail: lastFailure.message,
        hint: isQuota
          ? 'You have reached the per-minute or daily quota for this model. Switch to Gemini 3.6 Flash or 3.1 Flash Lite below, or wait a moment.'
          : lastFailure.hint,
      }
    : health && !health.ok
      ? {
          title: health.quotaExceeded
            ? `Quota Reached on ${health.model || 'selected model'}`
            : health.keyPresent
              ? 'Gemini is configured but not usable'
              : 'No Gemini API key on the server',
          detail: health.reason || 'The server could not reach the Gemini API.',
          hint: health.hint,
        }
      : null;

  if (!problem) {
    if (!health) return null;
    return (
      <div className="flex items-center justify-between gap-2 px-3 py-1.5 text-[11px] text-emerald-300/90 bg-emerald-950/30 border border-emerald-900/50 rounded-md">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" />
          <span>
            API reachable{health.model ? ` · ${health.model}` : ''}. Assistant and vision live.
          </span>
        </div>
        <button
          onClick={onRecheck}
          disabled={isChecking}
          className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium text-emerald-300 bg-emerald-950/60 hover:bg-emerald-900/60 border border-emerald-800/60 rounded transition-colors disabled:opacity-50"
          title="Re-run health probe"
        >
          {isChecking ? <Loader2 className="w-2.5 h-2.5 animate-spin" /> : <RefreshCw className="w-2.5 h-2.5" />}
          <span>Check</span>
        </button>
      </div>
    );
  }

  const suggestedFallback = health?.suggestedModel || (isQuota ? 'gemini-3.6-flash' : undefined);

  return (
    <div className="flex items-start gap-2.5 px-3 py-2.5 bg-red-950/40 border border-red-900/60 rounded-md">
      <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />

      <div className="flex-1 min-w-0 space-y-1">
        <div className="flex items-center gap-2 flex-wrap">
          <p className="text-xs font-semibold text-red-300">{problem.title}</p>
          {isQuota && (
            <span className="text-[9px] px-1.5 py-0.2 rounded bg-amber-950 text-amber-300 border border-amber-800/80 uppercase font-mono">
              Rate limited
            </span>
          )}
        </div>
        <p className="text-[11px] text-slate-300 leading-snug break-words">{problem.detail}</p>
        {problem.hint && (
          <p className="text-[11px] text-slate-400 leading-snug break-words">{problem.hint}</p>
        )}

        {/* Quick fallback switch buttons if quota exceeded */}
        {isQuota && onSelectModel && (
          <div className="flex items-center gap-2 pt-1 flex-wrap">
            <span className="text-[10px] text-slate-400 font-medium">Switch active model:</span>
            {suggestedFallback && (
              <button
                onClick={() => {
                  onSelectModel(suggestedFallback);
                  onRecheck();
                }}
                className="flex items-center gap-1 px-2 py-0.5 text-[10px] font-medium text-cyan-200 bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-700/80 rounded transition-colors"
              >
                <Sparkles className="w-2.5 h-2.5 text-cyan-400" />
                <span>Switch to {suggestedFallback}</span>
              </button>
            )}
            <button
              onClick={() => {
                onSelectModel('gemini-3.1-flash-lite');
                onRecheck();
              }}
              className="px-2 py-0.5 text-[10px] font-medium text-slate-300 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded transition-colors"
            >
              3.1 Flash Lite
            </button>
          </div>
        )}

        <p className="text-[10px] text-slate-500 pt-0.5">
          On-device detection and the recorded dataset labels are unaffected by this — they do not
          use the network.
        </p>
      </div>

      <div className="flex items-center gap-1 flex-shrink-0">
        <button
          onClick={onRecheck}
          disabled={isChecking}
          className="flex items-center gap-1 px-2 py-1 text-[10px] font-medium text-slate-200 bg-slate-800/80 hover:bg-slate-700 border border-slate-700 rounded transition-colors disabled:opacity-50"
        >
          {isChecking ? (
            <Loader2 className="w-3 h-3 animate-spin" />
          ) : (
            <RefreshCw className="w-3 h-3" />
          )}
          <span>Re-check</span>
        </button>
        <button
          onClick={onDismiss}
          className="p-1 text-slate-500 hover:text-slate-300 transition-colors"
          title="Dismiss"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
