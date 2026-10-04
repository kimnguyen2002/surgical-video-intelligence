import React from 'react';
import { ShieldAlert, Eye } from 'lucide-react';
import { FrameObservation } from '../types';
import { RISK_COLOR_MAP } from '../constants';
import { ActiveRisk, risksFor } from '../data/riskRules';

interface PanelsProps {
  currentObservation: FrameObservation | null;
  focusedInstrumentId: string | null;
  /** Whether anything is on the stage, so "no detections" can be explained. */
  hasVideo: boolean;
  /** The SurgVU task recorded at the playhead, if a library video is loaded. */
  recordedTaskLabel: string | null;
  recordedTaskDisplay: string | null;
  onSelectInstrument: (id: string) => void;
}

export const Panels: React.FC<PanelsProps> = ({
  currentObservation,
  focusedInstrumentId,
  hasVideo,
  recordedTaskLabel,
  recordedTaskDisplay,
  onSelectInstrument,
}) => {
  const instrumentsToDisplay = currentObservation?.instruments || [];

  /**
   * Risks are computed from the frame, not looked up from a fixed catalogue.
   *
   * Detected labels come back as display strings, so they are normalised back
   * to the snake_case class ids the rules are keyed on.
   */
  const activeRisks: ActiveRisk[] = hasVideo
    ? risksFor({
        recordedTaskLabel,
        recordedTaskDisplay,
        detectedToolLabels: instrumentsToDisplay.map((i) =>
          i.label.toLowerCase().replace(/[\/\s-]+/g, '_')
        ),
      })
    : [];

  return (
    <div className="flex flex-col space-y-4">
      {/* SECTION 1: ANATOMY AT RISK — recomputed from the frame, first
          because it is the thing a surgeon needs to see without hunting. */}
      <div className="bg-[#0b1017] p-3.5 rounded-lg border border-slate-800/80 flex flex-col space-y-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <ShieldAlert className="w-4 h-4 text-red-400" />
          <span>Anatomy at Risk</span>
          {activeRisks.length > 0 && (
            <span className="ml-auto text-[9px] font-normal normal-case text-slate-500 tracking-normal">
              {activeRisks.length} active
            </span>
          )}
        </div>

        {activeRisks.length === 0 ? (
          <div className="p-3 text-center text-xs text-slate-500 italic bg-[#0e141f] rounded border border-slate-800 leading-snug">
            {!hasVideo
              ? 'No video loaded. This panel follows the task recorded in the video and the instruments on screen, so there is nothing to show yet.'
              : recordedTaskLabel
                ? 'Nothing specific to this task and no instrument on screen that raises a hazard.'
                : 'No task is recorded at this timestamp and no instrument is detected. Start live detection, or seek to a labelled part of the case.'}
          </div>
        ) : (
          <div className="space-y-2.5">
            {activeRisks.map((risk) => {
              const colors = RISK_COLOR_MAP[risk.level] || RISK_COLOR_MAP.CAUTION;
              const isRecorded = risk.source === 'recorded';

              return (
                <div
                  key={risk.id}
                  id={`risk-card-${risk.id}`}
                  className={`p-2.5 rounded-md border ${colors.border} bg-[#0e141f] flex flex-col space-y-1.5 transition-all hover:brightness-105`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className={`text-xs font-bold ${colors.text} lowercase`}>
                      {risk.structure}
                    </span>

                    <span className="flex items-center gap-1 flex-shrink-0">
                      {/* Which signal raised this, in the app's usual colours:
                          green for a dataset fact, amber for a model output. */}
                      <span
                        className={`text-[9px] font-semibold px-1.5 py-0.5 rounded border ${
                          isRecorded
                            ? 'bg-emerald-950/70 text-emerald-400 border-emerald-700/70'
                            : 'bg-amber-950/70 text-amber-400 border-amber-700/70'
                        }`}
                      >
                        {isRecorded ? 'recorded task' : 'detected tool'}
                      </span>
                      <span
                        className={`text-[9px] font-bold px-1.5 py-0.5 rounded tracking-wider ${colors.badgeBg} ${colors.badgeText} border border-current`}
                      >
                        {risk.level}
                      </span>
                    </span>
                  </div>

                  <p className="text-[10px] text-slate-500 italic leading-snug">
                    Shown because {risk.reason}.
                  </p>

                  <p className="text-[11px] text-slate-300 leading-snug">{risk.mechanism}</p>

                  <p className="text-[11px] text-slate-200 leading-snug">
                    <span className="font-semibold text-slate-100">Protect:</span> {risk.protect}
                  </p>
                </div>
              );
            })}
          </div>
        )}

        <p className="text-[10px] text-slate-500 italic pt-1 border-t border-slate-800/60 leading-tight">
          Educational references keyed to the recorded task and the instruments on screen — not an
          observation of this subject's anatomy, and not a clinical protocol.
        </p>
      </div>
      {/* SECTION 2: PREDICTED — the live model output */}
      <div className="bg-[#0b1017] p-3.5 rounded-lg border border-slate-800/80 flex flex-col space-y-3">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <Eye className="w-4 h-4 text-amber-400" />
          <span>Predicted (model)</span>
          {currentObservation?.modelUsed && (
            <span className="ml-auto text-[9px] font-normal normal-case text-slate-500 tracking-normal font-mono">
              {currentObservation.modelUsed}
            </span>
          )}
        </div>

        {/* Scene summary */}
        <div className="p-2.5 rounded-md bg-[#0e141f] border border-slate-800 text-xs text-slate-200 leading-relaxed">
          {currentObservation?.summary ||
            (hasVideo
              ? 'No frame has been analysed yet. Press Analyse frames (live).'
              : 'No video is loaded, so there is nothing to analyse.')}
        </div>

        {/* Detected Instruments List */}
        <div className="space-y-2">
          {instrumentsToDisplay.length === 0 ? (
            <div className="p-3 text-center text-xs text-slate-500 italic bg-[#0e141f] rounded border border-slate-800">
              {currentObservation
                ? 'The model localized no instrument in this frame. An empty result is a real answer — check the recorded panel for what was actually installed.'
                : 'No predictions yet.'}
            </div>
          ) : (
            instrumentsToDisplay.map((inst) => {
              const isFocused = focusedInstrumentId === inst.id;

              return (
                <div
                  key={inst.id}
                  id={`observation-instrument-${inst.id}`}
                  onClick={() => onSelectInstrument(inst.id)}
                  className={`p-2.5 rounded-md border transition-all cursor-pointer ${
                    isFocused
                      ? 'border-amber-400 bg-amber-950/30 ring-1 ring-amber-400'
                      : 'border-slate-800/80 bg-[#0e141f] hover:border-slate-700 hover:bg-[#121927]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-amber-400">{inst.label}</span>

                    {/* A prediction without a percentage would be
                        indistinguishable from a recorded fact. */}
                    <span className="text-xs font-bold text-amber-400 flex-shrink-0">
                      {inst.confidence !== undefined ? `${inst.confidence}%` : 'no confidence given'}
                    </span>
                  </div>

                  {inst.description && (
                    <p className="text-[11px] text-slate-300 mt-1 leading-snug">
                      {inst.description}
                    </p>
                  )}

                  <div className="mt-1.5 text-[10px] text-amber-400/70">
                    {isFocused ? 'highlighted on the video' : 'click to highlight on the video'}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

    </div>
  );
};
