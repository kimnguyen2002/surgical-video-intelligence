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
      <div className="bg-card p-3.5 rounded-lg border border-line flex flex-col space-y-3">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-fg">
          <ShieldAlert className="w-4 h-4 text-risk" />
          <span>Anatomy at Risk</span>
          {activeRisks.length > 0 && (
            <span className="ml-auto text-[11px] font-normal text-fg2">
              {activeRisks.length} active
            </span>
          )}
        </div>

        {activeRisks.length === 0 ? (
          <div className="p-3 text-center text-xs text-fg2 italic bg-card rounded border border-line leading-snug">
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
                  className={`p-2.5 rounded-md border ${colors.border} bg-card flex flex-col space-y-1.5 transition-all hover:brightness-105`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className={`text-xs font-bold ${colors.text} lowercase`}>
                      {risk.structure}
                    </span>

                    <span className="flex items-center gap-1 flex-shrink-0">
                      {/* Which signal raised this, in the app's usual colours:
                          green for a dataset fact, amber for a model output. */}
                      <span
                        className={`text-[11px] font-semibold px-1.5 py-0.5 rounded border ${
                          isRecorded
                            ? 'bg-ok-soft text-ok border-ok-line'
                            : 'bg-warn-soft text-warn border-warn-line'
                        }`}
                      >
                        {isRecorded ? 'recorded task' : 'detected tool'}
                      </span>
                      <span
                        className={`text-[11px] font-bold px-1.5 py-0.5 rounded tracking-wider ${colors.badgeBg} ${colors.badgeText} border border-current`}
                      >
                        {risk.level}
                      </span>
                    </span>
                  </div>

                  <p className="text-[11px] text-fg2 italic leading-snug">
                    Shown because {risk.reason}.
                  </p>

                  <p className="text-xs text-fg2 leading-snug">{risk.mechanism}</p>

                  <p className="text-xs text-fg leading-snug">
                    <span className="font-semibold text-fg">Protect:</span> {risk.protect}
                  </p>
                </div>
              );
            })}
          </div>
        )}

        <p className="text-[11px] text-fg2 italic pt-1 border-t border-line leading-tight">
          Educational references keyed to the recorded task and the instruments on screen — not an
          observation of this subject's anatomy, and not a clinical protocol.
        </p>
      </div>
      {/* SECTION 2: PREDICTED — the live model output */}
      <div className="bg-card p-3.5 rounded-lg border border-line flex flex-col space-y-3">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-fg">
          <Eye className="w-4 h-4 text-warn" />
          <span>Predicted (model)</span>
          {currentObservation?.modelUsed && (
            <span className="ml-auto text-[11px] font-normal text-fg2 font-mono">
              {currentObservation.modelUsed}
            </span>
          )}
        </div>

        {/* Scene summary */}
        <div className="p-2.5 rounded-md bg-card border border-line text-xs text-fg leading-relaxed">
          {currentObservation?.summary ||
            (hasVideo
              ? 'No frame has been analysed yet. Press Analyse frames (live).'
              : 'No video is loaded, so there is nothing to analyse.')}
        </div>

        {/* Detected Instruments List */}
        <div className="space-y-2">
          {instrumentsToDisplay.length === 0 ? (
            <div className="p-3 text-center text-xs text-fg2 italic bg-card rounded border border-line">
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
                      ? 'border-warn-line bg-warn-soft ring-1 ring-warn-line'
                      : 'border-line bg-card hover:border-line hover:bg-hover'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-warn">{inst.label}</span>

                    {/* A prediction without a percentage would be
                        indistinguishable from a recorded fact. */}
                    <span className="text-xs font-bold text-warn flex-shrink-0">
                      {inst.confidence !== undefined ? `${inst.confidence}%` : 'no confidence given'}
                    </span>
                  </div>

                  {inst.description && (
                    <p className="text-xs text-fg2 mt-1 leading-snug">
                      {inst.description}
                    </p>
                  )}

                  <div className="mt-1.5 text-[11px] text-warn">
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
