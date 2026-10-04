import React from 'react';
import { ShieldCheck, Info } from 'lucide-react';
import { RecordedContext } from '../types';
import { armColor } from '../data/surgvuVocab';
import { formatClock } from '../services/groundTruth';

interface GroundTruthPanelProps {
  recorded: RecordedContext | null;
  currentTime: number;
}

/**
 * What the dataset records at the playhead.
 *
 * Deliberately styled apart from the Observations panel: solid green, no
 * percentages, no "click to show attention". A confidence figure here would be
 * a category error — the release is not estimating that a stapler was
 * installed, it is reporting it.
 */
export const GroundTruthPanel: React.FC<GroundTruthPanelProps> = ({ recorded, currentTime }) => {
  if (!recorded) {
    return (
      <div className="bg-card p-3.5 rounded-lg border border-line flex flex-col space-y-2">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-fg">
          <ShieldCheck className="w-4 h-4 text-fg2" />
          <span>Recorded (dataset)</span>
        </div>
        <p className="text-xs text-fg2 italic leading-snug">
          No annotated video is loaded. Attach a SurgVU case from the library to see what the
          release records, frame by frame.
        </p>
      </div>
    );
  }

  const untrainable = recorded.tools.filter((t) => !t.trainable);

  return (
    <div className="bg-card p-3.5 rounded-lg border border-ok-line flex flex-col space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-fg">
          <ShieldCheck className="w-4 h-4 text-ok" />
          <span>Recorded (dataset)</span>
        </div>
        <span className="text-[11px] font-mono text-fg2">
          case_{recorded.caseId} p{recorded.part} · {formatClock(currentTime)}
        </span>
      </div>

      {/* Labelled task */}
      <div className="flex items-center gap-2 text-xs">
        <span className="text-fg2">Task:</span>
        {recorded.taskDisplay ? (
          <span className="text-ok font-medium">{recorded.taskDisplay}</span>
        ) : (
          <span className="text-fg2 italic">no task labelled at this timestamp</span>
        )}
      </div>

      {/* Installed instruments */}
      {recorded.tools.length === 0 ? (
        <div className="p-2.5 text-xs text-fg2 italic bg-card rounded border border-line">
          The release records no instrument installed on any arm at this timestamp. That is a
          recorded fact, not a missing answer.
        </div>
      ) : (
        <div className="space-y-1.5">
          {recorded.tools.map((tool) => {
            const color = armColor(tool.arm);
            return (
              <div
                key={`${tool.arm}-${tool.label}`}
                className="p-2 rounded-md border border-ok-line bg-ok-soft flex items-start justify-between gap-2"
              >
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-ok flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: color.dot }}
                    />
                    <span className="truncate">{tool.display}</span>
                    {!tool.trainable && (
                      <span
                        className="text-[8px] px-1 py-px rounded bg-inset text-fg2 border border-line flex-shrink-0"
                        title="Outside the twelve classes the models are trained on"
                      >
                        untrained class
                      </span>
                    )}
                  </div>
                  {tool.commercial && (
                    <p className="text-[11px] text-fg2 mt-0.5 truncate">{tool.commercial}</p>
                  )}
                </div>
                <span className={`text-[11px] font-mono flex-shrink-0 ${color.text}`}>
                  {tool.arm}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {untrainable.length > 0 && (
        <p className="text-[11px] text-fg2 leading-snug flex items-start gap-1.5">
          <Info className="w-3 h-3 mt-px flex-shrink-0" />
          <span>
            {untrainable.map((t) => t.display).join(', ')} is recorded here but sits outside the
            twelve SurgVU training classes, so no model can be scored against it.
          </span>
        </p>
      )}

      <p className="text-[11px] text-fg2 italic border-t border-line pt-1.5 leading-tight">
        From the SurgVU 2024 release. No confidence is shown because none applies — this is not an
        estimate.
      </p>
    </div>
  );
};
