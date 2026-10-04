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
      <div className="bg-[#0b1017] p-3.5 rounded-lg border border-slate-800/80 flex flex-col space-y-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <ShieldCheck className="w-4 h-4 text-slate-600" />
          <span>Recorded (dataset)</span>
        </div>
        <p className="text-[11px] text-slate-500 italic leading-snug">
          No annotated video is loaded. Attach a SurgVU case from the library to see what the
          release records, frame by frame.
        </p>
      </div>
    );
  }

  const untrainable = recorded.tools.filter((t) => !t.trainable);

  return (
    <div className="bg-[#0b1017] p-3.5 rounded-lg border border-emerald-900/40 flex flex-col space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Recorded (dataset)</span>
        </div>
        <span className="text-[10px] font-mono text-slate-500">
          case_{recorded.caseId} p{recorded.part} · {formatClock(currentTime)}
        </span>
      </div>

      {/* Labelled task */}
      <div className="flex items-center gap-2 text-[11px]">
        <span className="text-slate-500">Task:</span>
        {recorded.taskDisplay ? (
          <span className="text-emerald-300 font-medium">{recorded.taskDisplay}</span>
        ) : (
          <span className="text-slate-500 italic">no task labelled at this timestamp</span>
        )}
      </div>

      {/* Installed instruments */}
      {recorded.tools.length === 0 ? (
        <div className="p-2.5 text-[11px] text-slate-400 italic bg-[#0e141f] rounded border border-slate-800">
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
                className="p-2 rounded-md border border-emerald-900/50 bg-emerald-950/20 flex items-start justify-between gap-2"
              >
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-emerald-300 flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: color.dot }}
                    />
                    <span className="truncate">{tool.display}</span>
                    {!tool.trainable && (
                      <span
                        className="text-[8px] px-1 py-px rounded bg-slate-800 text-slate-400 border border-slate-700 flex-shrink-0"
                        title="Outside the twelve classes the models are trained on"
                      >
                        untrained class
                      </span>
                    )}
                  </div>
                  {tool.commercial && (
                    <p className="text-[10px] text-slate-400 mt-0.5 truncate">{tool.commercial}</p>
                  )}
                </div>
                <span className={`text-[10px] font-mono flex-shrink-0 ${color.text}`}>
                  {tool.arm}
                </span>
              </div>
            );
          })}
        </div>
      )}

      {untrainable.length > 0 && (
        <p className="text-[10px] text-slate-500 leading-snug flex items-start gap-1.5">
          <Info className="w-3 h-3 mt-px flex-shrink-0" />
          <span>
            {untrainable.map((t) => t.display).join(', ')} is recorded here but sits outside the
            twelve SurgVU training classes, so no model can be scored against it.
          </span>
        </p>
      )}

      <p className="text-[10px] text-slate-500 italic border-t border-slate-800/60 pt-1.5 leading-tight">
        From the SurgVU 2024 release. No confidence is shown because none applies — this is not an
        estimate.
      </p>
    </div>
  );
};
