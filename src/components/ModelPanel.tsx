import React, { useState } from 'react';
import { Cpu, ChevronDown, ChevronRight, Info } from 'lucide-react';
import { LocalVisionInfo } from '../services/localVision';
import { VisionStatus } from './VideoStage';
import { toolDisplay, taskDisplay } from '../data/surgvuVocab';

interface ModelPanelProps {
  status: VisionStatus;
  info: LocalVisionInfo | null;
}

const METRIC_LABELS: Record<string, string> = {
  val_mAP: 'validation mAP',
  val_balanced_accuracy: 'balanced accuracy',
  val_accuracy: 'accuracy',
};

/**
 * What is actually running, and what it was trained on.
 *
 * The detector reaches 14 classes but was fitted to five clips of the public
 * cat1 subset, and six of those classes never occur in that data — so it
 * cannot predict them at all. A number like "78%" on a box says nothing about
 * that, and without this panel a small-data demonstrator reads as a validated
 * detector. Every checkpoint carries its own provenance note from training;
 * they are printed here verbatim rather than summarised.
 */
export const ModelPanel: React.FC<ModelPanelProps> = ({ status, info }) => {
  const [open, setOpen] = useState(false);

  if (!status.ready || !info) {
    return (
      <div className="bg-[#0b1017] p-3.5 rounded-lg border border-slate-800/80">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <Cpu className="w-4 h-4 text-slate-600" />
          <span>On-device models</span>
        </div>
        <p className="text-[11px] text-slate-500 italic mt-2 leading-snug">
          {status.loading
            ? 'Loading the surgical checkpoints into the browser…'
            : status.error
              ? status.error
              : 'Load a video to bring up the on-device models.'}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-[#0b1017] rounded-lg border border-slate-800/80">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-2 p-3.5 hover:bg-slate-800/30 transition-colors rounded-lg"
      >
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span>On-device models</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-slate-500 uppercase">{status.backend}</span>
          {open ? (
            <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
          )}
        </div>
      </button>

      {open && (
        <div className="px-3.5 pb-3.5 flex flex-col gap-2.5">
          <p className="text-[10px] text-slate-400 leading-snug">
            Three checkpoints from the surgical_main training run, exported to ONNX and executed in
            a worker on this machine. Frames never leave the browser, so there is no per-frame API
            call, no rate limit, and no network round trip.
          </p>

          {info.models.map((model) => (
            <div key={model.name} className="p-2 rounded-md bg-[#0e141f] border border-slate-800">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[11px] font-semibold text-slate-200 capitalize">
                  {model.name === 'detection'
                    ? 'Instrument detector (boxes)'
                    : model.name === 'tool'
                      ? 'Instrument presence'
                      : 'Surgical task'}
                </span>
                <span className="text-[10px] font-mono text-slate-500">{model.imageSize}px</span>
              </div>

              {model.metric && Object.keys(model.metric).length > 0 && (
                <p className="text-[10px] text-emerald-400/80 mt-0.5">
                  {Object.entries(model.metric)
                    .map(([k, v]) => `${METRIC_LABELS[k] || k} ${Number(v).toFixed(3)}`)
                    .join(' · ')}
                </p>
              )}

              <p className="text-[10px] text-slate-500 mt-1 leading-snug">
                {model.classes.length} classes:{' '}
                {model.classes
                  .slice(0, 4)
                  .map((c) => (model.name === 'task' ? taskDisplay(c) : toolDisplay(c)))
                  .join(', ')}
                {model.classes.length > 4 && `, +${model.classes.length - 4} more`}
              </p>

              {model.provenance && (
                <p className="text-[10px] text-amber-300/70 mt-1 leading-snug flex items-start gap-1">
                  <Info className="w-2.5 h-2.5 mt-0.5 flex-shrink-0" />
                  <span>{model.provenance}</span>
                </p>
              )}
            </div>
          ))}

          <p className="text-[10px] text-slate-500 italic border-t border-slate-800/60 pt-1.5 leading-tight">
            The detector has no published validation metric here because it was fitted to a handful
            of clips as a demonstrator. Read its boxes as a demonstration of the pipeline, not as a
            validated instrument detector.
          </p>
        </div>
      )}
    </div>
  );
};
