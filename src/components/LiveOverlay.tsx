import React, { useEffect, useState } from 'react';
import { DetectedInstrument } from '../types';
import { DetectionOverlay } from './DetectionOverlay';
import { liveVisionBus } from '../services/liveVisionBus';
import { LocalVisionResult } from '../services/localVision';
import { toolDisplay } from '../data/surgvuVocab';
import { ActivationOverlay } from './ActivationOverlay';

interface LiveOverlayProps {
  focusedInstrumentId: string | null;
  onSelectInstrument: (id: string) => void;
  /** Boxes to draw when live detection is off — the last held result. */
  fallback: DetectedInstrument[];
  isLive: boolean;
  /** Whether to paint the class activation heatmap under the boxes. */
  showActivation: boolean;
}

/**
 * Draws the live boxes, and only the live boxes.
 *
 * This component exists to be the *only* thing that re-renders at frame rate.
 * It subscribes straight to the vision bus rather than taking detections as
 * props, so a new result never travels through the app's state.
 */
export const LiveOverlay: React.FC<LiveOverlayProps> = ({
  focusedInstrumentId,
  onSelectInstrument,
  fallback,
  isLive,
  showActivation,
}) => {
  const [result, setResult] = useState<LocalVisionResult | null>(null);

  useEffect(() => liveVisionBus.subscribe(setResult), []);

  if (!isLive) {
    return (
      <DetectionOverlay
        instruments={fallback}
        focusedInstrumentId={focusedInstrumentId}
        onSelectInstrument={onSelectInstrument}
      />
    );
  }

  /*
   * A refused frame gets a notice, not an empty overlay.
   *
   * The worker already withheld every prediction for this frame, so without
   * this branch the overlay would simply be blank — and blank is what "the
   * detector found nothing" also looks like. Those are opposite claims: one
   * says the models looked and saw no instrument, the other says they were
   * never asked because this is not the kind of image they can judge.
   */
  if (result?.domain?.outOfDomain) {
    return (
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center p-6">
        <div className="max-w-sm rounded-lg border border-amber-600/70 bg-black/80 px-4 py-3 text-center shadow-2xl backdrop-blur-sm">
          <div className="text-xs font-semibold uppercase tracking-wider text-amber-300">
            Out of domain — no prediction
          </div>
          <p className="mt-1.5 text-[11px] leading-relaxed text-slate-300">
            This frame is unlike anything the models were trained on, so they were not asked
            to judge it. A sigmoid classifier cannot say &ldquo;I have never seen this&rdquo; —
            pointed at a face it reported a needle driver at 98%. This is the guard that
            stops it.
          </p>
          <div className="mt-1.5 font-mono text-[10px] text-slate-500">
            cosine distance {result.domain.distance.toFixed(3)}
          </div>
        </div>
      </div>
    );
  }

  // Identity is the class and slot, not a timestamp: a box keyed by
  // `Date.now()` would be a different element on every frame, so React would
  // tear down and rebuild the overlay 30 times a second and no CSS transition
  // would ever run.
  const instruments: DetectedInstrument[] = (result?.detections || []).map((d, idx) => ({
    id: `live-${d.label}-${idx}`,
    label: toolDisplay(d.label),
    confidence: d.confidence,
    box: d.box,
    type: 'predicted',
    description: '',
  }));

  return (
    <>
      {/* Beneath the boxes: the heatmap explains the presence classifier, the
          boxes come from the detector, and the boxes must stay readable. */}
      {showActivation && <ActivationOverlay map={result?.activation ?? null} />}
      <DetectionOverlay
        instruments={instruments}
        focusedInstrumentId={focusedInstrumentId}
        onSelectInstrument={onSelectInstrument}
      />
    </>
  );
};
