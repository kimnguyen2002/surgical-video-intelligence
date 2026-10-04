/**
 * The class activation heatmap, drawn over the video.
 *
 * The worker stitches three overlapping classifier crops into one map that
 * spans the full width of the frame, so the overlay is positioned by `region`
 * and covers the picture rather than a square in its middle. The map is
 * evidence for one class (the strongest one the classifier found), shown on a
 * single colour scale that fades in with evidence: weak regions stay clear,
 * strong ones run from cyan through yellow to red.
 *
 * The backbone's grid is coarse, so the map is interpolated bilinearly up to a
 * smooth image. That is display smoothing only; the values and the figure
 * printed in the corner are untouched.
 */

import React, { useEffect, useRef } from 'react';
import type { ActivationMap } from '../services/localVision';

interface ActivationOverlayProps {
  map: ActivationMap | null;
  opacity?: number;
}

/** How much larger than the source grid the canvas is drawn. */
const UPSAMPLE = 5;

const STOPS: [number, number, number, number][] = [
  [0.0, 0, 170, 235],
  [0.35, 70, 220, 120],
  [0.65, 250, 215, 40],
  [1.0, 235, 45, 40],
];

/** Below this the map is fully transparent, so the video stays clear. */
const FADE_START = 0.3;
/** Above this the colour is at full strength. */
const FADE_END = 0.75;

/**
 * Colour for a normalised value. Weak evidence is invisible and the colour
 * fades in smoothly from FADE_START, so only the regions the classifier
 * actually relied on are tinted. The fade is smooth (no hard edge), and it
 * changes only what is drawn: the values are untouched.
 */
function colorFor(value: number): [number, number, number, number] {
  const t = Math.min(1, Math.max(0, value));
  if (t <= FADE_START) return [0, 0, 0, 0];
  let i = 0;
  while (i < STOPS.length - 2 && t > STOPS[i + 1][0]) i++;
  const [t0, r0, g0, b0] = STOPS[i];
  const [t1, r1, g1, b1] = STOPS[i + 1];
  const u = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
  const x = Math.min(1, (t - FADE_START) / (FADE_END - FADE_START));
  const alpha = Math.round(235 * x * x * (3 - 2 * x));
  return [r0 + (r1 - r0) * u, g0 + (g1 - g0) * u, b0 + (b1 - b0) * u, alpha];
}

function sample(map: ActivationMap, x: number, y: number): number {
  const fx = Math.min(map.width - 1, Math.max(0, x));
  const fy = Math.min(map.height - 1, Math.max(0, y));
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const x1 = Math.min(map.width - 1, x0 + 1);
  const y1 = Math.min(map.height - 1, y0 + 1);
  const ax = fx - x0;
  const ay = fy - y0;
  const v = map.values;
  return (
    v[y0 * map.width + x0] * (1 - ax) * (1 - ay) +
    v[y0 * map.width + x1] * ax * (1 - ay) +
    v[y1 * map.width + x0] * (1 - ax) * ay +
    v[y1 * map.width + x1] * ax * ay
  );
}

export const ActivationOverlay: React.FC<ActivationOverlayProps> = ({ map, opacity = 0.8 }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !map) return;

    const w = map.width * UPSAMPLE;
    const h = map.height * UPSAMPLE;
    canvas.width = w;
    canvas.height = h;

    const context = canvas.getContext('2d');
    if (!context) return;

    const image = context.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const value = sample(map, (x + 0.5) / UPSAMPLE - 0.5, (y + 0.5) / UPSAMPLE - 0.5);
        const [r, g, b, a] = colorFor(value);
        const i = (y * w + x) * 4;
        image.data[i] = r;
        image.data[i + 1] = g;
        image.data[i + 2] = b;
        image.data[i + 3] = a;
      }
    }
    context.putImageData(image, 0, 0);
  }, [map]);

  if (!map) return null;

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <canvas
        ref={canvasRef}
        className="absolute"
        style={{
          left: `${map.region.x * 100}%`,
          top: `${map.region.y * 100}%`,
          width: `${map.region.width * 100}%`,
          height: `${map.region.height * 100}%`,
          opacity,
          imageRendering: 'auto',
        }}
      />
      <div className="absolute top-11 left-2.5 rounded border border-white/20 bg-black/80 px-2.5 py-1.5 text-[11px] text-slate-200 backdrop-blur-sm">
        <div className="font-mono">
          <span className="text-amber-300">CAM</span> · {map.label.replace(/_/g, ' ')} ·{' '}
          {map.confidence}%
        </div>
        <div className="mt-1 flex items-center gap-1.5 text-[10px] text-slate-400">
          <span>weak</span>
          <span
            className="h-1.5 w-20 rounded-full"
            style={{
              background:
                'linear-gradient(to right, rgba(0,170,235,0), rgb(0,170,235), rgb(70,220,120), rgb(250,215,40), rgb(235,45,40))',
            }}
          />
          <span>strong evidence</span>
        </div>
      </div>
    </div>
  );
};
