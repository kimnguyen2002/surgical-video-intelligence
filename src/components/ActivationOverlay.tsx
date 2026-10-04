/**
 * The class activation heatmap, drawn over the video.
 *
 * The map arrives as a 7x7 grid — that is the spatial resolution the backbone
 * actually has at its last stage, after five downsampling steps from 224px.
 * Presenting it larger than it is would be dishonest about the precision of
 * the explanation, so the grid is drawn at its true size onto a 7x7 canvas and
 * the browser's own bilinear filter scales it up. The blur is the data's, not
 * an effect added on top of it.
 *
 * It is positioned by `region` rather than stretched across the frame. The
 * classifier sees a centre crop, so a map drawn edge to edge would place the
 * hot region away from the pixels that actually produced it — wrong in a way
 * that still looks plausible, which is the worst failure mode an
 * explainability overlay can have.
 */

import React, { useEffect, useRef } from 'react';
import type { ActivationMap } from '../services/localVision';

interface ActivationOverlayProps {
  map: ActivationMap | null;
  opacity?: number;
}

/**
 * Transparent → cyan → amber → red, with the cold half discarded.
 *
 * Two choices here are about legibility rather than taste, and the first pass
 * got both wrong:
 *
 * **The cutoff is 0.35, not 0.15.** A CAM is normalised by its own maximum, so
 * a diffuse map stays diffuse: over half the grid sat above 0.15 and the
 * overlay tinted most of the frame, which communicates nothing. Showing only
 * the top two-thirds of the range is what makes "where" a readable answer.
 *
 * **Contrast is raised by a gamma.** `t ** 1.6` pushes the middle of the range
 * down, so the peak stands out against the shoulder instead of blending into
 * it. This changes the *appearance* of the map, never the values — the figure
 * printed in the corner and the underlying data are untouched.
 */
function colorFor(value: number): [number, number, number, number] {
  const CUTOFF = 0.35;
  if (value <= CUTOFF) return [0, 0, 0, 0];

  const t = Math.pow((value - CUTOFF) / (1 - CUTOFF), 1.6);
  const alpha = Math.round(60 + t * 170);

  if (t < 0.5) {
    // cyan -> amber
    const u = t / 0.5;
    return [
      Math.round(34 + u * (251 - 34)),
      Math.round(211 - u * (211 - 191)),
      Math.round(238 - u * (238 - 36)),
      alpha,
    ];
  }
  // amber -> red
  const u = (t - 0.5) / 0.5;
  return [251, Math.round(191 - u * 123), Math.round(36 + u * 32), alpha];
}

export const ActivationOverlay: React.FC<ActivationOverlayProps> = ({ map, opacity = 0.8 }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !map) return;

    canvas.width = map.width;
    canvas.height = map.height;

    const context = canvas.getContext('2d');
    if (!context) return;

    const image = context.createImageData(map.width, map.height);
    for (let i = 0; i < map.values.length; i++) {
      const [r, g, b, a] = colorFor(map.values[i]);
      image.data[i * 4] = r;
      image.data[i * 4 + 1] = g;
      image.data[i * 4 + 2] = b;
      image.data[i * 4 + 3] = a;
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
          // Let the browser scale 7x7 smoothly; `pixelated` would imply the
          // model reasons in blocks, which it does not.
          imageRendering: 'auto',
        }}
      />
      {/* Top-left, under the recorded badge: the bottom of the stage belongs
          to the player's own controls, and a label there sat on top of them. */}
      <div className="absolute top-11 left-2.5 rounded border border-slate-700/80 bg-black/80 px-2 py-1 font-mono text-[10px] text-slate-300 backdrop-blur-sm">
        <span className="text-amber-300">CAM</span> · {map.label.replace(/_/g, ' ')} ·{' '}
        {map.confidence}% · {map.width}&times;{map.height}
      </div>
    </div>
  );
};
