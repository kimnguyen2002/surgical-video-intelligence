import React, { useMemo, useState } from 'react';
import { Activity } from 'lucide-react';
import { armTracks, formatClock, taskIntervalsFor } from '../services/groundTruth';

interface GroundTruthTrackProps {
  caseId: string;
  part: number;
  /** The full part's duration, **not** the excerpt's. See the note below. */
  durationSeconds: number;
  /** Position within the full part, i.e. already offset for an excerpt. */
  currentTime: number;
  /**
   * The slice of the part that is actually playable, when a bundled excerpt
   * is loaded. Null when the whole part is available.
   *
   * The ribbon always draws the *whole* part, because that is the thing the
   * annotations describe — a five-hour operation with a stapler going in at
   * 00:07:24. Drawing only the excerpt on screen would throw away the one
   * view that makes the dataset legible. What the window does is shade the
   * reachable region, so it is obvious that seeking outside it will not work.
   */
  playableWindow: { start: number; end: number } | null;
  /** Receives a position in the full part's coordinate system. */
  onSeek: (seconds: number) => void;
}

/** Task bands get their own colours; the arms already have theirs. */
const TASK_COLORS: Record<string, string> = {
  suturing: '#22d3ee',
  uterine_horn: '#f472b6',
  rectal_artery_vein: '#fb7185',
  suspensory_ligaments: '#a78bfa',
  skills_application: '#4ade80',
  range_of_motion: '#facc15',
  retraction_collision_avoidance: '#fb923c',
  other: '#94a3b8',
};

/**
 * The recorded label track for one video part.
 *
 * This is the whole point of attaching a real case: instead of a scrubber with
 * nothing on it, the surgeon sees which instrument was on which arm for which
 * stretch of a five-hour operation, and can jump to the moment a stapler went
 * in. Every bar here is dataset fact — installation logs and task annotations
 * from the release — so nothing on this strip carries a confidence.
 *
 * Presence is per arm and genuinely overlapping: four arms can hold four
 * instruments at once, which is why this is four tracks and not one.
 */
export const GroundTruthTrack: React.FC<GroundTruthTrackProps> = ({
  caseId,
  part,
  durationSeconds,
  currentTime,
  playableWindow,
  onSeek,
}) => {
  const [hover, setHover] = useState<{ text: string; left: number } | null>(null);

  const tracks = useMemo(() => armTracks(caseId, part), [caseId, part]);
  const tasks = useMemo(() => taskIntervalsFor(caseId, part), [caseId, part]);

  const span = durationSeconds > 0 ? durationSeconds : 1;
  const pct = (seconds: number) => `${Math.max(0, Math.min(100, (seconds / span) * 100))}%`;
  const playheadPct = Math.max(0, Math.min(100, (currentTime / span) * 100));

  // Measured against the bar element itself, so the arm-label column does not
  // offset the result the way it would from the enclosing row.
  const seekFromEvent = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = (e.clientX - rect.left) / rect.width;
    const target = Math.max(0, Math.min(1, fraction)) * span;
    // Clicking outside the playable window would seek the element past its own
    // duration, which silently clamps to the end and leaves the playhead
    // somewhere the user did not click. Clamping here instead keeps the
    // playhead where the click can actually take it.
    if (playableWindow) {
      onSeek(Math.max(playableWindow.start, Math.min(playableWindow.end, target)));
      return;
    }
    onSeek(target);
  };

  if (tracks.length === 0 && tasks.length === 0) {
    return null;
  }

  return (
    <div className="bg-card p-3.5 rounded-lg border border-line flex flex-col space-y-2.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-fg">
          <Activity className="w-3.5 h-3.5 text-ok" />
          <span>Recorded labels — case_{caseId} part {part}</span>
        </div>
        <span className="text-[11px] text-ok border border-ok-line bg-ok-soft rounded px-1.5 py-0.5">
          dataset ground truth
        </span>
      </div>

      <div className="relative">
        {/*
          The reachable slice, when only an excerpt is bundled. Drawn behind
          everything and ignoring pointer events so it cannot intercept a seek.
        */}
        {playableWindow && (
          <div
            className="absolute inset-y-0 z-0 pointer-events-none border-x border-accent/40 bg-accent-soft"
            style={{
              left: pct(playableWindow.start),
              width: `${Math.max(0.4, ((playableWindow.end - playableWindow.start) / span) * 100)}%`,
            }}
            title={`Bundled excerpt: ${formatClock(playableWindow.start)} – ${formatClock(playableWindow.end)}`}
          />
        )}

        {/* Task band */}
        {tasks.length > 0 && (
          <div className="flex items-center gap-2 mb-1.5">
            <span className="w-11 flex-shrink-0 text-[11px] font-mono text-fg2">task</span>
            <div
              className="relative flex-1 h-4 bg-card rounded border border-line cursor-pointer overflow-hidden"
              onClick={seekFromEvent}
              onMouseLeave={() => setHover(null)}
            >
              {tasks.map((task, idx) => (
                <div
                  key={`${task.start}-${idx}`}
                  className="absolute top-0 bottom-0 opacity-70 hover:opacity-100 transition-opacity"
                  style={{
                    left: pct(task.start),
                    width: pct(task.end - task.start),
                    backgroundColor: TASK_COLORS[task.label] || '#94a3b8',
                  }}
                  onMouseEnter={(e) =>
                    setHover({
                      text: `${task.display} · ${formatClock(task.start)}–${formatClock(task.end)}`,
                      left: e.currentTarget.offsetLeft,
                    })
                  }
                />
              ))}
            </div>
          </div>
        )}

        {/* One row per robot arm */}
        {tracks.map((track) => (
          <div key={track.arm} className="flex items-center gap-2 mb-1">
            <span
              className={`w-11 flex-shrink-0 text-[11px] font-mono ${track.color.text}`}
              title={`Universal setup module ${track.arm.replace('USM', '')}`}
            >
              {track.arm}
            </span>
            <div
              className="relative flex-1 h-3.5 bg-card rounded border border-line cursor-pointer overflow-hidden"
              onClick={seekFromEvent}
              onMouseLeave={() => setHover(null)}
            >
              {track.intervals.map((interval, idx) => (
                <div
                  key={`${interval.start}-${idx}`}
                  className="absolute top-0 bottom-0 opacity-75 hover:opacity-100 transition-opacity"
                  style={{
                    left: pct(interval.start),
                    // A 30-second install in a 5-hour part is 0.16% wide and
                    // would be invisible; a floor keeps every record clickable.
                    minWidth: '2px',
                    width: pct(interval.end - interval.start),
                    backgroundColor: track.color.dot,
                  }}
                  onMouseEnter={(e) =>
                    setHover({
                      text: `${interval.display}${
                        interval.commercial ? ` (${interval.commercial})` : ''
                      } · ${formatClock(interval.start)}–${formatClock(interval.end)}`,
                      left: e.currentTarget.offsetLeft,
                    })
                  }
                />
              ))}
            </div>
          </div>
        ))}

        {/*
          Playhead across every track.

          The bars start after a 2.75rem arm label and a 0.5rem gap, so a raw
          percentage of the row's width lands to the left of where it belongs
          and drifts further out the closer the playhead gets to the end. The
          offset is subtracted from the span before the fraction is applied.
        */}
        <div
          className="absolute top-0 bottom-0 w-px bg-accent pointer-events-none"
          style={{ left: `calc(3.25rem + (100% - 3.25rem) * ${playheadPct / 100})` }}
        >
          <div className="w-1.5 h-1.5 rounded-full bg-accent -ml-[3px] -mt-0.5" />
        </div>
      </div>

      {hover && (
        <div className="text-[11px] text-fg2 font-mono bg-card border border-line rounded px-2 py-1 truncate">
          {hover.text}
        </div>
      )}

      <p className="text-[11px] text-fg2 leading-snug border-t border-line pt-1.5">
        Tool presence comes from the robot's installation log, not from vision — an instrument is
        recorded as present while it is installed, including when it is off-screen or occluded.
      </p>
    </div>
  );
};
