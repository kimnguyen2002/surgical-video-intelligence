import React, { useRef, useState } from 'react';
import {
  Upload,
  Camera,
  Play,
  Pause,
  X,
  Volume2,
  VolumeX,
  Maximize2,
  Film,
  MonitorPlay,
  ShieldCheck,
  Radio,
  Loader2,
  Cpu,
  AlertTriangle,
  Flame,
} from 'lucide-react';
import { FrameObservation, VideoSourceType } from '../types';
import { LiveOverlay } from './LiveOverlay';
import { formatClock } from '../services/groundTruth';

export interface VisionStatus {
  loading: boolean;
  ready: boolean;
  backend: string;
  medianMs: number;
  fps: number;
  error: string | null;
}

interface VideoStageProps {
  currentObservation: FrameObservation | null;
  focusedInstrumentId: string | null;
  isPlaying: boolean;
  isLive: boolean;
  vision: VisionStatus;
  videoFilename: string;
  videoUrl?: string;
  videoSourceType: VideoSourceType;
  recordedToolCount: number | null;
  showActivation: boolean;
  onToggleActivation: () => void;
  /** False when no CAM weights shipped; the toggle is then not offered. */
  activationAvailable: boolean;
  currentTime: number;
  duration: number;
  onTogglePlay: () => void;
  onToggleLive: () => void;
  onClear: () => void;
  onLoadVideoFile: (file: File) => void;
  onToggleCamera: () => void;
  onSelectInstrument: (id: string) => void;
  onSeek: (seconds: number) => void;
  onTimeUpdate: (seconds: number) => void;
  onDurationChange: (seconds: number) => void;
  /** The element failed to load its source. */
  onVideoError: () => void;
  /** A source loaded successfully. */
  onVideoLoaded: () => void;
  onPlayStateChange: (playing: boolean) => void;
}

/**
 * The video stage.
 *
 * The old toolbar had three controls for one job: "Analyse frame" for a single
 * shot, "Run loop" to repeat it, and a dropdown to choose how many seconds to
 * wait in between. All three existed because each analysis was a Gemini call —
 * slow enough that you had to ask for it deliberately, and metered enough that
 * you had to space them out. That is the wrong shape for an operating room:
 * the fastest available setting was one frame every two seconds, and the
 * twentieth call returned `RESOURCE_EXHAUSTED`.
 *
 * The models now run in the page, so there is nothing to meter and nothing to
 * wait for. One button starts detection and it runs continuously until
 * stopped, at whatever rate the machine sustains.
 */
export const VideoStage: React.FC<VideoStageProps> = ({
  currentObservation,
  focusedInstrumentId,
  isPlaying,
  isLive,
  vision,
  videoFilename,
  videoUrl,
  videoSourceType,
  recordedToolCount,
  showActivation,
  onToggleActivation,
  activationAvailable,
  currentTime,
  duration,
  onTogglePlay,
  onToggleLive,
  onClear,
  onLoadVideoFile,
  onToggleCamera,
  onSelectInstrument,
  onSeek,
  onTimeUpdate,
  onDurationChange,
  onVideoError,
  onVideoLoaded,
  onPlayStateChange,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [isMuted, setIsMuted] = useState(true);

  const hasVideo = videoSourceType !== 'none';
  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) containerRef.current.requestFullscreen().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  };

  return (
    <div className="flex flex-col space-y-2.5">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onLoadVideoFile(file);
          e.target.value = '';
        }}
      />

      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 bg-card p-1.5 rounded-lg border border-line">
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            id="btn-load-video"
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-fg bg-inset hover:bg-hover hover:text-white rounded-md transition-colors border border-line"
            title="Open any local video. SurgVU filenames bind to their labels automatically."
          >
            <Upload className="w-3.5 h-3.5 text-fg2" />
            <span>Load video</span>
          </button>

          <button
            id="btn-live-camera"
            onClick={onToggleCamera}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors border ${
              videoSourceType === 'camera'
                ? 'bg-accent-soft text-accent border-accent/40'
                : 'text-fg bg-inset hover:bg-hover hover:text-white border-line'
            }`}
            title="Connect an endoscope or webcam"
          >
            <Camera className="w-3.5 h-3.5" />
            <span>Live camera</span>
          </button>

          <button
            id="btn-play-pause"
            onClick={onTogglePlay}
            disabled={!hasVideo || videoSourceType === 'youtube'}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-fg bg-inset hover:bg-hover hover:text-white rounded-md transition-colors border border-line disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isPlaying ? (
              <>
                <Pause className="w-3.5 h-3.5 text-warn" />
                <span>Pause</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 text-ok fill-ok/30" />
                <span>Play</span>
              </>
            )}
          </button>

          {/* The one detection control */}
          <button
            id="btn-analyse-frame"
            onClick={onToggleLive}
            disabled={!hasVideo || videoSourceType === 'youtube' || vision.loading || !!vision.error}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-md transition-all border disabled:opacity-40 disabled:cursor-not-allowed ${
              isLive
                ? 'bg-ok-soft text-ok border-ok-line'
                : 'bg-accent-soft text-accent hover:bg-accent-soft border-accent/40'
            }`}
            title="Run the on-device models continuously over the video"
          >
            {vision.loading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Loading models…</span>
              </>
            ) : isLive ? (
              <>
                <Radio className="w-3.5 h-3.5 animate-pulse" />
                <span>Stop detection</span>
              </>
            ) : (
              <>
                <Radio className="w-3.5 h-3.5" />
                <span>Analyse frames (live)</span>
              </>
            )}
          </button>

          {/* Live performance readout — the claim of low latency, measured */}
          {vision.ready && (
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-mono rounded-md border border-line bg-card text-fg2"
              title={`Inference runs on ${vision.backend}, in a worker, entirely on this machine`}
            >
              <Cpu className="w-3 h-3 text-fg2" />
              <span className="uppercase">{vision.backend}</span>
              {isLive && vision.medianMs > 0 && (
                <>
                  <span className="text-fg2">│</span>
                  <span className={vision.medianMs <= 100 ? 'text-ok' : 'text-warn'}>
                    {vision.medianMs} ms
                  </span>
                  <span className="text-fg2">{vision.fps} fps</span>
                </>
              )}
            </div>
          )}

          {/* Class activation heatmap. Offered only when the weights shipped
              and detection is running — a toggle that does nothing is worse
              than no toggle. */}
          {activationAvailable && isLive && (
            <button
              id="btn-toggle-activation"
              onClick={onToggleActivation}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-md border transition-colors ${
                showActivation
                  ? 'border-warn-line bg-warn-soft text-warn'
                  : 'border-line bg-card text-fg2 hover:text-fg hover:bg-hover'
              }`}
              title="Class activation map: which regions drove the instrument-presence score. Computed from the forward pass, not from gradients."
            >
              <Flame className="w-3.5 h-3.5" />
              <span>Heatmap</span>
            </button>
          )}
        </div>

        <button
          id="btn-clear-stage"
          onClick={onClear}
          className="flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-fg2 hover:text-fg hover:bg-hover rounded-md transition-colors"
          title="Clear predictions and keyframes"
        >
          <X className="w-3.5 h-3.5" />
          <span>Clear</span>
        </button>
      </div>

      {vision.error && (
        <div className="flex items-start gap-2 px-3 py-2 text-xs text-warn bg-warn-soft border border-warn-line rounded-md">
          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-warn" />
          <span>
            On-device vision unavailable: {vision.error} Recorded dataset labels and the assistant
            still work.
          </span>
        </div>
      )}

      {/* Viewport */}
      <div
        ref={containerRef}
        id="video-viewport-container"
        className="relative w-full aspect-video bg-black rounded-lg overflow-hidden border border-line shadow-2xl flex items-center justify-center group"
      >
        {videoSourceType === 'youtube' && videoUrl ? (
          <iframe
            src={videoUrl}
            title={videoFilename}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            className="w-full h-full object-cover"
          />
        ) : (
          <video
            ref={videoRef}
            id="active-video-element"
            crossOrigin="anonymous"
            playsInline
            muted={isMuted}
            className={`w-full h-full object-contain ${hasVideo ? 'block' : 'hidden'}`}
            onTimeUpdate={(e) => onTimeUpdate(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => {
              const el = e.currentTarget;
              onVideoLoaded();
              // A camera stream reports Infinity; the manifest duration for a
              // library part is already correct, so only a real number wins.
              if (Number.isFinite(el.duration) && el.duration > 0) onDurationChange(el.duration);
            }}
            // Without this a bad URL failed in total silence.
            onError={onVideoError}
            onPlay={() => onPlayStateChange(true)}
            onPause={() => onPlayStateChange(false)}
          />
        )}

        {!hasVideo && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center px-8">
            <MonitorPlay className="w-10 h-10 text-slate-500" />
            <p className="text-sm font-medium text-slate-300">No video loaded</p>
            <p className="text-xs text-slate-400 max-w-sm leading-relaxed">
              Select a case part below to begin playback with recorded labels. Press{' '}
              <span className="font-mono text-slate-200">Analyse frames (live)</span> to run on-device detection.
            </p>
          </div>
        )}

        {hasVideo && (
          <LiveOverlay
            isLive={isLive}
            fallback={currentObservation?.instruments || []}
            focusedInstrumentId={focusedInstrumentId}
            onSelectInstrument={onSelectInstrument}
            showActivation={showActivation}
          />
        )}

        {/* Recorded badge */}
        {recordedToolCount !== null && (
          <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5 px-2 py-1 rounded bg-emerald-950/80 border border-emerald-500/50 text-[11px] text-emerald-300 backdrop-blur-sm">
            <ShieldCheck className="w-3 h-3" />
            <span>
              {recordedToolCount} recorded {recordedToolCount === 1 ? 'instrument' : 'instruments'}
            </span>
          </div>
        )}

        {/* Live badge */}
        {isLive && (
          <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5 px-2 py-1 rounded bg-black/70 border border-amber-400/60 text-[11px] text-amber-300 backdrop-blur-sm font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            <span>LIVE {vision.medianMs > 0 ? `${vision.medianMs}ms` : ''}</span>
          </div>
        )}

        {/* Scrubber */}
        {hasVideo && videoSourceType !== 'youtube' && (
          <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/90 via-black/60 to-transparent opacity-95 group-hover:opacity-100 transition-opacity flex flex-col gap-1.5">
            <div
              id="video-scrubber-track"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                onSeek(((e.clientX - rect.left) / rect.width) * duration);
              }}
              className="w-full h-1.5 bg-white/25 hover:h-2.5 rounded-full cursor-pointer transition-all relative overflow-hidden"
            >
              <div
                className="h-full bg-sky-400 rounded-full"
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            <div className="flex items-center justify-between text-xs text-slate-200">
              <div className="flex items-center gap-3">
                <button onClick={onTogglePlay} className="hover:text-white transition-colors">
                  {isPlaying ? (
                    <Pause className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4 fill-current" />
                  )}
                </button>
                <span className="font-mono text-white tracking-wider">
                  {formatClock(currentTime)} / {formatClock(duration)}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => setIsMuted(!isMuted)}
                  className="hover:text-white transition-colors"
                  title={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted ? (
                    <VolumeX className="w-4 h-4 text-slate-400" />
                  ) : (
                    <Volume2 className="w-4 h-4" />
                  )}
                </button>
                <button
                  onClick={toggleFullscreen}
                  className="hover:text-white transition-colors"
                  title="Fullscreen"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Source line */}
      <div className="flex items-center justify-between text-xs text-fg2 font-mono px-1">
        <div className="flex items-center gap-1.5 min-w-0">
          <Film className="w-3 h-3 text-fg2 flex-shrink-0" />
          <span className="truncate">{videoFilename}</span>
        </div>

        {currentObservation?.phase && (
          <span className="text-warn font-sans font-medium flex-shrink-0 ml-2">
            Predicted task: {currentObservation.phase}
          </span>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1.5 pt-1 text-xs text-fg2 border-t border-line">
        <div className="flex items-center gap-2">
          <span className="inline-block px-1.5 py-0.5 text-[11px] font-semibold text-ok border border-ok-line rounded bg-ok-soft">
            Recorded
          </span>
          <span className="text-fg2">— dataset fact, listed not boxed, no confidence</span>
        </div>

        <div className="flex items-center gap-2">
          <span className="inline-block px-1.5 py-0.5 text-[11px] font-semibold text-warn border border-dashed border-warn-line rounded bg-warn-soft">
            Predicted
          </span>
          <span className="text-fg2">— on-device model, boxed, always with a percentage</span>
        </div>
      </div>
    </div>
  );
};
