import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Stethoscope, Info, Settings as SettingsIcon, Sparkles, HelpCircle } from 'lucide-react';
import {
  SpecialtyId,
  FrameObservation,
  ChatMessage,
  RecordedContext,
  ApiFailure,
} from './types';
import { SURGICAL_KNOWLEDGE_BASE } from './data/knowledge';
import { initialDefaultState, AppState } from './store';
import { VideoStage, VisionStatus } from './components/VideoStage';
import { Panels } from './components/Panels';
import { ChatAssistant } from './components/ChatAssistant';
import { CaseLibrary, partId } from './components/CaseLibrary';
import { GroundTruthTrack } from './components/GroundTruthTrack';
import { GroundTruthPanel } from './components/GroundTruthPanel';
import { ApiStatusBanner } from './components/ApiStatusBanner';
import { ModelPanel } from './components/ModelPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { GuidedTour, tourSeen, type TourPhase } from './components/GuidedTour';
import { ThemeToggle } from './components/ThemeToggle';
import { ask, type TimelineEvent, type VideoFacts } from './assistant';
import { apiKey, checkKey, GeminiError, listModels, preferredModel } from './services/byokGemini';
import { LocalVision, LocalVisionResult } from './services/localVision';
import { liveVisionBus } from './services/liveVisionBus';
import { taskDisplay, toolDisplay } from './data/surgvuVocab';
import { MediaService } from './services/media';
import { dictation, recognitionSupported, speaker } from './services/voice';
import {
  findPart,
  formatClock,
  matchFileToPart,
  summariseFor,
  taskAt,
  taskIntervalsFor,
  toolIntervalsFor,
  toolsAt,
  truthAt,
} from './services/groundTruth';
import { getRemoteVideoUrl } from './data/surgvuCases';
import { clipFor } from './data/surgvuClips';
import { risksFor } from './data/riskRules';

const VIDEO_ELEMENT_ID = 'active-video-element';

const nowStamp = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

/** One sentence describing what the on-device models just returned. */
function describeLiveFrame(result: LocalVisionResult, boxCount: number): string {
  const parts: string[] = [];

  if (result.domain?.outOfDomain) {
    return (
      'This frame is outside the data these models were trained on, so no prediction was made. ' +
      `Cosine distance from the training distribution: ${result.domain.distance.toFixed(2)}.`
    );
  }

  if (boxCount === 0) {
    parts.push('The detector localized no instrument in this frame.');
  } else {
    const names = [...new Set(result.detections.map((d) => toolDisplay(d.label)))];
    parts.push(`Detector: ${boxCount} box${boxCount === 1 ? '' : 'es'} — ${names.join(', ')}.`);
  }

  if (result.task) {
    parts.push(
      `Task classifier: ${taskDisplay(result.task.label)} at ${result.task.confidence}%.`
    );
  }
  if (result.tools?.length) {
    parts.push(`Presence head: ${result.tools.map((t) => toolDisplay(t.label)).join(', ')}.`);
  }
  return parts.join(' ');
}

export default function App() {
  const [state, setState] = useState<AppState>(initialDefaultState);

  const [showSettings, setShowSettings] = useState(false);
  const [tourPhase, setTourPhase] = useState<TourPhase>(() => (tourSeen() ? 'closed' : 'welcome'));

  /**
   * Whether to paint the class activation heatmap.
   *
   * Off by default. It explains the *presence classifier*, while the boxes
   * come from the detector — two different models — so leaving it on by
   * default would invite reading the heatmap as an explanation of the boxes.
   */
  const [showActivation, setShowActivation] = useState(false);

  /**
   * Local video files the user attached, keyed `caseId/part`.
   *
   * Plain state holding a fresh Map per attach. An earlier version kept the
   * Map in a ref and forced a re-render by bumping a counter into the
   * library's `key` — which remounted the component and silently reset which
   * cases were expanded, so attaching the folder collapsed every case it had
   * just opened.
   */
  const [attachedFiles, setAttachedFiles] = useState<Map<string, File>>(new Map());

  const objectUrlRef = useRef<string | null>(null);

  /**
   * The on-device models. One instance for the life of the page — loading
   * 38 MB of ONNX and compiling its kernels is not something to repeat.
   */
  const visionRef = useRef<LocalVision | null>(null);
  if (visionRef.current === null) visionRef.current = new LocalVision();

  const [vision, setVision] = useState<VisionStatus>({
    loading: false,
    ready: false,
    backend: '',
    medianMs: 0,
    fps: 0,
    error: null,
  });

  /**
   * Wall-clock of the last panel update.
   *
   * Detections arrive at frame rate and go straight to the overlay through the
   * bus. The panels take a copy a few times a second — re-rendering the case
   * library and the chat transcript 30 times a second to update a summary
   * sentence would cost more than the inference does.
   */
  const lastPanelUpdateRef = useRef(0);

  /**
   * The most recent model output, held in a ref rather than state.
   *
   * The assistant needs whatever the models saw at the instant a question was
   * asked. Routing that through React state would either re-render the whole
   * tree at frame rate, or — with the 400 ms panel throttle below — hand the
   * assistant a result up to 400 ms stale. A ref gives it the current value
   * with no re-render at all.
   */
  const lastVisionRef = useRef<LocalVisionResult | null>(null);

  const activePartId =
    state.activeCaseId !== null && state.activePart !== null
      ? partId(state.activeCaseId, state.activePart)
      : null;

  // -------------------------------------------------------------------------
  // Ground truth at the playhead
  // -------------------------------------------------------------------------

  /**
   * What the release records right now.
   *
   * Recomputed as the video plays, independently of whether any model has run
   * — recorded facts do not depend on the API being reachable, and the panel
   * showing them must keep working when it is not.
   */
  /**
   * The playhead's position in the **original recording**.
   *
   * A bundled excerpt is cut from the middle of a case, so the player's clock
   * starts at 0:00 for frames that belong to, say, 1:14:50. Every annotation
   * lookup below goes through this value, never `state.currentTime`. Getting
   * this wrong does not produce an obvious error — it produces real
   * annotations from the wrong moment, which is indistinguishable from the app
   * working until you check it against the video.
   */
  const sourceTime = state.currentTime + state.timeOffset;

  const recordedTaskNow = useMemo(
    () =>
      state.activeCaseId !== null && state.activePart !== null
        ? taskAt(state.activeCaseId, state.activePart, sourceTime)
        : null,
    [state.activeCaseId, state.activePart, sourceTime]
  );

  const recordedNow: RecordedContext | null = useMemo(() => {
    if (state.activeCaseId === null || state.activePart === null) return null;
    const task = taskAt(state.activeCaseId, state.activePart, sourceTime);
    return {
      caseId: state.activeCaseId,
      part: state.activePart,
      tools: toolsAt(state.activeCaseId, state.activePart, sourceTime),
      taskDisplay: task?.display || null,
    };
  }, [state.activeCaseId, state.activePart, sourceTime]);

  /**
   * The whole-recording timeline, for "when does X happen?" questions.
   *
   * Built per part rather than per frame: it does not change as the video
   * plays, and rebuilding a few hundred entries on every `timeupdate` would
   * be work done thirty times a second for a value that is constant.
   */
  const timeline: TimelineEvent[] = useMemo(() => {
    if (state.activeCaseId === null || state.activePart === null) return [];
    const tools = toolIntervalsFor(state.activeCaseId, state.activePart).map((t) => ({
      kind: 'tool' as const,
      label: t.label,
      display: t.display,
      start: t.start,
      end: t.end,
      duration: Math.max(0, t.end - t.start),
      arm: t.arm,
    }));
    const tasks = taskIntervalsFor(state.activeCaseId, state.activePart).map((t) => ({
      kind: 'task' as const,
      label: t.label,
      display: t.display,
      start: t.start,
      end: t.end,
      duration: Math.max(0, t.end - t.start),
    }));
    return [...tasks, ...tools].sort((a, b) => a.start - b.start);
  }, [state.activeCaseId, state.activePart]);

  /**
   * Everything the assistant is allowed to assert, assembled fresh per question.
   *
   * Recorded annotations and model output are kept in separate fields all the
   * way down rather than merged into one list of "instruments". Merging them
   * would be more convenient here and would destroy the one distinction the
   * whole interface exists to preserve.
   */
  const buildFacts = useCallback((): VideoFacts => {
    const specialty = SURGICAL_KNOWLEDGE_BASE[state.specialtyId];
    const live = state.currentObservation;
    const groundTruth =
      state.activeCaseId !== null && state.activePart !== null
        ? truthAt(state.activeCaseId, state.activePart, sourceTime)
        : null;

    return {
      groundTruth,
      detections: lastVisionRef.current?.detections ?? [],
      predictedTools: lastVisionRef.current?.tools ?? [],
      predictedTask: lastVisionRef.current?.task ?? null,
      timeline,
      passages: [],
      risks: risksFor({
        recordedTaskLabel: recordedTaskNow?.label || null,
        recordedTaskDisplay: recordedTaskNow?.display || null,
        detectedToolLabels: (lastVisionRef.current?.detections ?? []).map((d) => d.label),
      }),
      timestamp: groundTruth ? sourceTime : null,
      title:
        state.activeCaseId !== null
          ? `SurgVU case ${state.activeCaseId}, part ${state.activePart}`
          : state.videoFilename,
      specialtyName: specialty?.name || 'General Surgery',
      outOfDomain: lastVisionRef.current?.domain?.outOfDomain ?? false,
    };
  }, [
    state.specialtyId,
    state.activeCaseId,
    state.activePart,
    state.videoFilename,
    state.currentObservation,
    sourceTime,
    timeline,
    recordedTaskNow,
    recordedNow,
  ]);

  const videoContext = useMemo(() => {
    if (state.activeCaseId === null || state.activePart === null) {
      return state.videoSourceType === 'none'
        ? undefined
        : `${state.videoFilename} — an unannotated source, so nothing about it is recorded.`;
    }
    return summariseFor(state.activeCaseId, state.activePart);
  }, [state.activeCaseId, state.activePart, state.videoFilename, state.videoSourceType]);

  // -------------------------------------------------------------------------
  // API health
  // -------------------------------------------------------------------------

  /**
   * Verify the visitor's own key, if they supplied one.
   *
   * Deliberately *not* run on mount when no key is configured. The app is
   * complete without one — the grounded assistant answers from the dataset at
   * zero cost — so probing an API nobody asked to use would spend a request to
   * learn something the app does not need to know, and would put a red banner
   * in front of a visitor for whom nothing is wrong.
   */
  const runHealthCheck = useCallback(async () => {
    if (!apiKey.configured) {
      setState((prev) => ({ ...prev, apiHealth: null, isCheckingHealth: false }));
      return;
    }

    setState((prev) => ({ ...prev, isCheckingHealth: true, lastFailure: null }));
    const result = await checkKey();

    // Branch rather than a ternary inside setState: the result is a
    // discriminated union, and TypeScript cannot narrow it through the two
    // arms of a conditional expression.
    if (result.ok) {
      setState((prev) => ({
        ...prev,
        isCheckingHealth: false,
        modelId: result.model,
        apiHealth: { ok: true, keyPresent: true, model: result.model },
        lastFailure: null,
      }));
      return;
    }

    const { failure } = result;
    setState((prev) => ({
      ...prev,
      isCheckingHealth: false,
      apiHealth: {
        ok: false,
        keyPresent: true,
        reason: `${failure.status} (${failure.code}): ${failure.message}`,
        hint: failure.hint,
        quotaExceeded: failure.code === 429,
      },
      lastFailure: failure,
      bannerDismissed: false,
    }));
  }, []);

  // Validate a key that was stored in a previous visit, once, on mount.
  useEffect(() => {
    if (apiKey.configured) void runHealthCheck();
  }, [runHealthCheck]);

  /**
   * Record a Gemini failure.
   *
   * This no longer stops anything: the only Gemini traffic left is the chat
   * assistant and voice, both of which the user initiates one message at a
   * time. Detection runs on device and is unaffected by the state of the key —
   * which is the point of moving it there.
   */
  const reportFailure = useCallback((failure: ApiFailure) => {
    setState((prev) => ({ ...prev, lastFailure: failure, bannerDismissed: false }));
  }, []);

  const asFailure = (err: unknown): ApiFailure =>
    err instanceof GeminiError
      ? err.failure
      : {
          code: 0,
          status: 'UNEXPECTED',
          message: err instanceof Error ? err.message : String(err),
        };

  // -------------------------------------------------------------------------
  // Video source
  // -------------------------------------------------------------------------

  const videoElement = () =>
    document.getElementById(VIDEO_ELEMENT_ID) as HTMLVideoElement | null;

  /** Move both the element and the state clock; they must not drift apart. */
  const handleSeek = useCallback((seconds: number) => {
    const el = videoElement();
    if (el && Number.isFinite(seconds)) {
      try {
        el.currentTime = seconds;
      } catch {
        // Seeking before metadata is ready throws; the timeupdate below fixes it.
      }
    }
    setState((prev) => ({ ...prev, currentTime: Math.max(0, seconds) }));
  }, []);

  /** Called from the video element itself, so it must not seek back. */
  const handleTimeUpdate = useCallback((seconds: number) => {
    setState((prev) => (prev.currentTime === seconds ? prev : { ...prev, currentTime: seconds }));
  }, []);

  const handleVideoLoaded = useCallback(() => {
    pendingUrlRef.current = null;
  }, []);

  const handleDurationChange = useCallback((seconds: number) => {
    setState((prev) => (prev.duration === seconds ? prev : { ...prev, duration: seconds }));
  }, []);

  const loadObjectUrl = useCallback((file: File) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    return url;
  }, []);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    },
    []
  );

  const handleAttachFiles = (files: File[]) => {
    setAttachedFiles((prev) => {
      const next = new Map(prev);
      for (const file of files) {
        const match = matchFileToPart(file.name);
        if (match) next.set(partId(match.caseId, match.part), file);
      }
      return next;
    });
  };

  /**
   * Play a case, from whichever source is actually available.
   *
   * The order is cheapest-and-most-faithful first:
   *
   * 1. A **local file** the visitor attached — the real, full part, read off
   *    their disk with nothing uploaded.
   * 2. The **bundled excerpt** — two minutes cut from the most label-dense
   *    window of that part, shipped with the app.
   * 3. A **remote store**, only if someone configured one for a private
   *    deployment.
   */
  const handleSelectPart = (caseId: string, part: number) => {
    const file = attachedFiles.get(partId(caseId, part));
    if (file) {
      loadLibraryPart(caseId, part, file);
      return;
    }
    if (loadBundledExcerpt(caseId, part)) return;
    loadRemotePart(caseId, part);
  };

  /**
   * Load the excerpt bundled for this part.
   *
   * Returns false when there is none, so the caller can fall through. The
   * excerpt's `sourceStartSeconds` goes into `timeOffset`, which is what keeps
   * every annotation lookup pointed at the moment in the *original* recording
   * that the viewer is actually watching.
   */
  const loadBundledExcerpt = (caseId: string, part: number): boolean => {
    const clip = clipFor(caseId, part);
    if (!clip) return false;
    if (state.activeCaseId === caseId && state.activePart === clip.part && state.isExcerpt) {
      return true;
    }

    const url = `${import.meta.env.BASE_URL}${clip.file}`;

    setState((prev) => ({
      ...prev,
      videoSourceType: 'library',
      videoFilename: clip.sourceFilename,
      videoUrl: url,
      activeCaseId: caseId,
      activePart: clip.part,
      currentTime: 0,
      duration: clip.durationSeconds,
      timeOffset: clip.sourceStartSeconds,
      isExcerpt: true,
      isPlaying: false,
      currentObservation: null,
      focusedInstrumentId: null,
      chatMessages: [
        ...prev.chatMessages,
        {
          id: `load-${Date.now()}`,
          sender: 'system',
          text:
            `Playing a **${clip.durationSeconds}-second excerpt** of case ${caseId}, part ${clip.part}, ` +
            `cut from ${formatClock(clip.sourceStartSeconds)} into the recording — the most ` +
            `densely-labelled window in the case.\n\n` +
            `The release records ${clip.tools.length} instrument${clip.tools.length === 1 ? '' : 's'} ` +
            `here (${clip.tools.join(', ')}) and the task **${clip.tasks.join(', ') || 'none'}**. ` +
            `Timestamps shown are the real ones from the full recording, not offsets into the clip.\n\n` +
            `*Attach your own copy of \`${clip.sourceFilename}\` in the library to watch the whole part instead.*`,
          timestamp: nowStamp(),
        },
      ],
    }));

    const el = videoElement();
    if (el) {
      el.crossOrigin = null;
      el.srcObject = null;
      el.src = url;
      el.load();
    }
    return true;
  };

  /** Stream a full part, only when a store has been deliberately configured. */
  const loadRemotePart = (caseId: string, part: number) => {
    if (state.activeCaseId === caseId && state.activePart === part) return;

    const url = getRemoteVideoUrl(caseId, part);
    if (!url) return;
    const manifest = findPart(caseId, part);

    setState((prev) => ({
      ...prev,
      videoSourceType: 'library',
      videoFilename: manifest?.filename || `case_${caseId}_video_part_${part}.mp4`,
      videoUrl: url,
      activeCaseId: caseId,
      activePart: part,
      currentTime: 0,
      duration: manifest?.durationSeconds || 0,
      timeOffset: 0,
      isExcerpt: false,
      isPlaying: false,
      currentObservation: null,
      focusedInstrumentId: null,
      chatMessages: [
        ...prev.chatMessages,
        {
          id: `load-${Date.now()}`,
          sender: 'system',
          text: `Streaming case ${caseId}, part ${part}. ${summariseFor(caseId, part)}`,
          timestamp: nowStamp(),
        },
      ],
    }));

    const el = videoElement();
    if (el) {
      el.crossOrigin = 'anonymous';
      el.srcObject = null;
      el.src = url;
      el.load();
    }
  };

  const loadLibraryPart = (caseId: string, part: number, file: File) => {
    // Re-selecting the part already on the stage would rewind it to 0:00 and
    // re-announce it in the chat. Clicking the row you are watching should do
    // nothing.
    if (state.activeCaseId === caseId && state.activePart === part) return;

    const manifest = findPart(caseId, part);
    const url = loadObjectUrl(file);

    setState((prev) => ({
      ...prev,
      videoSourceType: 'library',
      videoFilename: file.name,
      videoUrl: url,
      activeCaseId: caseId,
      activePart: part,
      currentTime: 0,
      // The manifest duration is exact — probed from the container — so the
      // ribbon can draw correctly before the element reports its own.
      duration: manifest?.durationSeconds || 0,
      isPlaying: false,
      currentObservation: null,
      focusedInstrumentId: null,
      chatMessages: [
        ...prev.chatMessages,
        {
          id: `load-${Date.now()}`,
          sender: 'system',
          text: `Loaded **${file.name}** — case ${caseId}, part ${part}. ${summariseFor(caseId, part)}`,
          timestamp: nowStamp(),
        },
      ],
    }));

    const el = videoElement();
    if (el) {
      el.srcObject = null;
      el.src = url;
      el.load();
    }
  };

  const handleLoadVideoFile = (file: File) => {
    const match = matchFileToPart(file.name);
    if (match) {
      // A SurgVU file dropped on the generic loader still deserves its labels.
      // `handleSelectPart` reads the state Map, which has not updated yet in
      // this tick, so the part is loaded directly from the file in hand.
      handleAttachFiles([file]);
      loadLibraryPart(match.caseId, match.part, file);
      return;
    }

    const url = loadObjectUrl(file);
    setState((prev) => ({
      ...prev,
      videoSourceType: 'file',
      videoFilename: file.name,
      videoUrl: url,
      activeCaseId: null,
      activePart: null,
      currentTime: 0,
      duration: 0,
      isPlaying: true,
      currentObservation: null,
    }));

    const el = videoElement();
    if (el) {
      el.srcObject = null;
      el.src = url;
      el.load();
      el.play().catch(() => {});
    }
  };

  /**
   * URLs that are web *pages*, not video files.
   *
   * A `<video>` element needs a URL that responds with video bytes. A YouTube
   * watch link responds with HTML, so `el.src = link` fails with
   * MEDIA_ERR_SRC_NOT_SUPPORTED and no explanation — which is what "it can't
   * obtain the video from url" was.
   */
  const PAGE_HOSTS = /(?:^|\.)(?:vimeo\.com|dailymotion\.com|drive\.google\.com|dropbox\.com)$/i;

  /** The URL currently being attempted, so an error can retry it once. */
  const pendingUrlRef = useRef<{ url: string; retriedWithoutCors: boolean } | null>(null);

  /**
   * Report why a video source failed, and retry once without CORS.
   *
   * The element carries `crossOrigin="anonymous"` because the detector has to
   * read pixels, and a tainted canvas throws. But demanding CORS makes the
   * *load itself* fail on any host that does not send
   * `Access-Control-Allow-Origin` — so a perfectly good MP4 would refuse to
   * play at all. Falling back to a plain request means the video plays; the
   * detector then cannot run on it, and that is said out loud rather than
   * surfacing later as an opaque worker error.
   */
  const handleVideoError = useCallback(() => {
    const el = videoElement();
    const pending = pendingUrlRef.current;

    if (el && pending && !pending.retriedWithoutCors) {
      pending.retriedWithoutCors = true;
      el.removeAttribute('crossorigin');
      el.src = pending.url;
      el.load();
      el.play().catch(() => {});

      setState((prev) => ({
        ...prev,
        chatMessages: [
          ...prev.chatMessages,
          {
            id: `cors-${Date.now()}`,
            sender: 'system',
            text: `That host does not send CORS headers, so I reloaded the video without them. It will play, but **live detection cannot run on it** — reading pixels from a cross-origin video is blocked by the browser. Detection works on the SurgVU cases and on any file you open from disk.`,
            timestamp: nowStamp(),
          },
        ],
      }));
      return;
    }

    if (!pending) return;
    pendingUrlRef.current = null;

    const code = el?.error?.code;
    const reason =
      code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
        ? 'The server did not return a playable video. The link is probably a web page rather than a video file, or the format is not supported.'
        : code === MediaError.MEDIA_ERR_NETWORK
          ? 'The download failed part-way through.'
          : code === MediaError.MEDIA_ERR_DECODE
            ? 'The file downloaded but could not be decoded.'
            : 'The browser could not load that URL.';

    reportFailure({
      code: 0,
      status: 'VIDEO_URL',
      message: reason,
      hint: 'Use a direct link that ends in .mp4 or .webm — one that plays on its own when pasted into a browser tab.',
    });
  }, [reportFailure]);

  const handleLoadUrl = (url: string) => {
    let parsed: URL | null = null;
    try {
      parsed = new URL(url);
    } catch {
      reportFailure({
        code: 0,
        status: 'VIDEO_URL',
        message: 'That is not a valid URL.',
        hint: 'Paste a full link starting with https://',
      });
      return;
    }

    const isYouTube = /(?:^|\.)(?:youtube\.com|youtu\.be)$/i.test(parsed.hostname);
    let videoId = '';
    if (isYouTube) {
      if (parsed.hostname.includes('youtu.be')) {
        videoId = parsed.pathname.slice(1);
      } else {
        videoId = parsed.searchParams.get('v') || '';
      }
    }

    if (videoId) {
      setState((prev) => ({
        ...prev,
        videoSourceType: 'youtube',
        videoFilename: `YouTube Video`,
        videoUrl: `https://www.youtube.com/embed/${videoId}?enablejsapi=1&origin=${window.location.origin}`,
        activeCaseId: null,
        activePart: null,
        currentTime: 0,
        duration: 0,
        isPlaying: false, // The iframe will likely autoplay if we add autoplay=1, but browser policies vary.
        currentObservation: null,
        focusedInstrumentId: null,
        chatMessages: [
          ...prev.chatMessages,
          {
            id: `load-${Date.now()}`,
            sender: 'system',
            text: `Loaded a YouTube video. It will play directly in the embedded player, but **live detection cannot run on it** because the browser blocks reading pixels out of a cross-origin iframe. The chat assistant is still available to answer manual questions about the surgery if you describe what you see.`,
            timestamp: nowStamp(),
          },
        ],
      }));
      return;
    }

    if (PAGE_HOSTS.test(parsed.hostname)) {
      reportFailure({
        code: 0,
        status: 'VIDEO_URL',
        message: `${parsed.hostname} serves a web page, not a video file, so the player has nothing to load.`,
        hint: 'Download the video and open it with "Load video", which also lets the detector run on it. An embedded player would not work either: the browser blocks reading pixels out of a third-party iframe, so detection could never run.',
      });
      return;
    }

    const filename = url.split('/').pop() || 'network_stream.mp4';
    const match = matchFileToPart(filename);
    const manifest = match ? findPart(match.caseId, match.part) : undefined;

    setState((prev) => ({
      ...prev,
      videoSourceType: match ? 'library' : 'url',
      videoFilename: filename,
      videoUrl: url,
      activeCaseId: match ? match.caseId : null,
      activePart: match ? match.part : null,
      currentTime: 0,
      duration: manifest?.durationSeconds || 0,
      isPlaying: true,
      currentObservation: null,
      focusedInstrumentId: null,
      chatMessages: match
        ? [
            ...prev.chatMessages,
            {
              id: `load-${Date.now()}`,
              sender: 'system',
              text: `Loaded SurgVU video **${filename}** from URL — case ${match.caseId}, part ${match.part}. ${summariseFor(match.caseId, match.part)}`,
              timestamp: nowStamp(),
            },
          ]
        : prev.chatMessages,
    }));

    pendingUrlRef.current = { url, retriedWithoutCors: false };

    const el = videoElement();
    if (el) {
      el.crossOrigin = 'anonymous';
      el.srcObject = null;
      el.src = url;
      el.load();
      el.play().catch(() => {});
    }
  };

  const handleToggleCamera = async () => {
    const el = videoElement();
    if (state.videoSourceType === 'camera') {
      if (el?.srcObject) {
        (el.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
        el.srcObject = null;
      }
      setState((prev) => ({
        ...prev,
        videoSourceType: 'none',
        videoFilename: 'no video loaded',
        isPlaying: false,
        currentObservation: null,
      }));
      return;
    }

    try {
      const stream = await MediaService.getCameraStream();
      setState((prev) => ({
        ...prev,
        videoSourceType: 'camera',
        videoFilename: 'live camera feed',
        activeCaseId: null,
        activePart: null,
        isPlaying: true,
        duration: 0,
        currentObservation: null,
      }));
      if (el) {
        el.src = '';
        el.srcObject = stream;
        el.play().catch(() => {});
      }
    } catch (err: any) {
      reportFailure({
        code: 0,
        status: 'CAMERA',
        message: err?.message || 'Camera access was refused.',
        hint: 'Grant camera permission in the browser, then try again.',
      });
    }
  };

  const handleTogglePlay = () => {
    const el = videoElement();
    setState((prev) => {
      const next = !prev.isPlaying;
      if (el) {
        if (next) el.play().catch(() => {});
        else el.pause();
      }
      return { ...prev, isPlaying: next };
    });
  };

  // -------------------------------------------------------------------------
  // Live detection, on device
  // -------------------------------------------------------------------------

  /**
   * Load the models once a video exists.
   *
   * Deferred rather than done at startup: the three graphs are 38 MB, and
   * someone who opens the page to read the case library should not pay for
   * them. By the time a video is on the stage the download has a reason.
   */
  useEffect(() => {
    if (state.videoSourceType === 'none') return;
    const local = visionRef.current!;
    if (local.ready || vision.loading) return;

    setVision((v) => ({ ...v, loading: true, error: null }));
    local
      .init()
      .then((info) => {
        setVision({
          loading: false,
          ready: true,
          backend: info.backend,
          medianMs: 0,
          fps: 0,
          error: null,
        });
      })
      .catch((err: Error) => {
        setVision({
          loading: false,
          ready: false,
          backend: '',
          medianMs: 0,
          fps: 0,
          error: err.message,
        });
      });
  }, [state.videoSourceType, vision.loading]);

  /** Route a worker result to the overlay immediately, the panels rarely. */
  const handleVisionResult = useCallback(
    (result: LocalVisionResult) => {
      lastVisionRef.current = result;
      liveVisionBus.publish(result);

      const now = performance.now();
      if (now - lastPanelUpdateRef.current < 400) return;
      lastPanelUpdateRef.current = now;

      const local = visionRef.current!;
      const stats = local.stats;
      setVision((v) =>
        v.medianMs === stats.medianMs && v.fps === stats.fps
          ? v
          : { ...v, medianMs: stats.medianMs, fps: stats.fps }
      );

      setState((prev) => {
        const instruments = result.detections.map((d, idx) => ({
          id: `live-${d.label}-${idx}`,
          label: toolDisplay(d.label),
          confidence: d.confidence,
          box: d.box,
          type: 'predicted' as const,
          description: '',
        }));

        // `task` and `tools` only ride along on the frames where the
        // classifiers ran, so the previous values are kept in between rather
        // than blanking the panel every intervening frame.
        const phase = result.task
          ? `${taskDisplay(result.task.label)} (${result.task.confidence}%)`
          : prev.currentObservation?.phase || '';

        const presence = result.tools
          ? result.tools.map((t) => `${toolDisplay(t.label)} ${t.confidence}%`)
          : prev.currentObservation?.anatomyIdentified || [];

        return {
          ...prev,
          currentObservation: {
            id: `live-${Math.round(result.timestampSeconds * 10)}`,
            timestampSeconds: result.timestampSeconds,
            timestampFormatted: formatClock(result.timestampSeconds),
            thumbnailUrl: '',
            phase: result.domain?.outOfDomain ? '' : phase,
            summary: describeLiveFrame(result, instruments.length),
            instruments,
            // Carried through so the panels can say *why* they are empty. On a
            // refused frame the worker already withheld every prediction, so
            // `instruments` is [] regardless — this is what turns "nothing
            // found" into "not judged".
            isOutOfDomain: result.domain?.outOfDomain ?? false,
            anatomyIdentified: result.domain?.outOfDomain ? [] : presence,
            risksPresent: [],
            recorded: null,
            modelUsed: `on-device · ${result.backend} · ${Math.round(result.totalMs)} ms`,
          },
        };
      });
    },
    []
  );

  useEffect(() => {
    const local = visionRef.current!;
    local.onResult = (result) => {
      // A frame got through, so any earlier capture error is no longer true.
      setVision((v) => (v.error ? { ...v, error: null } : v));
      handleVisionResult(result);
    };
    local.onError = (message) => setVision((v) => ({ ...v, error: message }));
    return () => {
      local.onResult = null;
      local.onError = null;
    };
  }, [handleVisionResult]);

  useEffect(() => () => visionRef.current?.dispose(), []);

  /**
   * The frame pump.
   *
   * requestAnimationFrame rather than a timer, because the aim is "as fresh as
   * possible" rather than any particular cadence. `submit` refuses while a
   * frame is still in flight, so a slow machine analyses fewer frames instead
   * of building a backlog — the boxes stay attached to a recent frame either
   * way. On a paused video the same frame is analysed once and then skipped,
   * so parking on a still costs nothing.
   */
  useEffect(() => {
    if (!state.isLiveDetecting) return;

    const local = visionRef.current!;
    let cancelled = false;
    let handle = 0;
    let lastSubmittedTime = -1;

    const pump = () => {
      if (cancelled) return;
      const el = videoElement();
      if (el && el.readyState >= 2 && !el.seeking) {
        const t = el.currentTime;
        if (!el.paused || t !== lastSubmittedTime) {
          lastSubmittedTime = t;
          void local.submit(el, t);
        }
      }
      handle = requestAnimationFrame(pump);
    };

    handle = requestAnimationFrame(pump);
    return () => {
      cancelled = true;
      cancelAnimationFrame(handle);
    };
  }, [state.isLiveDetecting]);

  const handleToggleLive = () => {
    // The side effects run here, not inside the updater. Publishing to the bus
    // from within `setState` reaches LiveOverlay's own setState while React is
    // still computing this component's next state — "Cannot update a component
    // while rendering a different component" — and StrictMode's double
    // invocation fires it twice.
    const next = !state.isLiveDetecting;
    if (next) {
      visionRef.current?.reset();
      lastPanelUpdateRef.current = 0;
    } else {
      liveVisionBus.publish(null);
    }
    setState((prev) => ({ ...prev, isLiveDetecting: next }));
  };

  const handleClear = () => {
    liveVisionBus.publish(null);
    setState((prev) => ({
      ...prev,
      currentObservation: null,
      timelineFrames: [],
      focusedInstrumentId: null,
      selectedTimelineFrameId: null,
    }));
  };

  // -------------------------------------------------------------------------
  // Assistant
  // -------------------------------------------------------------------------

  const handleSendMessage = async (text: string) => {
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text,
      timestamp: nowStamp(),
    };
    const history = [...state.chatMessages, userMsg];
    setState((prev) => ({ ...prev, chatMessages: history, isAssistantThinking: true }));

    // Only user and assistant turns, and only their text. System notices —
    // "playing an excerpt of case 003" — are interface chrome, not part of the
    // conversation, and feeding them back as turns makes the assistant answer
    // the app instead of the person.
    const turns = history
      .filter((m) => m.sender === 'user' || m.sender === 'assistant')
      .slice(-12)
      .map((m) => ({ role: m.sender === 'assistant' ? ('model' as const) : ('user' as const), text: m.text }));

    const answer = await ask(text, {
      facts: buildFacts(),
      role: state.role,
      specialty: state.specialtyId,
      // The question itself is the final turn; passing it twice would make the
      // model see it asked two ways.
      history: turns.slice(0, -1),
      useGenerative: state.useGenerative,
    });

    const messages: ChatMessage[] = [
      {
        id: `assistant-${Date.now()}`,
        sender: 'assistant',
        text: answer.text,
        timestamp: nowStamp(),
        source: answer.source,
      },
    ];

    // A failed generation does not replace the answer — the grounded text is
    // still there — but it is never swallowed either. The reader sees both.
    if (answer.warning) {
      messages.push({
        id: `warn-${Date.now()}`,
        sender: 'system',
        text: `The language model could not be reached, so the answer above comes from the recorded annotations instead.\n\n\`${answer.warning}\``,
        timestamp: nowStamp(),
      });
    }

    setState((prev) => ({
      ...prev,
      isAssistantThinking: false,
      chatMessages: [...prev.chatMessages, ...messages],
    }));

    if (state.isTtsEnabled) {
      speaker.speak(answer.text).catch(() => {
        // Synthesis failing is not worth interrupting the transcript for; the
        // text is on screen either way.
      });
    }
  };

  const handleToggleTts = () => {
    setState((prev) => {
      const next = !prev.isTtsEnabled;
      if (!next) speaker.stop();
      return { ...prev, isTtsEnabled: next };
    });
  };

  /**
   * Dictate a question with the browser's own recogniser.
   *
   * The previous build recorded audio and sent it to Gemini to transcribe,
   * because `webkitSpeechRecognition` fails inside the cross-origin iframe AI
   * Studio ran the app in. This build is a top-level document on its own
   * origin, where the browser recogniser works — and it costs nothing, needs
   * no key, and sends no audio to an API the visitor is paying for.
   */
  const [micStatus, setMicStatus] = useState<string | null>(null);

  const pushSystem = (text: string) =>
    setState((prev) => ({
      ...prev,
      chatMessages: [
        ...prev.chatMessages,
        { id: `mic-${Date.now()}`, sender: 'system', text, timestamp: nowStamp() },
      ],
    }));

  /**
   * Push to talk, transcribed on this machine.
   *
   * First press records; recording ends on its own after a short silence, or
   * on a second press. The transcript is sent as a question.
   */
  const handleToggleMic = () => {
    if (state.isMicListening) {
      void dictation.finish();
      return;
    }

    if (!recognitionSupported()) {
      pushSystem('This browser cannot record audio. Type the question instead.');
      return;
    }

    const done = () => {
      setMicStatus(null);
      setState((prev) => ({ ...prev, isMicListening: false, isMicStarting: false }));
    };

    void dictation.start({
      onStatus: (status) => {
        if (status === 'loading') {
          setMicStatus('Finishing the speech model download (about 40 MB, first time only)…');
          setState((prev) => ({ ...prev, isMicStarting: true }));
        } else if (status === 'listening') {
          setMicStatus('Listening… ask your question. It stops when you pause.');
          setState((prev) => ({ ...prev, isMicStarting: false, isMicListening: true }));
        } else {
          setMicStatus('Transcribing on this device…');
          setState((prev) => ({ ...prev, isMicListening: false, isMicStarting: true }));
        }
      },
      onFinal: (transcript) => {
        done();
        void handleSendMessage(transcript);
      },
      onError: (failure) => {
        done();
        pushSystem(`${failure.message}${failure.hint ? `\n\n${failure.hint}` : ''}`);
      },
      onEnd: done,
    });
  };

  /**
   * The banner reports the state of the visitor's *own* key, so it is only
   * shown once there is one. With no key there is nothing wrong to report —
   * the app is working as designed — and a persistent warning would say
   * otherwise.
   */
  const showBanner = !state.bannerDismissed && apiKey.configured;

  return (
    <div className="min-h-screen bg-page text-fg flex flex-col font-sans selection:bg-accent selection:text-on-accent">
      {/* TOP HEADER */}
      <header className="px-5 py-3 bg-page border-b border-line flex flex-wrap items-center justify-between gap-4 sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <a href="#/" className="flex items-center gap-3" title="Back to the overview">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-line bg-card text-accent">
              <Stethoscope className="h-[18px] w-[18px]" />
            </span>
            <span className="text-[12px] font-bold uppercase tracking-[0.22em] text-fg">
              Surgical Video Intelligence
            </span>
          </a>
          <p className="hidden md:flex text-xs text-fg3 items-center gap-1.5">
            <span>{vision.ready ? vision.backend : 'on-device'}</span>
            <span>·</span>
            <span>SurgVU 2024</span>
            <span>·</span>
            <span>
              {attachedFiles.size > 0
                ? `${attachedFiles.size} parts attached`
                : 'no parts attached'}
            </span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* About — the project, the dataset it plays, and the checkpoints it runs. */}
          <button
            onClick={() => setTourPhase('steps')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-fg2 hover:text-accent bg-inset hover:bg-hover border border-line rounded-md transition-colors shadow-sm"
          >
            <HelpCircle className="w-3.5 h-3.5 text-fg2" />
            <span>Guide</span>
          </button>

          <a
            data-tour="about"
            href="#/"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-fg2 hover:text-accent bg-inset hover:bg-hover border border-line rounded-md transition-colors shadow-sm"
          >
            <Info className="w-3.5 h-3.5 text-fg2" />
            <span>Overview</span>
          </a>

          <ThemeToggle />

          {/* Settings — audience, and the optional bring-your-own-key model. */}
          <button
            data-tour="settings"
            onClick={() => setShowSettings(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-fg bg-inset hover:bg-hover border border-line rounded-md transition-colors shadow-sm"
          >
            <SettingsIcon className="w-3.5 h-3.5 text-fg2" />
            <span>{state.role}</span>
            {state.useGenerative && apiKey.configured && (
              <span className="flex items-center gap-1 text-accent">
                <Sparkles className="w-3 h-3" />
              </span>
            )}
          </button>

        </div>
      </header>

      {/* API STATUS */}
      {showBanner && (
        <div className="px-4 pt-3 max-w-[1720px] w-full mx-auto">
          <ApiStatusBanner
            health={state.apiHealth}
            lastFailure={state.lastFailure}
            isChecking={state.isCheckingHealth}
            onRecheck={() => runHealthCheck()}
            onDismiss={() => setState((prev) => ({ ...prev, bannerDismissed: true }))}
            onSelectModel={(newModel) => {
              preferredModel.set(newModel);
              setState((prev) => ({ ...prev, modelId: newModel }));
              void runHealthCheck();
            }}
          />
        </div>
      )}

      <SettingsPanel
        open={showSettings}
        onClose={() => setShowSettings(false)}
        role={state.role}
        onRoleChange={(role) => setState((prev) => ({ ...prev, role }))}
        useGenerative={state.useGenerative}
        onUseGenerativeChange={(useGenerative) => setState((prev) => ({ ...prev, useGenerative }))}
        modelId={state.modelId}
        onModelChange={(modelId) => setState((prev) => ({ ...prev, modelId }))}
        onKeyChanged={() => void runHealthCheck()}
      />

      {/* MAIN 3-COLUMN DASHBOARD */}
      <main className="flex-1 p-4 max-w-[1720px] w-full mx-auto grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* LEFT: player, recorded ribbon, library, keyframes */}
        <div className="lg:col-span-6 flex flex-col space-y-4">
          <div data-tour="stage">
          <VideoStage
            showActivation={showActivation}
            onToggleActivation={() => setShowActivation((v) => !v)}
            activationAvailable={visionRef.current?.details?.activationMap ?? false}
            currentObservation={state.currentObservation}
            focusedInstrumentId={state.focusedInstrumentId}
            isPlaying={state.isPlaying}
            isLive={state.isLiveDetecting}
            vision={vision}
            videoFilename={state.videoFilename}
            videoSourceType={state.videoSourceType}
            recordedToolCount={recordedNow?.tools.length ?? null}
            currentTime={state.currentTime}
            duration={state.duration}
            onTogglePlay={handleTogglePlay}
            onToggleLive={handleToggleLive}
            onClear={handleClear}
            onLoadVideoFile={handleLoadVideoFile}
            onToggleCamera={handleToggleCamera}
            onSelectInstrument={(id) =>
              setState((p) => ({
                ...p,
                focusedInstrumentId: p.focusedInstrumentId === id ? null : id,
              }))
            }
            onSeek={handleSeek}
            onTimeUpdate={handleTimeUpdate}
            onDurationChange={handleDurationChange}
            onVideoError={handleVideoError}
            onVideoLoaded={handleVideoLoaded}
            onPlayStateChange={(playing) => setState((p) => ({ ...p, isPlaying: playing }))}
          />
          </div>

          {state.activeCaseId !== null && state.activePart !== null && (
            <div data-tour="ribbon">
            <GroundTruthTrack
              caseId={state.activeCaseId}
              part={state.activePart}
              // The *part's* duration, from the probed manifest. `state.duration`
              // is the element's, which for an excerpt is two minutes — every interval
              // in the case would then fall outside the ribbon's range and it
              // would render empty.
              durationSeconds={
                findPart(state.activeCaseId, state.activePart)?.durationSeconds || state.duration
              }
              currentTime={sourceTime}
              playableWindow={
                state.isExcerpt
                  ? { start: state.timeOffset, end: state.timeOffset + state.duration }
                  : null
              }
              // The ribbon speaks in part-time; the element only understands
              // its own clock.
              onSeek={(seconds) => handleSeek(seconds - state.timeOffset)}
            />
            </div>
          )}

          <div data-tour="library">
          <CaseLibrary
            attachedFiles={attachedFiles}
            activePartId={activePartId}
            onAttachFiles={handleAttachFiles}
            onSelectPart={handleSelectPart}
          />
          </div>
        </div>

        {/* CENTER: recorded facts, then predictions */}
        <div data-tour="facts" className="lg:col-span-3 flex flex-col space-y-4">
          <Panels
            currentObservation={state.currentObservation}
            focusedInstrumentId={state.focusedInstrumentId}
            hasVideo={state.videoSourceType !== 'none'}
            recordedTaskLabel={recordedTaskNow?.label || null}
            recordedTaskDisplay={recordedTaskNow?.display || null}
            onSelectInstrument={(id) =>
              setState((p) => ({
                ...p,
                focusedInstrumentId: p.focusedInstrumentId === id ? null : id,
              }))
            }
          />
          <GroundTruthPanel recorded={recordedNow} currentTime={sourceTime} />
          <ModelPanel status={vision} info={visionRef.current?.details || null} />
        </div>

        {/* RIGHT: assistant */}
        <div data-tour="assistant" className="lg:col-span-3 flex flex-col">
          <ChatAssistant
            currentObservation={state.currentObservation}
            messages={state.chatMessages}
            isTtsEnabled={state.isTtsEnabled}
            isMicListening={state.isMicListening}
            isMicStarting={state.isMicStarting}
            micStatus={micStatus}
            isThinking={state.isAssistantThinking}
            onSendMessage={handleSendMessage}
            onToggleTts={handleToggleTts}
            onToggleMic={handleToggleMic}
          />
        </div>
      </main>

      <GuidedTour phase={tourPhase} onPhaseChange={setTourPhase} />

      <footer className="px-5 py-2.5 text-[11px] text-fg2 border-t border-line text-center">
        Recorded labels from the SurgVU 2024 release · predictions from on-device models, always shown with a
        confidence · educational and research use only, not a medical device.
        {state.duration > 0 && ` · part length ${formatClock(state.duration)}`}
        {' · '}
        <a href="#/" className="text-fg3 hover:text-accent underline underline-offset-2 transition-colors">
          about this project
        </a>
      </footer>
    </div>
  );
}
