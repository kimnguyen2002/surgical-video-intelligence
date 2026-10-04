import {
  ApiFailure,
  ApiHealth,
  AssistantRole,
  ChatMessage,
  FrameObservation,
  SpecialtyId,
  VideoSourceType,
} from './types';

export interface AppState {
  specialtyId: SpecialtyId;
  /**
   * The Gemini model to use *if* the visitor supplied their own key.
   *
   * A plain string rather than a union of model ids. The union had to be
   * edited every time Google retired a model, and a stale entry failed as a
   * 404 at the worst moment. The list is now read from the key's own
   * catalogue at runtime, so the type cannot go out of date.
   */
  modelId: string;
  /**
   * Whether to spend the visitor's quota on generated prose.
   *
   * Off by default and meaningless without a key. The grounded assistant
   * answers factual questions about the video either way, so this is an
   * upgrade rather than a switch between working and broken.
   */
  useGenerative: boolean;
  role: AssistantRole;
  /**
   * Whether the on-device models are running over the video.
   *
   * This replaces `isAnalyzing` + `isLoopRunning` + `intervalSeconds`. Those
   * three existed because analysis was a metered network call: you asked for
   * one, or asked for one every N seconds. Inference is now local and free, so
   * there is only "running" and "not running".
   */
  isLiveDetecting: boolean;
  isPlaying: boolean;
  currentObservation: FrameObservation | null;
  timelineFrames: FrameObservation[];

  // --- Video source -----------------------------------------------------
  videoSourceType: VideoSourceType;
  videoFilename: string;
  videoUrl: string | null;
  /** Set only for a SurgVU library part, which is what unlocks ground truth. */
  activeCaseId: string | null;
  activePart: number | null;
  currentTime: number;
  duration: number;
  /**
   * Seconds to add to `currentTime` to get the position in the **original**
   * recording.
   *
   * Non-zero only while a bundled excerpt is playing: those are cut from the
   * middle of a case, so the player's clock starts at 0:00 while the frames
   * belong to, say, 1:14:50. Every annotation lookup goes through
   * `currentTime + timeOffset`. Reading the player's own clock instead would
   * be wrong by up to four hours and would look entirely plausible — the panel
   * would show real instruments from the opening of the case while the viewer
   * watches its middle.
   */
  timeOffset: number;
  /** True while the source is a bundled excerpt rather than a full part. */
  isExcerpt: boolean;

  // --- Assistant --------------------------------------------------------
  isTtsEnabled: boolean;
  isMicListening: boolean;
  /** Between the click and the browser granting the microphone. */
  isMicStarting: boolean;
  chatMessages: ChatMessage[];
  /** An answer is being composed. Only ever true while a generative call is in flight. */
  isAssistantThinking: boolean;
  focusedInstrumentId: string | null;
  selectedTimelineFrameId: string | null;

  // --- API status -------------------------------------------------------
  apiHealth: ApiHealth | null;
  isCheckingHealth: boolean;
  lastFailure: ApiFailure | null;
  bannerDismissed: boolean;
}

/**
 * The opening state of a fresh session.
 *
 * It is deliberately empty. The previous default shipped three instruments
 * with bounding boxes and confidence percentages — 88% monopolar curved
 * scissors, 82% force bipolar — drawn over a canvas cartoon before any model
 * had run or any video had loaded. Anyone opening the app saw a working
 * detector that did not exist.
 *
 * Nothing is asserted here until a real video is attached and a real call
 * returns.
 */
export const initialDefaultState: AppState = {
  specialtyId: 'general',
  modelId: '',
  useGenerative: false,
  role: 'Resident',
  isLiveDetecting: false,
  isPlaying: false,
  currentObservation: null,
  timelineFrames: [],

  videoSourceType: 'none',
  videoFilename: 'no video loaded',
  videoUrl: null,
  activeCaseId: null,
  activePart: null,
  currentTime: 0,
  duration: 0,
  timeOffset: 0,
  isExcerpt: false,

  isTtsEnabled: false,
  isMicListening: false,
  isMicStarting: false,
  chatMessages: [
    {
      id: 'welcome-msg',
      sender: 'assistant',
      text: `Ready — surgical video, on-device detection, and the dataset's own labels side by side.\n\nEverything runs in this browser: no server, no API key, nothing uploaded. Pick a case below to start, then ask me what the release records for any moment of it.`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ],
  isAssistantThinking: false,
  focusedInstrumentId: null,
  selectedTimelineFrameId: null,

  apiHealth: null,
  isCheckingHealth: false,
  lastFailure: null,
  bannerDismissed: false,
};
