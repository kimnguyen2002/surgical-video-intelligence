/**
 * Voice in and out, through the browser.
 *
 * This reverses a decision made in the previous build, and the reversal is
 * worth explaining because the old reasoning was correct for where that build
 * ran and wrong for where this one does.
 *
 * That build used Gemini for both directions. The Web Speech API had failed
 * there: `webkitSpeechRecognition` needs the embedding frame to grant a
 * microphone permission, and inside a **cross-origin iframe** — which is
 * exactly where AI Studio ran the app — it fails with a bare `not-allowed` and
 * no visible error, so the button simply did nothing. Routing speech through a
 * multimodal model sidestepped the iframe entirely.
 *
 * This build is served from its own origin as a top-level document. The iframe
 * constraint is gone, and with it the justification. Speech out uses the
 * browser's `speechSynthesis`. Speech in uses an on-device Whisper model
 * rather than `SpeechRecognition`, whose Chrome implementation streams audio
 * to a Google service (see `Dictation` below). Paying a
 * model to transcribe speech the browser already transcribes would be spending
 * someone's quota to solve a problem this deployment does not have.
 *
 * What is kept from the old reasoning
 * -----------------------------------
 * The failure modes it documented are real and are handled rather than
 * assumed away:
 *
 * - `speechSynthesis.getVoices()` returns an empty array until the voice list
 *   loads asynchronously, so the first `speak()` after page load would pick no
 *   voice. `voicesReady()` waits for the `voiceschanged` event once.
 * - Dictation does not use `SpeechRecognition`: Firefox lacks it and Chrome's
 *   streams audio to a server. `recognitionSupported()` checks only for a
 *   microphone, Web Audio, workers and WebAssembly.
 */

export interface VoiceFailure {
  message: string;
  hint?: string;
}

// ---------------------------------------------------------------------------
// Capability detection
// ---------------------------------------------------------------------------

/**
 * Dictation needs a microphone, Web Audio, a worker and WebAssembly: every
 * current browser, including Firefox, which has no SpeechRecognition at all.
 */
export function recognitionSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof AudioContext !== 'undefined' &&
    typeof Worker !== 'undefined' &&
    typeof WebAssembly !== 'undefined'
  );
}

export function synthesisSupported(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}

// ---------------------------------------------------------------------------
// Speech out
// ---------------------------------------------------------------------------

let voicesLoaded: Promise<SpeechSynthesisVoice[]> | null = null;

/**
 * Resolve once the browser has actually populated its voice list.
 *
 * Chrome returns `[]` from `getVoices()` on first call and fires
 * `voiceschanged` a moment later. Code that reads the list synchronously at
 * startup therefore picks nothing and speaks in the default voice — or, on
 * some platforms, not at all.
 */
function voicesReady(): Promise<SpeechSynthesisVoice[]> {
  if (!synthesisSupported()) return Promise.resolve([]);
  if (voicesLoaded) return voicesLoaded;

  voicesLoaded = new Promise((resolve) => {
    const existing = window.speechSynthesis.getVoices();
    if (existing.length > 0) {
      resolve(existing);
      return;
    }
    const timer = setTimeout(() => resolve(window.speechSynthesis.getVoices()), 1500);
    window.speechSynthesis.addEventListener(
      'voiceschanged',
      () => {
        clearTimeout(timer);
        resolve(window.speechSynthesis.getVoices());
      },
      { once: true }
    );
  });
  return voicesLoaded;
}

/**
 * Prefer a natural-sounding English voice.
 *
 * The default voice on macOS and Windows is a formant synthesiser that makes
 * surgical terminology hard to follow. The neural voices shipped alongside it
 * are markedly clearer, and they are identifiable by name.
 */
async function pickVoice(preferredLang: string): Promise<SpeechSynthesisVoice | null> {
  const voices = await voicesReady();
  if (voices.length === 0) return null;

  const langMatches = voices.filter((v) => v.lang.startsWith(preferredLang.slice(0, 2)));
  const pool = langMatches.length > 0 ? langMatches : voices;

  const preferred = ['Samantha', 'Google US English', 'Microsoft Aria', 'Microsoft Jenny', 'Daniel'];
  for (const name of preferred) {
    const found = pool.find((v) => v.name.includes(name));
    if (found) return found;
  }
  return pool.find((v) => v.localService) || pool[0];
}

export class Speaker {
  private current: SpeechSynthesisUtterance | null = null;

  /** Strip markdown so the reader does not pronounce asterisks and brackets. */
  private static clean(text: string): string {
    return text
      .replace(/```[\s\S]*?```/g, ' code block omitted. ')
      .replace(/[*_#`]/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/^[-•]\s*/gm, ', ')
      .replace(/https?:\/\/\S+/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  get speaking(): boolean {
    return synthesisSupported() && window.speechSynthesis.speaking;
  }

  async speak(text: string, lang = 'en-US'): Promise<void> {
    if (!synthesisSupported()) {
      throw { message: 'This browser has no speech synthesis.' } satisfies VoiceFailure;
    }

    this.stop();

    // Read-back happens between surgical steps, so length is capped rather
    // than letting a long answer become two minutes of audio nobody can skip
    // through.
    const spoken = Speaker.clean(text).slice(0, 1200);
    if (!spoken) return;

    const utterance = new SpeechSynthesisUtterance(spoken);
    utterance.lang = lang;
    utterance.rate = 1.02;
    utterance.pitch = 1;

    const voice = await pickVoice(lang);
    if (voice) utterance.voice = voice;

    this.current = utterance;
    return new Promise((resolve, reject) => {
      utterance.onend = () => {
        this.current = null;
        resolve();
      };
      utterance.onerror = (event) => {
        this.current = null;
        // `interrupted` and `canceled` are what `stop()` produces; they are the
        // expected outcome of the user pressing stop, not failures to report.
        if (event.error === 'interrupted' || event.error === 'canceled') resolve();
        else reject({ message: `Speech synthesis failed (${event.error}).` } satisfies VoiceFailure);
      };
      window.speechSynthesis.speak(utterance);
    });
  }

  stop(): void {
    if (!synthesisSupported()) return;
    window.speechSynthesis.cancel();
    this.current = null;
  }
}

// ---------------------------------------------------------------------------
// Speech in
// ---------------------------------------------------------------------------

export type DictationStatus = 'loading' | 'listening' | 'transcribing';

export interface DictationHandlers {
  /** Progress through loading the model, listening and transcribing. */
  onStatus?: (status: DictationStatus) => void;
  /** Fired once with the transcript. */
  onFinal: (text: string) => void;
  onError?: (failure: VoiceFailure) => void;
  onEnd?: () => void;
}

/** Stop this long after the speaker goes quiet. */
const TRAILING_SILENCE_MS = 1200;
/** Give up if nobody speaks within this long. */
const NO_SPEECH_TIMEOUT_MS = 7000;
/** A question, not a dictation session. */
const MAX_RECORDING_MS = 15000;

/**
 * Microphone dictation with an on-device Whisper model (see speech.worker.ts).
 *
 * Chrome's built-in recogniser streams audio to a Google service: it fails
 * without that connection ("could not reach its network service"), Firefox
 * does not have one at all, and it would contradict this app's promise that
 * nothing leaves the browser. Here the audio is recorded, transcribed in a
 * worker, and discarded.
 *
 * One question per press: recording stops on its own after a short silence,
 * or when the button is pressed again. The model (about 40 MB) downloads on
 * first use and is cached by the browser after that.
 */
export class Dictation {
  private worker: Worker | null = null;
  private ready: Promise<void> | null = null;
  private loaded = false;
  private nextId = 1;
  private pending = new Map<number, { resolve: (text: string) => void; reject: (e: Error) => void }>();

  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private chunks: Float32Array[] = [];
  private handlers: DictationHandlers | null = null;
  private heardSpeech = false;
  private active = false;
  private timers: number[] = [];

  get listening(): boolean {
    return this.active;
  }

  private loadModel(): Promise<void> {
    if (this.ready) return this.ready;
    const worker = new Worker(new URL('../workers/speech.worker.ts', import.meta.url), { type: 'module' });
    this.worker = worker;
    this.ready = new Promise<void>((resolve, reject) => {
      worker.onmessage = (event: MessageEvent) => {
        const msg = event.data;
        if (msg.type === 'ready') {
          this.loaded = true;
          resolve();
        }
        else if (msg.type === 'result') {
          this.pending.get(msg.id)?.resolve(msg.text);
          this.pending.delete(msg.id);
        } else if (msg.type === 'error') {
          if (msg.id && this.pending.has(msg.id)) {
            this.pending.get(msg.id)!.reject(new Error(msg.message));
            this.pending.delete(msg.id);
          } else {
            reject(new Error(msg.message));
          }
        }
      };
      worker.onerror = (event) => reject(new Error(event.message || 'The speech worker failed to start.'));
      worker.postMessage({ type: 'init', baseUrl: `${import.meta.env.BASE_URL}models/whisper/` });
    }).catch((err) => {
      // Let the next press try again rather than caching a failure forever.
      this.ready = null;
      this.worker?.terminate();
      this.worker = null;
      throw err;
    });
    return this.ready;
  }

  async start(handlers: DictationHandlers): Promise<void> {
    if (!recognitionSupported()) {
      handlers.onError?.({ message: 'This browser cannot record audio.', hint: 'Type the question instead.' });
      handlers.onEnd?.();
      return;
    }
    this.cancel();
    this.handlers = handlers;

    // Start loading the model, but do not wait for it: recording begins as
    // soon as the microphone is granted, so a first-time visitor's question
    // is captured while the model is still downloading.
    this.loadModel().catch(() => {});

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      if (this.handlers !== handlers) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      this.stream = stream;
    } catch (err: any) {
      const denied = err?.name === 'NotAllowedError' || err?.name === 'SecurityError';
      this.fail(
        denied
          ? { message: 'Microphone access was denied.', hint: 'Allow the microphone for this site in the address bar, then try again.' }
          : err?.name === 'NotFoundError'
            ? { message: 'No microphone was found.', hint: 'Connect one, or type the question instead.' }
            : { message: `Voice input could not start: ${err?.message || err}`, hint: 'Type the question instead.' }
      );
      return;
    }

    // The device's own rate; resampling to 16 kHz happens in the worker.
    // Forcing a 16 kHz AudioContext fails in Firefox when the mic runs at 48 kHz.
    const context = new AudioContext();
    const source = context.createMediaStreamSource(this.stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    this.context = context;
    this.processor = processor;
    this.chunks = [];
    this.heardSpeech = false;
    this.active = true;

    let noiseFloor = 0;
    let calibrated = 0;
    let lastVoiceAt = 0;
    const startedAt = performance.now();

    processor.onaudioprocess = (event) => {
      if (!this.active) return;
      const data = new Float32Array(event.inputBuffer.getChannelData(0));
      this.chunks.push(data);

      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const rms = Math.sqrt(sum / data.length);
      const now = performance.now();

      // The first ~300 ms set the room's noise floor.
      if (now - startedAt < 300) {
        noiseFloor = (noiseFloor * calibrated + rms) / (calibrated + 1);
        calibrated++;
        return;
      }
      const threshold = Math.max(0.012, noiseFloor * 2.5);
      if (rms > threshold) {
        this.heardSpeech = true;
        lastVoiceAt = now;
      } else if (this.heardSpeech && now - lastVoiceAt > TRAILING_SILENCE_MS) {
        void this.finish();
      }
    };
    source.connect(processor);
    processor.connect(context.destination);

    handlers.onStatus?.('listening');
    this.timers.push(
      window.setTimeout(() => {
        if (this.active && !this.heardSpeech) {
          this.fail({ message: 'Nothing was heard. Try again, closer to the microphone.' });
        }
      }, NO_SPEECH_TIMEOUT_MS),
      window.setTimeout(() => void this.finish(), MAX_RECORDING_MS)
    );
  }

  /** Stop recording and transcribe what was said. */
  async finish(): Promise<void> {
    if (!this.active) return;
    const handlers = this.handlers;
    const sampleRate = this.context?.sampleRate ?? 48000;
    const audio = concat(this.chunks);
    const heard = this.heardSpeech;
    this.release();

    if (!handlers) return;
    if (!heard) {
      handlers.onError?.({ message: 'Nothing was heard. Try again, closer to the microphone.' });
      handlers.onEnd?.();
      return;
    }

    try {
      if (!this.loaded) handlers.onStatus?.('loading');
      await this.loadModel();
      handlers.onStatus?.('transcribing');
      const id = this.nextId++;
      const text = await new Promise<string>((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        this.worker!.postMessage({ type: 'transcribe', id, audio, sampleRate }, [audio.buffer]);
      });
      const cleaned = text.replace(/\s+/g, ' ').trim();
      // Whisper's habit on near-silence is a stock phrase; treat it as nothing.
      if (!cleaned || /^(\[.*\]|\(.*\)|you\.?|thank you\.?|thanks for watching\.?)$/i.test(cleaned)) {
        handlers.onError?.({ message: 'I could not make out any words. Try again, a little closer to the microphone.' });
      } else {
        handlers.onFinal(cleaned);
      }
    } catch (err: any) {
      handlers.onError?.({ message: `Transcription failed: ${err?.message || err}`, hint: 'Type the question instead.' });
    }
    handlers.onEnd?.();
  }

  /** Stop without transcribing. */
  cancel(): void {
    const handlers = this.handlers;
    const wasActive = this.active;
    this.release();
    if (wasActive) handlers?.onEnd?.();
  }

  /** Kept for callers that used the old recogniser's API. */
  stop(): void {
    this.cancel();
  }

  private fail(failure: VoiceFailure) {
    const handlers = this.handlers;
    this.release();
    handlers?.onError?.(failure);
    handlers?.onEnd?.();
  }

  private release() {
    this.active = false;
    this.timers.forEach((t) => clearTimeout(t));
    this.timers = [];
    if (this.processor) this.processor.onaudioprocess = null;
    this.processor?.disconnect();
    this.processor = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    void this.context?.close().catch(() => {});
    this.context = null;
    this.chunks = [];
    this.handlers = null;
  }
}

function concat(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

export const speaker = new Speaker();
export const dictation = new Dictation();
