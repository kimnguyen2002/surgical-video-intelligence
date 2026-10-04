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
 * constraint is gone, and with it the justification: `SpeechRecognition` and
 * `speechSynthesis` work here, cost nothing, send no audio to any API the
 * visitor is paying for, and keep working with the network off. Paying a
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
 * - `SpeechRecognition` is unevenly supported — Chrome, Edge and Safari have
 *   it; Firefox does not. `recognitionSupported()` is checked before the
 *   microphone button is offered at all, rather than offering a control that
 *   silently does nothing.
 */

export interface VoiceFailure {
  message: string;
  hint?: string;
}

// ---------------------------------------------------------------------------
// Capability detection
// ---------------------------------------------------------------------------

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string; message?: string }) => void) | null;
  onend: (() => void) | null;
}

interface SpeechRecognitionEventLike {
  resultIndex: number;
  results: ArrayLike<
    ArrayLike<{ transcript: string; confidence: number }> & { isFinal: boolean }
  >;
}

function recognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

export function recognitionSupported(): boolean {
  return recognitionCtor() !== null;
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

export interface DictationHandlers {
  /** Fired repeatedly as the browser refines its guess. */
  onInterim?: (text: string) => void;
  /** Fired once with the settled transcript. */
  onFinal: (text: string) => void;
  onError?: (failure: VoiceFailure) => void;
  onEnd?: () => void;
}

/**
 * Microphone dictation, using the browser's own recogniser.
 *
 * `continuous` is off: this dictates one question, then stops. Leaving it on
 * means an open microphone for as long as the page is in the foreground, which
 * is both a privacy cost the demo has no reason to impose and a reliable way
 * to accumulate background chatter into the question box.
 */
export class Dictation {
  private recognition: SpeechRecognitionLike | null = null;
  private active = false;

  get listening(): boolean {
    return this.active;
  }

  start(handlers: DictationHandlers, lang = 'en-US'): void {
    const Ctor = recognitionCtor();
    if (!Ctor) {
      handlers.onError?.({
        message: 'This browser has no speech recognition.',
        hint: 'Chrome, Edge and Safari support it; Firefox does not. Type the question instead.',
      });
      return;
    }

    this.stop();

    const recognition = new Ctor();
    recognition.lang = lang;
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onresult = (event) => {
      let interim = '';
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const transcript = result[0]?.transcript || '';
        if (result.isFinal) final += transcript;
        else interim += transcript;
      }
      if (interim) handlers.onInterim?.(interim.trim());
      if (final.trim()) handlers.onFinal(final.trim());
    };

    recognition.onerror = (event) => {
      this.active = false;
      const failures: Record<string, VoiceFailure> = {
        'not-allowed': {
          message: 'Microphone access was denied.',
          hint: 'Allow the microphone for this site in the address bar, then try again.',
        },
        'service-not-allowed': {
          message: 'The browser blocked its speech service.',
          hint: 'This usually means the page is not on a secure origin. Type the question instead.',
        },
        'no-speech': { message: 'Nothing was heard. Try again, closer to the microphone.' },
        network: {
          message: 'The recogniser could not reach its network service.',
          hint: "Chrome's recogniser is server-backed and needs a connection. Type the question instead.",
        },
        aborted: { message: '' },
      };
      const failure = failures[event.error] || {
        message: `Speech recognition failed (${event.error}).`,
      };
      // An abort is what `stop()` causes; reporting it as an error would make
      // every deliberate cancellation look like a fault.
      if (failure.message) handlers.onError?.(failure);
    };

    recognition.onend = () => {
      this.active = false;
      handlers.onEnd?.();
    };

    this.recognition = recognition;
    this.active = true;
    try {
      recognition.start();
    } catch (err) {
      this.active = false;
      handlers.onError?.({
        message: err instanceof Error ? err.message : 'The recogniser would not start.',
      });
    }
  }

  stop(): void {
    if (!this.recognition) return;
    try {
      this.recognition.abort();
    } catch {
      // Already stopped.
    }
    this.recognition = null;
    this.active = false;
  }
}

export const speaker = new Speaker();
export const dictation = new Dictation();
