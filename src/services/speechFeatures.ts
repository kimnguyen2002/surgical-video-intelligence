/**
 * Whisper's audio front end and tokenizer, in plain TypeScript.
 *
 * A port of `WhisperFeatureExtractor` (Hugging Face) so the speech model can
 * run on the same onnxruntime-web the vision models use, instead of pulling in
 * a second ML runtime for one microphone button. Kept free of DOM and worker
 * APIs so it can be tested against the Python reference in Node.
 */

export const SAMPLE_RATE = 16000;
export const N_FFT = 400;
export const HOP = 160;
export const N_MELS = 80;
export const N_SAMPLES = 480000; // 30 s, the model's fixed window
export const N_FRAMES = 3000;
const N_BINS = N_FFT / 2 + 1; // 201

/**
 * Resample to 16 kHz. Each output sample averages the input span it covers,
 * which doubles as the anti-aliasing filter when downsampling from 44.1/48 kHz.
 */
export function resampleTo16k(input: Float32Array, rate: number): Float32Array {
  if (rate === SAMPLE_RATE) return input;
  const ratio = rate / SAMPLE_RATE;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const start = i * ratio;
    const end = Math.min(input.length, start + ratio);
    let sum = 0;
    let count = 0;
    for (let j = Math.floor(start); j < end; j++) {
      sum += input[j];
      count++;
    }
    out[i] = count > 0 ? sum / count : 0;
  }
  return out;
}

let tables: { cos: Float32Array; sin: Float32Array; window: Float32Array } | null = null;

function dftTables() {
  if (tables) return tables;
  const cos = new Float32Array(N_BINS * N_FFT);
  const sin = new Float32Array(N_BINS * N_FFT);
  for (let k = 0; k < N_BINS; k++) {
    for (let n = 0; n < N_FFT; n++) {
      const angle = (2 * Math.PI * k * n) / N_FFT;
      cos[k * N_FFT + n] = Math.cos(angle);
      sin[k * N_FFT + n] = Math.sin(angle);
    }
  }
  // Periodic Hann, as numpy/HF use for spectrograms.
  const window = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  tables = { cos, sin, window };
  return tables;
}

/**
 * Log-mel spectrogram, shape [80, 3000], matching WhisperFeatureExtractor:
 * pad to 30 s, centre frames with reflect padding, power spectrum, Slaney mel
 * filters, log10, clamp to (max - 8), then (x + 4) / 4.
 *
 * `melFilters` is row-major [80][201].
 */
export function logMelSpectrogram(audio: Float32Array, melFilters: Float32Array): Float32Array {
  const { cos, sin, window } = dftTables();

  const signal = new Float32Array(N_SAMPLES);
  signal.set(audio.subarray(0, N_SAMPLES));
  const realLength = Math.min(audio.length, N_SAMPLES);

  const half = N_FFT / 2;
  // Reflect padding on the 30 s padded signal, as numpy's centre=True does.
  const sample = (index: number): number => {
    if (index < 0) return signal[-index];
    if (index >= N_SAMPLES) return signal[2 * (N_SAMPLES - 1) - index];
    return signal[index];
  };

  const LOG_FLOOR = -10; // log10(1e-10): a frame of pure silence
  const out = new Float32Array(N_MELS * N_FRAMES).fill(LOG_FLOOR);

  // Frames entirely inside the zero padding are silent; skip the arithmetic.
  const lastFrame = Math.min(N_FRAMES, Math.ceil((realLength + half) / HOP) + 1);

  const frame = new Float32Array(N_FFT);
  const power = new Float32Array(N_BINS);
  for (let t = 0; t < lastFrame; t++) {
    const start = t * HOP - half;
    for (let n = 0; n < N_FFT; n++) frame[n] = sample(start + n) * window[n];

    for (let k = 0; k < N_BINS; k++) {
      let re = 0;
      let im = 0;
      const row = k * N_FFT;
      for (let n = 0; n < N_FFT; n++) {
        re += frame[n] * cos[row + n];
        im -= frame[n] * sin[row + n];
      }
      power[k] = re * re + im * im;
    }

    for (let m = 0; m < N_MELS; m++) {
      let energy = 0;
      const row = m * N_BINS;
      for (let k = 0; k < N_BINS; k++) energy += melFilters[row + k] * power[k];
      out[m * N_FRAMES + t] = Math.log10(Math.max(energy, 1e-10));
    }
  }

  let max = -Infinity;
  for (let i = 0; i < out.length; i++) if (out[i] > max) max = out[i];
  const floor = max - 8;
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], floor) + 4) / 4;
  return out;
}

let byteDecoder: Map<string, number> | null = null;

/** GPT-2's byte-to-unicode table, inverted: BPE token characters back to bytes. */
function bytesFromUnicode(): Map<string, number> {
  if (byteDecoder) return byteDecoder;
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  byteDecoder = new Map(bs.map((b, i) => [String.fromCharCode(cs[i]), b]));
  return byteDecoder;
}

/** Token ids to text, dropping special tokens. */
export function decodeTokens(ids: number[], vocab: string[], firstSpecial: number): string {
  const decoder = bytesFromUnicode();
  const bytes: number[] = [];
  for (const id of ids) {
    if (id >= firstSpecial) continue;
    for (const ch of vocab[id] ?? '') {
      const b = decoder.get(ch);
      if (b !== undefined) bytes.push(b);
    }
  }
  return new TextDecoder().decode(new Uint8Array(bytes)).trim();
}
