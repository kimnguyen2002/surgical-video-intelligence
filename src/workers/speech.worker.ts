/// <reference lib="webworker" />

/**
 * On-device speech recognition: Whisper tiny.en (int8 ONNX) on onnxruntime-web.
 *
 * Audio never leaves the page. The browser's built-in recogniser in Chrome
 * streams audio to a Google service, which fails without that connection and
 * contradicts this app's promise that nothing is uploaded.
 *
 * Greedy decoding with the non-cached decoder: a spoken question is a few
 * dozen tokens, so re-running the short sequence each step is cheaper to
 * maintain than past-key-value plumbing and fast enough at this length.
 */

import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';
import mjsUrl from 'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url';
import { decodeTokens, logMelSpectrogram, N_FRAMES, N_MELS, resampleTo16k } from '../services/speechFeatures';

ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };
ort.env.wasm.numThreads =
  typeof SharedArrayBuffer !== 'undefined' && (self as any).crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 2)
    : 1;
ort.env.logLevel = 'error';

interface SpeechConfig {
  startOfTranscript: number;
  noTimestamps: number;
  endOfText: number;
  firstSpecial: number;
  suppressTokens: number[];
  beginSuppressTokens: number[];
  maxNewTokens: number;
}

let encoder: ort.InferenceSession | null = null;
let decoder: ort.InferenceSession | null = null;
let config: SpeechConfig | null = null;
let vocab: string[] = [];
let melFilters: Float32Array | null = null;

async function init(baseUrl: string) {
  if (encoder) return;
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  };
  const [cfg, voc, mel, enc, dec] = await Promise.all([
    fetch(`${baseUrl}speech.json`).then((r) => r.json()),
    fetch(`${baseUrl}vocab.json`).then((r) => r.json()),
    fetch(`${baseUrl}mel_filters.bin`).then((r) => r.arrayBuffer()),
    ort.InferenceSession.create(`${baseUrl}encoder.onnx`, options),
    ort.InferenceSession.create(`${baseUrl}decoder.onnx`, options),
  ]);
  config = cfg;
  vocab = voc;
  melFilters = new Float32Array(mel);
  encoder = enc;
  decoder = dec;
}

async function transcribe(audio: Float32Array, sampleRate: number): Promise<string> {
  const pcm = resampleTo16k(audio, sampleRate);
  const features = logMelSpectrogram(pcm, melFilters!);

  // Fetch only the hidden state. The export also lists every layer's
  // attention matrix as an output (6 x 1500 x 1500 each); asking for them
  // would hold hundreds of megabytes for nothing.
  const encoded = await encoder!.run(
    { input_features: new ort.Tensor('float32', features, [1, N_MELS, N_FRAMES]) },
    ['last_hidden_state']
  );
  const hidden = encoded.last_hidden_state;

  const cfg = config!;
  const suppress = new Set(cfg.suppressTokens);
  const tokens = [cfg.startOfTranscript, cfg.noTimestamps];
  const generated: number[] = [];

  for (let step = 0; step < cfg.maxNewTokens; step++) {
    const ids = new ort.Tensor('int64', BigInt64Array.from(tokens.map((t) => BigInt(t))), [1, tokens.length]);
    const out = await decoder!.run({ input_ids: ids, encoder_hidden_states: hidden }, ['logits']);
    const logits = out.logits.data as Float32Array;
    const vocabSize = out.logits.dims[2];
    const offset = (tokens.length - 1) * vocabSize;

    let best = -1;
    let bestScore = -Infinity;
    for (let id = 0; id < vocabSize; id++) {
      if (suppress.has(id)) continue;
      if (step === 0 && cfg.beginSuppressTokens.includes(id)) continue;
      // Timestamps and other control tokens are never wanted in a transcript.
      if (id >= cfg.firstSpecial && id !== cfg.endOfText) continue;
      const score = logits[offset + id];
      if (score > bestScore) {
        bestScore = score;
        best = id;
      }
    }
    if (best === cfg.endOfText || best < 0) break;
    tokens.push(best);
    generated.push(best);
  }

  return decodeTokens(generated, vocab, cfg.firstSpecial);
}

self.onmessage = async (event: MessageEvent) => {
  const msg = event.data;
  try {
    if (msg.type === 'init') {
      await init(msg.baseUrl);
      (self as any).postMessage({ type: 'ready' });
      return;
    }
    if (msg.type === 'transcribe') {
      const started = performance.now();
      const text = await transcribe(msg.audio, msg.sampleRate);
      (self as any).postMessage({ type: 'result', id: msg.id, text, ms: performance.now() - started });
    }
  } catch (err: any) {
    (self as any).postMessage({ type: 'error', id: msg?.id, message: err?.message || String(err) });
  }
};
