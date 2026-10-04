import { describe, expect, it } from 'vitest';
import { decodeTokens, logMelSpectrogram, N_FRAMES, N_MELS, resampleTo16k } from './speechFeatures';

describe('speech front end', () => {
  it('produces Whisper-shaped features, flat for silence', () => {
    const mel = new Float32Array(N_MELS * 201).fill(0.01);
    const out = logMelSpectrogram(new Float32Array(16000), mel);
    expect(out.length).toBe(N_MELS * N_FRAMES);
    // Silence is the log floor everywhere, normalised by (x + 4) / 4.
    expect(new Set(Array.from(out.slice(0, 50)))).toEqual(new Set([(-10 + 4) / 4]));
  });

  it('resamples 48 kHz to a third of the length', () => {
    expect(resampleTo16k(new Float32Array(48000), 48000).length).toBe(16000);
  });

  it('decodes byte-level BPE and drops special tokens', () => {
    const vocab = ['What', 'Ġinstruments', '?'];
    expect(decodeTokens([0, 1, 2, 50257], vocab, 50257)).toBe('What instruments?');
  });
});
