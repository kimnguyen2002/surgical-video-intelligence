/**
 * Client for the in-browser vision worker.
 *
 * The contract this enforces is the one that makes live detection usable in an
 * operating room: **one frame in flight, newest wins, never queue.**
 *
 * Queueing is the obvious implementation and the wrong one. If inference takes
 * 40 ms and frames arrive every 16 ms, a queue grows without bound and the
 * boxes on screen fall further behind the video every second — the overlay
 * ends up describing an instrument position from a minute ago while looking
 * perfectly live. Dropping instead means throughput falls on a slow machine
 * but the boxes always describe a frame from ~one inference ago. Latency stays
 * bounded; only the frame rate gives way.
 */

/** A class activation map, computed from the forward pass. See export_cam.py. */
export interface ActivationMap {
  /** Row-major `height * width`, normalised to 0..1. */
  values: number[];
  width: number;
  height: number;
  label: string;
  confidence: number;
  /** Where it belongs on the source frame, as fractions. The classifier sees
   *  only a centre crop, so this is not the whole frame. */
  region: { x: number; y: number; width: number; height: number };
}

export interface LocalDetection {
  label: string;
  confidence: number;
  box: { ymin: number; xmin: number; ymax: number; xmax: number };
}

export interface LocalVisionResult {
  detections: LocalDetection[];
  tools?: { label: string; confidence: number }[];
  task?: { label: string; confidence: number };
  timestampSeconds: number;
  totalMs: number;
  backend: string;
  timings: Record<string, number>;
  /**
   * The out-of-domain verdict for this frame.
   *
   * When `outOfDomain` is true the three prediction fields above are empty —
   * the worker withholds them rather than trusting consumers to check this
   * flag first. `distance` is the cosine distance from the training
   * distribution's centroid, which the model panel shows so the refusal is a
   * number rather than an assertion.
   */
  domain?: { outOfDomain: boolean; distance: number };
  /** Null when the guard refused the frame, or when no class scored at all. */
  activation?: ActivationMap | null;
}

export interface LocalVisionInfo {
  backend: string;
  warmupMs: number;
  /** Null when no calibrated reference shipped; the guard is then inactive. */
  domainGuard: { threshold: number; dim: number } | null;
  /** Whether CAM weights shipped; false means the heatmap is unavailable. */
  activationMap: boolean;
  models: {
    name: string;
    classes: string[];
    imageSize: number;
    metric?: Record<string, number>;
    provenance?: string;
  }[];
}

/**
 * The classifiers answer "what procedure step is this" and "which instruments
 * are mounted" — state that moves on the scale of minutes. Running them on
 * every frame would double the per-frame cost to re-answer a question whose
 * answer has not changed. The detector, which locates instruments that move
 * continuously, runs on every frame.
 */
const CLASSIFIER_EVERY_N_FRAMES = 12;

export class LocalVision {
  private worker: Worker | null = null;
  private busy = false;
  private nextId = 1;
  private frameCount = 0;
  private info: LocalVisionInfo | null = null;

  private readonly recentLatencies: number[] = [];
  private lastResultAt = 0;
  private readonly recentIntervals: number[] = [];

  onResult: ((result: LocalVisionResult) => void) | null = null;
  onError: ((message: string) => void) | null = null;

  get ready(): boolean {
    return this.info !== null;
  }

  get details(): LocalVisionInfo | null {
    return this.info;
  }

  /** Median inference latency and achieved rate over the last ~30 frames. */
  get stats(): { medianMs: number; fps: number; samples: number } {
    if (this.recentLatencies.length === 0) return { medianMs: 0, fps: 0, samples: 0 };
    const sorted = [...this.recentLatencies].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const meanInterval =
      this.recentIntervals.length > 0
        ? this.recentIntervals.reduce((a, b) => a + b, 0) / this.recentIntervals.length
        : 0;
    return {
      medianMs: Math.round(median),
      fps: meanInterval > 0 ? Math.round((1000 / meanInterval) * 10) / 10 : 0,
      samples: this.recentLatencies.length,
    };
  }

  async init(): Promise<LocalVisionInfo> {
    if (this.info) return this.info;

    this.worker = new Worker(new URL('../workers/vision.worker.ts', import.meta.url), {
      type: 'module',
    });

    return new Promise<LocalVisionInfo>((resolve, reject) => {
      const timeout = window.setTimeout(
        () => reject(new Error('The vision models did not load within 90 seconds.')),
        90_000
      );

      this.worker!.onmessage = (event: MessageEvent) => {
        const msg = event.data;

        if (msg.type === 'ready') {
          clearTimeout(timeout);
          this.info = {
            backend: msg.backend,
            warmupMs: msg.warmupMs,
            domainGuard: msg.domainGuard ?? null,
            activationMap: Boolean(msg.activationMap),
            models: msg.models,
          };
          resolve(this.info);
          return;
        }

        if (msg.type === 'result') {
          this.busy = false;
          this.recordTiming(msg.totalMs);
          this.onResult?.(msg as LocalVisionResult);
          return;
        }

        if (msg.type === 'error') {
          this.busy = false;
          if (!this.info) {
            clearTimeout(timeout);
            reject(new Error(msg.message));
          } else {
            this.onError?.(msg.message);
          }
        }
      };

      this.worker!.onerror = (event) => {
        clearTimeout(timeout);
        this.busy = false;
        reject(new Error(event.message || 'The vision worker failed to start.'));
      };

      // Vite serves `public/` from the site root in dev and copies it into the
      // build, so this resolves the same way in both.
      this.worker!.postMessage({ type: 'init', baseUrl: '/models/' });
    });
  }

  private recordTiming(totalMs: number) {
    this.recentLatencies.push(totalMs);
    if (this.recentLatencies.length > 30) this.recentLatencies.shift();

    const now = performance.now();
    if (this.lastResultAt > 0) {
      this.recentIntervals.push(now - this.lastResultAt);
      if (this.recentIntervals.length > 30) this.recentIntervals.shift();
    }
    this.lastResultAt = now;
  }

  /**
   * Offer a frame. Returns false if one is already in flight — the caller does
   * not retry, it simply offers the next one.
   */
  async submit(source: HTMLVideoElement, timestampSeconds: number): Promise<boolean> {
    if (!this.worker || !this.info || this.busy) return false;
    // A seeking element, or one whose source was just swapped, can pass a
    // readyState check and still have no decodable frame a moment later.
    if (source.readyState < 2 || source.videoWidth === 0 || source.seeking) return false;

    this.busy = true;
    try {
      // createImageBitmap is the cheap part of this: it hands the worker a
      // GPU-backed handle instead of copying pixels through the main thread.
      const bitmap = await createImageBitmap(source);
      const runClassifiers = this.frameCount % CLASSIFIER_EVERY_N_FRAMES === 0;
      this.frameCount++;

      this.worker.postMessage(
        {
          type: 'infer',
          id: this.nextId++,
          bitmap,
          timestampSeconds,
          runTool: runClassifiers,
          runTask: runClassifiers,
        },
        [bitmap]
      );
      return true;
    } catch (err: any) {
      this.busy = false;
      const msg = err?.message || String(err);
      // "The image source is not usable": the frame vanished between the check
      // and the capture (seek, source switch, buffering). That is a dropped
      // frame, not a broken pipeline; the next one will be captured normally.
      if (err?.name === 'InvalidStateError' || /not usable/i.test(msg)) return false;
      if (msg.includes('Non-origin-clean') || msg.includes('SecurityError')) {
        this.onError?.(
          'Video source origin is restricted by CORS. Configure CORS on your cloud bucket or attach the local file.'
        );
      } else {
        this.onError?.(msg || 'Could not capture a frame from the video.');
      }
      return false;
    }
  }

  reset() {
    this.recentLatencies.length = 0;
    this.recentIntervals.length = 0;
    this.lastResultAt = 0;
    this.frameCount = 0;
  }

  dispose() {
    this.worker?.terminate();
    this.worker = null;
    this.info = null;
    this.busy = false;
  }
}
