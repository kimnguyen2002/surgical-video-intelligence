/// <reference lib="webworker" />
/**
 * In-browser surgical vision, off the main thread.
 *
 * This replaces the per-frame Gemini call. That call cost a network round trip
 * — hundreds of milliseconds before the model even started — and burned one
 * free-tier request per frame, so twenty frames hit
 * `RESOURCE_EXHAUSTED 429`. Neither is survivable for a live loop.
 *
 * The three checkpoints from `surgical_main/checkpoints/` were trained on this
 * dataset and are small (ConvNeXtV2-Atto and YOLO11n, ~2.6-3.4M parameters
 * each). Exported to ONNX they run here, in the page, on frames that never
 * leave the machine. There is no quota and no round trip.
 *
 * Everything runs in a worker because the main thread is playing video. A
 * 20 ms inference on the main thread is a dropped frame; here it is invisible.
 */

import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from 'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm?url';
import mjsUrl from 'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs?url';

ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl };

/**
 * Threads need SharedArrayBuffer, which needs cross-origin isolation, which
 * an app running inside AI Studio's iframe does not get. Asking for threads
 * anyway makes session creation fail outright, so the count is what the
 * environment can actually honour.
 */
ort.env.wasm.numThreads =
  typeof SharedArrayBuffer !== 'undefined' && (self as any).crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 2)
    : 1;
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

interface ModelEntry {
  name: string;
  file: string;
  imageSize: number;
  classes: string[];
  activation?: 'sigmoid' | 'softmax';
  metric?: Record<string, number>;
  provenance?: string;
}

interface Manifest {
  models: ModelEntry[];
}

let manifest: Manifest | null = null;
/**
 * The out-of-domain reference: a centroid in the tool classifier's own
 * embedding space, and the cosine distance beyond which a frame is refused.
 *
 * See scripts/build_domain_guard.py for how it is calibrated. The short
 * version: a sigmoid classifier scores every class on every input and has no
 * way to say "I have never seen anything like this", so pointed at a face it
 * reported needle driver at 0.98. This is the representation of that sentence.
 */
interface DomainReference {
  centroid: number[];
  threshold: number;
  dim: number;
}

let domainReference: DomainReference | null = null;

/**
 * The last domain verdict, reused on frames where the classifier did not run.
 *
 * The embedding comes from the tool classifier, which runs every twelfth frame
 * — so after a source change the verdict can be up to ~0.6 s stale. That is
 * the right trade: "is this surgical video" is a property of the source, not
 * of the frame, and re-answering it 20 times a second would double the
 * per-frame cost to re-confirm something that changes when the user switches
 * input and at no other time.
 */
let lastDomain: { outOfDomain: boolean; distance: number } | null = null;

/**
 * Effective class-activation weights: the classifier's weight matrix with the
 * head layer-norm's per-channel scale folded in.
 *
 * `CAM[c, s] = sum_k weights[c, k] * features[k, s]`, which is exact for this
 * architecture — see scripts/export_cam.py for the derivation. No gradients
 * are involved, which is why this works at all in a runtime that has none.
 */
interface CamWeights {
  weights: Float32Array;
  numClasses: number;
  numChannels: number;
}

let camWeights: CamWeights | null = null;

/**
 * The most recent heatmap, reused on the eleven frames in twelve where the
 * classifier does not run. Without this the overlay would flash on for one
 * frame and off for eleven.
 */
let lastActivation: ActivationMap | null = null;

let detector: ort.InferenceSession | null = null;
let toolNet: ort.InferenceSession | null = null;
let taskNet: ort.InferenceSession | null = null;
let backend = 'unknown';

/** One canvas per input size, reused — allocating per frame would dominate. */
const canvases = new Map<number, OffscreenCanvas>();

function canvasFor(size: number): OffscreenCanvas {
  let canvas = canvases.get(size);
  if (!canvas) {
    canvas = new OffscreenCanvas(size, size);
    canvases.set(size, canvas);
  }
  return canvas;
}

/**
 * Classifier preprocessing, matching `default_transform(train=False)` in
 * ai/training/dataset.py: scale the *shorter* side to `size * 1.14`, then take
 * the centre square.
 *
 * Squashing the 16:9 frame into a square instead would be simpler and wrong —
 * the model never saw horizontally compressed anatomy, and the browser would
 * quietly disagree with the checkpoint for no visible reason.
 */
function preprocessClassifier(bitmap: ImageBitmap, size: number, focusX = 0.5): ort.Tensor {
  const canvas = canvasFor(size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

  const scale = (size * 1.14) / Math.min(bitmap.width, bitmap.height);
  const scaledW = bitmap.width * scale;
  const scaledH = bitmap.height * scale;

  // focusX 0.5 is the centre crop the checkpoint was trained on. 0 and 1 slide
  // the same crop to the left and right edge, which the heatmap uses to look
  // at the whole frame rather than only its middle.
  const offsetX = scaledW > size ? -(scaledW - size) * focusX : (size - scaledW) / 2;
  ctx.drawImage(bitmap, offsetX, (size - scaledH) / 2, scaledW, scaledH);
  const { data } = ctx.getImageData(0, 0, size, size);

  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    out[i] = (data[p] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    out[plane + i] = (data[p + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    out[2 * plane + i] = (data[p + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }
  return new ort.Tensor('float32', out, [1, 3, size, size]);
}

interface Letterbox {
  tensor: ort.Tensor;
  scale: number;
  padX: number;
  padY: number;
}

/**
 * Detector preprocessing: Ultralytics letterbox — fit inside the square and
 * pad the remainder with grey (114), keeping the whole frame.
 *
 * The classifiers crop to the centre; the detector must not. An instrument
 * entering from the edge of the field is exactly what it is there to find.
 * The scale and padding come back with the tensor so boxes can be mapped
 * home afterwards.
 */
function preprocessDetector(bitmap: ImageBitmap, size: number): Letterbox {
  const canvas = canvasFor(size);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const drawW = bitmap.width * scale;
  const drawH = bitmap.height * scale;
  const padX = (size - drawW) / 2;
  const padY = (size - drawH) / 2;

  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(bitmap, padX, padY, drawW, drawH);
  const { data } = ctx.getImageData(0, 0, size, size);

  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    out[i] = data[p] / 255;
    out[plane + i] = data[p + 1] / 255;
    out[2 * plane + i] = data[p + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', out, [1, 3, size, size]), scale, padX, padY };
}

interface RawBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
}

/**
 * Decode the YOLO head and suppress duplicates.
 *
 * The graph is exported without built-in NMS, so it emits every anchor:
 * `[1, 4 + numClasses, anchors]`, box coordinates as centre-x/centre-y/w/h in
 * letterboxed pixels. Several neighbouring anchors fire on one instrument —
 * about 7 raw boxes per frame where there are two or three instruments — so
 * without suppression the overlay would stack duplicates on every tool.
 */
function decodeDetections(
  output: ort.Tensor,
  numClasses: number,
  scoreThreshold: number,
  iouThreshold: number
): RawBox[] {
  const data = output.data as Float32Array;
  const [, channels, anchors] = output.dims as number[];

  const candidates: RawBox[] = [];
  for (let a = 0; a < anchors; a++) {
    let best = 0;
    let bestId = -1;
    for (let c = 4; c < channels; c++) {
      const score = data[c * anchors + a];
      if (score > best) {
        best = score;
        bestId = c - 4;
      }
    }
    if (bestId < 0 || best < scoreThreshold) continue;

    const cx = data[0 * anchors + a];
    const cy = data[1 * anchors + a];
    const w = data[2 * anchors + a];
    const h = data[3 * anchors + a];
    candidates.push({
      x1: cx - w / 2,
      y1: cy - h / 2,
      x2: cx + w / 2,
      y2: cy + h / 2,
      score: best,
      classId: bestId,
    });
  }

  candidates.sort((a, b) => b.score - a.score);

  const kept: RawBox[] = [];
  for (const box of candidates) {
    let overlaps = false;
    for (const k of kept) {
      if (k.classId !== box.classId) continue;
      const ix = Math.max(0, Math.min(k.x2, box.x2) - Math.max(k.x1, box.x1));
      const iy = Math.max(0, Math.min(k.y2, box.y2) - Math.max(k.y1, box.y1));
      const inter = ix * iy;
      const union =
        (k.x2 - k.x1) * (k.y2 - k.y1) + (box.x2 - box.x1) * (box.y2 - box.y1) - inter;
      if (union > 0 && inter / union > iouThreshold) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) kept.push(box);
    if (kept.length >= 20) break;
  }
  return kept;
}

async function createSession(url: string): Promise<[ort.InferenceSession, string]> {
  // WebGPU first: it is several times faster than WASM here and, unlike
  // threaded WASM, needs no cross-origin isolation. WASM is the fallback that
  // always works.
  try {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
    });
    return [session, 'webgpu'];
  } catch {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return [session, `wasm x${ort.env.wasm.numThreads}`];
  }
}

async function init(baseUrl: string) {
  manifest = await (await fetch(`${baseUrl}manifest.json`)).json();

  // The guard is optional: a build without a calibrated reference still runs,
  // it just cannot refuse anything. Failing hard here would make a missing
  // 5 KB file take the whole detector down with it.
  try {
    const response = await fetch(`${baseUrl}domain_reference.json`);
    if (response.ok) {
      const reference = (await response.json()) as DomainReference;
      if (Array.isArray(reference.centroid) && reference.centroid.length > 0) {
        domainReference = reference;
      }
    }
  } catch {
    domainReference = null;
  }

  // Same contract as the domain reference: optional, and its absence costs
  // the heatmap rather than the detector.
  try {
    const response = await fetch(`${baseUrl}cam_weights.json`);
    if (response.ok) {
      const payload = await response.json();
      const binary = atob(payload.weightsBase64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      camWeights = {
        weights: new Float32Array(bytes.buffer),
        numClasses: payload.numClasses,
        numChannels: payload.numChannels,
      };
    }
  } catch {
    camWeights = null;
  }
  const entry = (name: string) => manifest!.models.find((m) => m.name === name)!;

  const [det, detBackend] = await createSession(`${baseUrl}${entry('detection').file}`);
  detector = det;
  backend = detBackend;

  const [tool] = await createSession(`${baseUrl}${entry('tool').file}`);
  toolNet = tool;
  const [task] = await createSession(`${baseUrl}${entry('task').file}`);
  taskNet = task;

  // First inference compiles kernels and allocates buffers, and on WebGPU that
  // can take a second. Doing it here means the first real frame is fast rather
  // than being the slowest one the surgeon ever sees.
  const warm = new OffscreenCanvas(64, 64);
  warm.getContext('2d')!.fillRect(0, 0, 64, 64);
  const bitmap = warm.transferToImageBitmap();
  const start = performance.now();
  await runInference(bitmap, true, true);
  const warmupMs = performance.now() - start;

  // Warmup runs on a blank grey canvas, which is emphatically out of domain —
  // and leaves `lastDomain` saying so. The classifier only re-runs every
  // twelfth frame, so without this the first second of every session would
  // refuse real surgical video on the strength of a synthetic warmup frame.
  lastDomain = null;
  lastActivation = null;

  return {
    backend,
    warmupMs,
    domainGuard: domainReference
      ? { threshold: domainReference.threshold, dim: domainReference.dim }
      : null,
    activationMap: camWeights !== null,
    models: manifest!.models.map((m) => ({
      name: m.name,
      classes: m.classes,
      imageSize: m.imageSize,
      metric: m.metric,
      provenance: m.provenance,
    })),
  };
}

/**
 * Cosine distance from the training distribution, and the verdict.
 *
 * Both vectors are L2-normalised, so `1 - dot` is the cosine distance. The
 * centroid arrives normalised from the calibration script; the embedding is
 * normalised here.
 */
function judgeDomain(embedding: Float32Array): { outOfDomain: boolean; distance: number } {
  const centroid = domainReference!.centroid;
  const n = Math.min(centroid.length, embedding.length);

  let norm = 0;
  for (let i = 0; i < n; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm) || 1;

  let dot = 0;
  for (let i = 0; i < n; i++) dot += (embedding[i] / norm) * centroid[i];

  const distance = 1 - dot;
  return { outOfDomain: distance > domainReference!.threshold, distance };
}

export interface ActivationMap {
  /** Row-major, `height * width`, normalised to 0..1. */
  values: number[];
  width: number;
  height: number;
  /** The class the map explains, and its probability. */
  label: string;
  confidence: number;
  /**
   * Where the map belongs on the source frame, as fractions of its width and
   * height. Horizontally it spans the whole frame (three overlapping crops are
   * stitched); vertically it spans the band the classifier's crop can see,
   * which is about 88% of the height because the checkpoint was trained with
   * the shorter side scaled to 1.14x the input.
   */
  region: { x: number; y: number; width: number; height: number };
}

/** The raw, non-negative class evidence for one crop, on the backbone's own grid. */
interface RawCam {
  values: Float32Array;
  width: number;
  height: number;
}

/**
 * Class activation for one crop, from the forward pass alone.
 *
 * Negative contributions are clamped away: a CAM is evidence *for* the class,
 * and keeping the negative lobe would paint regions that argued against it in
 * the same colour ramp as regions that argued for. The values are left
 * un-normalised so that crops can be compared on one scale before stitching.
 */
function rawCam(features: ort.Tensor, classIndex: number): RawCam | null {
  if (!camWeights) return null;

  const dims = features.dims as number[];
  if (dims.length !== 4) return null;
  const [, channels, height, width] = dims;
  if (channels !== camWeights.numChannels) return null;

  const data = features.data as Float32Array;
  const plane = height * width;
  const weightOffset = classIndex * camWeights.numChannels;

  const map = new Float32Array(plane);
  for (let k = 0; k < channels; k++) {
    const weight = camWeights.weights[weightOffset + k];
    if (weight === 0) continue;
    const base = k * plane;
    for (let i = 0; i < plane; i++) map[i] += weight * data[base + i];
  }
  for (let i = 0; i < plane; i++) if (map[i] < 0) map[i] = 0;
  return { values: map, width, height };
}

function sampleBilinear(cam: RawCam, u: number, v: number): number {
  const x = Math.min(cam.width - 1, Math.max(0, u * cam.width - 0.5));
  const y = Math.min(cam.height - 1, Math.max(0, v * cam.height - 0.5));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(cam.width - 1, x0 + 1);
  const y1 = Math.min(cam.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const at = (xx: number, yy: number) => cam.values[yy * cam.width + xx];
  return (
    at(x0, y0) * (1 - fx) * (1 - fy) +
    at(x1, y0) * fx * (1 - fy) +
    at(x0, y1) * (1 - fx) * fy +
    at(x1, y1) * fx * fy
  );
}

/** Output grid width for the stitched map; the height follows the frame's aspect. */
const STITCH_WIDTH = 64;

/**
 * Stitch crops taken at several horizontal positions into one map in frame
 * coordinates, feathering the overlaps so no seam shows, then normalise the
 * whole thing by a single maximum.
 */
function stitchActivation(
  tiles: { cam: RawCam; focusX: number }[],
  bitmap: ImageBitmap,
  inputSize: number,
  label: string,
  confidence: number
): ActivationMap | null {
  const scale = (inputSize * 1.14) / Math.min(bitmap.width, bitmap.height);
  const cropW = Math.min(1, inputSize / scale / bitmap.width);
  const cropH = Math.min(1, inputSize / scale / bitmap.height);
  const top = (1 - cropH) / 2;

  const outW = STITCH_WIDTH;
  const outH = Math.max(8, Math.round((outW * cropH * bitmap.height) / bitmap.width));
  const values = new Array<number>(outW * outH).fill(0);

  let max = 0;
  for (let oy = 0; oy < outH; oy++) {
    const v = (oy + 0.5) / outH;
    for (let ox = 0; ox < outW; ox++) {
      const fx = (ox + 0.5) / outW;
      let sum = 0;
      let weight = 0;
      for (const { cam, focusX } of tiles) {
        const left = focusX * (1 - cropW);
        if (fx < left || fx > left + cropW) continue;
        const u = (fx - left) / cropW;
        // Triangular feather: a crop's own edge is where it is least reliable.
        const w = 0.1 + (1 - Math.abs(2 * u - 1));
        sum += w * sampleBilinear(cam, u, v);
        weight += w;
      }
      const value = weight > 0 ? sum / weight : 0;
      values[oy * outW + ox] = value;
      if (value > max) max = value;
    }
  }

  // Every position argued against the class: nothing to show, and a flat map
  // normalised by a near-zero max would be amplified noise.
  if (max <= 1e-6) return null;
  for (let i = 0; i < values.length; i++) values[i] /= max;

  return {
    values,
    width: outW,
    height: outH,
    label,
    confidence,
    region: { x: 0, y: top, width: 1, height: cropH },
  };
}

async function runInference(bitmap: ImageBitmap, runTool: boolean, runTask: boolean) {
  const entry = (name: string) => manifest!.models.find((m) => m.name === name)!;
  const timings: Record<string, number> = {};

  const detEntry = entry('detection');
  let t0 = performance.now();
  const box = preprocessDetector(bitmap, detEntry.imageSize);
  timings.preprocess = performance.now() - t0;

  t0 = performance.now();
  const detOut = await detector!.run({ [detector!.inputNames[0]]: box.tensor });
  timings.detect = performance.now() - t0;

  // 0.25/0.45 are Ultralytics' own defaults. Measured over 60 real SurgVU
  // frames this yields ~1.4 boxes per frame at a mean score of 0.63; raising
  // the threshold to 0.4 drops it below one box per frame. This checkpoint
  // under-detects rather than over-detects, so the standard threshold is the
  // right default rather than a stricter one.
  const raw = decodeDetections(
    detOut[detector!.outputNames[0]],
    detEntry.classes.length,
    0.25,
    0.45
  );

  // Undo the letterbox, then express the box as 0-1000 of the source frame —
  // the scale the overlay draws in.
  const detections = raw.map((b) => {
    const x1 = ((b.x1 - box.padX) / box.scale / bitmap.width) * 1000;
    const y1 = ((b.y1 - box.padY) / box.scale / bitmap.height) * 1000;
    const x2 = ((b.x2 - box.padX) / box.scale / bitmap.width) * 1000;
    const y2 = ((b.y2 - box.padY) / box.scale / bitmap.height) * 1000;
    return {
      label: detEntry.classes[b.classId] || `class_${b.classId}`,
      confidence: Math.round(b.score * 100),
      box: {
        xmin: Math.max(0, Math.min(1000, x1)),
        ymin: Math.max(0, Math.min(1000, y1)),
        xmax: Math.max(0, Math.min(1000, x2)),
        ymax: Math.max(0, Math.min(1000, y2)),
      },
    };
  });

  let tools: { label: string; confidence: number }[] | undefined;
  let activation: ActivationMap | null = null;
  if (runTool && toolNet) {
    const e = entry('tool');
    t0 = performance.now();
    const out = await toolNet.run({ [toolNet.inputNames[0]]: preprocessClassifier(bitmap, e.imageSize) });
    timings.tool = performance.now() - t0;

    // By name, not by index. The graph now has two outputs — `logits` and the
    // `embedding` the domain guard reads — and output order is not part of the
    // ONNX contract. Indexing [0] happened to work and would have started
    // scoring instruments from a 320-dim feature vector the day it did not.
    const logits = out.logits.data as Float32Array;
    tools = [];
    for (let i = 0; i < e.classes.length; i++) {
      const p = 1 / (1 + Math.exp(-logits[i]));
      if (p > 0.5) tools.push({ label: e.classes[i], confidence: Math.round(p * 100) });
    }
    tools.sort((a, b) => b.confidence - a.confidence);

    if (domainReference && out.embedding) {
      lastDomain = judgeDomain(out.embedding.data as Float32Array);
    }

    // Explain the strongest class, which is the one the panel is showing. A
    // map per class would be 14 passes over the feature map for 13 heatmaps
    // nobody asked to see.
    if (camWeights && out.features && tools.length > 0) {
      let topIndex = 0;
      for (let i = 1; i < e.classes.length; i++) {
        if (logits[i] > logits[topIndex]) topIndex = i;
      }
      const centre = rawCam(out.features, topIndex);
      if (centre) {
        const tiles = [{ cam: centre, focusX: 0.5 }];
        // A wide frame is three crops wide. The two extra passes run only on the
        // one frame in twelve the classifier runs on, so the cost is small.
        if (bitmap.width > bitmap.height * 1.05) {
          for (const focusX of [0, 1]) {
            const side = await toolNet.run({
              [toolNet.inputNames[0]]: preprocessClassifier(bitmap, e.imageSize, focusX),
            });
            const cam = side.features ? rawCam(side.features, topIndex) : null;
            if (cam) tiles.push({ cam, focusX });
          }
        }
        activation = stitchActivation(
          tiles,
          bitmap,
          e.imageSize,
          e.classes[topIndex],
          Math.round((1 / (1 + Math.exp(-logits[topIndex]))) * 100)
        );
      }
    }
  }

  let task: { label: string; confidence: number } | undefined;
  if (runTask && taskNet) {
    const e = entry('task');
    t0 = performance.now();
    const out = await taskNet.run({ [taskNet.inputNames[0]]: preprocessClassifier(bitmap, e.imageSize) });
    timings.task = performance.now() - t0;
    const logits = out[taskNet.outputNames[0]].data as Float32Array;

    let max = -Infinity;
    for (let i = 0; i < e.classes.length; i++) max = Math.max(max, logits[i]);
    let sum = 0;
    const probs = new Float32Array(e.classes.length);
    for (let i = 0; i < e.classes.length; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    let bestIdx = 0;
    for (let i = 1; i < e.classes.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
    task = { label: e.classes[bestIdx], confidence: Math.round((probs[bestIdx] / sum) * 100) };
  }

  if (activation) lastActivation = activation;
  bitmap.close();

  /*
   * Refusal is enforced here, not in the interface.
   *
   * The rule the whole app is built on is that a refused frame has nothing
   * drawn on it. Sending the boxes anyway and trusting every consumer to check
   * a flag first makes that rule a convention — one `if` forgotten in one
   * component and a confident box lands on a human face. Withholding the
   * values means the interface cannot draw them, because it never had them.
   */
  const domain = lastDomain;
  if (domain?.outOfDomain) {
    return {
      detections: [],
      tools: [],
      task: undefined,
      timings,
      // No heatmap either: a map explaining a prediction that was refused
      // would be explaining a conclusion the app declined to draw.
      activation: null,
      domain: { outOfDomain: true, distance: domain.distance },
    };
  }

  return {
    detections,
    tools,
    task,
    timings,
    activation: activation ?? lastActivation,
    domain: domain
      ? { outOfDomain: false, distance: domain.distance }
      : { outOfDomain: false, distance: 0 },
  };
}

self.onmessage = async (event: MessageEvent) => {
  const msg = event.data;
  try {
    if (msg.type === 'init') {
      const info = await init(msg.baseUrl);
      (self as any).postMessage({ type: 'ready', ...info });
      return;
    }

    if (msg.type === 'infer') {
      if (!detector) {
        msg.bitmap.close();
        (self as any).postMessage({ type: 'error', id: msg.id, message: 'Models are not loaded yet.' });
        return;
      }
      const start = performance.now();
      const result = await runInference(msg.bitmap, msg.runTool, msg.runTask);
      (self as any).postMessage({
        type: 'result',
        id: msg.id,
        timestampSeconds: msg.timestampSeconds,
        totalMs: performance.now() - start,
        backend,
        ...result,
      });
    }
  } catch (err: any) {
    msg?.bitmap?.close?.();
    (self as any).postMessage({ type: 'error', id: msg?.id, message: err?.message || String(err) });
  }
};
