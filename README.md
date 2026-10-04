# Surgical Video Intelligence

Browser-native understanding of robotic surgical video, on the **SurgVU 2024**
release. Instrument detection, instrument-presence and surgical-task
recognition all run **in the page**, on frames that never leave the machine.
There is no server, no API key, and no per-frame cost.

> Built for surgical education and computer-vision research.
> **Not a medical device.** It does not provide diagnosis, treatment
> recommendations, or clinical decision support.

---

## What it does

Load a case and you see, on the same frame and never blurred together:

- **What the dataset records** — which instruments the da Vinci system logged
  as installed, and which surgical task is annotated, at this exact moment.
- **What the models predict** — localised instrument boxes with confidences,
  from three ONNX graphs executing in a Web Worker.
- **Where the disagreement is** — because the two are drawn differently, on
  purpose, and one of them is a fact.

An assistant answers questions about what is on screen. By default it answers
from the dataset's own annotations rather than from a language model, which for
*"which instrument is in use?"* is not a compromise — it is the better answer.

---

## The one rule the interface is built around

**Every claim on screen states where it came from, and claims of different
kinds never look alike.**

| Kind | Meaning | How it is drawn |
|---|---|---|
| **Recorded** | Written into the SurgVU release by its authors. | Solid green, listed not boxed, and **no confidence** — none applies. |
| **Predicted** | Produced by a model here, now. | Dashed amber box, **always with a percentage**. |
| **Refused** | The models have no competence over this input. | Nothing drawn; the interface says why. |

### Why "refused" exists

During development the live camera was pointed at a human face. The instrument
classifier reported **needle driver at 0.98 confidence**, and the attention
heatmap landed on the person's cheek.

Nothing was broken. A multi-label classifier ending in a sigmoid produces a
score for every class on every input; it has no way to represent *"I have never
seen anything like this."* It had only ever seen endoscopic surgical frames, so
it scored a face as though it were one.

That is worse than being wrong. It is **confidently wrong on input the model
has no business judging**, and it is visually indistinguishable from a correct
prediction. On a platform whose entire purpose is explaining what a model sees,
that single behaviour would invalidate everything else.

#### How the guard works

The tensor feeding the tool classifier's final layer — its own pooled,
layer-normalised feature vector — is exposed as a second ONNX output. That
costs nothing: the values are already computed on every frame, and the file
does not grow. A frame is refused when the cosine distance from the centroid of
360 real SurgVU frames exceeds a calibrated threshold, and the worker then
withholds every prediction rather than flagging them, so the interface cannot
draw a box it never received.

#### What it does and does not catch

Calibrated against 24 probes (`npm run build:domain`), with real SurgVU frames
spanning 0.063–0.433:

| Probe category | Nearest probe | Separates? |
|---|---|---|
| Photographs of people | 0.599 | yes |
| Synthetic patterns | 0.573 | yes |
| Abstract renders | 0.562 | yes |
| Natural photographs | 0.467 | yes |
| **Non-endoscopic biological tissue** | **0.328** | **no** |

The motivating case is caught with a wide margin: a human face sits at 0.60
against a threshold of 0.356.

**Tissue is not.** Retina fundus images, immunohistochemistry, histology slides
and a kidney slice land at 0.33–0.38, overlapping the surgical distribution's
upper tail, and one of them is accepted. That is the honest limit of this
approach: a 3.4M-parameter instrument classifier's embedding separates
*surgical video* from *not photographs of a room*, but it does not separate
endoscopic surgery from other close-up biological imagery.

So the guard should be read as catching **the camera pointed at the wrong
thing**, which is the failure that actually occurs, and not as a claim that the
models know endoscopy from histology. The calibration script prints the
overlap, writes the per-category breakdown into
`public/models/domain_reference.json`, and warns when the populations are not
separable rather than quietly picking a threshold that implies they are.

### The second application of the rule

The release's instrument labels come from the **robot's installation log** —
they record that an instrument was *mounted on an arm*, not that it is *visible
on screen*. An instrument stays "present" while off-screen or hidden behind
tissue.

So the assistant says **"installed"** or **"mounted on an arm"**, and never
**"visible in frame"**, unless a bounding box confirms it. This is enforced in
the prompt layer and in the grounded answerer's wording, not left to chance.

---

## What it costs to run: nothing

That is a design constraint, not a side effect, and it is worth being explicit
about how it was reached — because an earlier version of this project cost real
money in two separate ways and both are now gone.

| | Before | Now |
|---|---|---|
| **Per-frame analysis** | One Gemini call per frame. Twenty frames returned `RESOURCE_EXHAUSTED 429`. | Three ONNX graphs in a worker. ~35 ms/frame. No quota to exhaust. |
| **Video delivery** | 8.9 GB streamed from a Cloud Storage bucket — egress billed per visitor, growing with exactly the attention the project was published to attract. | 33.6 MB of excerpts served as static files, plus local-folder attachment for the full release. |
| **Assistant** | Server-held API key, every visitor spending the author's quota. | Deterministic grounded answerer in the browser. Optional: the visitor supplies **their own** key. |
| **Hosting** | Node process holding a secret. | A directory of static files. |

There is no `.env` to populate, no secret in CI, and no key in this repository —
because there is no server for one to live on.

### The optional language model

A visitor who wants generated prose can paste their **own** Gemini API key
under *Settings → Language model*. It is stored in their browser and sent
directly to Google. It never reaches any server belonging to this project,
which is a property of the architecture rather than a promise in a policy.

Even then, factual lookups stay on the grounded path. Paraphrasing a dataset
value through a language model cannot make it more correct, can make it less,
and costs a request.

---

## The three answer paths

Tried in this order, which is the design:

1. **Grounded** (`src/assistant/grounded.ts`) — structured facts: the
   installation log, the task intervals, the detector's output, the timeline.
   Free, instant, offline, and for factual questions simply better.
2. **Extractive** (`src/assistant/extractive.ts`) — sentences ranked out of the
   bundled surgical knowledge base and quoted verbatim, with citations. Free,
   and traceable to indexed material.
3. **Generative** (`src/services/byokGemini.ts`) — only with the visitor's own
   key, and only ever *on top of* the grounding from the first two.

Every assistant turn in the transcript carries a badge — `recorded`, `quoted`
or `generated` — so a reader can tell which they are looking at without asking.
The three are equally fluent and only one of them is evidence.

**Why the extractive path reads only retrieved passages:** an earlier version
ranked sentences from the whole system prompt, which contains the safety
constraints. Asking *"what instrument is in use?"* returned *"Never use
profanity, slurs, demeaning language…"* — it was quoting its own instructions
back as findings. The fix was not a better ranking function; it was to make the
instruction layers structurally unreachable from the answer path.

---

## The models

Three graphs, **36 MB total**, in `public/models/`. They ship with the
repository; nothing is downloaded and nothing is trained at first run.

| Model | Architecture | Job | Held-out metric |
|---|---|---|---|
| `detection.onnx` | YOLO11n (2.6M params) | Where each instrument is | none published — see below |
| `tool.onnx` | ConvNeXtV2-Atto (3.4M) | Which of 14 instruments are present | mAP **0.971**, but read the caveat |
| `task.onnx` | ConvNeXtV2-Atto (3.4M) | Which of 8 surgical tasks | balanced accuracy **0.776** over 20 unseen cases |

### Read these numbers honestly

The presence and detection checkpoints were fitted to **five clips** of the
public cat1 subset and validated on two held-out clips. **Six of their fourteen
classes never occur in that data and cannot be predicted at all.** An mAP of
0.971 over a validation set that narrow is a statement about the pipeline, not
about the detector. The *On-device models* panel in the app prints each
checkpoint's own provenance note next to its score, so the caveat travels with
the number instead of living only here.

Measured over 150 frames sampled evenly across the six bundled excerpts
(`npm run bench:clips`), at Ultralytics' default 0.25 threshold:

| | |
|---|---|
| Frames with at least one box | **79%** |
| Boxes per frame | **1.23** |
| Mean confidence | **0.61** |
| Distinct classes predicted | **6 of 14** |

So roughly one frame in five gets no box at all. That is the checkpoint
under-detecting rather than the clips being too compressed — it is why the
threshold stays at the default rather than being raised, and why an empty
result is rendered as *"the detector localized no instrument in this frame"*
rather than as a blank panel. An empty answer is a real answer and the
interface says so.

The task classifier is the one with a defensible number: it was split **by
case**, so the 0.776 balanced accuracy is over twenty recordings the model
never saw.

### Explaining a prediction, without gradients

The heatmap is **CAM** (Zhou et al., 2016), not Grad-CAM, and the difference is
the point. Grad-CAM needs a backward pass; ONNX Runtime Web has no autograd,
and shipping a training runtime to a web page to produce a heatmap would cost
more than everything else here combined.

It is also unnecessary for this architecture. Grad-CAM exists because modern
networks put several layers between the last convolution and the logit, so
per-channel importance has to be recovered from gradients. ConvNeXtV2's head is
global average pooling, a layer norm, and one linear layer — the importance is
already in the weight matrix.

Writing `x[k,s]` for the final feature map and folding the head through:

```
CAM[c, s] = Σ_k ( W[c,k] · γ[k] ) · x[k, s]
```

where `γ` is the layer norm's per-channel scale. **Folding in `γ` is not
optional.** It sits inside the sum over channels, and on this checkpoint it
spans 0.24–2.77 — an 11.6× range. Using the raw `W` and ignoring it, which is
the obvious shortcut, silently reweights every channel by however much the norm
had learned to amplify or suppress it.

The remaining terms (`μ`, `σ`, the biases) are per-frame scalars that shift and
scale the map uniformly and vanish when it is normalised for display. So the
decomposition is exact, and `scripts/export_cam.py` checks it: summing the CAM
over space reconstructs the model's own logits to **2.9 × 10⁻⁴**, which is
float32 rounding.

The map is 7×7 — the backbone's true spatial resolution at its last stage —
and it is drawn at that size and scaled up, rather than smoothed to imply a
precision it does not have. It is positioned over the classifier's **centre
crop**, not the whole frame, because that is the only region the classifier
saw.

### Measured latency

End to end in the browser, on a video frame, including preprocessing and box
decoding:

| Backend | Per frame | Rate |
|---|---|---|
| WASM, 4 threads (cross-origin isolated) | **~35 ms** | ~22 fps |
| WASM, single thread (no isolation) | ~98 ms | ~10 fps |

The toolbar prints the live figure and which backend took the job, so the
number above is checkable rather than asserted.

Four things make that possible, and the last two are the interesting ones:

- **Small models.** 2.6M and 3.4M parameters.
- **A worker.** The main thread is playing video; a 20 ms inference there is a
  dropped frame.
- **One frame in flight, newest wins.** Frames are *dropped*, never queued.
  Queueing is the obvious implementation and the wrong one: if inference takes
  40 ms and frames arrive every 16 ms, the overlay falls further behind the
  video every second while looking perfectly live. Dropping bounds latency and
  lets frame rate give way instead.
- **Cadence split.** The detector runs on every frame. The presence and task
  classifiers run every twelfth — they answer questions whose answers change
  over minutes.

---

## The dataset

**SurgVU** — Surgical Video Understanding, [arXiv:2501.09209](https://arxiv.org/html/2501.09209v1).
Recordings from da Vinci robotic surgical training sessions.

The release is ~168 GB and carries its own data-use terms. **It is not in this
repository.** What is here:

| | Bundled | Not bundled |
|---|---|---|
| Annotations | 259 instrument intervals and 43 task intervals across 6 cases / 9 parts (24.4 h of footage), covering 13 of 14 instrument classes and 7 of 8 task classes | the other 149 cases |
| Video | 6 excerpts, 75 s each, 33.6 MB total | 8.9 GB of full parts |

### How the excerpts were chosen

Not from the start of each case. The first ninety seconds of a robotic
procedure is trocar placement: a still, dark field with nothing installed and
no task labelled. An excerpt taken from `t=0` would show the detector finding
nothing and the ground-truth panel saying nothing, which reads as a broken app
rather than an accurate one.

So `scripts/build_clips.py` scores every candidate window by how much
*labelled* activity it contains — distinct instruments, distinct tasks,
instrument changes inside the window, and the fraction of the window with any
task annotated — and takes the best. All six land at **100% task coverage**
with four to seven instruments each.

### The timestamp trap

An excerpt is cut from the middle of a case, so the player's clock starts at
0:00 for frames belonging to, say, 1:14:50. Every annotation lookup therefore
goes through `currentTime + timeOffset`, never the player's own clock.

This is the single most dangerous bug this codebase can have, because it does
not fail loudly: reading the wrong clock produces *real* annotations from the
*wrong moment* — confident, precisely-timed and entirely wrong — and looks
exactly like the app working.

### Attaching the real dataset

Click **Attach dataset folder** in the case library and pick your local
`surgvu24_videos_only` directory. Files are read straight off disk through
`URL.createObjectURL`: **nothing is uploaded**, and a five-hour part seeks as
fast as the disk can serve it.

Files are matched to cases **by filename, never by pick order**. Playing
`case_003_video_part_002.mp4` against case 002's labels would produce
confident, precisely-timed, entirely wrong ground truth, so a filename not in
the manifest is rejected rather than guessed at.

### Three schema traps, for anyone regenerating the labels

Each one fails **silently** — training still converges and the resulting model
is meaningless.

1. **The two CSVs have different column names.** `tools.csv` uses
   `install_case_part` / `install_case_time`; `tasks.csv` uses `start_part` /
   `start_time`. A parser written for one raises on the other. Before this was
   fixed, *every task label in all 155 cases failed to parse.*
2. **The "part" column is a bare number and the case id is not in the file.**
   `install_case_part` contains `1.0` — the part number. The case id appears
   only in the directory name. Reading `1.0` as a case id, which is the natural
   guess, makes every timestamp lookup miss.
3. **Times are formatted differently in each file.** `tools.csv` uses
   `00:43:36.978000`; `tasks.csv` uses float seconds (`3226.251309`).

Additionally, `tools.csv` contains 1,449 rows labelled `nan(camera in)` — the
endoscope being inserted, not an instrument. Kept, it becomes a phantom
fifteenth class present in most frames.

All handled in `ml/ai/training/labels.py` and pinned by twelve tests in
`ml/tests/test_labels_real_schema.py`.

---

## Running it

```bash
npm install
npm run dev          # http://localhost:3000
```

That is the whole setup. No key, no `.env`, no backend.

| Script | What it does |
|---|---|
| `npm run dev` | Vite dev server, with the cross-origin isolation headers |
| `npm run build` | Static bundle into `dist/` |
| `npm run preview` | Serve the built bundle locally |
| `npm run lint` | `tsc --noEmit`, in strict mode |
| `npm run build:clips` | Re-cut the excerpts from a local copy of the dataset |
| `npm run build:data` | Regenerate the bundled annotations |
| `npm run build:models` | Re-export the checkpoints to ONNX |

### Deploying

Any static host. For Vercel, `vercel.json` is already configured — import the
repository and deploy; there is nothing to set.

The one thing a host **must** get right is the pair of headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

Without them `SharedArrayBuffer` is unavailable, onnxruntime-web falls back to
a single thread, and inference goes from ~35 ms to ~98 ms per frame. It will
still work — it will just look like a slow machine rather than a missing
header, which is why the toolbar names the backend it actually got.

`credentialless` rather than `require-corp`: the stricter value blocks
cross-origin subresources that send no CORP header, which includes the Google
Fonts stylesheet.

---

## Repository map

```
src/
  assistant/          the three answer paths
    grounded.ts         structured facts — the default
    extractive.ts       sentence ranking over retrieved passages
    retrieval.ts        lexical retrieval with IDF over the knowledge base
    prompt.ts           layered system prompt for the generative path
    intents.ts          question classification
  services/
    localVision.ts      worker client: one frame in flight, newest wins
    groundTruth.ts      binary-searched annotation lookup
    byokGemini.ts       the visitor's own key, never ours
    voice.ts            Web Speech API, in and out
  components/
    ActivationOverlay.tsx  the CAM heatmap, positioned over the centre crop
  workers/
    vision.worker.ts    ONNX sessions, pre/post-processing, NMS, CAM, domain guard
  data/                 generated: cases, labels, clips, vocabulary
ml/
  ai/training/          the PyTorch pipeline the checkpoints came from
  ai/datasets/          SurgVU label parsing — the part with the schema traps
  ai/llm/grounded.py    the Python original of src/assistant/grounded.ts
  checkpoints/          the .pt files export_models.py converts
  tests/                113 tests over label parsing and the training pipeline
scripts/
  build_clips.py        excerpt selection and encoding
  build_surgvu_data.py  annotation extraction
  build_domain_guard.py out-of-domain calibration, with its own probe set
  export_cam.py         effective CAM weights, with the derivation
  benchmark_clips.py    what the detector actually does on the excerpts
  export_models.py      PyTorch → ONNX
```

`ml/` does not run in production, is not imported by the application and is not
part of the build. It is in the repository because a checkpoint without the code
that produced it is an unverifiable artefact. See [ml/README.md](ml/README.md).

---

## Known limitations

Stated here rather than discovered later:

- **Retrieval is lexical, not semantic.** It finds passages sharing *words*
  with the question, not meaning. For a few hundred passages of controlled
  surgical vocabulary this is close to as good and costs nothing to ship; for a
  larger or noisier corpus it would not be.
- **The out-of-domain guard does not separate tissue.** It reliably rejects
  faces, rooms and synthetic input; it overlaps with non-endoscopic biological
  imagery. Measured, not assumed — see the table above.
- **Six of fourteen instrument classes are unpredictable** — they do not occur
  in the data the presence and detection checkpoints were fitted to.
- **The detector has no published held-out metric.** Treat its boxes as a
  demonstration of the pipeline.
- **Speech recognition is browser-dependent.** Chrome, Edge and Safari have it;
  Firefox does not. The microphone is not offered where it is unsupported,
  rather than offered and silently inert.
- **WebGPU is attempted first and falls back to WASM**, so the reported backend
  varies by machine. The toolbar says which one took the job.

---

## Licence and attribution

Code: MIT, see [LICENSE](LICENSE).

The SurgVU dataset is **not** covered by that licence and is not redistributed
here beyond the short excerpts described above. It remains subject to its own
terms; obtain it from the challenge organisers and honour the conditions
attached.

This is research and educational software. It is not a medical device, has not
been clinically validated, and must not be used to inform the care of a
patient.
