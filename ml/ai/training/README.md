# SurgVU Training Pipeline

Complete PyTorch pipeline for the two tasks in
[SurgVU (arXiv:2501.09209v1)](https://arxiv.org/html/2501.09209v1):

| Task | Type | Classes | Source labels |
|---|---|---|---|
| **Tool presence** | Multi-label frame classification | 12 instruments | `tools.csv` |
| **Step recognition** | Temporal clip classification | 8 surgical tasks | `tasks.csv` |
| **Instrument detection** | Bounding-box detection | 14 instruments | cat1 `*_coco.json` |
| **Weak localisation** | Grad-CAM → boxes, scored on cat1 | shared 12 | `tools.csv` + cat1 |
| **VQA** | Open-ended question answering | — | cat2 `*.json` |

The dataset is 280 clips from 155 robotic training sessions — 720p at 60 fps,
over 840 hours, roughly 18 million labelled frames from the da Vinci endoscope
channel.

## What is actually on this machine

`ai/datasets` discovers the corpora wherever they sit — they are ~168 GB, so
nothing is copied. Check what it found:

```bash
node scripts/python.cjs -c "import json;from ai.datasets import registry;print(json.dumps(registry.summary(),indent=2))"
```

| Corpus | Present | Annotation |
|---|---|---|
| `surgvu24` | 155 cases, 81 with video, 144 parts, 168 GB | tool + task intervals |
| `cat1` | 7 clips, 5,178 frames, 11,323 boxes | COCO bounding boxes |
| `cat2` | 7 cases | question / accepted answers |

Override the search with `SURGVU_VIDEO_ROOT`, `SURGVU_LABEL_ROOT`,
`SURGVU_CAT1_ROOT`, `SURGVU_CAT2_ROOT`.

> [!IMPORTANT]
> **cat1 is positives-only.** 98.8% of its frames are annotated and every
> annotated frame carries at least one box — there are no empty-scene
> negatives anywhere in it. A detector trained on it learns background only
> from the non-box regions of frames that do contain an instrument, and will
> lean toward always predicting something. That is a property of the
> annotations, not a bug in the trainer.

> [!NOTE]
> The cat1 clips ship **already decoded at exactly 1 fps**, so a COCO
> `image_id` is the frame index and the second offset simultaneously. No
> timestamp arithmetic is needed for that corpus.

> [!NOTE]
> Run everything as modules **from the repository root** so that `ai.*` imports
> resolve:
> ```bash
> python -m ai.training.<script>
> ```

---

## Verify the pipeline first

Both trainers have a `--smoke-test` mode that runs the entire loop — model,
loss, scheduler, AMP path, metrics, checkpointing, and Grad-CAM — on synthetic
data. It takes about fifteen seconds and needs no dataset. Use it to confirm
the pipeline is sound before committing a GPU and a 100 GB download to it.

```bash
python -m ai.training.train_tool_detection --smoke-test
python -m ai.training.train_step_recognition --smoke-test --clip-len 4
```

---

## 1. Get the data

```bash
python -m ai.training.download_dataset --check                       # sizes only
python -m ai.training.download_dataset --dest data/surgvu --only labels --extract
python -m ai.training.download_dataset --dest data/surgvu --extract  # + videos (~100 GB)
```

Downloads resume if interrupted. Start with `--only labels` (a few MB) to
inspect the schema before committing the bandwidth.

## 2. Extract frames

```bash
python -m ai.training.extract_frames --dataset data/surgvu --fps 1
python -m ai.training.extract_frames --dataset data/surgvu --limit 3   # quick trial
```

Videos are decoded **once** into JPEGs plus a manifest. Training directly from
60 fps video spends nearly all its time in the decoder, and consecutive frames
are near-duplicates that cost compute without adding information. At 1 fps,
840 hours still yields ~3M frames.

The manifest records each frame's `(case, part, timestamp)` and each part's
duration. Both are required to resolve interval labels correctly.

## 3. Train

```bash
# Tool detection
python -m ai.training.train_tool_detection \
    --model convnextv2_tiny.fcmae_ft_in22k_in1k \
    --batch-size 32 --epochs 30

# Step recognition, warm-started from the tool detector (recommended)
python -m ai.training.train_step_recognition \
    --model cnn_gru --clip-len 16 \
    --init-from checkpoints/tool_best.pt
```

Any timm backbone works for tool detection: `convnextv2_*`, `efficientnetv2_*`,
`vit_*`, `swin_*`, `resnet*`.

## 3b. Detection and localisation

Two independent routes to a labelled box on screen, reported side by side
because they answer different questions and fail in different ways.

```bash
# Supervised: train on cat1's real boxes (clip-disjoint split)
python -m ai.training.export_detection --train-stride 2
python -m ai.training.train_detection --epochs 30 --imgsz 384

# Weakly supervised: Grad-CAM from a presence classifier, scored on cat1 boxes
python -m ai.training.localize --checkpoint checkpoints/tool_best.pt
```

**Read the supervised number carefully.** It is trained on five clips of the
*public cat1 test set* and validated on two held-out clips. Six of the fourteen
classes never occur in the data and can be neither learned nor scored. It is a
small-data demonstrator, not a validated detector, and the serving layer labels
it as such wherever its output appears.

**Expect the weakly-supervised number to be low**, and report it anyway.
Grad-CAM marks whatever region drove a classification, which for an instrument
is often the tissue interaction rather than the extent a human would box; and a
classifier has no notion of instance count, so two overlapping needle drivers
produce one blob and the second scores as a miss. Both routes are scored by the
same `detection_metrics` code — two mAPs from different implementations are not
a comparison.

## 3c. Surgical task recognition — what is being *done*

Tool presence says "a needle driver is mounted". This says "the surgeon is
suturing", which is the question a trainee actually asks, and `tasks.csv` has
labelled it across all 155 cases.

```bash
python -m ai.training.extract_task_frames --cases 200 --per-interval 15
python -m ai.training.train_task --epochs 12 --image-size 176
```

**Extraction seeks; it does not decode.** `extract_frames` walks a video front
to back calling `grab()` on ~1 million frames — about 11 minutes for one 4-hour
part, a full day for the 144 parts on disk, and nearly all of it discarded.
`tasks.csv` states exactly when each activity happens and those intervals are
roughly a third of a case, so seeking straight to sampled timestamps inside
them is ~150× faster (2 cases in 8 seconds) and yields a **balanced** set by
construction: frames are drawn per interval, so a 33-minute suturing run does
not swamp a 2-minute range-of-motion run.

Frames are sampled away from interval edges. SurgVU task boundaries are coarse,
and a frame taken at the exact start of an interval is as likely to show the
previous activity as the labelled one.

**Balanced accuracy selects the checkpoint.** Suturing is 41% of the data, so
plain accuracy rewards a model that predicts it and gives up on the rest.

## 3d. Out-of-domain guard — required before live camera

Without it, a webcam pointed at a face returns a needle driver at 0.98 with a
Grad-CAM heatmap over the person's cheek. The classifier has only ever seen
endoscopic frames and a sigmoid head cannot say "I have never seen this".

```bash
python -m ai.training.build_domain_reference --verify
```

`--verify` measures both sides — non-surgical probes must be refused *and*
held-out surgical frames must still pass. Measured here: training frames
0.07–0.61, threshold 0.593, held-out surgical 0.24, face 0.894, noise 0.91.

The threshold is a high percentile **scaled by a margin** into the gap between
the populations, not a raw percentile — p99 sits inside the training spread by
construction and refuses real frames.

## 4. Evaluate and predict

```bash
python -m ai.training.evaluate --task tool --checkpoint checkpoints/tool_best.pt \
    --split test --gradcam-samples 12

python -m ai.training.predict --image frame.jpg --checkpoint checkpoints/tool_best.pt
python -m ai.training.predict --video case.mp4 --checkpoint checkpoints/tool_best.pt \
    --interval 2 --output timeline.json

python -m ai.training.export_onnx --checkpoint checkpoints/tool_best.pt --verify
```

Checkpoints in `checkpoints/` are picked up by the running platform
automatically, or press **Reload models** in the developer console.

---

## Classes

**Instruments (12)** — needle driver, cadiere forceps, prograsp forceps,
monopolar curved scissors, bipolar forceps, stapler, force bipolar, vessel
sealer, permanent cautery hook/spatula, clip applier, tip-up fenestrated
grasper, grasping retractor.

**Tasks (8)** — suturing, uterine horn, rectal artery/vein manipulation,
suspensory ligaments, general skills application, range of motion, retraction
and collision avoidance, other.

---

## How the labels actually work

The two files do **not** share a schema. This tripped the original parser, and
the failure was silent in the way that matters — training still ran.

```
tools.csv: index, install_case_part, install_case_time, uninstall_case_part,
           uninstall_case_time, arm, commercial_toolname, groundtruth_toolname
tasks.csv: index, start_part, start_time, stop_part, stop_time,
           groundtruth_taskname
```

Three differences to keep in mind:

- **The part column is a bare number** — `1.0` in `tools.csv`, `1` in
  `tasks.csv` — not a composite `case_042_video_part_003` string. The case id
  appears *only* in the directory name (`labels/case_002/`), which is why
  `load_label_rows` takes the case from the path.
- **Times are formatted differently.** `tools.csv` uses `00:43:36.978000`;
  `tasks.csv` uses float seconds (`3226.251309`). `parse_time` accepts both.
- **`nan(camera in)` is not an instrument.** It marks the endoscope being
  inserted — 1,449 rows — and is dropped, along with 144 blank rows. Kept, it
  becomes a phantom thirteenth class present in most frames.

Verified across all 155 cases: 0 parse failures, 10,508 tool intervals,
1,178 task intervals spanning all 8 task classes.

A row says "this tool was installed on this arm at (part P1, time T1) and
removed at (part P2, time T2)". Three consequences shape `labels.py`:

**Intervals can span video parts.** A tool installed near the end of part 3 and
removed in part 5 covers all of part 4. Resolving that requires each part's
duration, which is why labels are resolved against the frame manifest rather
than in isolation. Dropping the middle part would silently label those frames
"no tool" — the loss would still fall, and the result would be meaningless.

**Several arms carry tools at once.** Tool presence is genuinely multi-label:
the labels at a timestamp are the union across arms. Hence sigmoid per class,
never softmax.

**Tool labels come from installation logs, not from vision.** A tool counts as
present while it is installed — including while it is occluded or entirely
outside the endoscopic field of view. This is *weak supervision*: a model that
"wrongly" predicts absent for an off-screen installed tool is being penalised
for being visually correct. That is a property of the dataset, and it is why
per-class metrics here should be read as an upper bound on visual difficulty
rather than as clean detection accuracy.

---

## Why the pipeline is built this way

**Splits are by case, never by frame.** Frames within one operation share
patient, lighting, camera, and instruments. A random frame-level split puts
near-duplicates of validation frames into training and reports accuracy that
collapses on genuinely unseen cases. `split_cases()` partitions case IDs with a
fixed seed.

**mAP selects the tool checkpoint, not accuracy or loss.** With 12 classes where
most are absent in most frames, a model predicting "absent" everywhere exceeds
90% accuracy and is worthless. Average precision integrates precision across the
full recall range and is unmoved by that shortcut. There is a test asserting
exactly this.

**Balanced accuracy selects the step checkpoint.** `other` dominates the task
labels, so plain accuracy rewards a model that predicts it and gives up on
everything else.

**Per-class `pos_weight` on the BCE loss.** Tool prevalence spans orders of
magnitude — needle drivers appear constantly, staplers rarely. Unweighted BCE is
minimised by ignoring rare tools. Weights are capped at 50 so a class with a
handful of examples cannot destabilise training.

**Label smoothing on the temporal task.** Task boundaries in SurgVU are coarse
and short transitions are frequently unlabelled. A clip straddling a boundary
genuinely is partly both, so training it to 100% confidence is training on
noise.

**Warmup then cosine decay.** Fine-tuning a pretrained backbone at full learning
rate from step zero destroys pretrained features before the randomly-initialised
head produces a useful gradient. Step recognition additionally freezes the frame
encoder for the first two epochs for the same reason.

**Surgical-specific augmentation.** Horizontal flips and colour jitter are safe.
Vertical flips and rotations are not — endoscopic video has a consistent horizon,
and training on upside-down frames wastes capacity on an orientation that never
occurs. `RandomErasing` stands in for smoke, glare, and tissue occlusion.

**Warm-starting step recognition from the tool detector** usually beats ImageNet
initialisation. Tool detection has far more supervision and learns exactly the
features the temporal task needs — instruments, tissue, and their interaction.

---

## Model architectures

**Tool detection** — a timm backbone with a dropout + linear multi-label head.
Sigmoid per class.

**Step recognition** — three backends, in descending order of capability and
ascending order of how likely they are to run on the machine in front of you:

| `--model` | Description |
|---|---|
| `cnn_gru` *(default)* | Shared 2-D encoder → bidirectional GRU → attention pooling. Trains on one consumer GPU; the encoder can be warm-started from the tool detector. |
| `timesformer` / `videomae` | True video transformers via HuggingFace. Best accuracy, heaviest requirements. |
| `mean_pool` | Same encoder, mean pooling, no temporal model. An **ablation baseline** — if a temporal model cannot beat it, the temporal modelling is not earning its complexity. |

Attention pooling rather than mean pooling because a decisive two-second clip
application inside a sixteen-frame window gets diluted by averaging.

---

## Explainability

Grad-CAM is shared with the live platform (`ai/explainability/gradcam.py`), so
a heatmap produced during evaluation and one shown in the dashboard come from
exactly the same code path.

Two details make it correct across architectures:

- **Transformers need reshaping.** A ConvNeXt layer emits `(B, C, H, W)`; a
  ViT/Swin layer emits `(B, N, C)` tokens that must be folded back to a spatial
  grid (dropping the class token) before pooling. Without that, the map is
  meaningless.
- **Multi-label needs per-class attribution.** Backward runs from a *single*
  class logit, so the heatmap answers "where is the evidence for *this*
  instrument" rather than blending all twelve.

---

## Configuration

Everything in `config.py` is overridable by environment variable:

```bash
SURGVU_DIR=/data/surgvu SURGVU_FPS=2 SURGVU_BS=64 SURGVU_LR=5e-5 \
  python -m ai.training.train_tool_detection
```

| Variable | Default | Notes |
|---|---|---|
| `SURGVU_DIR` | `data/surgvu` | Dataset root |
| `SURGVU_FRAMES` | `data/surgvu_frames` | Extracted frames + manifest |
| `SURGVU_OUT` | `checkpoints` | Checkpoint output |
| `SURGVU_FPS` | `1.0` | Extraction sampling rate |
| `SURGVU_IMG` | `224` | Input resolution |
| `SURGVU_BS` | `32` | Batch size |
| `SURGVU_LR` | `1e-4` | Learning rate |
| `SURGVU_EPOCHS` | `30` | Max epochs (early stopping at 7 without improvement) |
| `SURGVU_CLIP` | `16` | Frames per temporal clip |

---

## Hardware

| Setup | Tool detection (30 epochs, 1 fps) | Notes |
|---|---|---|
| RTX 4090 / A100 | ~6–10 h | AMP on, batch 64+ |
| RTX 3090 | ~12–18 h | AMP on, batch 32 |
| Apple Silicon (MPS) | days | Works; AMP is disabled on MPS |
| CPU | impractical | Use `--smoke-test` to validate only |

Mixed precision is enabled on CUDA only — the MPS autocast path is still
inconsistent enough to distrust for a long run.

---

## Experiment tracking

Every run registers itself with `ai/evaluation`, recording model, config,
dataset, git revision, and capability tier. Browse and compare runs at
`/experiments` in the app.

Comparison refuses to rank runs that differ in dataset, code revision, or
hardware tier, and says why. Two numbers produced under different conditions
are not a result.

---

## Files

| File | Purpose |
|---|---|
| `export_detection.py` | cat1 → YOLO dataset, clip-disjoint split |
| `train_detection.py` | Supervised bounding-box detector (Ultralytics) |
| `train_presence.py` | Instrument-presence classifier — **what Grad-CAM needs** |
| `extract_task_frames.py` | Seek-based extraction of `tasks.csv` intervals |
| `train_task.py` | Surgical task recognition (single-label, balanced accuracy) |
| `build_domain_reference.py` | Calibrates the out-of-domain guard |
| `detection_metrics.py` | IoU, per-class AP, mAP@50 and mAP@50-95 |
| `localize.py` | Grad-CAM → boxes, scored against cat1 |
| `config.py` | Configuration and class vocabularies |
| `labels.py` | CSV parsing, interval resolution, fast timestamp lookup |
| `extract_frames.py` | Video → JPEG frames + manifest |
| `dataset.py` | `SurgVUToolDataset`, `SurgVUTaskDataset`, augmentation, case splits |
| `models.py` | Tool and step architectures, checkpoint save/load |
| `metrics.py` | mAP, per-class P/R/F1, confusion matrices — no sklearn dependency |
| `train_tool_detection.py` | Multi-label training loop |
| `train_step_recognition.py` | Temporal training loop |
| `evaluate.py` | Evaluation reports and Grad-CAM sample rendering |
| `predict.py` | Single-image and whole-video inference |
| `export_onnx.py` | ONNX export with numerical verification |
| `download_dataset.py` | Resumable dataset download |
| `gradcam.py` | Re-export of the shared explainability implementation |

---

## Citation

```bibtex
@article{surgvu2025,
  title  = {SurgVU: Surgical Video Understanding Dataset},
  journal= {arXiv preprint arXiv:2501.09209},
  year   = {2025},
  url    = {https://arxiv.org/html/2501.09209v1}
}
```

> [!IMPORTANT]
> Models trained here are for **surgical education and computer-vision
> research**. They are not validated for clinical use and must not inform
> patient care.
