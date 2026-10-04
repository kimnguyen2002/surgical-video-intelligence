# `ml/` — where the checkpoints came from

**Nothing in this directory runs in production.** It is not imported by the
application, not part of the build, and not served to anyone. The app is a
static page; this is the Python it was built out of.

It is in the repository because a checkpoint without the code that produced it
is an unverifiable artefact. `public/models/*.onnx` are claims; this is the
evidence for them.

## What is here

```
ai/training/      the PyTorch pipeline: datasets, transforms, the three
                  training entry points, metrics, ONNX export
ai/datasets/      SurgVU label parsing — the part with the schema traps
ai/llm/           grounded.py and knowledge.py, the Python originals of the
                  browser assistant in src/assistant/
ai/explainability/  the Grad-CAM implementation the browser's CAM replaced
ai/rag/, ai/vectorstore/, ai/embeddings/   retrieval, from the server era
ai/agents/, ai/vision/, ai/voice/, ai/knowledge/   the rest of the platform
checkpoints/      the three .pt files that export_models.py converts
tests/            113 tests, mostly over label parsing and the training pipeline
```

## Running the tests

```bash
cd ml
pip install -r requirements.txt   # only the core + vision blocks are needed
python3 -m pytest tests -q
```

113 pass, 4 skip when optional dependencies are absent.

The ones worth looking at are in `tests/test_labels_real_schema.py`. They pin
the three ways SurgVU's annotations will silently mis-parse — different column
names between the two CSVs, a `part` column that looks like a case id, and two
incompatible time formats — each of which produces a model that trains happily
and means nothing. See the dataset section of the top-level README.

## What was removed

The tests that exercised HTTP endpoints (`test_api.py`, `test_security.py`,
`test_agents.py`) and the `client` fixture in `conftest.py`. They tested a
FastAPI backend that this project no longer has, and a test that cannot fail
because the thing it tests was deleted is worse than no test.

The modules those tests covered are still here. They are history rather than
dependencies: `ai/llm/grounded.py` in particular is the original of
`src/assistant/grounded.ts`, and reading them side by side is the clearest
account of what moving this system into the browser actually cost.

## Regenerating the ONNX graphs

```bash
python3 scripts/export_models.py     # from the repository root
```

That is the whole command. `tool.onnx` carries two graph outputs beyond
`logits` — `embedding`, which the out-of-domain guard measures distance in, and
`features`, which the activation heatmap is computed from — and a fresh export
overwrites the file and takes both with it.

The failure that would cause is quiet: the worker treats a missing output as
"that capability did not ship" and carries on, so the app keeps detecting
instruments while the guard stops refusing anything and the heatmap button
disappears. Nothing errors. So `export_models.py` re-runs `export_cam.py` and
`build_domain_guard.py` itself rather than leaving it to a line in a README.
Both are idempotent and can also be run on their own:

```bash
python3 scripts/export_cam.py          # CAM weights, with a numeric self-check
python3 scripts/build_domain_guard.py  # re-calibrates against its probe set
python3 scripts/benchmark_clips.py     # what the detector does on the excerpts
```
