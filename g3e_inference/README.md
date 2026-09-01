# g3e-inference

Production inference for **G3E-1** (YOLO detection) + **G3E-2** (Qwen2.5-VL
+ LoRA semantic reasoning). This is the standalone, `pip install`-able
package meant to be integrated directly into the G3E app — it does not
depend on anything else in the `g3e-models` training repo.

```bash
pip install g3e-inference
```

```python
from g3e_inference import G3EPipeline

pipeline = G3EPipeline()  # zero-config
event = pipeline.run("frame.jpg", output_dir="./events/evt_001")
print(event.model_dump_json(by_alias=True, indent=2))
```

That's the whole integration surface for the common case. Everything below
documents exactly what that call does, what it needs, and what it hands
back, so integrating this into the app behaves exactly as planned — no
surprises.

---

## 1. How model resolution works (read this first)

Two models are needed: G3E-1's YOLO weights (a single `.pt` file) and
G3E-2's LoRA adapter (a small directory of files) on top of the
`Qwen/Qwen2.5-VL-3B-Instruct` base model.

**Resolution policy — local if available, Hugging Face Hub otherwise:**

```
                    Was a local path given,
                    AND does it exist?
                       yes |        | no
                           v        v
                    use it directly   resolve from Hugging Face Hub
                    (NO network call)  (huggingface_hub's own cache:
                                        first call downloads + caches,
                                        every call after that on this
                                        machine is offline)
```

This is implemented once, in `g3e_inference/artifacts.py`, and used
identically by both `G3E1Detector` and `G3E2Reasoner` — see
`resolve_weights_file()` and `resolve_adapter_dir()`.

**What this means in practice for the app:**
- **First ever prediction on a machine**: needs network access once, to
  pull both models from Hugging Face Hub. This can take a few minutes
  (the Qwen2.5-VL-3B base model alone is several GB).
- **Every prediction after that**: fully offline. The models live in
  `~/.cache/huggingface/hub` (or wherever `$HF_HOME` points), and nothing
  in this library re-downloads them once cached.
- **Pre-warming the cache** (e.g. as a Docker build step or deploy script,
  so the first real request from a user isn't slow):

  ```bash
  g3e-download
  ```

  This is installed as a console script by `pip install g3e-inference`
  and does nothing except trigger the same resolution logic ahead of time.

- **Using your own local checkpoint instead of the published model**
  (e.g. while iterating on a new fine-tune before publishing it):

  ```python
  from g3e_inference import G3EPipeline, G3E1Detector, G3E2Reasoner

  pipeline = G3EPipeline(
      detector=G3E1Detector(weights="./my_local/best.pt"),
      reasoner=G3E2Reasoner(adapter="./my_local/g3e2_adapter"),
  )
  ```

  If `weights`/`adapter` point at something that doesn't actually exist on
  disk, resolution silently falls through to the Hugging Face repo — it
  will not raise just because your local override was wrong, so
  double-check the path if you expected it to be used.

## 2. Configuration — which HF repos it pulls from

Set these as real environment variables in whatever runs the app
(container env, `.env` file, deployment platform config):

| Variable | Default | Purpose |
|---|---|---|
| `G3E1_HF_REPO` | `Godsave22/g3e1-yolo` *(placeholder — update this)* | G3E-1 weights repo |
| `G3E1_HF_FILENAME` | `best.pt` | Which file in that repo |
| `G3E2_HF_REPO` | `Godsave22/g3e2-lora-v1` *(placeholder — update this)* | G3E-2 LoRA adapter repo |
| `G3E2_BASE_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Base model the adapter was trained against — only change this if you fine-tuned against a different base |
| `HF_TOKEN` | unset | Only needed if any of the above repos are private |

**The two placeholder defaults above must be updated to your actual
published repo ids before this is used for real** — see
`g3e_inference/defaults.py`. Nothing in this library will error if you
forget; it will just try to download from a repo that doesn't exist (or
isn't yours), so set these explicitly in your deployment environment.

## 3. Full input contract

### `G3EPipeline.run(image_path, output_dir, event_id=None)`

| Argument | Type | Required | Notes |
|---|---|---|---|
| `image_path` | `str` | yes | Path to a JPG/PNG file on disk. Any resolution — both models resize internally. |
| `output_dir` | `str` or `Path` | yes | Created if it doesn't exist. Two files are written here (see below). |
| `event_id` | `str` | no | Auto-generated (`evt_<12 hex chars>`) if omitted. |

There is currently no in-memory-bytes input path (e.g. passing a numpy
array or raw bytes instead of a file path) — if your app has frames in
memory (e.g. from a live OpenCV capture loop), write the frame to a
temporary file first:

```python
import cv2, tempfile

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    cv2.imwrite(tmp.name, frame)  # frame: a numpy array from cv2.VideoCapture
    event = pipeline.run(tmp.name, output_dir="./events/evt_001")
```

### Using `G3E1Detector`/`G3E2Reasoner` independently

You don't have to go through `G3EPipeline` — both are usable standalone:

```python
from g3e_inference import G3E1Detector, G3E2Reasoner

detector = G3E1Detector()
g3e1_output = detector.predict("frame.jpg")   # -> G3E1Output

reasoner = G3E2Reasoner()
detections_as_dicts = [d.model_dump(by_alias=True) for d in g3e1_output.objects]
g3e2_output = reasoner.predict("frame.jpg", detections_as_dicts)  # -> G3E2Output
```

`G3E2Reasoner.predict()` expects `detections` as a **list of plain
dicts**, each with a `"class"` key (not `"class_name"`) — matching G3E-1's
raw wire format exactly:

```json
[{"class": "person", "class_id": 0, "confidence": 0.98, "bbox": [120, 80, 420, 620]}]
```

`bbox` is always pixel `[x1, y1, x2, y2]` — never normalized 0-1, never
`[x, y, w, h]`.

## 4. Full output contract

`G3EPipeline.run()` produces three things, always together:

**1. Returns a `G3EEvent`** (pydantic model, `g3e_inference.schemas`):

```json
{
  "event_id": "evt_b33668ab4658",
  "timestamp": "2026-08-26T12:00:00Z",
  "image": {
    "original": "frame.jpg",
    "annotated": "./events/evt_001/annotated.jpg"
  },
  "g3e1": {
    "objects": [
      {"class": "person", "class_id": 0, "confidence": 0.98, "bbox": [120.0, 80.0, 420.0, 620.0]},
      {"class": "knife", "class_id": 4, "confidence": 0.91, "bbox": [300.0, 270.0, 355.0, 350.0]}
    ]
  },
  "g3e2": {
    "state": "potential_threat",
    "severity": "high",
    "description": "A person appears to be holding a knife.",
    "reason": "Weapon detected near a person.",
    "recommended_action": "alert_user",
    "confidence": null
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `event_id` | `str` | `evt_<12 hex chars>` unless you passed one in |
| `timestamp` | `str` | ISO 8601, UTC, `Z`-suffixed |
| `image.original` | `str` | Echoes back whatever `image_path` you passed |
| `image.annotated` | `str` | Path to the rendered frame — see below |
| `g3e1.objects` | `list[DetectedObject]` | Empty list if nothing detected — never `null` |
| `g3e1.objects[].class` | `str` | One of the 8 frozen v1 classes: `person, fire, smoke, gun, knife, car, dog, cat` |
| `g3e1.objects[].bbox` | `list[float]` (len 4) | Pixel `[x1, y1, x2, y2]` |
| `g3e2.state` | `str` | One of: `normal, caution, hazard, potential_threat` |
| `g3e2.severity` | `str` | One of: `none, low, medium, high, critical` |
| `g3e2.recommended_action` | `str` | e.g. `none, log_and_monitor, alert_user, alert_immediately, review` |
| `g3e2.confidence` | `float \| null` | Currently always `null` — G3E-2 doesn't emit a confidence score today; reserved for a future version |

**Always call `.model_dump_json(by_alias=True)`** (or `g3e_inference.to_json(event)`,
which does this for you) when serializing an event yourself.
`DetectedObject.class_name` aliases to `"class"` on the wire —
`by_alias=True` is what actually produces `"class"` instead of
`"class_name"` in the JSON. `G3EPipeline.run()` and `save_event()` already
do this correctly internally; this only matters if you call
`.model_dump_json()` directly on the returned object yourself.

**2. Writes `{output_dir}/annotated.jpg`** — the source frame with each
detection's box + `"CLASS NN%"` label drawn on it (OpenCV, spec section
9). Use this for a human-facing alert/dashboard image.

**3. Writes `{output_dir}/event.json`** — the same event as above,
already serialized correctly (`by_alias=True` applied). Read this
directly if your app polls a directory rather than consuming the return
value in-process.

## 5. Error behavior

| Failure | What happens |
|---|---|
| `image_path` doesn't exist / isn't a readable image | `ValueError` from the renderer (`cv2` couldn't read it) — check the path before calling |
| G3E-2's raw output isn't valid JSON (rare; more likely with an undertrained model) | `ValueError` from `parse_model_output`, with the raw text included in the message — do not silently swallow this; it signals the model needs more training, not a parsing bug to work around |
| Model repo doesn't exist / network unavailable on first-ever call | Whatever `huggingface_hub` raises (typically `RepositoryNotFoundError` or a connection error) — propagates up uncaught; wrap `G3EPipeline()` construction or the first `.run()` call in your own retry/error handling if your app needs graceful degradation |
| Private repo, no token set | `huggingface_hub`'s own 401/403 error — set `HF_TOKEN` |

This library does not currently catch and wrap these into a single
G3E-specific exception type — they surface as whatever the underlying
library (`cv2`, `huggingface_hub`, `json`) raises. If your app wants a
uniform error type at the integration boundary, catch broadly around
`pipeline.run(...)` on your side.

## 6. CLI

Installed as console scripts:

```bash
g3e-predict --image frame.jpg --output-dir ./events/evt_001
g3e-download                          # pre-fetch both models
g3e-download --skip-g3e1              # only pre-fetch G3E-2
```

## 7. What this package does NOT include

Deliberately out of scope for `g3e-inference` (they live in the separate
`g3e-models` training repo instead): dataset preparation, class balancing,
training scripts for either model, and the `ultralytics`/`peft` training
loops. This package is inference-only — it consumes already-trained models
from Hugging Face Hub (or a local override) and does nothing else.

## 8. Publishing this package

```bash
pip install build twine
python -m build
twine upload dist/*
```

Bump `version` in `pyproject.toml` before each publish. Nothing else in
this repo needs to change for a version bump — the training-side
`g3e-models` repo and this package are versioned independently.

## 9. Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

30 tests, fully offline — no GPU, no real model download, no network
access. Covers artifact resolution (including the exact local-vs-HF
precedence rules), event assembly and serialization, OpenCV rendering, and
full pipeline orchestration via fake detector/reasoner objects (dependency
injection — see `tests/test_pipeline.py`).
