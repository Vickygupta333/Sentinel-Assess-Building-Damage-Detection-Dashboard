# Sentinel Assess — Building Damage Detection

A working, end-to-end implementation of the pipeline described in
*Xu et al., "Building Damage Detection in Satellite Imagery Using
Convolutional Neural Networks" (NeurIPS 2019)*: upload matched
before/after satellite (or drone/aerial) images of an area, and the system
detects buildings, crops matched pre/post patches, and scores each one for
disaster damage.

It runs **fully offline out of the box** — no pretrained weights, no
internet access, no GPU required — using a classical computer-vision
building detector and a dependency-free heuristic damage scorer. It also
ships the *real* paper-faithful CNN (Twin-Tower Subtract) and a training
script, so you can swap in a trained model once you have labeled data.

```
project/
├── backend/
│   ├── main.py          FastAPI service (the API)
│   ├── detector.py      Building detection (OpenCV + optional Faster R-CNN)
│   ├── pipeline.py       Patch cropping, normalization, orchestration
│   ├── model.py          Heuristic scorer + real TwinTowerCNN (PyTorch)
│   ├── train.py          Training script for the real CNN
│   └── requirements.txt
├── frontend/
│   └── index.html        Self-contained web UI (no build step)
└── tests/
    └── test_pipeline.py  Unit + API tests (pytest)
```

---

## 1. Quick start (5 minutes)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Use NVIDIA hosted free-tier vision model instead of local scoring:

```bash
cd backend
set DAMAGE_MODEL=nvidia
set NVIDIA_API_KEY=nvapi-your-key
set NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct
uvicorn main:app --reload --port 8000
```

You should see `Uvicorn running on http://127.0.0.1:8000`. Check it:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_mode":"heuristic"}
```

### Frontend

The frontend is a single static HTML file — no npm install, no build step.

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500` in a browser. Click **"Load a synthetic demo
scene"** to try it immediately without any real imagery, or upload your own
matched pre/post images and click **Run assessment**.

> If your backend isn't on `http://localhost:8000`, set
> `window.SENTINEL_API_BASE = "https://your-api-host"` in a small
> `<script>` tag before `index.html`'s closing `</body>`, or edit the
> `API_BASE` constant near the top of the `<script>` block.

### Run the tests

```bash
cd backend
pip install pytest
python -m pytest ../tests -q
```

All 6 tests should pass — they cover patch cropping, histogram
equalization, building detection, the heuristic damage scorer, the full
pipeline, and the `/api/infer` endpoint.

---

## 2. How it works (pipeline)

This mirrors Figure 1 of the paper:

```
pre/post images
      │
      ▼
[1] Building Detection   — find every building-like structure on the pre image
      │
      ▼
[2] Patch Cropping        — crop a matched pre + post patch around each building
      │
      ▼
[3] Normalization          — per-channel histogram equalization on both patches
      │
      ▼
[4] Damage Scoring          — compare pre/post patch → score in [0, 1]
      │
      ▼
Per-building damage report (JSON) → rendered on the map/canvas
```

### Step 1 — Building detection (`backend/detector.py`)

Two implementations are provided:

- **`detect_buildings_cv` (default, active)** — a classical detector using
  Canny edge detection + contour finding + non-max suppression. It needs no
  pretrained weights and no internet access, which is why the whole app
  works offline. It finds real rectangular structures in the image (not a
  static grid), but is far less accurate than a trained detector.
- **`detect_buildings_frcnn` (production path)** — torchvision's pretrained
  Faster R-CNN, exactly as described in the paper and the original project
  plan. Requires `pip install torch torchvision` and internet access (or a
  cached checkpoint) to download COCO weights on first run. In practice
  you'd fine-tune this on a building-footprint dataset (e.g. via
  [xBD](https://xview2.org)) rather than using raw COCO classes, since COCO
  has no "building" class.

To switch, change the `detector=` argument passed to `run_assessment()` in
`backend/main.py`.

### Step 2 — Patch cropping (`backend/pipeline.py::crop_patch`)

For every detected building, a square patch is cropped from both the pre-
and post-event image, centered on the building's centroid. The paper uses
161×161 px at 0.3 m resolution (≈50 m × 50 m on the ground); the code
defaults to the same `PATCH_SIZE = 161`. Patches that fall near the image
edge are reflection-padded rather than dropped, so every detected building
gets a usable crop.

### Step 3 — Normalization (`backend/pipeline.py::histogram_equalize`)

Per-channel histogram equalization, exactly as described in the paper's
data-generation section — this is the *only* preprocessing step, kept
deliberately minimal so it stays practical in a real disaster-response
scenario where heavy manual cleanup isn't feasible.

### Step 4 — Damage scoring (`backend/model.py`)

Two scorers, selected via the `DAMAGE_MODEL` environment variable:

- **`heuristic` (default)** — no ML framework required. Blends three
  classical signals between the pre/post patch: raw pixel difference (the
  spirit of the paper's "subtract" combination), edge-density change
  (collapsed roofs lose regular edges), and color-histogram shift (exposed
  rubble/soil looks different from intact roofing). This is a *stand-in*,
  not a trained model — good enough to make the whole app runnably
  end-to-end, not good enough for real damage assessment.
- **`cnn`** — the real **Twin-Tower Subtract (TTS)** architecture from the
  paper (`backend/model.py::build_twin_tower_cnn`). Two weight-shared
  convolutional towers independently encode the pre/post patch, their
  feature maps are subtracted element-wise, and a small head produces a
  sigmoid damage score. This is the architecture that scored best in the
  paper (AUC ≈ 0.83 on the Haiti dataset), beating single-tower and
  concatenation variants.

  To use it:
  ```bash
  pip install torch torchvision
  export DAMAGE_MODEL=cnn
  export MODEL_WEIGHTS=/path/to/tts_model.pt
  uvicorn main:app --reload --port 8000
  ```

---

## 3. Training the real CNN on real data

The heuristic scorer is a placeholder. To get results resembling the
paper's, you need labeled data and to train `TwinTowerCNN`.

1. **Get imagery + labels.** Pre/post satellite images (e.g. via
   [Google Earth Engine](https://earthengine.google.com) or a commercial
   provider) and [UNOSAT/HDX](https://data.humdata.org) damage
   assessments for a real disaster. The paper combines "Severe Damage" and
   "Destroyed" into one positive class, and treats detected-but-unlabeled
   buildings as negatives.

2. **Generate patches.** Reuse `backend/pipeline.py`'s `crop_patch` and
   `histogram_equalize` to turn each labeled building into a 161×161
   pre/post patch pair. Save them as
   ```
   data_dir/
     labels.csv          # patch_id,label
     pre/<patch_id>.png
     post/<patch_id>.png
   ```

3. **Train.**
   ```bash
   cd backend
   pip install torch torchvision
   python train.py --data_dir /path/to/data_dir --epochs 20 --out tts_model.pt
   ```

4. **Serve it.** Point the API at the checkpoint as shown above
   (`DAMAGE_MODEL=cnn`, `MODEL_WEIGHTS=tts_model.pt`).

5. **Cross-region tip from the paper:** a model trained on one disaster
   generalizes poorly to a new one (AUC drops from ~0.8 to ~0.6). Training
   on *two or more* disasters, then fine-tuning on a small sample (~10%)
   of the target region, recovers most of the accuracy — worth
   replicating if you're deploying across multiple events.

---

## 4. API reference

### `GET /health`
```json
{ "status": "ok", "model_mode": "heuristic" }
```

When running with NVIDIA mode:

```json
{ "status": "ok", "model_mode": "nvidia", "nvidia_model": "meta/llama-3.2-11b-vision-instruct" }
```

### `POST /api/detect`
Multipart form field: `image` (single file). Returns detected building
boxes on that image only (no damage scoring).

```json
{
  "image_size": [400, 400],
  "buildings": [
    { "id": "b0", "bbox": [40, 40, 100, 100], "centroid": [70, 70] }
  ]
}
```

### `POST /api/infer`
Multipart form fields: `pre_image`, `post_image`. Runs the full pipeline.

```json
{
  "image_size": [400, 400],
  "model_mode": "heuristic",
  "summary": { "total_buildings": 4, "damaged": 1, "undamaged": 3 },
  "buildings": [
    {
      "id": "b0",
      "bbox": [40, 40, 100, 100],
      "centroid": [70, 70],
      "damage_score": 0.83,
      "damaged": true
    }
  ]
}
```

---

## 5. Deploying beyond localhost

- **Containerize:** wrap `backend/` in a `Dockerfile` (Python 3.11-slim +
  `pip install -r requirements.txt` + `CMD uvicorn main:app --host 0.0.0.0
  --port 8000`), push to a registry, deploy to any container platform
  (Cloud Run, ECS, App Runner…).
- **CI:** add a GitHub Actions workflow that runs `pytest ../tests` on
  every push and builds the Docker image on merge to `main`.
- **Serve the frontend statically:** `frontend/index.html` needs no build
  step — host it on any static file server or CDN (S3 + CloudFront, GitHub
  Pages, Netlify) and point `SENTINEL_API_BASE` at your deployed API.
- **Security, before going further than a demo:** put the API behind
  HTTPS, add auth (API keys or OAuth) to `/api/infer` and `/api/detect`,
  restrict `CORSMiddleware`'s `allow_origins` in `main.py` to your actual
  frontend domain, and store any satellite-provider or Earth Engine
  credentials in a secrets manager rather than environment files checked
  into git.
- **Scale:** the pipeline processes one image pair per request; for large
  areas, tile the input imagery and fan out requests across multiple
  container instances (e.g. AWS Batch or a Kubernetes Job per tile), then
  merge the per-tile building lists.

---

## 6. Known limitations (read before using on real imagery)

- The default building detector (`detect_buildings_cv`) is a classical
  contour detector, not a trained model — it will miss irregular or
  densely packed buildings and can pick up non-building rectangular
  shapes (parking lots, containers). Swap in `detect_buildings_frcnn`
  (or better, a detector fine-tuned on building footprints) for anything
  beyond a demo.
- The default damage scorer (`heuristic_damage_score`) is a
  pixel/edge/color heuristic, not a trained classifier — it's there so the
  whole app is runnable with zero setup, not because it's an accurate
  damage assessment tool. Train and serve the real `TwinTowerCNN` (§3)
  before using this for anything operational.
- Pre/post images must already be roughly aligned (same geographic area,
  similar scale). The pipeline resizes post to match pre's dimensions but
  does not perform geometric registration.
