"""
Damage Assessment API.

Endpoints:
  GET  /health          — liveness check
  POST /api/detect       — upload one image, get back detected building boxes
  POST /api/infer         — upload pre + post images, get back per-building
                            damage scores (the main endpoint the frontend uses)

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

By default the app scores damage with a dependency-free heuristic (see
model.py) so it works with zero ML frameworks installed and zero internet
access. To use the real trained CNN instead:
    export DAMAGE_MODEL=cnn
    export MODEL_WEIGHTS=/path/to/checkpoint.pt
"""
from __future__ import annotations

import io
import os

import numpy as np
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from detector import detect_buildings_cv
from model import heuristic_damage_score, load_cnn_scorer, load_nvidia_vision_scorer
from pipeline import run_assessment

app = FastAPI(title="Building Damage Assessment API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- scorer selection -------------------------------------------------------
MODEL_MODE = os.environ.get("DAMAGE_MODEL", "heuristic")
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_API_BASE = os.environ.get("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")

if MODEL_MODE == "cnn":
    if not MODEL_WEIGHTS:
        raise RuntimeError("DAMAGE_MODEL=cnn requires MODEL_WEIGHTS to point at a checkpoint")
    score_fn = load_cnn_scorer(MODEL_WEIGHTS)
elif MODEL_MODE == "nvidia":
    if not NVIDIA_API_KEY:
        raise RuntimeError("DAMAGE_MODEL=nvidia requires NVIDIA_API_KEY")
    score_fn = load_nvidia_vision_scorer(
        api_key=NVIDIA_API_KEY,
        model_name=NVIDIA_MODEL,
        api_base=NVIDIA_API_BASE,
    )
else:
    score_fn = heuristic_damage_score


def _read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        # PIL cannot decode every browser-accepted photo format. Fall back to
        # OpenCV so common JPEG/PNG/WebP variants still work.
        arr = np.frombuffer(data, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is not None:
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        raise HTTPException(
            400,
            f"Could not decode image: {e}. Content-Type was: {upload.content_type or 'unknown'}",
        )


@app.get("/health")
def health():
    out = {"status": "ok", "model_mode": MODEL_MODE}
    if MODEL_MODE == "nvidia":
        out["nvidia_model"] = NVIDIA_MODEL
    return out


@app.post("/api/detect")
async def detect(image: UploadFile = File(...)):
    """Detect buildings in a single (pre-disaster) image."""
    img = _read_image(image)
    boxes = detect_buildings_cv(img)
    return JSONResponse(
        {
            "image_size": [img.shape[1], img.shape[0]],
            "buildings": [
                {"id": b.id, "bbox": [b.x1, b.y1, b.x2, b.y2], "centroid": list(b.centroid)}
                for b in boxes
            ],
        }
    )


@app.post("/api/infer")
async def infer(pre_image: UploadFile = File(...), post_image: UploadFile = File(...)):
    """Full pipeline: detect buildings on the pre-image, score damage using
    matched pre/post patches."""
    try:
        pre = _read_image(pre_image)
        post = _read_image(post_image)

        results = run_assessment(pre, post, score_fn=score_fn)
        message = "ok"
        if not results:
            message = "No buildings were detected in the pre-image. Try a clearer image with visible rooftops or use the demo scene."

        num_damaged = sum(1 for r in results if r.damaged)
        return JSONResponse(
            {
                "image_size": [pre.shape[1], pre.shape[0]],
                "model_mode": MODEL_MODE,
                "nvidia_model": NVIDIA_MODEL if MODEL_MODE == "nvidia" else None,
                "message": message,
                "summary": {
                    "total_buildings": len(results),
                    "damaged": num_damaged,
                    "undamaged": len(results) - num_damaged,
                },
                "buildings": [
                    {
                        "id": r.id,
                        "bbox": r.bbox,
                        "centroid": r.centroid,
                        "damage_score": r.damage_score,
                        "damaged": r.damaged,
                    }
                    for r in results
                ],
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("ERROR in /api/infer:")
        traceback.print_exc()
        raise HTTPException(500, f"Pipeline error: {str(e)}")
