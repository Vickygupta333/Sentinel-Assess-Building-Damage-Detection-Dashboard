"""
Damage classification models.

This module ships two scorers:

1. `heuristic_damage_score` — a dependency-free (no PyTorch needed) stand-in
   scorer used by default so the whole app runs out of the box. It mimics
   the spirit of the paper's "Twin-Tower Subtract" (TTS) idea — compare
   pre/post patches and turn their difference into a score — but uses
   classical image statistics (structural pixel difference, edge-density
   change, color histogram shift) instead of a trained CNN.

2. `TwinTowerCNN` — the real architecture described in Xu et al. 2019
   ("Building Damage Detection in Satellite Imagery Using Convolutional
   Neural Networks"), section 3, model (d) TTS. Two identical convolutional
   towers independently encode the pre- and post-disaster patch, their
   feature maps are subtracted element-wise, and the result is fed through
   a small classification head ending in a sigmoid. Requires PyTorch.

Swap the heuristic for the trained CNN once you have labeled data (see
train.py) by setting DAMAGE_MODEL=cnn and MODEL_WEIGHTS=<path> in the
environment — see backend/main.py for the wiring.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np


# ---------------------------------------------------------------------------
# 1. Heuristic scorer (default — no ML framework required)
# ---------------------------------------------------------------------------
def heuristic_damage_score(pre_patch: np.ndarray, post_patch: np.ndarray) -> float:
    """Score how "damaged" a building looks by comparing its pre/post crops.

    pre_patch, post_patch: HxWx3 uint8 arrays, same shape, already aligned.
    Returns a float in [0, 1] — higher means more likely damaged.
    """
    if pre_patch.shape != post_patch.shape:
        # Resize post to match pre if the detector produced slightly
        # different crop sizes.
        import cv2

        post_patch = cv2.resize(post_patch, (pre_patch.shape[1], pre_patch.shape[0]))

    pre = pre_patch.astype(np.float32) / 255.0
    post = post_patch.astype(np.float32) / 255.0

    # (a) Raw pixel difference — analogous to the TTS "subtract" step.
    pixel_diff = np.abs(pre - post).mean()

    # (b) Edge-density change — collapsed/rubbled buildings lose their
    # regular roofline edges, or gain a lot of chaotic new edges from debris.
    import cv2

    def edge_density(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges.mean() / 255.0

    edge_change = abs(edge_density(pre) - edge_density(post))

    # (c) Color histogram shift — exposed rubble/soil vs. roofing material
    # shows up as a shift in the dominant color distribution.
    def hist(img):
        h = cv2.calcHist(
            [(img * 255).astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        return cv2.normalize(h, h).flatten()

    hist_dist = cv2.compareHist(hist(pre), hist(post), cv2.HISTCMP_BHATTACHARYYA)

    # Weighted blend, then squash to [0, 1]. Weights were picked so that a
    # visually obvious "before/after" difference lands above the 0.5
    # decision threshold used elsewhere in the app.
    raw = 0.45 * pixel_diff + 0.25 * edge_change + 0.30 * hist_dist
    score = 1.0 / (1.0 + np.exp(-8.0 * (raw - 0.28)))  # logistic squash centered near typical raw values
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. Real CNN (paper-faithful) — optional, requires torch
# ---------------------------------------------------------------------------
def _lazy_import_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401

        return torch, nn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for TwinTowerCNN. Install with "
            "`pip install torch torchvision` (see requirements.txt)."
        ) from e


def build_twin_tower_cnn():
    """Factory so importing this module never requires torch unless called."""
    torch, nn = _lazy_import_torch()

    class ConvTower(nn.Module):
        """Shared-weight feature extractor applied to pre and post patches."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def forward(self, x):
            return self.features(x)

    class TwinTowerCNN(nn.Module):
        """Twin-Tower Subtract (TTS) architecture, Xu et al. 2019, Fig 2(d).

        Two identical convolutional towers (shared weights) independently
        encode the 3-channel pre- and post-disaster patch. Their feature
        maps are subtracted element-wise, then passed through one more
        conv block and two FC layers to a sigmoid damage score.
        """

        def __init__(self, input_size: int = 161):
            super().__init__()
            self.tower = ConvTower()

            # Second half: shared for both TTC/TTS in the paper — one more
            # conv block followed by 2 FC layers and a sigmoid.
            self.post_conv = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, pre, post):
            feat_pre = self.tower(pre)
            feat_post = self.tower(post)
            diff = torch.abs(feat_pre - feat_post)  # "subtract" combination
            x = self.post_conv(diff)
            logit = self.classifier(x)
            return torch.sigmoid(logit).squeeze(-1)

    return TwinTowerCNN()


def load_cnn_scorer(weights_path: str):
    """Returns a `score(pre_patch, post_patch) -> float` callable backed by a
    trained TwinTowerCNN checkpoint."""
    torch, _ = _lazy_import_torch()
    model = build_twin_tower_cnn()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    def _to_tensor(patch: np.ndarray):
        t = torch.from_numpy(patch.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

    @torch.no_grad()
    def score(pre_patch: np.ndarray, post_patch: np.ndarray) -> float:
        pre_t = _to_tensor(pre_patch)
        post_t = _to_tensor(post_patch)
        out = model(pre_t, post_t)
        return float(out.item())

    return score


# ---------------------------------------------------------------------------
# 3. NVIDIA API vision scorer (hosted model) — optional, requires internet
# ---------------------------------------------------------------------------
def _png_data_url(patch: np.ndarray) -> str:
    import cv2

    ok, enc = cv2.imencode(".png", cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode image patch as PNG")
    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_score_from_text(text: str) -> float:
    # Preferred response shape: {"damage_score": 0.73}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "damage_score" in parsed:
            return float(np.clip(float(parsed["damage_score"]), 0.0, 1.0))
    except Exception:
        pass

    # Fallback: grab the first decimal-like number from the model text.
    match = re.search(r"([01](?:\.\d+)?)", text)
    if not match:
        raise ValueError(f"Could not parse damage score from model output: {text!r}")
    return float(np.clip(float(match.group(1)), 0.0, 1.0))


def load_nvidia_vision_scorer(
    api_key: str,
    model_name: str = "meta/llama-3.2-11b-vision-instruct",
    api_base: str = "https://integrate.api.nvidia.com/v1",
):
    """Returns a `score(pre_patch, post_patch) -> float` callable backed by
    NVIDIA's hosted OpenAI-compatible chat/completions API.

    This path is useful when you want a cloud-hosted model (including free
    tier usage limits) instead of local training/checkpoints.
    """
    if not api_key:
        raise ValueError("NVIDIA API key is required for DAMAGE_MODEL=nvidia")

    endpoint = api_base.rstrip("/") + "/chat/completions"

    def score(pre_patch: np.ndarray, post_patch: np.ndarray) -> float:
        pre_data_url = _png_data_url(pre_patch)
        post_data_url = _png_data_url(post_patch)

        prompt = (
            "You are a building damage assessment model. "
            "Compare a pre-disaster and post-disaster image patch of the same building. "
            "Return ONLY strict JSON: {\"damage_score\": <number between 0 and 1>} "
            "where 0 means no visible damage and 1 means severe destruction."
        )

        payload = {
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": 80,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "Pre-disaster patch:"},
                        {"type": "image_url", "image_url": {"url": pre_data_url}},
                        {"type": "text", "text": "Post-disaster patch:"},
                        {"type": "image_url", "image_url": {"url": post_data_url}},
                    ],
                }
            ],
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"NVIDIA API HTTP {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"NVIDIA API request failed: {e.reason}") from e

        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected NVIDIA API response: {raw}") from e

        return _extract_score_from_text(content)

    return score
