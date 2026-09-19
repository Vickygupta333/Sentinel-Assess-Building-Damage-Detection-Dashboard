"""
End-to-end damage-assessment pipeline, mirroring Figure 1 of the paper:

    pre/post images -> building detection -> crop pre/post patches
    -> normalize (histogram equalization) -> damage score per building

Framework-agnostic: works with the built-in heuristic scorer out of the box,
or a trained TwinTowerCNN if `MODEL_MODE=cnn` is configured in main.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import cv2

from detector import BuildingBox, detect_buildings_cv

PATCH_SIZE = 161  # matches the paper's 50m x 50m crop at 0.3m resolution


@dataclass
class BuildingResult:
    id: str
    bbox: List[int]
    centroid: List[int]
    damage_score: float
    damaged: bool


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """Per-channel histogram equalization, as used in the paper to normalize
    pixel intensity ranges across different satellite images/sensors."""
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(c) for c in channels]
    return cv2.merge(eq_channels)


def crop_patch(image: np.ndarray, centroid, size: int = PATCH_SIZE) -> np.ndarray:
    """Crop a size x size patch centered on `centroid`, zero-padding at
    image edges so every building yields a fixed-size tensor."""
    h, w = image.shape[:2]
    cx, cy = centroid
    half = size // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    crop = image[y1c:y2c, x1c:x2c]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )
    return cv2.resize(crop, (size, size))


def run_assessment(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    threshold: float = 0.5,
    detector: Callable[[np.ndarray], List[BuildingBox]] = detect_buildings_cv,
) -> List[BuildingResult]:
    """Run the full pipeline and return one BuildingResult per detected
    building."""
    # Resize post to match pre if they differ (common with real downloads).
    if post_image.shape[:2] != pre_image.shape[:2]:
        post_image = cv2.resize(post_image, (pre_image.shape[1], pre_image.shape[0]))

    pre_eq = histogram_equalize(pre_image)
    post_eq = histogram_equalize(post_image)

    boxes = detector(pre_eq)

    results: List[BuildingResult] = []
    for box in boxes:
        pre_patch = crop_patch(pre_eq, box.centroid)
        post_patch = crop_patch(post_eq, box.centroid)
        score = score_fn(pre_patch, post_patch)
        results.append(
            BuildingResult(
                id=box.id,
                bbox=[box.x1, box.y1, box.x2, box.y2],
                centroid=list(box.centroid),
                damage_score=round(score, 4),
                damaged=score >= threshold,
            )
        )
    return results
