"""
Building detection.

Two detectors are provided:

1. `detect_buildings_cv` (default) — a classical computer-vision detector
   using edge detection + contours. It needs no pretrained weights and no
   internet access, so the app runs fully offline out of the box. It finds
   rectangular, roof-like structures on the pre-disaster image. It won't
   match a trained Faster R-CNN's accuracy, but is a genuine structural
   detector (not a static grid), so results vary with real image content.

2. `detect_buildings_frcnn` — the production path described in the project
   plan: torchvision's pretrained Faster R-CNN. Requires `torch` +
   `torchvision` and (on first run) internet access to download COCO
   weights. Swap to this once you have connectivity / a fine-tuned
   building-detection checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import cv2


@dataclass
class BuildingBox:
    id: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def centroid(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


def _non_max_suppression(boxes: List[BuildingBox], iou_thresh: float = 0.3) -> List[BuildingBox]:
    if not boxes:
        return []
    arr = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=np.float32)
    areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(arr[i, 0], arr[order[1:], 0])
        yy1 = np.maximum(arr[i, 1], arr[order[1:], 1])
        xx2 = np.minimum(arr[i, 2], arr[order[1:], 2])
        yy2 = np.minimum(arr[i, 3], arr[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return [boxes[i] for i in keep]


def detect_buildings_cv(
    image: np.ndarray,
    min_area: int = 400,
    max_area_frac: float = 0.2,
) -> List[BuildingBox]:
    """Detect roof-like rectangular structures using edges + contours.

    image: HxWx3 uint8 RGB array (the pre-disaster image).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[BuildingBox] = []
    max_area = h * w * max_area_frac
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        # Filter out very thin slivers (roads, shadows) — buildings are
        # roughly box-shaped.
        aspect = bw / float(bh + 1e-6)
        if aspect > 4 or aspect < 0.25:
            continue
        boxes.append(BuildingBox(id=f"b{idx}", x1=x, y1=y, x2=x + bw, y2=y + bh))

    boxes = _non_max_suppression(boxes)
    # Re-number sequentially after NMS for clean, stable IDs.
    for i, b in enumerate(boxes):
        b.id = f"b{i}"
    return boxes


_frcnn_model = None


def detect_buildings_frcnn(image: np.ndarray, score_thresh: float = 0.5) -> List[BuildingBox]:
    """Production-grade detector using torchvision's pretrained Faster R-CNN.

    Requires torch + torchvision, and network access on first call to
    download ImageNet/COCO-pretrained weights. Prefer fine-tuning on an
    actual building-footprint dataset (e.g. via the xBD dataset) before
    relying on this in production — the plain COCO model only knows COCO's
    80 classes and does not have a "building" class, so in practice you'd
    fine-tune the box head or swap in a footprint-specific checkpoint.
    """
    global _frcnn_model
    import torch
    from torchvision import models, transforms

    if _frcnn_model is None:
        _frcnn_model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()

    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image)
    with torch.no_grad():
        preds = _frcnn_model([tensor])[0]

    boxes: List[BuildingBox] = []
    keep_idx = (preds["scores"] > score_thresh).nonzero().squeeze(-1)
    for i, idx in enumerate(keep_idx.tolist()):
        x1, y1, x2, y2 = preds["boxes"][idx].tolist()
        boxes.append(BuildingBox(id=f"b{i}", x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)))
    return boxes
