"""
Basic tests. Run with:
    cd backend && pytest ../tests -q
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np
from fastapi.testclient import TestClient

from pipeline import crop_patch, histogram_equalize, run_assessment
from detector import detect_buildings_cv
from model import heuristic_damage_score


def make_synthetic_scene(seed=0, damage=False):
    """Create a synthetic 'satellite image' with a few rectangular
    building-like blobs, for tests that don't require real imagery."""
    rng = np.random.default_rng(seed)
    img = (rng.integers(60, 90, size=(400, 400, 3))).astype(np.uint8)  # ground texture
    buildings = [(40, 40, 90, 90), (150, 60, 220, 130), (250, 200, 320, 260)]
    for (x1, y1, x2, y2) in buildings:
        color = rng.integers(150, 220, size=3)
        img[y1:y2, x1:x2] = color
        if damage:
            # scatter debris-like noise over one building to simulate damage
            if (x1, y1) == (150, 60):
                noise = rng.integers(0, 255, size=(y2 - y1, x2 - x1, 3)).astype(np.uint8)
                img[y1:y2, x1:x2] = noise
    return img, buildings


def test_crop_patch_shape():
    img, _ = make_synthetic_scene()
    patch = crop_patch(img, (200, 200), size=161)
    assert patch.shape == (161, 161, 3)


def test_histogram_equalize_shape_preserved():
    img, _ = make_synthetic_scene()
    eq = histogram_equalize(img)
    assert eq.shape == img.shape


def test_detect_buildings_finds_boxes():
    img, buildings = make_synthetic_scene()
    boxes = detect_buildings_cv(img)
    assert len(boxes) >= 1


def test_heuristic_scores_damage_higher_than_no_change():
    pre, _ = make_synthetic_scene(seed=1, damage=False)
    post_same, _ = make_synthetic_scene(seed=1, damage=False)
    post_damaged, _ = make_synthetic_scene(seed=1, damage=True)

    patch_pre = crop_patch(pre, (185, 95))
    patch_post_same = crop_patch(post_same, (185, 95))
    patch_post_damaged = crop_patch(post_damaged, (185, 95))

    score_same = heuristic_damage_score(patch_pre, patch_post_same)
    score_damaged = heuristic_damage_score(patch_pre, patch_post_damaged)

    assert score_damaged > score_same


def test_run_assessment_end_to_end():
    pre, _ = make_synthetic_scene(seed=2, damage=False)
    post, _ = make_synthetic_scene(seed=2, damage=True)
    results = run_assessment(pre, post, score_fn=heuristic_damage_score)
    assert len(results) >= 1
    for r in results:
        assert 0.0 <= r.damage_score <= 1.0


def test_api_health_and_infer():
    os = __import__("os")
    os.environ.setdefault("DAMAGE_MODEL", "heuristic")
    import importlib
    import main as api_main

    importlib.reload(api_main)
    client = TestClient(api_main.app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    import io
    from PIL import Image

    pre_img, _ = make_synthetic_scene(seed=3, damage=False)
    post_img, _ = make_synthetic_scene(seed=3, damage=True)

    def to_bytes(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        buf.seek(0)
        return buf

    resp = client.post(
        "/api/infer",
        files={
            "pre_image": ("pre.png", to_bytes(pre_img), "image/png"),
            "post_image": ("post.png", to_bytes(post_img), "image/png"),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "buildings" in data
    assert "summary" in data
