"""Quick pipeline test"""
import numpy as np
from detector import detect_buildings_cv
from model import heuristic_damage_score
from pipeline import run_assessment

# Create synthetic images
pre = np.random.randint(60, 90, (400, 400, 3), dtype=np.uint8)
post = np.random.randint(60, 90, (400, 400, 3), dtype=np.uint8)

# Add some building-like structures
pre[40:100, 40:100] = 150
post[40:100, 40:100] = 100  # simulate damage

try:
    results = run_assessment(pre, post, score_fn=heuristic_damage_score)
    print(f"✓ Pipeline works! Detected {len(results)} buildings")
    if results:
        print(f"  First building: {results[0]}")
except Exception as e:
    print(f"✗ Pipeline error: {e}")
    import traceback
    traceback.print_exc()
