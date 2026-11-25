"""Create a tiny synthetic 4-domain dataset for the smoke test only."""
import os
import sys

import numpy as np
from PIL import Image

root = sys.argv[1]
rng = np.random.default_rng(0)
classes = ["c0", "c1", "c2"]
base = np.array([[200, 40, 40], [40, 200, 40], [40, 40, 200]])
for d in range(4):
    for ci, c in enumerate(classes):
        os.makedirs(os.path.join(root, f"d{d}", c), exist_ok=True)
        for i in range(16):
            img = np.clip(base[ci] * (1 - 0.15 * d) + rng.normal(0, 25 + 15 * d, (64, 64, 3)), 0, 255)
            Image.fromarray(img.astype(np.uint8)).save(os.path.join(root, f"d{d}", c, f"{i}.png"))
print("fake data written to", root)
