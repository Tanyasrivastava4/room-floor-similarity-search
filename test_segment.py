"""
test_segment.py
----------------
Runs floor segmentation on all query images and saves a green-overlay
visualization for each, so we can visually confirm the mask is only
covering the floor - not furniture/rugs/walls.
"""

import cv2
import numpy as np
import os
from segment import extract_floor, get_floor_mask

os.makedirs("outputs/viz", exist_ok=True)

for i in range(1, 6):
    path = f"data/query/{i}.jpg"
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path} - check your folder/file names")
        continue

    raw_mask, method = get_floor_mask(img)
    patch, cropped_mask, _ = extract_floor(img)

    green = np.zeros_like(img)
    green[:, :, 1] = 255
    overlay = np.where(raw_mask[..., None] == 1,
                        cv2.addWeighted(img, 0.5, green, 0.5, 0),
                        img)

    cv2.imwrite(f"outputs/viz/overlay_query{i}.jpg", overlay)
    cv2.imwrite(f"outputs/viz/patch_query{i}.jpg", patch)

    print(f"query {i}: method={method}, floor coverage={raw_mask.mean():.2%}")

print("\nSaved overlays and patches to outputs/viz/")
print("Open outputs/viz/overlay_query1.jpg (etc.) and check the green area is ONLY the floor.")