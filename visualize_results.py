"""
visualize_results.py
----------------------
Builds one image per query: the query room photo on the left, and its
top-5 ranked product thumbnails on the right with similarity scores
labeled. This is the human-checkable evidence for the assessment -
a reviewer can look at this and immediately judge whether the ranking
makes visual sense.
"""

import os
import json
import cv2
import numpy as np

RESULTS_JSON = "outputs/results/all_results.json"
PRODUCT_DIR = "data/sku"
QUERY_DIR = "data/query"
OUTPUT_DIR = "outputs/viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THUMB_SIZE = 220
QUERY_DISPLAY_HEIGHT = 500
PADDING = 15
LABEL_HEIGHT = 30


def load_and_resize(path, target_w=None, target_h=None):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if target_h is not None:
        scale = target_h / h
        img = cv2.resize(img, (int(w * scale), target_h))
    elif target_w is not None:
        scale = target_w / w
        img = cv2.resize(img, (target_w, int(h * scale)))
    return img


def make_thumb_with_label(product_id, score, size=THUMB_SIZE):
    path = os.path.join(PRODUCT_DIR, f"{product_id}.jpg")
    img = cv2.imread(path)
    img = cv2.resize(img, (size, size))

    canvas = np.full((size + LABEL_HEIGHT, size, 3), 255, dtype=np.uint8)
    canvas[:size, :, :] = img

    label = f"#{product_id}  score={score:.3f}"
    cv2.putText(canvas, label, (5, size + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def build_grid_for_query(query_num, ranking, top_k=5):
    query_path = os.path.join(QUERY_DIR, f"{query_num}.jpg")
    query_img = load_and_resize(query_path, target_h=QUERY_DISPLAY_HEIGHT)

    thumbs = []
    for entry in ranking[:top_k]:
        thumb = make_thumb_with_label(entry["product_id"], entry["final_score"])
        thumbs.append(thumb)

    thumb_row_h = THUMB_SIZE + LABEL_HEIGHT
    cols = 3
    rows = int(np.ceil(top_k / cols))
    grid_w = cols * THUMB_SIZE + (cols - 1) * PADDING
    grid_h = rows * thumb_row_h + (rows - 1) * PADDING

    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = idx // cols, idx % cols
        y0 = r * (thumb_row_h + PADDING)
        x0 = c * (THUMB_SIZE + PADDING)
        grid[y0:y0 + thumb_row_h, x0:x0 + THUMB_SIZE] = thumb

    total_h = max(query_img.shape[0], grid_h)
    total_w = query_img.shape[1] + grid_w + PADDING * 2

    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    canvas[:query_img.shape[0], :query_img.shape[1]] = query_img
    gx0 = query_img.shape[1] + PADDING
    canvas[:grid_h, gx0:gx0 + grid_w] = grid

    cv2.putText(canvas, f"Query {query_num}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Top 5 matches", (gx0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


def main():
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)

    for q in range(1, 6):
        ranking = all_results[f"query_{q}"]["ranking"]
        grid = build_grid_for_query(q, ranking, top_k=5)
        out_path = os.path.join(OUTPUT_DIR, f"result_grid_query{q}.jpg")
        cv2.imwrite(out_path, grid)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()