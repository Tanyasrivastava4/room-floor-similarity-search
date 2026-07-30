# Floor Similarity Matcher

Given a room photo and a catalog of flooring product images, this system extracts the floor region from the room photo and ranks all catalog products by visual similarity to that floor — based on color, texture, grain pattern, and overall material feel.

Built for a technical assessment on image matching / visual similarity (VTPL).

---

## Problem Understanding & Assumptions

The task provides:
- **20 product images** — clean, top-down, studio-style photos of flooring, already essentially "floor only."
- **5 query images** — full room photos (sofas, walls, windows, rugs, lighting) where the floor is only *part* of the frame.

The core insight driving the whole design: **comparing a full room photo directly to a clean product photo is the wrong comparison.** A naive whole-image similarity approach would be dominated by "does this room look like a studio product shot" (never true) rather than "does the floor material match." So the first and most important design decision is:

> **Isolate the floor pixels in the query image before computing any similarity.**

Everything else in the pipeline follows from that.


## Pipeline Overview

```
Room photo (query)
   │
   ▼
[1] Floor segmentation  ──────────────►  Floor-only patch
   │  (SegFormer, ADE20K-pretrained,
   │   with GrabCut + bottom-crop
   │   fallbacks)
   ▼
[2] Feature extraction (applied to floor patch AND to each of the 20 product images)
   │  ├── CLIP embedding (512-dim)       — pattern / material / texture
   │  └── HSV color histogram             — literal color palette
   ▼
[3] Similarity scoring
   │  final_score = 0.7 × CLIP_cosine_similarity + 0.3 × color_histogram_correlation
   ▼
[4] Ranking — sort all 20 products by final_score, descending
   ▼
Ranked list + visual grid (query photo next to top-5 matches)
```

---

## Step 1: Floor Segmentation

Product images need no segmentation — they're already floor-only. Query images do.

**Two-tier design:**

1. **Primary: SegFormer** (`nvidia/segformer-b0-finetuned-ade-512-512`), a transformer-based semantic segmentation model pretrained on **ADE20K**, a 150-class scene-parsing dataset with an explicit `"floor"` class (id=3, verified directly against `model.config.id2label`). This model was actually trained to understand *scenes*, so it correctly separates floor from furniture, rugs, and walls regardless of camera angle — verified visually on all 5 query images (see `outputs/viz/overlay_query*.jpg`).

2. **Fallback: OpenCV GrabCut**, used only if SegFormer's weights can't be downloaded/loaded, or if it returns an implausible (near-empty) mask. Seeded with a **graded confidence mask** (bottom of the image = likely floor, top = likely background) rather than a plain rectangle — an earlier version seeded with a rectangle and it over-included furniture sitting inside that rectangle (e.g. a couch got pulled into the "floor" mask). The graded mask fixed this by giving GrabCut a much more accurate prior.

3. **Last resort: bottom-fraction crop** — a pure geometric heuristic (bottom 35% of the image), guaranteeing the pipeline never crashes even if both learned/classical segmentation fail.

In this run, **SegFormer succeeded on all 5 query images** — no fallback was triggered. Segmentation quality was manually verified by overlaying the predicted mask in green on the original photo (`outputs/viz/overlay_query*.jpg`) and confirming it hugs the floor boundary without bleeding onto furniture.

---

## Step 2: Feature Representation (Turning Images Into Numbers)

Two complementary signals are computed for every image (20 products + each query's floor patch):

### CLIP embedding (`openai/clip-vit-base-patch32`)
CLIP was trained on hundreds of millions of image-text pairs, so it has a strong general sense of visual "meaning" — pattern, grain style, glossiness, plank layout — without ever being fine-tuned on flooring specifically. This matters because we only have 20 products and 5 queries — nowhere near enough to train anything from scratch, so an off-the-shelf, well-generalizing model is the right tool. It's also reasonably tolerant of imperfect segmentation (a bit of shadow or a stray furniture edge in the crop), since it was trained on noisy real-world images, not clean isolated crops.

### HSV color histogram (Hue + Saturation, 50×60 bins)
CLIP's embedding is a single global vector optimized for broad semantic similarity, and can sometimes weight pattern/shape over exact color tone. For flooring, color/shade (light oak vs. dark walnut vs. gray-washed) is often the most immediately obvious thing a person judges similarity by. So we add a classical HSV histogram as a second, independent, fully interpretable signal.

**Why HSV and not RGB:** HSV separates brightness (Value) from color identity (Hue). This makes hue comparisons more robust to lighting differences between the studio-lit product photos and the naturally-lit room photos — a wood floor under warm ambient light and the same floor under neutral studio light will have similar Hue even though their raw RGB values differ substantially. We deliberately **exclude the Value channel** from the histogram for this reason.

For query images, the histogram is computed **only over floor pixels** (using the segmentation mask), so furniture/wall pixels never contaminate the color signal.

---

## Step 3: Similarity Computation

```
final_score = 0.7 × cosine_similarity(CLIP_query, CLIP_product)
            + 0.3 × histogram_correlation(HSV_query, HSV_product)
```

- CLIP similarity via cosine similarity (standard for embedding comparison).
- Color similarity via OpenCV's histogram correlation (`cv2.HISTCMP_CORREL`), rescaled from `[-1, 1]` to `[0, 1]` so both signals combine on the same scale.
- **The 70/30 weighting is a heuristic default**, not tuned on labeled data (none was available). CLIP is weighted higher because it captures pattern/texture/material jointly and was empirically the stronger, more consistent signal; the color histogram is used to catch cases where CLIP treats two differently-colored-but-similarly-patterned floors as more similar than they should be.

**Validated impact of segmentation:** on a sanity-check pair (product 1 vs. query 1), CLIP cosine similarity rose from **0.829 (full room image)** to **0.922 (segmented floor-only patch)** — confirming that whole-image comparison was diluting the true floor-similarity signal with irrelevant room content (walls, sofa, fireplace).

---

## Step 4: Ranking & Output

For each of the 5 queries, **all 20 products are ranked** by `final_score`, descending (not just the top 5 — the full ranking of all 20 is computed and saved for every query). Output is saved as:
- `outputs/results/query_{n}_ranking.csv` — full ranked list of all 20 products per query
- `outputs/results/all_results.json` — everything, including per-signal scores (CLIP score, color score, final score) for every product/query pair
- `outputs/viz/result_grid_query{n}.jpg` — visual grid: query photo next to its top-5 product matches with scores labeled, for quick human review

---

## Results Summary

| Query | Top match | Score | Visual read |
|---|---|---|---|
| 1 | Product 9 | 0.944 | Strong — closely matches medium-brown wide-plank floor |
| 2 | Product 7 | 0.877 | Reasonable, but #4 (product 2) is a visibly weaker match |
| 3 | Product 2 | 0.897 | Weakest of the five — top match doesn't visually align well with the light, rustic query floor |
| 4 | Product 16 | 0.915 | Strong — close light-oak match |
| 5 | Product 11 | 0.893 | Top match visually diverges from the smooth, warm-toned query floor |

Full per-query rankings (all 20 products, not just top 5) are in `outputs/results/`.

---


## How to Run

```bash
# 1. Create environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow opencv-python-headless scikit-image scikit-learn numpy

# 3. Folder structure expected
#    data/sku/1.jpg ... 20.jpg      (product images)
#    data/query/1.jpg ... 5.jpg     (query images)

# 4. Run the full pipeline
python pipeline.py

# 5. Generate visual result grids
python visualize_results.py

# Optional: verify floor segmentation visually
python test_segment.py
```

Outputs are written to `outputs/results/` (CSV + JSON rankings, all 20 products per query) and `outputs/viz/` (segmentation overlays + result grids).

---

## Code Structure

```
segment.py                    # floor segmentation (SegFormer + GrabCut + crop fallback)
clip_features.py              # CLIP embedding extraction (product + query)
color_features.py             # HSV histogram extraction + comparison
pipeline.py                   # full ranking pipeline, all 20 products x all 5 queries
visualize_results.py          # builds query-vs-top5 visual grids
test_segment.py                # segmentation sanity check / visualization
```

---

## Trade-offs Considered

| Decision | Chosen approach | Alternative considered | Why |
|---|---|---|---|
| Segmentation | SegFormer (learned) | Fixed bottom-crop heuristic | Learned model adapts to camera angle/furniture placement; heuristic is faster but fails on atypical framings |
| Embedding | CLIP (pretrained, zero-shot) | Fine-tuned flooring-specific model | No labeled training data available; CLIP generalizes well without fine-tuning |
| Similarity signal | Hybrid (CLIP + color histogram) | CLIP alone | Color histogram catches cases where CLIP over-weights pattern over color |
| Weighting | Fixed heuristic (0.7/0.3) | Learned weighting | No labeled validation data to learn optimal weights from |
