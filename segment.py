"""
segment.py
----------
Floor extraction for query (room) images.

Product images are already clean floor-only shots, so they need no
segmentation. Query images are full rooms (furniture, walls, shadows),
so we must isolate the floor region before computing similarity -
otherwise we'd be comparing "whole room" to "clean product photo",
which is the wrong comparison.

TWO-TIER DESIGN
----------------
1. PRIMARY: SegFormer, a transformer-based semantic segmentation
   model pretrained on ADE20K (a 150-class scene-parsing dataset that
   includes an explicit "floor" class, id=3). Robust to camera angle,
   furniture placement, and partial occlusion because it was actually
   trained to understand scenes, not just seeded by a geometric guess.

2. FALLBACK: OpenCV GrabCut, seeded with a graded confidence mask
   (bottom of image = likely floor, top = likely background). Used
   only if SegFormer's weights can't be downloaded/loaded, or if
   SegFormer produces an implausible (near-empty) mask.

3. LAST RESORT: a plain bottom-fraction crop, so the pipeline never
   breaks even if both of the above fail.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------
# Tier 1: SegFormer (ADE20K) - loaded lazily so importing this module
# doesn't require internet access if it's never actually called.
# ---------------------------------------------------------------
_segformer_processor = None
_segformer_model = None
_SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
_ADE20K_FLOOR_CLASS_ID = 3  # confirmed via model.config.id2label[3] == "floor"


def _load_segformer():
    global _segformer_processor, _segformer_model
    if _segformer_model is None:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        _segformer_processor = SegformerImageProcessor.from_pretrained(_SEGFORMER_MODEL_NAME)
        _segformer_model = SegformerForSemanticSegmentation.from_pretrained(_SEGFORMER_MODEL_NAME)
        _segformer_model.eval()
    return _segformer_processor, _segformer_model


def segformer_floor_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Run SegFormer and return a binary floor mask (uint8, 0/1)."""
    import torch
    from PIL import Image

    processor, model = _load_segformer()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, 150, h', w') - low resolution

    h, w = image_bgr.shape[:2]
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    pred_mask = upsampled.argmax(dim=1)[0]  # (h, w) class id per pixel

    floor_mask = (pred_mask == _ADE20K_FLOOR_CLASS_ID).numpy().astype(np.uint8)
    return floor_mask


# ---------------------------------------------------------------
# Tier 2: GrabCut fallback (classical CV, no downloads required)
# ---------------------------------------------------------------
def _largest_component_mask(mask_bin: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask_bin
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


def grab_cut_floor_mask(image_bgr: np.ndarray, iterations: int = 5) -> np.ndarray:
    h, w = image_bgr.shape[:2]

    seed = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    seed[int(h * 0.80):, :] = cv2.GC_PR_FGD
    seed[int(h * 0.88):, :] = cv2.GC_FGD
    seed[:int(h * 0.20), :] = cv2.GC_BGD

    mask = seed.copy()
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image_bgr, mask, None, bgd_model, fgd_model,
                iterations, cv2.GC_INIT_WITH_MASK)

    floor_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    floor_mask = _largest_component_mask(floor_mask)

    kernel = np.ones((9, 9), np.uint8)
    floor_mask = cv2.erode(floor_mask, kernel, iterations=1)

    return floor_mask


def bottom_crop_mask(image_bgr: np.ndarray, fraction: float = 0.35) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    start_y = int(h * (1 - fraction))
    mask[start_y:, :] = 1
    return mask


def is_mask_plausible(mask: np.ndarray, min_fraction: float = 0.08) -> bool:
    return (mask.sum() / mask.size) >= min_fraction


# ---------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------
def get_floor_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Returns (raw_mask, method_used) - mask is full image size."""
    try:
        mask = segformer_floor_mask(image_bgr)
        if not is_mask_plausible(mask):
            raise ValueError("SegFormer mask too small/empty")
        return mask, "segformer"
    except Exception as e:
        print(f"  [segment] SegFormer unavailable/failed ({e}); trying GrabCut fallback")

    try:
        mask = grab_cut_floor_mask(image_bgr)
        if not is_mask_plausible(mask):
            raise ValueError("GrabCut mask too small/empty")
        return mask, "grabcut_fallback"
    except Exception as e2:
        print(f"  [segment] GrabCut also failed ({e2}); using bottom-crop fallback")

    mask = bottom_crop_mask(image_bgr)
    return mask, "bottom_crop_fallback"


def extract_floor(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Returns (floor_patch_bgr, cropped_mask, method_used).

    floor_patch_bgr is the original image with non-floor pixels
    zeroed out, then cropped to the mask's bounding box.
    """
    mask, method = get_floor_mask(image_bgr)

    masked = image_bgr.copy()
    masked[mask == 0] = 0

    ys, xs = np.where(mask == 1)
    if len(ys) == 0:
        return image_bgr, np.ones(image_bgr.shape[:2], np.uint8), "no_mask_full_image"

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    cropped = masked[y0:y1 + 1, x0:x1 + 1]
    cropped_mask = mask[y0:y1 + 1, x0:x1 + 1]

    return cropped, cropped_mask, method