"""
color_features.py
HSV color histogram - the classical, no-download-required half of
our hybrid similarity signal. Complements CLIP by giving an explicit,
interpretable measure of color palette (light oak vs dark walnut vs
gray-washed, etc.), robust to lighting differences because Hue is
mostly separated from brightness in HSV space.

For query images, the histogram is computed ONLY over floor pixels
(using the mask from segment.extract_floor) - the cropped patch has
non-floor pixels zeroed out, and we explicitly exclude those zeroed
pixels from the histogram via OpenCV's mask parameter, so they don't
skew the color distribution toward black.
"""

import cv2
import numpy as np

from segment import extract_floor

# Bins: 50 for Hue (0-180 in OpenCV), 60 for Saturation (0-256).
# We deliberately leave Value (brightness) out of the histogram - it's
# the channel most affected by lighting differences between studio
# product shots and ambient-lit room photos, and Hue+Saturation alone
# already captures the color palette we care about.
_H_BINS = 50
_S_BINS = 60
_H_RANGE = [0, 180]
_S_RANGE = [0, 256]


def get_color_histogram(image_bgr: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Returns a normalized, flattened Hue-Saturation histogram."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    cv_mask = None
    if mask is not None:
        cv_mask = (mask * 255).astype(np.uint8)

    hist = cv2.calcHist(
        [hsv], [0, 1], cv_mask,
        [_H_BINS, _S_BINS], _H_RANGE + _S_RANGE
    )
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Correlation-based similarity, rescaled from [-1, 1] to [0, 1]
    so it combines cleanly with cosine similarity from CLIP later."""
    correlation = cv2.compareHist(
        hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL
    )
    return (correlation + 1) / 2


def get_product_histogram(image_path: str) -> np.ndarray:
    """Product images are already floor-only - use the full image, no mask."""
    image_bgr = cv2.imread(image_path)
    return get_color_histogram(image_bgr, mask=None)


def get_query_histogram(image_path: str) -> tuple[np.ndarray, str]:
    """Query images need floor extraction first, and we mask out
    non-floor (zeroed) pixels so they don't skew the color histogram."""
    image_bgr = cv2.imread(image_path)
    floor_patch, floor_mask, method = extract_floor(image_bgr)
    hist = get_color_histogram(floor_patch, mask=floor_mask)
    return hist, method


if __name__ == "__main__":
    print(" Sanity test: product 1 vs query 1 (floor-only, color only) ")
    product_hist = get_product_histogram("data/sku/1.jpg")
    query_hist, method = get_query_histogram("data/query/1.jpg")

    print("Product histogram shape:", product_hist.shape)
    print("Query histogram shape:", query_hist.shape)
    print("Segmentation method used for query:", method)

    similarity = compare_histograms(product_hist, query_hist)
    print("Color similarity (product 1 vs query 1, floor-only):", similarity)
