"""
clip_features.py
Extracts CLIP image embeddings for:
  - product images (used as-is, they're already floor-only)
  - query images (first passed through segment.extract_floor() to
    isolate the floor region, THEN embedded)

This ensures CLIP is only ever looking at floor pixels, never at
furniture/walls, which is the whole point of the segmentation step.
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from segment import extract_floor

_MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model + processor...")
_model = CLIPModel.from_pretrained(_MODEL_NAME)
_processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
_model.eval()
print("CLIP ready.")


def _bgr_to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def get_clip_embedding_from_array(image_bgr: np.ndarray) -> torch.Tensor:
    """Returns a 512-dim CLIP embedding for an in-memory BGR image (numpy array)."""
    pil_image = _bgr_to_pil(image_bgr)
    inputs = _processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = _model.get_image_features(**inputs)
        embedding = outputs.pooler_output
    return embedding.squeeze(0)  # shape (512,)


def get_product_embedding(image_path: str) -> torch.Tensor:
    """Product images are already floor-only - embed directly, no segmentation."""
    image_bgr = cv2.imread(image_path)
    return get_clip_embedding_from_array(image_bgr)


def get_query_embedding(image_path: str) -> tuple[torch.Tensor, str]:
    """Query images need floor extraction first. Returns (embedding, method_used)."""
    image_bgr = cv2.imread(image_path)
    floor_patch, _, method = extract_floor(image_bgr)
    embedding = get_clip_embedding_from_array(floor_patch)
    return embedding, method


if __name__ == "__main__":
    import torch.nn.functional as F

    print("\n--- Sanity test: product 1 vs query 1 (floor-only now) ---")
    product_emb = get_product_embedding("data/sku/1.jpg")
    query_emb, method = get_query_embedding("data/query/1.jpg")

    print("Product embedding shape:", product_emb.shape)
    print("Query embedding shape:", query_emb.shape)
    print("Segmentation method used for query:", method)

    similarity = F.cosine_similarity(product_emb.unsqueeze(0), query_emb.unsqueeze(0))
    print("Cosine similarity (product 1 vs query 1, floor-only):", similarity.item())





