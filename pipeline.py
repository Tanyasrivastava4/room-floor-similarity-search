"""
pipeline.py
Full ranking pipeline: for each of the 5 query images, ranks all 20
product images by combined floor similarity.

final_score = 0.7 * CLIP_similarity + 0.3 * color_histogram_similarity

Weights are a heuristic default (documented in README) - CLIP given
more weight since it captures pattern/material/texture jointly and is
generally the stronger signal; color histogram fine-tunes/corrects
based on literal color palette.
"""

import os
import json
import torch
import torch.nn.functional as F

from clip_features import get_product_embedding, get_query_embedding
from color_features import get_product_histogram, get_query_histogram, compare_histograms

CLIP_WEIGHT = 0.7
COLOR_WEIGHT = 0.3

PRODUCT_DIR = "data/sku"
QUERY_DIR = "data/query"
NUM_PRODUCTS = 20
NUM_QUERIES = 5

OUTPUT_DIR = "outputs/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_product_index():
    """Precompute CLIP embeddings + color histograms for all 20
    products ONCE. This is the scalability-conscious design point:
    in a real catalog, product features are computed once and cached,
    not recomputed per query."""
    print("Building product feature index (20 products)...")
    index = {}
    for i in range(1, NUM_PRODUCTS + 1):
        path = os.path.join(PRODUCT_DIR, f"{i}.jpg")
        emb = get_product_embedding(path)
        hist = get_product_histogram(path)
        index[i] = {"embedding": emb, "histogram": hist}
        print(f"  product {i}/{NUM_PRODUCTS} indexed")
    return index


def rank_products_for_query(query_path: str, product_index: dict):
    query_emb, seg_method = get_query_embedding(query_path)
    query_hist, _ = get_query_histogram(query_path)

    scores = []
    for product_id, feats in product_index.items():
        clip_sim = F.cosine_similarity(
            query_emb.unsqueeze(0), feats["embedding"].unsqueeze(0)
        ).item()
        color_sim = compare_histograms(query_hist, feats["histogram"])

        final_score = CLIP_WEIGHT * clip_sim + COLOR_WEIGHT * color_sim

        scores.append({
            "product_id": product_id,
            "clip_similarity": round(clip_sim, 4),
            "color_similarity": round(color_sim, 4),
            "final_score": round(final_score, 4),
        })

    scores.sort(key=lambda x: x["final_score"], reverse=True)
    for rank, entry in enumerate(scores, start=1):
        entry["rank"] = rank

    return scores, seg_method


def main():
    product_index = build_product_index()

    all_results = {}
    for q in range(1, NUM_QUERIES + 1):
        query_path = os.path.join(QUERY_DIR, f"{q}.jpg")
        print(f"\nRanking products for query {q}...")
        ranked, seg_method = rank_products_for_query(query_path, product_index)

        all_results[f"query_{q}"] = {
            "segmentation_method": seg_method,
            "ranking": ranked,
        }

        print(f"  Top 5 for query {q} (segmented via {seg_method}):")
        for entry in ranked[:5]:
            print(f"    #{entry['rank']}: product {entry['product_id']} "
                  f"(final={entry['final_score']}, clip={entry['clip_similarity']}, "
                  f"color={entry['color_similarity']})")

        # per-query CSV
        csv_path = os.path.join(OUTPUT_DIR, f"query_{q}_ranking.csv")
        with open(csv_path, "w") as f:
            f.write("rank,product_id,final_score,clip_similarity,color_similarity\n")
            for entry in ranked:
                f.write(f"{entry['rank']},{entry['product_id']},"
                        f"{entry['final_score']},{entry['clip_similarity']},"
                        f"{entry['color_similarity']}\n")

    # full JSON with everything
    json_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDone. Per-query CSVs and all_results.json saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
