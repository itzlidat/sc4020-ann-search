"""Measure Annoy at its library-default search_k on both datasets.

Passing search_k=-1 makes Annoy inspect n_trees * k nodes (50 * 10 = 500
here), which is what an untuned deployment gets. Reuses the main Annoy
benchmark (same n_trees, warm-up pass and harness) with the sweep replaced
by the default, and writes a separate results file so annoy_results.json is
untouched. Each run builds a fresh index, and Annoy's trees are random, so
recall can differ slightly from the main sweep's build.
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import methods.annoy_method as annoy_method  # noqa: E402
from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402
from eval.harness import load_ground_truth  # noqa: E402

DEFAULT_SEARCH_K = -1


def main() -> None:
    annoy_method.SEARCH_K_SWEEP = [DEFAULT_SEARCH_K]
    K = annoy_method.K
    effective = annoy_method.N_TREES * K
    results = {}

    sift_dir = os.path.join(PROJECT_ROOT, "data", "sift")
    sift_base = load_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    sift_query = load_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    sift_gt = load_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))[:, :K]
    results["sift1m"] = annoy_method.benchmark_dataset(
        "sift1m", sift_base, sift_query, sift_gt, metric="euclidean"
    )
    del sift_base, sift_query, sift_gt

    wiki_dir = os.path.join(PROJECT_ROOT, "data", "wiki")
    wiki_base = np.load(os.path.join(wiki_dir, "wiki_base_embeddings.npy"))
    wiki_query = np.load(os.path.join(wiki_dir, "wiki_query_embeddings.npy"))
    wiki_gt = load_ground_truth(
        wiki_query / np.linalg.norm(wiki_query, axis=1, keepdims=True),
        wiki_base / np.linalg.norm(wiki_base, axis=1, keepdims=True),
        k=K,
    )
    results["wikipedia"] = annoy_method.benchmark_dataset(
        "wikipedia", wiki_base, wiki_query, wiki_gt, metric="angular"
    )

    for per_search_k in results.values():
        for metrics in per_search_k.values():
            metrics["search_k_effective"] = effective

    path = os.path.join(PROJECT_ROOT, "results", "annoy_default_search_k.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    main()
