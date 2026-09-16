"""Annoy (Approximate Nearest Neighbors Oh Yeah) random projection trees.

Builds a forest of binary trees, each recursively splitting the space
with random hyperplanes. At search time, multiple trees are traversed and
their candidate leaves merged, giving approximate nearest neighbours with
a small, disk-friendly index.

Running this file as a script benchmarks Annoy on SIFT1M and Wikipedia
sentence embeddings across several search_k values (index built once per
dataset at a fixed n_trees), using the shared eval/harness.py utilities.
Groupmates implementing LSH/PQ/IVF/HNSW should follow the same
load -> build -> search -> measure -> record pattern.

Note on search_k: Annoy's get_nns_by_vector defaults search_k to
n_trees * k, which is far too small a search budget on large datasets
(e.g. n_trees=50, k=10 -> search_k=500 against 1M vectors) and starves
recall artificially. search_k is swept explicitly here instead.
"""

import gc
import json
import os
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np
from annoy import AnnoyIndex

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402
from eval.harness import (  # noqa: E402
    compute_recall_at_k,
    load_ground_truth,
    measure_index_size,
    measure_query_time,
)


class Annoy:
    """Random-projection-tree approximate nearest neighbour index."""

    def __init__(self, num_trees: int = 10, metric: str = "angular") -> None:
        self.num_trees = num_trees
        self.metric = metric
        self.index: AnnoyIndex | None = None
        self.dim: int | None = None

    def build_index(self, vectors: np.ndarray) -> None:
        """Build the forest of random projection trees over the base vectors.

        Args:
            vectors: Array of shape (n, d) containing the base embeddings.
        """
        self.dim = vectors.shape[1]
        self.index = AnnoyIndex(self.dim, self.metric)
        for i, vec in enumerate(vectors):
            self.index.add_item(i, vec)
        self.index.build(self.num_trees)

    def search(self, query_vector: np.ndarray, k: int = 10, search_k: int = -1) -> List[int]:
        """Return the ids of approximate k nearest neighbours to query_vector.

        Args:
            query_vector: Array of shape (d,) representing the query.
            k: Number of neighbours to return.
            search_k: Number of nodes to inspect during search. Larger
                values trade query time for recall. -1 uses Annoy's
                default (n_trees * k), which is often too small.

        Returns:
            List of the k nearest neighbour ids, ordered nearest first.
        """
        return self.index.get_nns_by_vector(query_vector, k, search_k)

    def save(self, path: str) -> None:
        """Persist the built index to disk (used for index-size measurement)."""
        self.index.save(path)


# --------------------------------------------------------------------------- #
# Benchmark script
# --------------------------------------------------------------------------- #

N_TREES = 50
SEARCH_K_SWEEP = [1000, 5000, 20000, 100000]
K = 10


def _print_row(dataset: str, search_k: int, metrics: Dict[str, float]) -> None:
    print(
        f"  {dataset:<10} | search_k={search_k:<7} | "
        f"recall@{K}={metrics['recall_at_10']:.4f} | "
        f"avg_query_time={metrics['avg_query_time_ms']:.3f} ms | "
        f"build_time={metrics['build_time_sec']:.2f} s | "
        f"index_size={metrics['index_size_mb']:.2f} MB"
    )


def benchmark_dataset(
    dataset_name: str,
    base: np.ndarray,
    queries: np.ndarray,
    ground_truth_ids: np.ndarray,
    metric: str,
) -> Dict[str, Dict[str, float]]:
    """Build one Annoy index (n_trees=N_TREES) and sweep search_k at query time.

    n_trees mainly governs index build quality/size, not query-time recall,
    so it's held fixed here; search_k is the actual recall/latency knob.

    Args:
        dataset_name: Label used in printed output and the results dict.
        base: Array of shape (n_base, d) of base vectors to index.
        queries: Array of shape (n_queries, d) of query vectors.
        ground_truth_ids: Array of shape (n_queries, K) of true neighbour ids.
        metric: Annoy distance metric ("euclidean" or "angular").

    Returns:
        Dict mapping str(search_k) -> metric dict for this dataset.
    """
    print(f"\n=== {dataset_name} (metric={metric}, n_trees={N_TREES}) ===")
    results: Dict[str, Dict[str, float]] = {}

    model = Annoy(num_trees=N_TREES, metric=metric)

    start = time.perf_counter()
    model.build_index(base)
    build_time_sec = time.perf_counter() - start

    with tempfile.NamedTemporaryFile(suffix=".ann", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        index_size_mb = measure_index_size(tmp_path)
    finally:
        os.remove(tmp_path)

    for search_k in SEARCH_K_SWEEP:
        # Collect retrieved ids as a side effect of the harness's timed search
        # loop, so we run each query only once but still get both timing and
        # recall from the shared eval harness.
        retrieved_ids: List[List[int]] = []

        def timed_search(query_vector: np.ndarray, _model=model, _search_k=search_k) -> List[int]:
            ids = _model.search(query_vector, k=K, search_k=_search_k)
            retrieved_ids.append(ids)
            return ids

        _, avg_query_time_sec = measure_query_time(timed_search, queries)
        recall = compute_recall_at_k(retrieved_ids, ground_truth_ids, k=K)

        metrics = {
            "recall_at_10": recall,
            "avg_query_time_ms": avg_query_time_sec * 1000,
            "build_time_sec": build_time_sec,
            "index_size_mb": index_size_mb,
        }
        results[str(search_k)] = metrics
        _print_row(dataset_name, search_k, metrics)

    del model
    gc.collect()

    return results


def main() -> None:
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # --- SIFT1M (ground truth shipped with the dataset) ---
    sift_dir = os.path.join(PROJECT_ROOT, "data", "sift")
    sift_base = load_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    sift_query = load_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    sift_gt = load_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))[:, :K]

    all_results["sift1m"] = benchmark_dataset(
        "sift1m", sift_base, sift_query, sift_gt, metric="euclidean"
    )

    del sift_base, sift_query, sift_gt
    gc.collect()

    # --- Wikipedia (no precomputed ground truth: compute via brute-force
    # cosine similarity, i.e. Euclidean distance over L2-normalized vectors) ---
    wiki_base = np.load(os.path.join(PROJECT_ROOT, "data", "wiki_base_embeddings.npy"))
    wiki_query = np.load(os.path.join(PROJECT_ROOT, "data", "wiki_query_embeddings.npy"))

    wiki_base_norm = wiki_base / np.linalg.norm(wiki_base, axis=1, keepdims=True)
    wiki_query_norm = wiki_query / np.linalg.norm(wiki_query, axis=1, keepdims=True)
    wiki_gt = load_ground_truth(wiki_query_norm, wiki_base_norm, k=K)
    del wiki_base_norm, wiki_query_norm
    gc.collect()

    all_results["wikipedia"] = benchmark_dataset(
        "wikipedia", wiki_base, wiki_query, wiki_gt, metric="angular"
    )

    del wiki_base, wiki_query, wiki_gt
    gc.collect()

    # --- Save + final summary ---
    results_path = os.path.join(PROJECT_ROOT, "results", "annoy_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\n=== Final summary ===")
    for dataset_name, per_search_k in all_results.items():
        for search_k_str, metrics in per_search_k.items():
            _print_row(dataset_name, int(search_k_str), metrics)


if __name__ == "__main__":
    main()
