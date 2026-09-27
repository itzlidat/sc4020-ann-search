"""Exact linear-scan nearest-neighbour search.

Set OPTIMISED below to switch between two exact implementations:

    True  -> precomputed norms + vectorised distance + argpartition
    False -> direct squared differences + complete argsort
"""

import gc
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# Repository setup
# --------------------------------------------------------------------------- #

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402
from eval.harness import (  # noqa: E402
    compute_recall_at_k,
    load_ground_truth,
    measure_query_time,
)

# --------------------------------------------------------------------------- #
# Dataset paths
# --------------------------------------------------------------------------- #

SIFT_DIR = os.path.join(PROJECT_ROOT, "data", "sift")

SIFT_BASE = os.path.join(SIFT_DIR, "sift_base.fvecs")
SIFT_QUERY = os.path.join(SIFT_DIR, "sift_query.fvecs")
SIFT_GT = os.path.join(SIFT_DIR, "sift_groundtruth.ivecs")

WIKI_BASE = os.path.join(
    PROJECT_ROOT,
    "data",
    "wiki_base_embeddings.npy",
)
WIKI_QUERY = os.path.join(
    PROJECT_ROOT,
    "data",
    "wiki_query_embeddings.npy",
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# ONE-LINE SWITCH:
# True  = optimised exact linear scan
# False = non-optimised exact linear scan
OPTIMISED = True

K_VALUES = [1, 10, 100]
MAX_QUERIES = None  # Set to e.g. 100 for a quick test.

RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "linear_scan_results.json",
)


class LinearScan:
    """Exact nearest-neighbour search with an optimisation switch."""

    def __init__(self, optimised: bool = True) -> None:
        self.optimised = optimised
        self.vectors: np.ndarray | None = None
        self.vector_norms: np.ndarray | None = None

    def build_index(self, vectors: np.ndarray) -> None:
        """Store vectors and precompute norms for the optimised version."""

        self.vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

        if self.optimised:
            self.vector_norms = np.sum(
                self.vectors * self.vectors,
                axis=1,
                dtype=np.float32,
            )

    def _search_optimised(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> List[int]:
        """Exact search using norms and partial sorting."""

        assert self.vectors is not None
        assert self.vector_norms is not None

        q = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        q_norm = np.dot(q, q)

        # ||q-x||^2 = ||q||^2 + ||x||^2 - 2q.x
        distances = np.maximum(
            q_norm
            + self.vector_norms
            - 2.0 * np.dot(self.vectors, q),
            0.0,
        )

        k = min(k, len(distances))

        candidate_ids = np.argpartition(
            distances,
            k - 1,
        )[:k]

        candidate_ids = candidate_ids[
            np.argsort(distances[candidate_ids])
        ]

        return candidate_ids.tolist()

    def _search_non_optimised(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> List[int]:
        """Exact search using direct distances and complete sorting."""

        assert self.vectors is not None

        q = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        distances = np.sum(
            (self.vectors - q) * (self.vectors - q),
            axis=1,
            dtype=np.float32,
        )

        # Deliberately sort all N distances for an implementation-level
        # comparison with the optimised version.
        ids = np.argsort(distances)[
            : min(k, len(distances))
        ]

        return ids.tolist()

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[int]:
        """Return exact nearest-neighbour ids, nearest first."""

        if self.vectors is None:
            raise RuntimeError("Index has not been built.")

        if self.optimised:
            return self._search_optimised(
                query_vector,
                k,
            )

        return self._search_non_optimised(
            query_vector,
            k,
        )


# --------------------------------------------------------------------------- #
# Benchmark helpers
# --------------------------------------------------------------------------- #

def _limit_queries(
    queries: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if MAX_QUERIES is None:
        return queries, ground_truth

    n = min(MAX_QUERIES, len(queries))

    if ground_truth is None:
        return queries[:n], None

    return queries[:n], ground_truth[:n]


def _print_row(
    dataset: str,
    metrics: Dict[str, float | bool],
) -> None:
    print(
        f"  {dataset:<10} | "
        f"mode={'optimised' if metrics['optimised'] else 'non-optimised':<13} | "
        f"R@1={metrics['recall_at_1']:.4f} | "
        f"R@10={metrics['recall_at_10']:.4f} | "
        f"R@100={metrics['recall_at_100']:.4f} | "
        f"avg_query_time={metrics['avg_query_time_ms']:.3f} ms | "
        f"QPS={metrics['qps']:.2f}"
    )


def benchmark_dataset(
    dataset_name: str,
    base: np.ndarray,
    queries: np.ndarray,
    ground_truth_ids: np.ndarray,
) -> Dict[str, float | bool]:
    """Build the exact scanner and measure latency and recall."""

    mode = "optimised" if OPTIMISED else "non-optimised"

    print(
        f"\n=== {dataset_name} "
        f"(linear scan, {mode}) ==="
    )

    model = LinearScan(
        optimised=OPTIMISED,
    )

    build_start = time.perf_counter()

    model.build_index(base)

    build_time_sec = (
        time.perf_counter()
        - build_start
    )

    retrieved_ids: List[List[int]] = []
    max_k = max(K_VALUES)

    def timed_search(
        query_vector: np.ndarray,
    ) -> List[int]:
        ids = model.search(
            query_vector,
            k=max_k,
        )
        retrieved_ids.append(ids)
        return ids

    _, avg_query_time_sec = measure_query_time(
        timed_search,
        queries,
    )

    recalls = {
        k: compute_recall_at_k(
            retrieved_ids,
            ground_truth_ids,
            k=k,
        )
        for k in K_VALUES
    }

    metrics: Dict[str, float | bool] = {
        "optimised": OPTIMISED,
        "build_time_sec": float(build_time_sec),
        "avg_query_time_ms": float(
            avg_query_time_sec * 1000
        ),
        "qps": float(
            1.0 / avg_query_time_sec
        ),
        "recall_at_1": float(recalls[1]),
        "recall_at_10": float(recalls[10]),
        "recall_at_100": float(recalls[100]),
    }

    _print_row(
        dataset_name,
        metrics,
    )

    del model, retrieved_ids
    gc.collect()

    return metrics


# --------------------------------------------------------------------------- #
# Dataset runners -- same datasets as Annoy
# --------------------------------------------------------------------------- #

def run_sift() -> Dict[str, float | bool]:
    """Benchmark exact linear search on the repository's SIFT1M dataset."""

    base = load_fvecs(SIFT_BASE)
    queries = load_fvecs(SIFT_QUERY)
    ground_truth = load_ivecs(SIFT_GT)

    queries, ground_truth = _limit_queries(
        queries,
        ground_truth,
    )

    if base.shape[1] != queries.shape[1]:
        raise ValueError(
            "SIFT base/query dimensions do not match."
        )

    if len(queries) != len(ground_truth):
        raise ValueError(
            "SIFT query/ground-truth counts do not match."
        )

    print(
        f"SIFT1M: base={base.shape}, "
        f"queries={queries.shape}, "
        f"ground_truth={ground_truth.shape}"
    )

    return benchmark_dataset(
        "sift1m",
        base,
        queries,
        ground_truth,
    )


def run_wikipedia() -> Dict[str, float | bool]:
    """Benchmark exact search on the repository's Wikipedia embeddings.

    This follows Annoy exactly at the dataset level:
    wiki_base_embeddings.npy and wiki_query_embeddings.npy are loaded,
    then both are L2-normalised for cosine/angular nearest-neighbour search.
    Ground truth is generated exactly because no Wikipedia GT file is
    supplied in the repository.
    """

    base = np.load(WIKI_BASE)
    queries = np.load(WIKI_QUERY)

    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError(
            "Wikipedia embeddings must be 2-D arrays."
        )

    if base.shape[1] != queries.shape[1]:
        raise ValueError(
            "Wikipedia base/query dimensions do not match."
        )

    base_norms = np.linalg.norm(
        base,
        axis=1,
        keepdims=True,
    )
    query_norms = np.linalg.norm(
        queries,
        axis=1,
        keepdims=True,
    )

    if np.any(base_norms == 0) or np.any(query_norms == 0):
        raise ValueError(
            "Wikipedia embeddings contain zero vectors and cannot be "
            "L2-normalised."
        )

    # Same normalisation used by Annoy's Wikipedia benchmark.
    base = base / base_norms
    queries = queries / query_norms

    queries, _ = _limit_queries(queries)

    ground_truth = load_ground_truth(
        queries,
        base,
        k=max(K_VALUES),
    )

    print(
        f"Wikipedia: base={base.shape}, "
        f"queries={queries.shape}, "
        f"ground_truth={ground_truth.shape}"
    )

    return benchmark_dataset(
        "wikipedia",
        base,
        queries,
        ground_truth,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    mode = "optimised" if OPTIMISED else "non-optimised"

    print("=" * 80)
    print(f"EXACT LINEAR SCAN | mode={mode}")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"SIFT base:    {SIFT_BASE}")
    print(f"SIFT query:   {SIFT_QUERY}")
    print(f"SIFT GT:      {SIFT_GT}")
    print(f"Wiki base:    {WIKI_BASE}")
    print(f"Wiki query:   {WIKI_QUERY}")

    all_results = {
        "sift1m": run_sift(),
        "wikipedia": run_wikipedia(),
    }

    os.makedirs(
        os.path.dirname(RESULTS_FILE),
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(f"\nSaved results to {RESULTS_FILE}")

    print("\n=== Final summary ===")

    for dataset_name, metrics in all_results.items():
        _print_row(
            dataset_name,
            metrics,
        )


if __name__ == "__main__":
    main()
