"""Exact linear-scan nearest-neighbour search.

"""

import gc
import json
import os
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np

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

# ONE-LINE SWITCH: True = optimised, False = non-optimised
OPTIMISED = True
K = 10
MAX_QUERIES = None


class LinearScan:
    """Exact brute-force nearest-neighbour index."""

    def __init__(self, optimised: bool = True) -> None:
        self.optimised = optimised
        self.vectors: np.ndarray | None = None
        self.vector_norms: np.ndarray | None = None

    def build_index(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.optimised:
            self.vector_norms = np.sum(
                self.vectors * self.vectors, axis=1, dtype=np.float32
            )

    def search(self, q: np.ndarray, k: int = 10) -> List[int]:
        if self.vectors is None:
            raise RuntimeError("Index has not been built.")
        q = np.asarray(q, dtype=np.float32)
        k = min(k, len(self.vectors))

        if self.optimised:
            if self.vector_norms is None:
                raise RuntimeError("Database norms are missing.")
            q_norm = np.dot(q, q)
            distances = np.maximum(
                q_norm + self.vector_norms - 2.0 * np.dot(self.vectors, q),
                0.0,
            )
            ids = np.argpartition(distances, k - 1)[:k]
            ids = ids[np.argsort(distances[ids])]
        else:
            distances = np.sum(
                (self.vectors - q) * (self.vectors - q),
                axis=1,
                dtype=np.float32,
            )
            ids = np.argsort(distances)[:k]

        return ids.tolist()

    def save(self, path: str) -> None:
        """Save the stored vectors so the shared harness can measure size."""
        if self.vectors is None:
            raise RuntimeError("Index has not been built.")
        with open(path, "wb") as f:
            np.save(f, self.vectors)


def _limit(q: np.ndarray, gt: np.ndarray | None = None):
    if MAX_QUERIES is None:
        return q, gt
    n = min(MAX_QUERIES, len(q))
    return q[:n], None if gt is None else gt[:n]


def benchmark_dataset(name: str, base: np.ndarray, queries: np.ndarray,
                      gt: np.ndarray) -> Dict[str, float | bool]:
    model = LinearScan(OPTIMISED)
    start = time.perf_counter()
    model.build_index(base)
    build_time = time.perf_counter() - start

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        index_size = measure_index_size(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    retrieved: List[List[int]] = []

    def timed_search(q: np.ndarray) -> List[int]:
        ids = model.search(q, K)
        retrieved.append(ids)
        return ids

    _, avg_sec = measure_query_time(timed_search, queries)
    recall = compute_recall_at_k(retrieved, gt, k=K)

    metrics = {
        "optimised": OPTIMISED,
        "recall_at_10": float(recall),
        "avg_query_time_ms": float(avg_sec * 1000),
        "build_time_sec": float(build_time),
        "index_size_mb": float(index_size),
        "qps": float(1.0 / avg_sec),
    }

    print(
        f"  {name:<10} | mode={'optimised' if OPTIMISED else 'non-optimised':<13} | "
        f"recall@10={recall:.4f} | avg_query_time={avg_sec * 1000:.3f} ms | "
        f"build_time={build_time:.2f} s | index_size={index_size:.2f} MB"
    )
    del model, retrieved
    gc.collect()
    return metrics


def run_sift() -> Dict[str, float | bool]:
    d = os.path.join(PROJECT_ROOT, "data", "sift")
    base = load_fvecs(os.path.join(d, "sift_base.fvecs"))
    query = load_fvecs(os.path.join(d, "sift_query.fvecs"))
    gt = load_ivecs(os.path.join(d, "sift_groundtruth.ivecs"))[:, :K]
    query, gt = _limit(query, gt)
    return benchmark_dataset("sift1m", base, query, gt)


def run_wikipedia() -> Dict[str, float | bool]:
    base = np.load(os.path.join(PROJECT_ROOT, "data", "wiki", "wiki_base_embeddings.npy"))
    query = np.load(os.path.join(PROJECT_ROOT, "data", "wiki", "wiki_query_embeddings.npy"))

    base_norm = np.linalg.norm(base, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    if np.any(base_norm == 0) or np.any(query_norm == 0):
        raise ValueError("Wikipedia embeddings contain zero vectors.")

    # Explicitly normalise rather than relying on the source files being unit norm.
    base = (base / base_norm).astype(np.float32)
    query = (query / query_norm).astype(np.float32)
    query, _ = _limit(query)
    gt = load_ground_truth(query, base, k=K)
    return benchmark_dataset("wikipedia", base, query, gt)


def main() -> None:
    print(f"EXACT LINEAR SCAN | {'optimised' if OPTIMISED else 'non-optimised'}")
    results = {"sift1m": run_sift(), "wikipedia": run_wikipedia()}
    path = os.path.join(PROJECT_ROOT, "results", "linear_scan_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
