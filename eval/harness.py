"""Shared evaluation utilities for comparing ANN search methods.

Provides ground-truth computation, recall scoring, query timing, and
index size measurement so every method in methods/ is benchmarked the
same way.
"""

import os
import time
from typing import Callable, Sequence, Tuple

import numpy as np


def load_ground_truth(
    query_embeddings: np.ndarray, base_embeddings: np.ndarray, k: int = 10
) -> np.ndarray:
    """Compute true top-k nearest neighbours via brute-force search.

    Use this for datasets without precomputed ground truth (e.g. our
    Wikipedia sentence embedding set). For SIFT1M, prefer the dataset's
    shipped ground-truth file if available.

    Args:
        query_embeddings: Array of shape (n_queries, d).
        base_embeddings: Array of shape (n_base, d).
        k: Number of nearest neighbours to compute per query.

    Returns:
        Array of shape (n_queries, k) with the ids (row indices into
        base_embeddings) of the true nearest neighbours, ordered nearest
        first.
    """
    # Squared Euclidean distance, expanded to avoid an (n_queries, n_base, d) tensor.
    query_sq = np.sum(query_embeddings**2, axis=1, keepdims=True)
    base_sq = np.sum(base_embeddings**2, axis=1)
    distances = query_sq + base_sq - 2 * query_embeddings @ base_embeddings.T

    k = min(k, base_embeddings.shape[0])
    top_k_unsorted = np.argpartition(distances, k - 1, axis=1)[:, :k]
    row_indices = np.arange(distances.shape[0])[:, None]
    order = np.argsort(distances[row_indices, top_k_unsorted], axis=1)
    return top_k_unsorted[row_indices, order]


def compute_recall_at_k(
    retrieved_ids: Sequence[Sequence[int]],
    ground_truth_ids: Sequence[Sequence[int]],
    k: int = 10,
) -> float:
    """Compute average recall@k across a batch of queries.

    Recall@k for one query is the fraction of the true top-k neighbours
    that appear anywhere in that query's retrieved ids.

    Args:
        retrieved_ids: Per-query lists/arrays of ids returned by an ANN method.
        ground_truth_ids: Per-query lists/arrays of true nearest-neighbour ids.
        k: Number of neighbours considered per query.

    Returns:
        Mean recall@k over all queries, in [0, 1].
    """
    if len(retrieved_ids) != len(ground_truth_ids):
        raise ValueError("retrieved_ids and ground_truth_ids must have the same length")

    recalls = []
    for retrieved, truth in zip(retrieved_ids, ground_truth_ids):
        truth_k = set(truth[:k])
        if not truth_k:
            continue
        retrieved_k = set(retrieved[:k])
        recalls.append(len(retrieved_k & truth_k) / len(truth_k))

    return float(np.mean(recalls)) if recalls else 0.0


def measure_query_time(
    search_fn: Callable[[np.ndarray], object], queries: np.ndarray
) -> Tuple[list, float]:
    """Time a search function over a batch of queries.

    Args:
        search_fn: Callable that takes a single query vector and returns
            its search results (e.g. a method's `.search` bound method).
        queries: Array of shape (n_queries, d).

    Returns:
        Tuple of (per_query_times, average_time), where per_query_times is
        a list of per-query wall-clock durations in seconds and
        average_time is their mean.
    """
    per_query_times = []
    for query in queries:
        start = time.perf_counter()
        search_fn(query)
        per_query_times.append(time.perf_counter() - start)

    average_time = float(np.mean(per_query_times)) if per_query_times else 0.0
    return per_query_times, average_time


def measure_index_size(index_object_or_path) -> float:
    """Measure the size of a built index in megabytes.

    Args:
        index_object_or_path: Either a filesystem path to a saved index
            (str/os.PathLike), or an in-memory object exposing a NumPy
            array (via a `.vectors`/`.codes` attribute) whose `.nbytes`
            can be used as an approximation.

    Returns:
        Size of the index in megabytes.
    """
    if isinstance(index_object_or_path, (str, os.PathLike)):
        if os.path.isfile(index_object_or_path):
            return os.path.getsize(index_object_or_path) / (1024**2)
        if os.path.isdir(index_object_or_path):
            total = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(index_object_or_path)
                for f in files
            )
            return total / (1024**2)
        raise FileNotFoundError(f"No such file or directory: {index_object_or_path}")

    for attr in ("vectors", "codes"):
        value = getattr(index_object_or_path, attr, None)
        if isinstance(value, np.ndarray):
            return value.nbytes / (1024**2)

    raise TypeError(
        "index_object_or_path must be a path to a saved index, or an object "
        "with a `.vectors`/`.codes` NumPy array attribute"
    )
