import gc
import json
import os
import sys
import tempfile
import time
from typing import Dict, List

import hnswlib
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


class HNSW:
    """Hierarchical Navigable Small World (HNSW) graph algorithm wrapper."""

    def __init__(
        self,
        space: str = "l2",
        dim: int = 128,
        M: int = 16,
        ef_construction: int = 200,
    ) -> None:
        self.space = space  # 'l2' for SIFT1M, 'ip' (inner product) for Wikipedia
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.index: hnswlib.Index | None = None

    def build_index(self, vectors: np.ndarray) -> None:
        """Build the HNSW graph over base vectors."""
        num_elements = vectors.shape[0]
        self.dim = vectors.shape[1]

        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.index.add_items(vectors, np.arange(num_elements))

    def search(
        self, query_vector: np.ndarray, k: int = 10, ef_search: int = 50
    ) -> List[int]:
        """Return the ids of approximate k nearest neighbours."""
        self.index.set_ef(ef_search)
        labels, _ = self.index.knn_query(query_vector.reshape(1, -1), k=k)
        return labels[0].tolist()

    def save(self, path: str) -> None:
        """Save built index to disk for measuring index size."""
        self.index.save_index(path)


# --------------------------------------------------------------------------- #
# Benchmark script
# --------------------------------------------------------------------------- #

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH_SWEEP = [16, 32, 64, 128, 256, 512]
K = 10


def _print_row(dataset: str, ef_search: int, metrics: Dict[str, float]) -> None:
    print(
        f"  {dataset:<10} | ef_search={ef_search:<5} | "
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
    space: str,
) -> Dict[str, Dict[str, float]]:
    """Build one HNSW index and sweep ef_search at query time."""
    print(
        f"\n=== {dataset_name} (space={space}, M={M}, efConstruction={EF_CONSTRUCTION}) ==="
    )
    results: Dict[str, Dict[str, float]] = {}

    model = HNSW(
        space=space,
        dim=base.shape[1],
        M=M,
        ef_construction=EF_CONSTRUCTION,
    )

    start = time.perf_counter()
    model.build_index(base)
    build_time_sec = time.perf_counter() - start

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        index_size_mb = measure_index_size(tmp_path)
    finally:
        os.remove(tmp_path)

    for ef_search in EF_SEARCH_SWEEP:
        retrieved_ids: List[List[int]] = []

        def timed_search(
            query_vector: np.ndarray, _model=model, _ef=ef_search
        ) -> List[int]:
            ids = _model.search(query_vector, k=K, ef_search=_ef)
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
        results[str(ef_search)] = metrics
        _print_row(dataset_name, ef_search, metrics)

    del model
    gc.collect()

    return results


def main() -> None:
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    # --- SIFT1M (Euclidean / L2 distance) ---
    sift_dir = os.path.join(PROJECT_ROOT, "data", "sift")
    sift_base = load_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    sift_query = load_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    sift_gt = load_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))[:, :K]

    all_results["sift1m"] = benchmark_dataset(
        "sift1m", sift_base, sift_query, sift_gt, space="l2"
    )

    del sift_base, sift_query, sift_gt
    gc.collect()

    # --- Wikipedia (Cosine Similarity via Inner Product over L2-normalized vectors) ---
    wiki_base = np.load(os.path.join(PROJECT_ROOT, "data", "wiki_base_embeddings.npy"))
    wiki_query = np.load(os.path.join(PROJECT_ROOT, "data", "wiki_query_embeddings.npy"))

    wiki_base_norm = wiki_base / np.linalg.norm(wiki_base, axis=1, keepdims=True)
    wiki_query_norm = wiki_query / np.linalg.norm(wiki_query, axis=1, keepdims=True)
    wiki_gt = load_ground_truth(wiki_query_norm, wiki_base_norm, k=K)

    all_results["wikipedia"] = benchmark_dataset(
        "wikipedia", wiki_base_norm, wiki_query_norm, wiki_gt, space="ip"
    )

    del wiki_base, wiki_query, wiki_gt, wiki_base_norm, wiki_query_norm
    gc.collect()

    # --- Save results ---
    results_path = os.path.join(PROJECT_ROOT, "results", "hnsw_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()