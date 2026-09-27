"""Random Projection LSH approximate nearest-neighbour search.

Repository-compatible implementation following the same dataset and
load/build/search/measure/record pattern as methods/annoy.py.

Datasets used by the repository:
    SIFT1M
        data/sift/sift_base.fvecs
        data/sift/sift_query.fvecs
        data/sift/sift_groundtruth.ivecs

    Wikipedia sentence embeddings
        data/wiki_base_embeddings.npy
        data/wiki_query_embeddings.npy

Wikipedia has no supplied ground-truth file, so exact ground truth is
generated from the L2-normalised base/query embeddings, matching the
cosine/angular setup used by Annoy.

Run from the repository root:
    python methods/lsh_random_projection.py
"""

import gc
import json
import os
import sys
import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

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
# Dataset paths -- deliberately identical to methods/annoy.py
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

N_TABLES = 20
N_HASHES = 4
RANDOM_SEED = 42
BUILD_CHUNK_SIZE = 100_000

# Optional candidate cap. None means all retrieved candidates are reranked.
MAX_CANDIDATES = None

# Bucket width is the main LSH search-quality/speed parameter.
# Values are kept dataset-specific because projection scales differ.
BUCKET_WIDTH_SWEEP = {
    "sift1m": [4.0, 8.0, 16.0, 32.0],
    "wikipedia": [0.5, 1.0, 1.5, 2.0],
}

K_VALUES = [1, 10, 100]
MAX_QUERIES = None  # Set to e.g. 100 for a quick test.

RESULTS_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "lsh_results.json",
)


class RandomProjectionLSH:
    """Random-projection LSH index for approximate nearest neighbours."""

    def __init__(
        self,
        num_tables: int = N_TABLES,
        num_hashes: int = N_HASHES,
        bucket_width: float = 1.0,
        random_seed: int = RANDOM_SEED,
        max_candidates: int | None = MAX_CANDIDATES,
    ) -> None:
        if num_tables <= 0:
            raise ValueError("num_tables must be positive.")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive.")
        if bucket_width <= 0:
            raise ValueError("bucket_width must be positive.")

        self.num_tables = num_tables
        self.num_hashes = num_hashes
        self.bucket_width = bucket_width
        self.random_seed = random_seed
        self.max_candidates = max_candidates

        self.dim: int | None = None
        self.vectors: np.ndarray | None = None
        self.vector_norms: np.ndarray | None = None
        self.projections: np.ndarray | None = None
        self.offsets: np.ndarray | None = None

        self.tables: List[
            DefaultDict[Tuple[int, ...], List[int]]
        ] = []

        self.last_candidate_count = 0

    def build_index(self, vectors: np.ndarray) -> None:
        """Build the random-projection hash tables."""

        self.vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )
        self.vector_norms = np.sum(
            self.vectors * self.vectors,
            axis=1,
            dtype=np.float32,
        )
        self.dim = self.vectors.shape[1]

        rng = np.random.default_rng(self.random_seed)

        self.projections = rng.normal(
            size=(
                self.num_tables,
                self.num_hashes,
                self.dim,
            )
        ).astype(np.float32)

        self.offsets = rng.uniform(
            0.0,
            self.bucket_width,
            size=(
                self.num_tables,
                self.num_hashes,
            ),
        ).astype(np.float32)

        self.tables = [
            defaultdict(list)
            for _ in range(self.num_tables)
        ]

        n_vectors = len(self.vectors)

        for start in range(
            0,
            n_vectors,
            BUILD_CHUNK_SIZE,
        ):
            end = min(
                start + BUILD_CHUNK_SIZE,
                n_vectors,
            )
            chunk = self.vectors[start:end]

            for table_id in range(self.num_tables):
                projection = self.projections[table_id]
                offset = self.offsets[table_id]

                hash_values = np.floor(
                    (
                        chunk @ projection.T
                        + offset
                    )
                    / self.bucket_width
                ).astype(np.int64)

                table = self.tables[table_id]

                for local_id, key_array in enumerate(hash_values):
                    # Use the complete tuple as the dictionary key.
                    key = tuple(int(x) for x in key_array)
                    table[key].append(start + local_id)

    def _candidate_ids(
        self,
        query_vector: np.ndarray,
    ) -> Set[int]:
        """Return the union of vectors in matching buckets."""

        if self.projections is None or self.offsets is None:
            raise RuntimeError("Index has not been built.")

        q = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        candidates: Set[int] = set()

        for table_id in range(self.num_tables):
            hash_values = np.floor(
                (
                    self.projections[table_id] @ q
                    + self.offsets[table_id]
                )
                / self.bucket_width
            ).astype(np.int64)

            key = tuple(int(x) for x in hash_values)

            candidates.update(
                self.tables[table_id].get(
                    key,
                    [],
                )
            )

        return candidates

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> List[int]:
        """Return approximate k-nearest-neighbour ids."""

        if self.vectors is None or self.vector_norms is None:
            raise RuntimeError("Index has not been built.")

        candidates = self._candidate_ids(query_vector)

        if (
            self.max_candidates is not None
            and len(candidates) > self.max_candidates
        ):
            rng = np.random.default_rng(self.random_seed)

            candidate_array = np.fromiter(
                candidates,
                dtype=np.int64,
            )

            candidates = set(
                rng.choice(
                    candidate_array,
                    size=self.max_candidates,
                    replace=False,
                ).tolist()
            )

        self.last_candidate_count = len(candidates)

        if not candidates:
            return []

        candidate_ids = np.fromiter(
            candidates,
            dtype=np.int64,
        )

        q = np.asarray(
            query_vector,
            dtype=np.float32,
        )
        q_norm = np.dot(q, q)

        candidate_vectors = self.vectors[candidate_ids]
        candidate_norms = self.vector_norms[candidate_ids]

        distances = np.maximum(
            q_norm
            + candidate_norms
            - 2.0 * np.dot(candidate_vectors, q),
            0.0,
        )

        k = min(k, len(candidate_ids))

        selected = np.argpartition(
            distances,
            k - 1,
        )[:k]

        selected = selected[
            np.argsort(distances[selected])
        ]

        return candidate_ids[selected].tolist()

    def estimated_array_memory_mb(self) -> float:
        """Estimate memory used by the main NumPy index arrays."""

        arrays = [
            self.vectors,
            self.vector_norms,
            self.projections,
            self.offsets,
        ]

        total_bytes = sum(
            array.nbytes
            for array in arrays
            if array is not None
        )

        return total_bytes / (1024 ** 2)


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
    bucket_width: float,
    metrics: Dict[str, float],
) -> None:
    print(
        f"  {dataset:<10} | "
        f"w={bucket_width:<6g} | "
        f"R@1={metrics['recall_at_1']:.4f} | "
        f"R@10={metrics['recall_at_10']:.4f} | "
        f"R@100={metrics['recall_at_100']:.4f} | "
        f"avg_query_time={metrics['avg_query_time_ms']:.3f} ms | "
        f"QPS={metrics['qps']:.2f} | "
        f"candidates={metrics['avg_candidates']:.0f} "
        f"({metrics['candidate_pct']:.2f}%)"
    )


def benchmark_dataset(
    dataset_name: str,
    base: np.ndarray,
    queries: np.ndarray,
    ground_truth_ids: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Build LSH indexes and sweep bucket width."""

    widths = BUCKET_WIDTH_SWEEP[dataset_name]

    print(
        f"\n=== {dataset_name} "
        f"(tables={N_TABLES}, hashes/table={N_HASHES}) ==="
    )

    results: Dict[str, Dict[str, float]] = {}

    for bucket_width in widths:
        print(
            f"\nBuilding LSH index: "
            f"bucket_width={bucket_width}"
        )

        model = RandomProjectionLSH(
            num_tables=N_TABLES,
            num_hashes=N_HASHES,
            bucket_width=bucket_width,
            random_seed=RANDOM_SEED,
            max_candidates=MAX_CANDIDATES,
        )

        build_start = time.perf_counter()
        model.build_index(base)
        build_time_sec = time.perf_counter() - build_start

        retrieved_ids: List[List[int]] = []
        candidate_counts: List[int] = []

        max_k = max(K_VALUES)

        def timed_search(
            query_vector: np.ndarray,
        ) -> List[int]:
            ids = model.search(
                query_vector,
                k=max_k,
            )
            retrieved_ids.append(ids)
            candidate_counts.append(
                model.last_candidate_count
            )
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

        avg_candidates = (
            float(np.mean(candidate_counts))
            if candidate_counts
            else 0.0
        )

        candidate_pct = (
            100.0 * avg_candidates / len(base)
            if len(base) > 0
            else 0.0
        )

        metrics = {
            "num_tables": float(N_TABLES),
            "num_hashes": float(N_HASHES),
            "bucket_width": float(bucket_width),
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
            "avg_candidates": avg_candidates,
            "candidate_pct": candidate_pct,
            "index_array_memory_mb": float(
                model.estimated_array_memory_mb()
            ),
        }

        results[str(bucket_width)] = metrics
        _print_row(dataset_name, bucket_width, metrics)

        del model
        gc.collect()

    return results


# --------------------------------------------------------------------------- #
# Dataset runners -- same datasets as Annoy
# --------------------------------------------------------------------------- #

def run_sift() -> Dict[str, Dict[str, float]]:
    """Benchmark LSH on the repository's SIFT1M dataset."""

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


def run_wikipedia() -> Dict[str, Dict[str, float]]:
    """Benchmark LSH on the repository's Wikipedia embeddings.

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
    print("=" * 80)
    print("RANDOM PROJECTION LSH")
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
    for dataset_name, per_width in all_results.items():
        for width_str, metrics in per_width.items():
            _print_row(
                dataset_name,
                float(width_str),
                metrics,
            )


if __name__ == "__main__":
    main()
