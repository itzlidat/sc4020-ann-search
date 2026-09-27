"""Random Projection LSH approximate nearest-neighbour search.

Shared repository interface:
    build_index(vectors)
    search(q, k) -> list[int]
    save(path)

Uses the same SIFT1M and Wikipedia datasets as methods/annoy_method.py.

Run from repository root:
    python methods/lsh.py

---------------------------------------------------------------
STANDARDISED PARAMETERS
---------------------------------------------------------------

SIFT1M:
    Standard:
        tables       = 10
        hashes/table = 8
        width        = 744.6261

    Low:
        tables       = 10
        hashes/table = 8
        width        = 496.4174

    Medium:
        tables       = 10
        hashes/table = 8
        width        = 744.6261

    High:
        tables       = 10
        hashes/table = 8
        width        = 992.8348

These widths were selected empirically from the SIFT1M
projection-scale calibration and parameter experiments.

WIKIPEDIA:
    Wikipedia embeddings are explicitly L2-normalised.
    Therefore, SIFT1M widths are NOT reused for Wikipedia.

---------------------------------------------------------------
PARAMETER SWEEP
---------------------------------------------------------------

Set:

    RUN_PARAMETER_SWEEP = False

for the normal standardised experiment.

Set:

    RUN_PARAMETER_SWEEP = True

to run the full parameter sweep.

The sweep is useful for analysing the recall/speed/candidate
trade-off, but should normally be switched OFF for the final
standardised comparison.
"""

import gc
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple

import numpy as np


# ==============================================================
# PROJECT PATH
# ==============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ==============================================================
# PROJECT IMPORTS
# ==============================================================

from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402

from eval.harness import (  # noqa: E402
    compute_recall_at_k,
    load_ground_truth,
    measure_index_size,
    measure_query_time,
)


# ==============================================================
# GENERAL SETTINGS
# ==============================================================

K = 10

RANDOM_SEED = 42

BUILD_CHUNK_SIZE = 100_000

MAX_QUERIES = None


# ==============================================================
# MAIN SWITCH
# ==============================================================

# --------------------------------------------------------------
# False = use only the standardised parameters
#
# True = run the full parameter sweep
# --------------------------------------------------------------

RUN_PARAMETER_SWEEP = False


# ==============================================================
# STANDARDISED SIFT1M PARAMETERS
# ==============================================================

# These are the parameters we found from the SIFT1M
# calibration and experiments.

STANDARD_SIFT_TABLES = 10

STANDARD_SIFT_HASHES = 8

STANDARD_SIFT_WIDTH = 744.6261


# --------------------------------------------------------------
# Additional standardised SIFT operating points.
#
# These are retained so that we can show the speed/recall
# trade-off without performing the entire parameter sweep.
# --------------------------------------------------------------

SIFT_STANDARD_CONFIGS = {
    "low": {
        "num_tables": 10,
        "num_hashes": 8,
        "bucket_width": 496.4174,
    },

    "medium": {
        "num_tables": 10,
        "num_hashes": 8,
        "bucket_width": 744.6261,
    },

    "high": {
        "num_tables": 10,
        "num_hashes": 8,
        "bucket_width": 992.8348,
    },
}


# ==============================================================
# SIFT1M PARAMETER SWEEP
# ==============================================================

# --------------------------------------------------------------
# These sweep values are based on the measured SIFT1M
# projection scale:
#
# projection std ≈ 496.4174
#
# Width multipliers:
#
# 0.25 -> 124.1044
# 0.50 -> 248.2087
# 0.75 -> 372.3131
# 1.00 -> 496.4174
# 1.50 -> 744.6261
# 2.00 -> 992.8348
# 3.00 -> 1489.2523
# --------------------------------------------------------------

SIFT_PROJECTION_SCALE = 496.4174

SIFT_WIDTH_MULTIPLIERS = [
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
]


# Number of tables used in the table-count sweep.

SIFT_TABLE_SWEEP = [
    5,
    10,
    20,
    40,
]


# Hashes/table used in the main width and table sweeps.

SIFT_MAIN_HASHES = 8


# Secondary hash-count sweep.

SIFT_HASH_SWEEP_TABLES = [
    10,
    20,
    40,
]

SIFT_HASHES_SWEEP = [
    2,
    4,
    8,
]


# ==============================================================
# WIKIPEDIA PARAMETERS
# ==============================================================

# Wikipedia embeddings are normalised, so their numerical
# projection scale is completely different from raw SIFT1M.

WIKI_STANDARD_CONFIGS = {
    "low": {
        "num_tables": 20,
        "num_hashes": 4,
        "bucket_width": 1.0,
    },

    "medium": {
        "num_tables": 20,
        "num_hashes": 4,
        "bucket_width": 1.5,
    },

    "high": {
        "num_tables": 40,
        "num_hashes": 4,
        "bucket_width": 2.0,
    },
}


# Wikipedia parameter sweep.

WIKI_TABLE_SWEEP = [
    5,
    10,
    20,
    40,
]

WIKI_WIDTH_SWEEP = [
    0.5,
    1.0,
    1.5,
    2.0,
]

WIKI_HASHES = 4


# ==============================================================
# RANDOM PROJECTION LSH
# ==============================================================

class RandomProjectionLSH:
    """Random-projection LSH with multiple hash tables."""

    def __init__(
        self,
        num_tables: int = STANDARD_SIFT_TABLES,
        num_hashes: int = STANDARD_SIFT_HASHES,
        bucket_width: float = STANDARD_SIFT_WIDTH,
        random_seed: int = RANDOM_SEED,
    ) -> None:

        if (
            num_tables <= 0
            or num_hashes <= 0
            or bucket_width <= 0
        ):
            raise ValueError(
                "LSH parameters must be positive."
            )

        self.num_tables = int(num_tables)

        self.num_hashes = int(num_hashes)

        self.bucket_width = float(bucket_width)

        self.random_seed = int(random_seed)

        self.vectors: np.ndarray | None = None

        self.vector_norms: np.ndarray | None = None

        self.projections: np.ndarray | None = None

        self.offsets: np.ndarray | None = None

        self.tables: List[
            DefaultDict[
                Tuple[int, ...],
                List[int]
            ]
        ] = []

        self.last_candidate_count = 0


    # ==========================================================
    # BUILD INDEX
    # ==========================================================

    def build_index(
        self,
        vectors: np.ndarray
    ) -> None:

        self.vectors = np.ascontiguousarray(
            vectors,
            dtype=np.float32,
        )

        self.vector_norms = np.sum(
            self.vectors * self.vectors,
            axis=1,
            dtype=np.float32,
        )

        dim = self.vectors.shape[1]

        rng = np.random.default_rng(
            self.random_seed
        )

        # ------------------------------------------------------
        # Gaussian random projections
        # ------------------------------------------------------

        self.projections = rng.normal(
            size=(
                self.num_tables,
                self.num_hashes,
                dim,
            )
        ).astype(np.float32)

        # ------------------------------------------------------
        # Random offsets in [0, bucket_width)
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # Build hash tables in chunks
        # ------------------------------------------------------

        for start in range(
            0,
            len(self.vectors),
            BUILD_CHUNK_SIZE,
        ):

            end = min(
                start + BUILD_CHUNK_SIZE,
                len(self.vectors),
            )

            chunk = self.vectors[start:end]

            for t in range(
                self.num_tables
            ):

                hashes = np.floor(
                    (
                        chunk
                        @ self.projections[t].T
                        + self.offsets[t]
                    )
                    / self.bucket_width
                ).astype(np.int64)

                table = self.tables[t]

                for local_id, key_values in enumerate(
                    hashes
                ):

                    key = tuple(
                        int(v)
                        for v in key_values
                    )

                    table[key].append(
                        start + local_id
                    )


    # ==========================================================
    # SEARCH
    # ==========================================================

    def search(
        self,
        q: np.ndarray,
        k: int = K,
    ) -> List[int]:

        if (
            self.vectors is None
            or self.vector_norms is None
        ):
            raise RuntimeError(
                "Index has not been built."
            )

        if (
            self.projections is None
            or self.offsets is None
        ):
            raise RuntimeError(
                "LSH parameters have not been built."
            )

        q = np.asarray(
            q,
            dtype=np.float32,
        )

        bucket_arrays: List[np.ndarray] = []

        # ------------------------------------------------------
        # Retrieve matching buckets from every table
        # ------------------------------------------------------

        for t in range(
            self.num_tables
        ):

            hashes = np.floor(
                (
                    self.projections[t]
                    @ q
                    + self.offsets[t]
                )
                / self.bucket_width
            ).astype(np.int64)

            key = tuple(
                int(v)
                for v in hashes
            )

            ids = self.tables[t].get(
                key
            )

            if ids:
                bucket_arrays.append(
                    np.asarray(
                        ids,
                        dtype=np.int64,
                    )
                )

        # ------------------------------------------------------
        # No candidates
        # ------------------------------------------------------

        if not bucket_arrays:

            self.last_candidate_count = 0

            return []


        # ------------------------------------------------------
        # Union candidates from all tables
        #
        # np.unique is intentionally used here rather than a
        # Python set because this is the standardised candidate
        # union used by the project.
        # ------------------------------------------------------

        candidates = np.unique(
            np.concatenate(
                bucket_arrays
            )
        )

        self.last_candidate_count = len(
            candidates
        )


        # ------------------------------------------------------
        # Exact reranking
        # ------------------------------------------------------

        q_norm = np.dot(
            q,
            q,
        )

        candidate_vectors = (
            self.vectors[candidates]
        )

        candidate_norms = (
            self.vector_norms[candidates]
        )

        distances = np.maximum(
            q_norm
            + candidate_norms
            - 2.0
            * np.dot(
                candidate_vectors,
                q,
            ),
            0.0,
        )


        # ------------------------------------------------------
        # Top-k
        # ------------------------------------------------------

        k = min(
            k,
            len(candidates),
        )

        if k <= 0:
            return []

        selected = np.argpartition(
            distances,
            k - 1,
        )[:k]

        selected = selected[
            np.argsort(
                distances[selected]
            )
        ]

        return candidates[
            selected
        ].tolist()


    # ==========================================================
    # SAVE
    # ==========================================================

    def save(
        self,
        path: str
    ) -> None:

        """Save the built index."""

        if (
            self.vectors is None
            or self.projections is None
            or self.offsets is None
        ):
            raise RuntimeError(
                "Index has not been built."
            )

        keys = np.empty(
            self.num_tables,
            dtype=object,
        )

        values = np.empty(
            self.num_tables,
            dtype=object,
        )

        for t, table in enumerate(
            self.tables
        ):

            keys[t] = np.asarray(
                list(table.keys()),
                dtype=object,
            )

            values[t] = np.asarray(
                list(table.values()),
                dtype=object,
            )

        with open(
            path,
            "wb",
        ) as f:

            np.savez(
                f,
                vectors=self.vectors,
                vector_norms=self.vector_norms,
                projections=self.projections,
                offsets=self.offsets,
                table_keys=keys,
                table_values=values,
            )


# ==============================================================
# QUERY LIMIT
# ==============================================================

def _limit(
    q: np.ndarray,
    gt: np.ndarray | None = None
):

    if MAX_QUERIES is None:
        return q, gt

    n = min(
        MAX_QUERIES,
        len(q),
    )

    return (
        q[:n],
        None if gt is None else gt[:n],
    )


# ==============================================================
# SINGLE CONFIGURATION BENCHMARK
# ==============================================================

def benchmark_configuration(
    dataset_name: str,
    configuration_name: str,
    base: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    num_tables: int,
    num_hashes: int,
    width: float,
) -> Dict[str, float]:

    print()

    print(
        f"=== {dataset_name} | "
        f"{configuration_name} | "
        f"tables={num_tables}, "
        f"hashes/table={num_hashes}, "
        f"width={width:.4f} ==="
    )


    # ----------------------------------------------------------
    # Build
    # ----------------------------------------------------------

    model = RandomProjectionLSH(
        num_tables=num_tables,
        num_hashes=num_hashes,
        bucket_width=width,
        random_seed=RANDOM_SEED,
    )

    start = time.perf_counter()

    model.build_index(
        base
    )

    build_time = (
        time.perf_counter()
        - start
    )


    # ----------------------------------------------------------
    # Measure index size
    # ----------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".npz",
        delete=False,
    ) as tmp:

        tmp_path = tmp.name

    try:

        model.save(
            tmp_path
        )

        index_size = measure_index_size(
            tmp_path
        )

    finally:

        if os.path.exists(
            tmp_path
        ):
            os.remove(
                tmp_path
            )


    # ----------------------------------------------------------
    # Search
    # ----------------------------------------------------------

    retrieved: List[List[int]] = []

    candidates: List[int] = []


    def timed_search(
        q: np.ndarray
    ) -> List[int]:

        ids = model.search(
            q,
            K,
        )

        retrieved.append(
            ids
        )

        candidates.append(
            model.last_candidate_count
        )

        return ids


    _, avg_sec = measure_query_time(
        timed_search,
        queries,
    )


    # ----------------------------------------------------------
    # Recall
    # ----------------------------------------------------------

    recall = compute_recall_at_k(
        retrieved,
        gt,
        k=K,
    )


    # ----------------------------------------------------------
    # Candidate statistics
    # ----------------------------------------------------------

    avg_candidates = (
        float(
            np.mean(candidates)
        )
        if candidates
        else 0.0
    )

    candidate_pct = (
        100.0
        * avg_candidates
        / len(base)
    )


    # ----------------------------------------------------------
    # QPS
    # ----------------------------------------------------------

    qps = (
        1.0 / avg_sec
        if avg_sec > 0
        else 0.0
    )


    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    metrics = {

        "configuration":
            configuration_name,

        "num_tables":
            int(num_tables),

        "num_hashes":
            int(num_hashes),

        "bucket_width":
            float(width),

        "recall_at_10":
            float(recall),

        "avg_query_time_ms":
            float(
                avg_sec * 1000
            ),

        "build_time_sec":
            float(
                build_time
            ),

        "index_size_mb":
            float(
                index_size
            ),

        "avg_candidates":
            avg_candidates,

        "candidate_pct":
            float(
                candidate_pct
            ),

        "qps":
            float(
                qps
            ),
    }


    # ----------------------------------------------------------
    # Print
    # ----------------------------------------------------------

    print(
        f"  recall@10="
        f"{recall:.4f}"
        f" | avg_query_time="
        f"{avg_sec * 1000:.3f} ms"
        f" | build_time="
        f"{build_time:.2f} s"
        f" | index_size="
        f"{index_size:.2f} MB"
        f" | candidates="
        f"{avg_candidates:.0f}"
        f" ({candidate_pct:.2f}%)"
    )


    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------

    del model
    del retrieved
    del candidates

    gc.collect()


    return metrics


# ==============================================================
# STANDARDISED BENCHMARK
# ==============================================================

def benchmark_standard_configs(
    name: str,
    base: np.ndarray,
    queries: np.ndarray,
    gt: np.ndarray,
    configs: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:

    results = {}


    for config_name, config in configs.items():

        metrics = benchmark_configuration(
            dataset_name=name,
            configuration_name=config_name,
            base=base,
            queries=queries,
            gt=gt,
            num_tables=int(
                config["num_tables"]
            ),
            num_hashes=int(
                config["num_hashes"]
            ),
            width=float(
                config["bucket_width"]
            ),
        )

        results[
            config_name
        ] = metrics


    return results


# ==============================================================
# SIFT PARAMETER SWEEP
# ==============================================================

def run_sift_parameter_sweep(
    base: np.ndarray,
    query: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, Dict[str, float]]:

    print()
    print("=" * 70)
    print("SIFT1M PARAMETER SWEEP")
    print("=" * 70)

    results = {}


    # ----------------------------------------------------------
    # Stage 1:
    # Width sweep
    #
    # 10 tables × 8 hashes
    # ----------------------------------------------------------

    print()
    print(
        "Stage 1: bucket-width sweep"
    )

    widths = [
        SIFT_PROJECTION_SCALE * multiplier
        for multiplier in SIFT_WIDTH_MULTIPLIERS
    ]


    for width in widths:

        multiplier = (
            width
            / SIFT_PROJECTION_SCALE
        )

        name = (
            f"width_{multiplier:.2f}x"
        )

        results[name] = benchmark_configuration(
            dataset_name="sift1m",
            configuration_name=name,
            base=base,
            queries=query,
            gt=gt,
            num_tables=10,
            num_hashes=8,
            width=width,
        )


    # ----------------------------------------------------------
    # Stage 2:
    # Table-count sweep
    #
    # Use the medium width.
    # ----------------------------------------------------------

    print()
    print(
        "Stage 2: table-count sweep"
    )

    medium_width = (
        SIFT_PROJECTION_SCALE
        * 1.50
    )


    for num_tables in SIFT_TABLE_SWEEP:

        name = (
            f"tables_{num_tables}"
        )

        results[name] = benchmark_configuration(
            dataset_name="sift1m",
            configuration_name=name,
            base=base,
            queries=query,
            gt=gt,
            num_tables=num_tables,
            num_hashes=SIFT_MAIN_HASHES,
            width=medium_width,
        )


    # ----------------------------------------------------------
    # Stage 3:
    # Hash-count sweep
    #
    # Use medium width.
    # ----------------------------------------------------------

    print()
    print(
        "Stage 3: hashes-per-table sweep"
    )


    for num_tables in SIFT_HASH_SWEEP_TABLES:

        for num_hashes in SIFT_HASHES_SWEEP:

            name = (
                f"tables_{num_tables}"
                f"_hashes_{num_hashes}"
            )

            results[name] = benchmark_configuration(
                dataset_name="sift1m",
                configuration_name=name,
                base=base,
                queries=query,
                gt=gt,
                num_tables=num_tables,
                num_hashes=num_hashes,
                width=medium_width,
            )


    return results


# ==============================================================
# WIKIPEDIA PARAMETER SWEEP
# ==============================================================

def run_wikipedia_parameter_sweep(
    base: np.ndarray,
    query: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, Dict[str, float]]:

    print()
    print("=" * 70)
    print("WIKIPEDIA PARAMETER SWEEP")
    print("=" * 70)

    results = {}


    for num_tables in WIKI_TABLE_SWEEP:

        for width in WIKI_WIDTH_SWEEP:

            name = (
                f"tables_{num_tables}"
                f"_width_{width:g}"
            )

            results[name] = benchmark_configuration(
                dataset_name="wikipedia",
                configuration_name=name,
                base=base,
                queries=query,
                gt=gt,
                num_tables=num_tables,
                num_hashes=WIKI_HASHES,
                width=width,
            )


    return results


# ==============================================================
# SIFT1M
# ==============================================================

def run_sift():

    print()
    print("=" * 70)
    print("SIFT1M - RANDOM PROJECTION LSH")
    print("=" * 70)


    d = os.path.join(
        PROJECT_ROOT,
        "data",
        "sift",
    )


    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    base = load_fvecs(
        os.path.join(
            d,
            "sift_base.fvecs",
        )
    )

    query = load_fvecs(
        os.path.join(
            d,
            "sift_query.fvecs",
        )
    )

    gt = load_ivecs(
        os.path.join(
            d,
            "sift_groundtruth.ivecs",
        )
    )[:, :K]


    query, gt = _limit(
        query,
        gt,
    )


    print(
        f"Base shape       : {base.shape}"
    )

    print(
        f"Query shape      : {query.shape}"
    )

    print(
        f"Ground truth     : {gt.shape}"
    )


    # ----------------------------------------------------------
    # Parameter sweep
    # ----------------------------------------------------------

    if RUN_PARAMETER_SWEEP:

        print()
        print(
            "PARAMETER SWEEP ENABLED"
        )

        return {
            "mode": "parameter_sweep",
            "results":
                run_sift_parameter_sweep(
                    base,
                    query,
                    gt,
                ),
        }


    # ----------------------------------------------------------
    # Standardised configurations
    # ----------------------------------------------------------

    print()
    print(
        "PARAMETER SWEEP DISABLED"
    )

    print(
        "Running standardised LSH configurations."
    )


    return {
        "mode": "standard",
        "results":
            benchmark_standard_configs(
                name="sift1m",
                base=base,
                queries=query,
                gt=gt,
                configs=SIFT_STANDARD_CONFIGS,
            ),
    }


# ==============================================================
# WIKIPEDIA
# ==============================================================

def run_wikipedia():

    print()
    print("=" * 70)
    print("WIKIPEDIA - RANDOM PROJECTION LSH")
    print("=" * 70)


    base = np.load(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "wiki_base_embeddings.npy",
        )
    )

    query = np.load(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "wiki_query_embeddings.npy",
        )
    )


    # ----------------------------------------------------------
    # Explicit L2 normalisation
    # ----------------------------------------------------------

    base_norm = np.linalg.norm(
        base,
        axis=1,
        keepdims=True,
    )

    query_norm = np.linalg.norm(
        query,
        axis=1,
        keepdims=True,
    )


    if np.any(base_norm == 0):
        raise ValueError(
            "Wikipedia base embeddings contain zero vectors."
        )

    if np.any(query_norm == 0):
        raise ValueError(
            "Wikipedia query embeddings contain zero vectors."
        )


    base = (
        base / base_norm
    ).astype(
        np.float32
    )

    query = (
        query / query_norm
    ).astype(
        np.float32
    )


    query, _ = _limit(
        query
    )


    print(
        f"Base shape       : {base.shape}"
    )

    print(
        f"Query shape      : {query.shape}"
    )


    # ----------------------------------------------------------
    # Ground truth
    # ----------------------------------------------------------

    gt = load_ground_truth(
        query,
        base,
        k=K,
    )


    # ----------------------------------------------------------
    # Parameter sweep
    # ----------------------------------------------------------

    if RUN_PARAMETER_SWEEP:

        print()
        print(
            "PARAMETER SWEEP ENABLED"
        )

        return {
            "mode": "parameter_sweep",
            "results":
                run_wikipedia_parameter_sweep(
                    base,
                    query,
                    gt,
                ),
        }


    # ----------------------------------------------------------
    # Standardised configurations
    # ----------------------------------------------------------

    print()
    print(
        "PARAMETER SWEEP DISABLED"
    )

    print(
        "Running standardised LSH configurations."
    )


    return {
        "mode": "standard",
        "results":
            benchmark_standard_configs(
                name="wikipedia",
                base=base,
                queries=query,
                gt=gt,
                configs=WIKI_STANDARD_CONFIGS,
            ),
    }


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "SC4020 ANN SEARCH"
    )
    print(
        "RANDOM PROJECTION LSH"
    )
    print("=" * 70)

    print()

    print(
        "Parameter sweep:",
        "ON" if RUN_PARAMETER_SWEEP else "OFF",
    )


    # ----------------------------------------------------------
    # Run datasets
    # ----------------------------------------------------------

    results = {
        "sift1m": run_sift(),
        "wikipedia": run_wikipedia(),
    }


    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------

    path = os.path.join(
        PROJECT_ROOT,
        "results",
        "lsh_results.json",
    )


    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )


    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )


    print()
    print("=" * 70)

    print(
        "Saved results to:"
    )

    print(
        path
    )

    print("=" * 70)


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()