"""
Benchmark PQ, IVF, and IVF+PQ on SIFT1M and Wikipedia embeddings.

Metrics:
- Recall@10
- Average query latency (ms)
- Index size (MB)

Results are saved to:
results/pq_ivf_results.json
"""

from pathlib import Path
import json
import tempfile

import numpy as np

from eval.harness import (
    compute_recall_at_k,
    measure_query_time,
    measure_index_size,
)

from methods.PQ import PQ
from methods.IVF import IVF
from methods.PQ_IVF import IVFPQ

from fvecs_loader import load_fvecs, load_ivecs


# =================================================
# Dataset paths
# =================================================

# IMPORTANT:
# The SIFT1M and Wikipedia datasets are NOT stored in this GitHub repo
# because the files are too large.
#
# Before running this script, replace the two paths below with the
# locations where you downloaded/saved the datasets on your computer.


# Example:
# SIFT_DIR = Path("/Users/yourname/Downloads/sift")
SIFT_DIR = Path(
    "/PATH/TO/YOUR/SIFT/FOLDER"
)


# Example:
# WIKI_DIR = Path("/Users/yourname/Downloads/wiki data")
WIKI_DIR = Path(
    "/PATH/TO/YOUR/WIKIPEDIA/FOLDER"
)

# =================================================
# Project paths
# =================================================

# Automatically locate the GitHub repository root.
# Assumes this file is:
# sc4020-ann-search/eval/run_pq_ivf.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Results produced by this script will still be saved
# inside the GitHub project's results/ folder.
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "pq_ivf_results.json"


# =================================================
# Helper functions
# =================================================

def normalize_vectors(vectors):
    """L2-normalize vectors for cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(norms, 1e-12)

    return vectors / norms


def get_index_size(model):
    """Save index temporarily and measure its size."""
    with tempfile.NamedTemporaryFile(
        suffix=".index",
        delete=False,
    ) as tmp:
        temp_path = tmp.name

    try:
        model.save(temp_path)
        size_mb = measure_index_size(temp_path)

    finally:
        Path(temp_path).unlink(missing_ok=True)

    return size_mb


def compute_ground_truth_batched(
    queries,
    base,
    k=10,
    batch_size=50,
):
    """
    Exact L2 ground truth in batches to avoid creating
    one huge query-by-database distance matrix.
    """

    results = []

    base_sq = np.sum(
        base ** 2,
        axis=1,
    )

    total_queries = len(queries)

    for start in range(0, total_queries, batch_size):
        end = min(
            start + batch_size,
            total_queries,
        )

        query_batch = queries[start:end]

        query_sq = np.sum(
            query_batch ** 2,
            axis=1,
            keepdims=True,
        )

        distances = (
            query_sq
            + base_sq
            - 2 * query_batch @ base.T
        )

        ids = np.argpartition(
            distances,
            kth=k - 1,
            axis=1,
        )[:, :k]

        rows = np.arange(
            len(query_batch)
        )[:, None]

        top_distances = distances[
            rows,
            ids,
        ]

        order = np.argsort(
            top_distances,
            axis=1,
        )

        ids = ids[
            rows,
            order,
        ]

        results.append(ids)

        print(
            f"Ground truth: "
            f"{end}/{total_queries} queries"
        )

    return np.vstack(results)


def benchmark_method(
    name,
    model,
    base,
    train,
    queries,
    ground_truth,
    k=10,
):
    """Build and evaluate one ANN method."""

    print(f"\n========== {name} ==========")
    print("Building index...")

    model.build_index(
        base,
        train_vectors=train,
    )

    print("Index built.")

    retrieved = []

    def timed_search(query):
        ids = model.search(
            query,
            k=k,
        )

        retrieved.append(ids)

        return ids

    # Query latency
    _, avg_time_sec = measure_query_time(
        timed_search,
        queries,
    )

    # Recall
    recall = compute_recall_at_k(
        retrieved,
        ground_truth,
        k=k,
    )

    # Index size
    index_size_mb = get_index_size(model)

    result = {
        "recall_at_10": float(recall),
        "avg_query_time_ms": float(
            avg_time_sec * 1000
        ),
        "index_size_mb": float(
            index_size_mb
        ),
    }

    print(
        f"{name} recall@10:       "
        f"{result['recall_at_10']:.4f}"
    )

    print(
        f"{name} avg query time:  "
        f"{result['avg_query_time_ms']:.4f} ms"
    )

    print(
        f"{name} index size:      "
        f"{result['index_size_mb']:.4f} MB"
    )

    return result


# =================================================
# SIFT1M
# =================================================

def run_sift():
    print("\n\n==============================")
    print("SIFT1M BENCHMARK")
    print("==============================")

    base = load_fvecs(
        SIFT_DIR / "sift_base.fvecs"
    )

    queries = load_fvecs(
        SIFT_DIR / "sift_query.fvecs"
    )

    train = load_fvecs(
        SIFT_DIR / "sift_learn.fvecs"
    )

    ground_truth = load_ivecs(
        SIFT_DIR / "sift_groundtruth.ivecs"
    )[:, :10]

    print("Base:        ", base.shape)
    print("Queries:     ", queries.shape)
    print("Train:       ", train.shape)
    print("Ground truth:", ground_truth.shape)

    results = {
        "dataset": {
            "base_vectors": len(base),
            "query_vectors": len(queries),
            "train_vectors": len(train),
            "dimensions": base.shape[1],
        }
    }

    # PQ
    results["PQ"] = {
        "parameters": {
            "m": 8,
            "nbits": 8,
        },
        **benchmark_method(
            "PQ",
            PQ(
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
        ),
    }

    # IVF
    results["IVF"] = {
        "parameters": {
            "nlist": 1024,
            "nprobe": 8,
        },
        **benchmark_method(
            "IVF",
            IVF(
                nlist=1024,
                nprobe=8,
            ),
            base,
            train,
            queries,
            ground_truth,
        ),
    }

    # IVF + PQ
    results["IVF+PQ"] = {
        "parameters": {
            "nlist": 1024,
            "nprobe": 8,
            "m": 8,
            "nbits": 8,
        },
        **benchmark_method(
            "IVF+PQ",
            IVFPQ(
                nlist=1024,
                nprobe=8,
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
        ),
    }

    return results


# =================================================
# Wikipedia
# =================================================

def run_wiki():
    print("\n\n==============================")
    print("WIKIPEDIA BENCHMARK")
    print("==============================")

    base = np.load(
        WIKI_DIR / "wiki_base_embeddings.npy"
    ).astype(np.float32)

    queries = np.load(
        WIKI_DIR / "wiki_query_embeddings.npy"
    ).astype(np.float32)

    train = np.load(
        WIKI_DIR / "wiki_train_embeddings.npy"
    ).astype(np.float32)

    print("Base:   ", base.shape)
    print("Queries:", queries.shape)
    print("Train:  ", train.shape)

    # Wikipedia uses cosine similarity.
    # L2 search on normalized vectors gives the same ranking.
    base = normalize_vectors(base)
    queries = normalize_vectors(queries)
    train = normalize_vectors(train)

    groundtruth_path = (
        WIKI_DIR / "wiki_groundtruth.npy"
    )

    # Reuse ground truth if already generated
    if groundtruth_path.exists():

        print(
            "\nLoading saved Wikipedia "
            "ground truth..."
        )

        ground_truth = np.load(
            groundtruth_path
        )

    else:

        print(
            "\nComputing exact Wikipedia "
            "ground truth..."
        )

        ground_truth = (
            compute_ground_truth_batched(
                queries,
                base,
                k=10,
                batch_size=50,
            )
        )

        np.save(
            groundtruth_path,
            ground_truth,
        )

        print(
            "Ground truth saved to:",
            groundtruth_path,
        )

    print(
        "Ground truth:",
        ground_truth.shape,
    )

    results = {
        "dataset": {
            "base_vectors": len(base),
            "query_vectors": len(queries),
            "train_vectors": len(train),
            "dimensions": base.shape[1],
        }
    }

    # PQ
    results["PQ"] = {
        "parameters": {
            "m": 8,
            "nbits": 8,
        },
        **benchmark_method(
            "PQ",
            PQ(
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
        ),
    }

    # IVF
    results["IVF"] = {
        "parameters": {
            "nlist": 512,
            "nprobe": 8,
        },
        **benchmark_method(
            "IVF",
            IVF(
                nlist=512,
                nprobe=8,
            ),
            base,
            train,
            queries,
            ground_truth,
        ),
    }

    # IVF + PQ
    results["IVF+PQ"] = {
        "parameters": {
            "nlist": 512,
            "nprobe": 8,
            "m": 8,
            "nbits": 8,
        },
        **benchmark_method(
            "IVF+PQ",
            IVFPQ(
                nlist=512,
                nprobe=8,
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
        ),
    }

    return results


# =================================================
# Main
# =================================================

if __name__ == "__main__":

    all_results = {
        "SIFT1M": run_sift(),
        "Wikipedia": run_wiki(),
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=4,
        )

    print("\n\n==============================")
    print("BENCHMARK COMPLETE")
    print("==============================")

    print(
        "Results saved to:",
        OUTPUT_PATH,
    )

    print(
        "\nNote: query timing includes a very "
        "small amount of Python overhead from "
        "storing returned neighbour IDs. The "
        "same evaluation structure is used for "
        "PQ, IVF, and IVF+PQ."
    )
