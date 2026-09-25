"""
Benchmark PQ (ADC and SDC), IVF, and IVF+PQ
on SIFT1M and Wikipedia embeddings.

Metrics:
- Recall@10
- Average query latency (ms)
- Index size (MB)

PQ distance calculations:
- ADC: Asymmetric Distance Calculation
- SDC: Symmetric Distance Calculation

Results are saved to:
results/pq_ivf_results.json
"""

from pathlib import Path
import json
import tempfile
import matplotlib.pyplot as plt

import numpy as np

from harness import (
    compute_recall_at_k,
    measure_query_time,
    measure_index_size,
)

from PQ import PQ
from IVF import IVF
from PQ_IVF import IVFPQ

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

PROJECT_ROOT = Path.cwd()

# Results produced by this script will still be saved
# inside the GitHub project's results/ folder.
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "pq_ivf_results2.json"


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


def benchmark_method(
    name,
    model,
    base,
    train,
    queries,
    ground_truth,
    k=10,
    search_type="default",
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

        # PQ symmetric distance calculation
        if search_type == "symmetric":
            ids = model.search_symmetric(
                query,
                k=k,
            )

        # Normal search:
        # - PQ asymmetric
        # - IVF
        # - IVF+PQ
        else:
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



def plot_results(all_results):
    """Plot benchmark results for SIFT1M and Wikipedia."""

    methods = ["PQ-ADC","PQ-SDC", "IVF", "IVF+PQ"]
    labels = ["PQ\nADC", "PQ\nSDC","IVF","IVF+PQ"]

    datasets = ["SIFT1M", "Wikipedia"]

    # =================================================
    # Recall@10
    # =================================================

    x = np.arange(len(methods))
    width = 0.35

    sift_recall = [
        all_results["SIFT1M"][method]["recall_at_10"]
        for method in methods
    ]

    wiki_recall = [
        all_results["Wikipedia"][method]["recall_at_10"]
        for method in methods
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        sift_recall,
        width,
        label="SIFT1M",
    )

    plt.bar(
        x + width / 2,
        wiki_recall,
        width,
        label="Wikipedia",
    )

    plt.xlabel("Method")
    plt.ylabel("Recall@10")
    plt.title("Recall@10 Comparison")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    recall_path = RESULTS_DIR / "recall_at_10_2.png"
    plt.savefig(recall_path, dpi=300)
    plt.show()


    # =================================================
    # Average query latency
    # =================================================

    sift_latency = [
        all_results["SIFT1M"][method]["avg_query_time_ms"]
        for method in methods
    ]

    wiki_latency = [
        all_results["Wikipedia"][method]["avg_query_time_ms"]
        for method in methods
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        sift_latency,
        width,
        label="SIFT1M",
    )

    plt.bar(
        x + width / 2,
        wiki_latency,
        width,
        label="Wikipedia",
    )

    plt.xlabel("Method")
    plt.ylabel("Average Query Time (ms)")
    plt.title("Average Query Latency Comparison")
    plt.xticks(x, methods)
    plt.legend()
    plt.tight_layout()

    latency_path = RESULTS_DIR / "query_latency_2.png"
    plt.savefig(latency_path, dpi=300)
    plt.show()


    # =================================================
    # Index size
    # =================================================

    sift_size = [
        all_results["SIFT1M"][method]["index_size_mb"]
        for method in methods
    ]

    wiki_size = [
        all_results["Wikipedia"][method]["index_size_mb"]
        for method in methods
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width / 2,
        sift_size,
        width,
        label="SIFT1M",
    )

    plt.bar(
        x + width / 2,
        wiki_size,
        width,
        label="Wikipedia",
    )

    plt.xlabel("Method")
    plt.ylabel("Index Size (MB)")
    plt.title("Index Size Comparison")
    plt.xticks(x, methods)
    plt.legend()
    plt.tight_layout()

    size_path = RESULTS_DIR / "index_size_2.png"
    plt.savefig(size_path, dpi=300)
    plt.show()

    print("\nGraphs saved to:")
    print(recall_path)
    print(latency_path)
    print(size_path)


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
    
    # PQ - Asymmetric Distance Calculation (ADC)

    results["PQ-ADC"] = {
        "parameters": {
            "m": 8,
            "nbits": 8,
            "distance_type": "asymmetric",
        },
        **benchmark_method(
            "PQ (Asymmetric)",
            PQ(
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
            search_type="default",
        ),
    }

    # PQ - Symmetric Distance Calculation (SDC

    results["PQ-SDC"] = {
        "parameters": {
            "m": 8,
            "nbits": 8,
            "distance_type": "symmetric",
        },
        **benchmark_method(
            "PQ (Symmetric)",
            PQ(
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
            search_type="symmetric",
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

    # Load Wikipedia embeddings
    base = np.load(
        WIKI_DIR / "wiki_base_embeddings.npy"
    ).astype(np.float32)

    queries = np.load(
        WIKI_DIR / "wiki_query_embeddings.npy"
    ).astype(np.float32)

    train = np.load(
        WIKI_DIR / "wiki_train_embeddings.npy"
    ).astype(np.float32)

    # Load precomputed ground truth
    ground_truth = np.load(
        WIKI_DIR / "wiki_groundtruth.npy"
    )[:, :10]

    print("Base:        ", base.shape)
    print("Queries:     ", queries.shape)
    print("Train:       ", train.shape)
    print("Ground truth:", ground_truth.shape)

    # Wikipedia uses cosine similarity.
    # L2 search on normalized vectors gives the same ranking.
    base = normalize_vectors(base)
    queries = normalize_vectors(queries)
    train = normalize_vectors(train)

    results = {
        "dataset": {
            "base_vectors": len(base),
            "query_vectors": len(queries),
            "train_vectors": len(train),
            "dimensions": base.shape[1],
        }
    }

    # PQ - Asymmetric Distance Calculation (ADC)

    results["PQ-ADC"] = {
        "parameters": {
            "m": 8,
            "nbits": 8,
            "distance_type": "asymmetric",
        },
        **benchmark_method(
            "PQ (Asymmetric)",
            PQ(
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
            search_type="default",
        ),
    }
    # PQ - Symmetric Distance Calculation (SD

    results["PQ-SDC"] = {
        "parameters": {
            "m": 8,
            "nbits": 8,
            "distance_type": "symmetric",
        },
        **benchmark_method(
            "PQ (Symmetric)",
            PQ(
                m=8,
                nbits=8,
            ),
            base,
            train,
            queries,
            ground_truth,
            search_type="symmetric",
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

    # Plot benchmark results
    plot_results(all_results)

    print(
        "\nNote: query timing includes a very "
        "small amount of Python overhead from "ok 
        "storing returned neighbour IDs. The "
        "same evaluation structure is used for "
        "PQ, IVF, and IVF+PQ."
    )
