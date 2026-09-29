"""
Benchmark PQ (ADC and SDC), IVF and IVF+PQ on SIFT1M and Wikipedia.

Run from the repo root:
    python -m eval.run_pq_ivf                 # both datasets
    python -m eval.run_pq_ivf --datasets sift # one dataset

Metrics: recall@10, average query latency (ms), index size (MB), build time (s).

Output (shared schema, same as Annoy/HNSW), one file per method:
    results/pq_results.json
    results/ivf_results.json
    results/ivf_pq_results.json
    {"sift1m": {"<param>": {recall_at_10, avg_query_time_ms,
                            build_time_sec, index_size_mb, ...}}}
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # no blocking plt.show(); we save PNGs instead
import matplotlib.pyplot as plt

# Make `methods`, `eval`, `data` importable even if run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.harness import (  # noqa: E402
    compute_recall_at_k,
    measure_query_time,
    measure_index_size,
)
from methods.pq import PQ  # noqa: E402
from methods.ivf import IVF  # noqa: E402
from methods.ivf_pq import IVFPQ  # noqa: E402
from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402

# =================================================
# Project paths
# =================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset directories are assigned from command-line
# arguments in main().
SIFT_DIR = None
WIKI_DIR = None

# Dataset keys must match what Annoy/HNSW use so the combined plot works.
SIFT_KEY = "sift1m"
WIKI_KEY = "wiki"  # <- change if the Annoy/HNSW files use another name

K = 10

# Sweeps
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]
SIFT_CFG = {"nlist": 1024, "pq_ms": [8, 16, 32]}       # d = 128
WIKI_CFG = {"nlist": 512, "pq_ms": [8, 16, 48, 96]}    # d = 384 (m must divide d)
NBITS = 8


# =================================================
# Helpers
# =================================================

def normalize_vectors(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x / norms


def get_index_size(model):
    """Save index to a temp file and measure it."""
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as tmp:
        path = tmp.name
    try:
        model.save(path)
        return float(measure_index_size(path))
    finally:
        Path(path).unlink(missing_ok=True)


def set_nprobe(model, nprobe):
    """Change nprobe on an already-built index (no rebuild)."""
    done = False
    if hasattr(model, "nprobe"):
        model.nprobe = nprobe
        done = True
    inner = getattr(model, "index", None)
    if inner is not None and hasattr(inner, "nprobe"):
        inner.nprobe = nprobe
        done = True
    if not done:
        raise AttributeError(
            "Could not find nprobe on the model. Expose `nprobe` or "
            "`index.nprobe` in methods/ivf.py and methods/ivf_pq.py."
        )


def build(model, base, train):
    t0 = time.perf_counter()
    model.build_index(base, train_vectors=train)
    return time.perf_counter() - t0


def evaluate(search_fn, queries, ground_truth, k=K):
    """Return (recall@k, avg query ms)."""
    retrieved = []

    def timed_search(q):
        ids = search_fn(q, k=k)
        retrieved.append(ids)
        return ids

    _, avg_sec = measure_query_time(timed_search, queries)

    # If the harness does warm-up calls, keep only the real pass.
    retrieved = retrieved[-len(queries):]

    recall = compute_recall_at_k(retrieved, ground_truth, k=k)
    return float(recall), float(avg_sec * 1000)


def entry(recall, ms, build_sec, size_mb, **params):
    return {
        "recall_at_10": recall,
        "avg_query_time_ms": ms,
        "build_time_sec": float(build_sec),
        "index_size_mb": float(size_mb),
        **params,
    }


def wiki_ground_truth(base, queries, k=K):
    """Exact L2 top-k on normalised vectors (same as cosine ranking).
    Cached on disk. If eval.harness has load_ground_truth, prefer that so
    everyone shares one ground truth."""
    cache = WIKI_DIR / "wiki_groundtruth_l2norm.npy"
    if cache.exists():
        return np.load(cache)[:, :k]
    import faiss
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, gt = index.search(queries, k)
    np.save(cache, gt)
    return gt


# =================================================
# Per-dataset benchmark
# =================================================

def run_dataset(base, train, queries, gt, cfg):
    d = base.shape[1]
    out = {"PQ": {}, "IVF": {}, "IVF+PQ": {}}
    nlist = cfg["nlist"]

    # ---- PQ: build once per m, evaluate ADC and SDC on same index ----
    for m in cfg["pq_ms"]:
        if d % m != 0:
            print(f"skip PQ m={m}: d={d} not divisible")
            continue
        print(f"\n[PQ] m={m}")
        model = PQ(m=m, nbits=NBITS)
        b = build(model, base, train)
        size = get_index_size(model)

        r, ms = evaluate(model.search, queries, gt)
        out["PQ"][f"m{m}_nbits{NBITS}_adc"] = entry(
            r, ms, b, size, m=m, nbits=NBITS, distance_type="asymmetric")
        print(f"  ADC recall={r:.4f} {ms:.4f} ms")

        r, ms = evaluate(model.search_symmetric, queries, gt)
        out["PQ"][f"m{m}_nbits{NBITS}_sdc"] = entry(
            r, ms, b, size, m=m, nbits=NBITS, distance_type="symmetric")
        print(f"  SDC recall={r:.4f} {ms:.4f} ms")

    # ---- IVF: build once, sweep nprobe ----
    print(f"\n[IVF] nlist={nlist}")
    model = IVF(nlist=nlist, nprobe=1)
    b = build(model, base, train)
    size = get_index_size(model)
    for nprobe in NPROBES:
        if nprobe > nlist:
            continue
        set_nprobe(model, nprobe)
        r, ms = evaluate(model.search, queries, gt)
        out["IVF"][f"nlist{nlist}_nprobe{nprobe}"] = entry(
            r, ms, b, size, nlist=nlist, nprobe=nprobe)
        print(f"  nprobe={nprobe:<4} recall={r:.4f} {ms:.4f} ms")

    # ---- IVF+PQ: one build per m, sweep nprobe ----
    for m in cfg["pq_ms"]:
        if d % m != 0:
            continue
        print(f"\n[IVF+PQ] nlist={nlist} m={m}")
        model = IVFPQ(nlist=nlist, nprobe=1, m=m, nbits=NBITS)
        b = build(model, base, train)
        size = get_index_size(model)
        for nprobe in NPROBES:
            if nprobe > nlist:
                continue
            set_nprobe(model, nprobe)
            r, ms = evaluate(model.search, queries, gt)
            out["IVF+PQ"][f"nlist{nlist}_nprobe{nprobe}_m{m}"] = entry(
                r, ms, b, size, nlist=nlist, nprobe=nprobe, m=m, nbits=NBITS)
            print(f"  nprobe={nprobe:<4} recall={r:.4f} {ms:.4f} ms")

    return out


def load_sift():
    """Load SIFT1M from the directory supplied by the user."""

    required_files = [
        "sift_base.fvecs",
        "sift_query.fvecs",
        "sift_learn.fvecs",
        "sift_groundtruth.ivecs",
    ]

    for filename in required_files:
        path = SIFT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"\nCould not find:\n"
                f"    {path}\n\n"
                f"Please check that --sift-dir points to "
                f"your SIFT1M folder."
            )

    base = load_fvecs(SIFT_DIR / "sift_base.fvecs")
    queries = load_fvecs(SIFT_DIR / "sift_query.fvecs")
    train = load_fvecs(SIFT_DIR / "sift_learn.fvecs")
    gt = load_ivecs(
        SIFT_DIR / "sift_groundtruth.ivecs"
    )[:, :K]

    return base, train, queries, gt


def load_wiki():
    """Load Wikipedia embeddings from the directory supplied by the user."""

    required_files = [
        "wiki_base_embeddings.npy",
        "wiki_query_embeddings.npy",
        "wiki_train_embeddings.npy",
    ]

    for filename in required_files:
        path = WIKI_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"\nCould not find:\n"
                f"    {path}\n\n"
                f"Please check that --wiki-dir points to "
                f"your Wikipedia dataset folder."
            )

    def f32(name):
        return np.load(WIKI_DIR / name).astype(np.float32)

    base = normalize_vectors(
        f32("wiki_base_embeddings.npy")
    )

    queries = normalize_vectors(
        f32("wiki_query_embeddings.npy")
    )

    train = normalize_vectors(
        f32("wiki_train_embeddings.npy")
    )

    gt = wiki_ground_truth(base, queries)

    return base, train, queries, gt


# =================================================
# Saving and plotting
# =================================================

FILE_FOR = {"PQ": "pq", "IVF": "ivf", "IVF+PQ": "ivf_pq"}


def save_results(results):
    """results = {dataset_key: {method: {param: entry}}} -> one file/method."""
    for method, stem in FILE_FOR.items():
        payload = {ds: res[method] for ds, res in results.items()}
        path = RESULTS_DIR / f"{stem}_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print("saved", path)


def plot_tradeoff(results):
    """Recall@10 vs latency (log x) per dataset, plus index sizes printed."""
    for ds, res in results.items():
        plt.figure(figsize=(7, 5))
        for method, marker in [("IVF", "o"), ("IVF+PQ", "s"), ("PQ", "^")]:
            pts = sorted(
                (e["avg_query_time_ms"], e["recall_at_10"])
                for e in res[method].values()
            )
            if pts:
                xs, ys = zip(*pts)
                plt.plot(xs, ys, marker=marker, label=method)
        plt.xscale("log")
        plt.xlabel("Average query time (ms, log)")
        plt.ylabel("Recall@10")
        plt.ylim(0, 1.02)
        plt.title(f"Recall vs latency ({ds})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"pq_ivf_recall_vs_latency_{ds}.png", dpi=300)
        plt.close()


# =================================================
# Main
# =================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Benchmark PQ, IVF and IVF+PQ."
    )

    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["sift", "wiki"],
        choices=["sift", "wiki"],
        help="Datasets to benchmark.",
    )

    ap.add_argument(
        "--sift-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "sift",
        help=(
            "Folder containing the SIFT1M files. "
            "Default: data/sift"
        ),
    )

    ap.add_argument(
        "--wiki-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "wiki",
        help=(
            "Folder containing the Wikipedia embedding files. "
            "Default: data/wiki"
        ),
    )

    args = ap.parse_args()

    # Make dataset directories available to the
    # loading functions.
    SIFT_DIR = args.sift_dir.expanduser().resolve()
    WIKI_DIR = args.wiki_dir.expanduser().resolve()

    results = {}

    if "sift" in args.datasets:

        print("\n===== SIFT1M =====")
        print("Loading from:", SIFT_DIR)

        base, train, queries, gt = load_sift()

        print(
            base.shape,
            train.shape,
            queries.shape,
            gt.shape,
        )

        results[SIFT_KEY] = run_dataset(
            base,
            train,
            queries,
            gt,
            SIFT_CFG,
        )

    if "wiki" in args.datasets:

        print("\n===== Wikipedia =====")
        print("Loading from:", WIKI_DIR)

        base, train, queries, gt = load_wiki()

        print(
            base.shape,
            train.shape,
            queries.shape,
            gt.shape,
        )

        results[WIKI_KEY] = run_dataset(
            base,
            train,
            queries,
            gt,
            WIKI_CFG,
        )

    save_results(results)
    plot_tradeoff(results)

    print("\nDone.")
