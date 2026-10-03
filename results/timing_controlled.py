"""Controlled re-timing of the report's headline configurations.

Wikipedia (default): linear scan, LSH "low", IVF (nlist 512, nprobe 16), HNSW
(ef_search 64) and Annoy (search_k 20,000 and 100,000).
SIFT1M (--sift): linear scan, IVF (nlist 1024, nprobe 16), HNSW (ef_search 32)
and Annoy (search_k 5,000).

Each configuration is timed for ROUNDS rounds, interleaved so every method in
a round shares the same machine conditions; the start position rotates each
round. Same single-threaded setup, preprocessing, query set, harness timing
path and Annoy warm-up as the main benchmark. HNSW and Annoy are built once
(single-threaded, as in the main run) and cached, so reruns only time. Each
round also records recall@10 (against the shipped ground truth for SIFT1M,
the linear-scan ids for Wikipedia), the load average and the power source.

    python results/timing_controlled.py [--sift] --build-only   # build/cache
    python results/timing_controlled.py [--sift]                # time

Writes results/timing_controlled.json (Wikipedia) or
results/timing_controlled_sift1m.json.
"""

import os

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"  # README thread setup; must precede the numpy import

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import hnswlib
import numpy as np
from annoy import AnnoyIndex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402
from eval.harness import compute_recall_at_k, measure_query_time  # noqa: E402
from methods.annoy_method import Annoy, N_TREES, SEARCH_K_SWEEP  # noqa: E402
from methods.hnsw import EF_CONSTRUCTION, HNSW, M  # noqa: E402
from methods.ivf import IVF  # noqa: E402
from methods.linear_scan import LinearScan  # noqa: E402
from methods.lsh import RandomProjectionLSH, WIKI_STANDARD_CONFIGS  # noqa: E402

ROUNDS = 5
K = 10
CACHE = Path(tempfile.gettempdir()) / "ann_timing_cache"
SIFT = "--sift" in sys.argv
DS = "sift1m" if SIFT else "wikipedia"
HNSW_SPACE, ANNOY_METRIC = ("l2", "euclidean") if SIFT else ("ip", "angular")


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def load_data():
    """(base, query, train, ground truth or None), preprocessed as in the main run."""
    if SIFT:
        d = PROJECT_ROOT / "data" / "sift"
        return (load_fvecs(d / "sift_base.fvecs"), load_fvecs(d / "sift_query.fvecs"),
                load_fvecs(d / "sift_learn.fvecs"), load_ivecs(d / "sift_groundtruth.ivecs")[:, :K])
    d = PROJECT_ROOT / "data" / "wiki"
    return (unit(np.load(d / "wiki_base_embeddings.npy")), unit(np.load(d / "wiki_query_embeddings.npy")),
            unit(np.load(d / "wiki_train_embeddings.npy")), None)


def load_hnsw(base):
    path = CACHE / f"{DS}_hnsw.bin"
    model = HNSW(space=HNSW_SPACE, dim=base.shape[1], M=M, ef_construction=EF_CONSTRUCTION)
    if path.exists():
        model.index = hnswlib.Index(space=HNSW_SPACE, dim=base.shape[1])
        model.index.load_index(str(path), max_elements=len(base))
    else:
        model.build_index(base)
        model.save(str(path))
    model.index.set_num_threads(1)
    return model


def load_annoy(base):
    path = CACHE / f"{DS}_annoy.ann"
    model = Annoy(num_trees=N_TREES, metric=ANNOY_METRIC)
    if not path.exists():
        model.build_index(base)
        model.save(str(path))  # Annoy re-maps the saved file, as in the main run
    else:
        model.dim = base.shape[1]
        model.index = AnnoyIndex(model.dim, ANNOY_METRIC)
        model.index.load(str(path))
    return model


def power_source():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return out.splitlines()[0].split("'")[1] if "'" in out else out.strip()


def main():
    faiss.omp_set_num_threads(1)
    CACHE.mkdir(exist_ok=True)
    base, query, train, gt = load_data()

    hnsw = load_hnsw(base)
    annoy = load_annoy(base)
    if "--build-only" in sys.argv:
        print("cached indexes in", CACHE)
        return

    linear = LinearScan(True)
    linear.build_index(base)
    ivf = IVF(nlist=1024 if SIFT else 512, nprobe=16)
    ivf.build_index(base, train_vectors=train)
    for q in query:  # same untimed Annoy warm-up as methods/annoy_method.py
        annoy.search(q, k=K, search_k=min(SEARCH_K_SWEEP))

    if SIFT:
        configs = {
            "linear_scan": lambda q: linear.search(q, K),
            "ivf_nprobe16": lambda q: ivf.search(q, K),
            "hnsw_ef32": lambda q: hnsw.search(q, k=K, ef_search=32),
            "annoy_search_k5000": lambda q: annoy.search(q, k=K, search_k=5_000),
        }
    else:
        low = WIKI_STANDARD_CONFIGS["low"]
        lsh = RandomProjectionLSH(low["num_tables"], low["num_hashes"], low["bucket_width"])
        lsh.build_index(base)
        configs = {
            "linear_scan": lambda q: linear.search(q, K),
            "lsh_low": lambda q: lsh.search(q, K),
            "ivf_nprobe16": lambda q: ivf.search(q, K),
            "hnsw_ef64": lambda q: hnsw.search(q, k=K, ef_search=64),
            "annoy_search_k20000": lambda q: annoy.search(q, k=K, search_k=20_000),
            "annoy_search_k100000": lambda q: annoy.search(q, k=K, search_k=100_000),
        }
    names = list(configs)
    rounds = []
    for r in range(ROUNDS):
        order = names[r % len(names):] + names[:r % len(names)]
        ms, ids = {}, {}
        for name in order:
            got = []
            _, avg = measure_query_time(lambda q, f=configs[name]: got.append(f(q)), query)
            ms[name], ids[name] = avg * 1000, got
        truth = gt if gt is not None else ids["linear_scan"]
        recall = {n: compute_recall_at_k(ids[n], truth, k=K) for n in names}
        rounds.append({"order": order, "avg_query_time_ms": ms, "recall_at_10": recall,
                       "loadavg": os.getloadavg(), "power": power_source()})
        print(f"round {r + 1}: " + "  ".join(f"{n} {ms[n]:.3f}" for n in names), flush=True)

    path = PROJECT_ROOT / "results" / ("timing_controlled_sift1m.json" if SIFT else "timing_controlled.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dataset": DS, "rounds": rounds}, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
