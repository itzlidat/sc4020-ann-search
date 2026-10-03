"""Sensitivity of IVF, PQ and IVF+PQ to k-means initialisation.

IVF learns its coarse centroids and PQ its codebooks with k-means, which
FAISS seeds with a fixed default (1234). This retrains each index with
several seeds at fixed operating points and records recall@10, query time
and (for IVF) list imbalance, so the run-to-run spread caused by
initialisation alone can be compared with the gaps between methods.

Indexes are built exactly as in methods/ivf.py, pq.py and ivf_pq.py (same
FAISS classes, nlist, m, nbits, training sets); only the k-means seed
changes. Single-threaded like the main benchmark. Writes
results/kmeans_seed_sensitivity.json.
"""

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import eval.run_PQ_IVF as pqivf  # noqa: E402

SEEDS = [1234, 1, 2, 3, 4]  # 1234 is FAISS's default, i.e. the main run
K = pqivf.K
# Fixed operating points: IVF at a low and a moderate nprobe, PQ at a middle m,
# IVF+PQ at its largest m.
POINTS = {
    "sift1m": {"ivf_nprobe": [1, 16], "pq_m": 16, "ivfpq_m": 32, "ivfpq_nprobe": 32},
    "wikipedia": {"ivf_nprobe": [1, 16], "pq_m": 48, "ivfpq_m": 96, "ivfpq_nprobe": 32},
}


def search_fn(index):
    return lambda q, k=K: index.search(np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32), k)[1][0].tolist()


def run(dataset, base, train, queries, gt, nlist, pts):
    d = base.shape[1]
    out = {"ivf": {}, "pq": {}, "ivfpq": {}}
    for seed in SEEDS:
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
        ivf.cp.seed = seed
        ivf.train(train)
        ivf.add(base)
        row = {"imbalance_factor": float(ivf.invlists.imbalance_factor())}
        for nprobe in pts["ivf_nprobe"]:
            ivf.nprobe = nprobe
            r, ms = pqivf.evaluate(search_fn(ivf), queries, gt)
            row[f"nprobe{nprobe}"] = {"recall_at_10": r, "avg_query_time_ms": ms}
        out["ivf"][str(seed)] = row

        pq = faiss.IndexPQ(d, pts["pq_m"], pqivf.NBITS)
        pq.pq.cp.seed = seed
        pq.train(train)
        pq.add(base)
        r, ms = pqivf.evaluate(search_fn(pq), queries, gt)
        out["pq"][str(seed)] = {"recall_at_10": r, "avg_query_time_ms": ms}

        ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, pts["ivfpq_m"], pqivf.NBITS)
        ivfpq.cp.seed = seed
        ivfpq.pq.cp.seed = seed
        ivfpq.train(train)
        ivfpq.add(base)
        ivfpq.nprobe = pts["ivfpq_nprobe"]
        r, ms = pqivf.evaluate(search_fn(ivfpq), queries, gt)
        out["ivfpq"][str(seed)] = {"recall_at_10": r, "avg_query_time_ms": ms}

        print(f"  {dataset} seed={seed:<5} IVF nprobe{pts['ivf_nprobe']}: "
              f"{[round(row[f'nprobe{n}']['recall_at_10'], 4) for n in pts['ivf_nprobe']]} "
              f"imbalance={row['imbalance_factor']:.3f} | PQ m={pts['pq_m']}: "
              f"{out['pq'][str(seed)]['recall_at_10']:.4f} | IVF+PQ m={pts['ivfpq_m']}: "
              f"{out['ivfpq'][str(seed)]['recall_at_10']:.4f}", flush=True)
    return {"settings": {"nlist": nlist, **pts, "nbits": pqivf.NBITS}, **out}


def main():
    faiss.omp_set_num_threads(1)
    pqivf.SIFT_DIR = PROJECT_ROOT / "data" / "sift"
    pqivf.WIKI_DIR = PROJECT_ROOT / "data" / "wiki"
    results = {"seeds": SEEDS}

    base, train, queries, gt = pqivf.load_sift()
    results["sift1m"] = run("sift1m", base, train, queries, gt, pqivf.SIFT_CFG["nlist"], POINTS["sift1m"])
    del base, train, queries, gt

    base, train, queries, gt = pqivf.load_wiki()
    results["wikipedia"] = run("wikipedia", base, train, queries, gt, pqivf.WIKI_CFG["nlist"], POINTS["wikipedia"])

    path = PROJECT_ROOT / "results" / "kmeans_seed_sensitivity.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
