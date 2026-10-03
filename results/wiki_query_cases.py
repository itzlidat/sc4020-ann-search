"""Per-query success and failure analysis on Wikipedia for IVF and IVF+PQ.

Builds IVF (nlist 512) and IVF+PQ (m 96) exactly as in the main benchmark
(FAISS default k-means seed), evaluates every query at one operating point
each, and records per query:
  - recall@10 for both methods;
  - s1 / s10: cosine similarity of the true 1st / 10th nearest neighbour
    (how tight the query's neighbourhood is);
  - n_cells: how many IVF cells the true top-10 are spread over;
  - coverage: share of the true top-10 lying in the cells IVF probes. IVF
    computes exact distances inside probed cells, so its recall equals this.
It also picks one success and one failure example by fixed rules and stores
their sentences. Writes results/wiki_query_cases.json. Recall only; no timing.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import eval.run_PQ_IVF as pqivf  # noqa: E402

K = pqivf.K
IVF_NPROBE = 8
IVFPQ_M, IVFPQ_NPROBE = 96, 32
N_SHOW = 3  # neighbours shown per example


def per_query_recall(retrieved, gt):
    return np.array([len(set(r[:K]) & set(g[:K])) / K for r, g in zip(retrieved, gt)])


def main():
    faiss.omp_set_num_threads(1)
    pqivf.WIKI_DIR = PROJECT_ROOT / "data" / "wiki"
    base, train, queries, gt = pqivf.load_wiki()
    d, nlist = base.shape[1], pqivf.WIKI_CFG["nlist"]

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist, faiss.METRIC_L2)
    ivf.train(train)
    ivf.add(base)
    ivf.nprobe = IVF_NPROBE
    ivf_ids = ivf.search(queries, K)[1]

    ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, IVFPQ_M, pqivf.NBITS)
    ivfpq.train(train)
    ivfpq.add(base)
    ivfpq.nprobe = IVFPQ_NPROBE
    ivfpq_ids = ivfpq.search(queries, K)[1]

    rec_ivf = per_query_recall(ivf_ids, gt)
    rec_ivfpq = per_query_recall(ivfpq_ids, gt)

    # Neighbourhood tightness and how the true neighbours fall into IVF cells.
    sims = np.einsum("qd,qkd->qk", queries, base[gt])          # cosine (unit vectors)
    s1, s10 = sims[:, 0], sims[:, K - 1]
    nb_cell = ivf.quantizer.search(base[gt.ravel()], 1)[1].reshape(gt.shape)
    probed = ivf.quantizer.search(queries, IVF_NPROBE)[1]
    n_cells = np.array([len(set(row)) for row in nb_cell])
    coverage = np.array([np.isin(nb_cell[i], probed[i]).mean() for i in range(len(queries))])

    # Fixed selection rules. Success: perfect recall for both methods, tightest
    # neighbourhood (highest s10). Failure: lowest IVF+PQ recall, ties broken by
    # lowest s10.
    both = np.where((rec_ivf == 1.0) & (rec_ivfpq == 1.0))[0]
    success = int(both[np.argmax(s10[both])])
    worst = np.where(rec_ivfpq == rec_ivfpq.min())[0]
    failure = int(worst[np.argmin(s10[worst])])

    with open(PROJECT_ROOT / "data" / "wiki" / "wiki_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    def example(i):
        def listing(ids):
            return [{"id": int(j), "cosine": float(queries[i] @ base[j]), "sentence": sentences["base"][j]}
                    for j in ids[:N_SHOW]]
        return {
            "query_index": i, "query": sentences["query"][i],
            "recall_ivf": float(rec_ivf[i]), "recall_ivfpq": float(rec_ivfpq[i]),
            "s1": float(s1[i]), "s10": float(s10[i]), "n_cells": int(n_cells[i]),
            "coverage": float(coverage[i]),
            "true_top": listing(gt[i]), "ivfpq_top": listing(ivfpq_ids[i]),
        }

    # Mean recall by quartile of s10 (loosest to tightest neighbourhoods).
    edges = np.quantile(s10, [0, 0.25, 0.5, 0.75, 1])
    q_idx = np.clip(np.searchsorted(edges, s10, side="right") - 1, 0, 3)
    quartiles = [{"s10_range": [float(edges[b]), float(edges[b + 1])],
                  "mean_recall_ivf": float(rec_ivf[q_idx == b].mean()),
                  "mean_recall_ivfpq": float(rec_ivfpq[q_idx == b].mean()),
                  "mean_n_cells": float(n_cells[q_idx == b].mean()),
                  "n": int((q_idx == b).sum())} for b in range(4)]

    out = {
        "settings": {"nlist": nlist, "ivf_nprobe": IVF_NPROBE, "ivfpq_m": IVFPQ_M,
                     "ivfpq_nprobe": IVFPQ_NPROBE, "nbits": pqivf.NBITS},
        "mean_recall_ivf": float(rec_ivf.mean()), "mean_recall_ivfpq": float(rec_ivfpq.mean()),
        "ivf_recall_equals_coverage": bool(np.allclose(rec_ivf, coverage)),
        "per_query": {"recall_ivf": rec_ivf.tolist(), "recall_ivfpq": rec_ivfpq.tolist(),
                      "s1": s1.tolist(), "s10": s10.tolist(), "n_cells": n_cells.tolist(),
                      "coverage": coverage.tolist()},
        "s10_quartiles": quartiles,
        "examples": {"success": example(success), "failure": example(failure)},
    }
    path = PROJECT_ROOT / "results" / "wiki_query_cases.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"mean recall IVF {out['mean_recall_ivf']:.4f}, IVF+PQ {out['mean_recall_ivfpq']:.4f}; "
          f"IVF recall == coverage: {out['ivf_recall_equals_coverage']}")
    for qt in quartiles:
        print("  s10 {:.3f}-{:.3f}: IVF {:.3f} IVF+PQ {:.3f} cells {:.1f}".format(
            *qt["s10_range"], qt["mean_recall_ivf"], qt["mean_recall_ivfpq"], qt["mean_n_cells"]))
    print("success", success, "failure", failure, "saved", path)


if __name__ == "__main__":
    main()
