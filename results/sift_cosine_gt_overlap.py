"""How far SIFT1M's cosine neighbours differ from its euclidean ground truth.

The Annoy metric ablation (annoy_metric_comparison.py) scores the angular
index against the shipped euclidean ground truth. This counts how many of the
10 x 10,000 true-neighbour slots change when neighbours are ranked by cosine
instead, i.e. the most recall the angular index could lose to the mismatch
alone. Also records the spread of base-vector norms. Writes
results/sift_cosine_gt_overlap.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402

K = 10
CHUNK = 200


def main():
    d = PROJECT_ROOT / "data" / "sift"
    base = load_fvecs(d / "sift_base.fvecs")
    query = load_fvecs(d / "sift_query.fvecs")
    gt = load_ivecs(d / "sift_groundtruth.ivecs")[:, :K]

    norms = np.linalg.norm(base, axis=1)
    base_u = (base / norms[:, None]).astype(np.float32)
    query_u = (query / np.linalg.norm(query, axis=1, keepdims=True)).astype(np.float32)

    shared = 0
    for i in range(0, len(query_u), CHUNK):
        sims = query_u[i:i + CHUNK] @ base_u.T
        top = np.argpartition(-sims, K - 1, axis=1)[:, :K]
        shared += sum(len(set(t) & set(g)) for t, g in zip(top, gt[i:i + CHUNK]))
    slots = gt.size

    out = {
        "slots": int(slots),
        "differing_slots": int(slots - shared),
        "max_recall_loss_points": 100 * (slots - shared) / slots,
        "base_norm": {"mean": float(norms.mean()), "std": float(norms.std()),
                      "p1": float(np.percentile(norms, 1)), "p99": float(np.percentile(norms, 99))},
    }
    print(out)
    with open(PROJECT_ROOT / "results" / "sift_cosine_gt_overlap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
