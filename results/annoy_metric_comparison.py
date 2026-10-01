"""Controlled ablation: does switching Annoy's metric (euclidean -> angular)
on the SAME SIFT1M data reproduce the recall/speed gap seen on Wikipedia?

Builds two Annoy indexes over the identical SIFT1M base/query vectors
(128-dim, n_trees=50) -- one with metric="euclidean", one with
metric="angular" -- and sweeps the same search_k values used in the main
Annoy benchmark. Recall is scored against the same (Euclidean) SIFT1M
ground truth in both cases, so dimensionality and the "true" neighbours
are held constant and metric is the only thing that changes.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.fvecs_loader import load_fvecs, load_ivecs  # noqa: E402
from methods.annoy_method import K, benchmark_dataset  # noqa: E402


def main() -> None:
    sift_dir = os.path.join(PROJECT_ROOT, "data", "sift")
    sift_base = load_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    sift_query = load_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    sift_gt = load_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))[:, :K]

    all_results = {}
    for metric in ["euclidean", "angular"]:
        per_search_k = benchmark_dataset(
            f"sift1m_{metric}", sift_base, sift_query, sift_gt, metric=metric
        )
        for metrics in per_search_k.values():
            metrics["metric"] = metric
        all_results[metric] = per_search_k

    results_path = os.path.join(PROJECT_ROOT, "results", "annoy_sift1m_metric_comparison.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    search_ks = sorted({int(sk) for per in all_results.values() for sk in per})
    header = (
        f"{'search_k':>9} | {'euclidean recall':>17} | {'euclidean ms':>13} | "
        f"{'angular recall':>15} | {'angular ms':>11}"
    )
    print("\n=== SIFT1M metric comparison (128-dim, n_trees=50) ===")
    print(header)
    print("-" * len(header))
    for search_k in search_ks:
        e = all_results["euclidean"][str(search_k)]
        a = all_results["angular"][str(search_k)]
        print(
            f"{search_k:>9} | {e['recall_at_10']:>17.4f} | {e['avg_query_time_ms']:>13.3f} | "
            f"{a['recall_at_10']:>15.4f} | {a['avg_query_time_ms']:>11.3f}"
        )


if __name__ == "__main__":
    main()
