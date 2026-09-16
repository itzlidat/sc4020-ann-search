"""Plot Annoy's recall/latency tradeoff curves from results/annoy_results.json.

For each dataset, draws recall@10 vs avg_query_time_ms as a line with
points labeled by the search_k value that produced them.
"""

import json
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(RESULTS_DIR, "annoy_results.json")

plt.style.use("seaborn-v0_8-whitegrid")

DATASET_TITLES = {"sift1m": "SIFT1M", "wikipedia": "Wikipedia"}


def plot_tradeoff(dataset: str, per_search_k: dict) -> str:
    """Plot recall@10 vs avg_query_time_ms for one dataset, labeled by search_k.

    Args:
        dataset: Dataset key (e.g. "sift1m").
        per_search_k: Dict mapping str(search_k) -> metric dict.

    Returns:
        Path the figure was saved to.
    """
    points = sorted(
        ((int(search_k_str), metrics) for search_k_str, metrics in per_search_k.items()),
        key=lambda p: p[0],
    )
    x = [metrics["avg_query_time_ms"] for _, metrics in points]
    y = [metrics["recall_at_10"] for _, metrics in points]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, marker="o", linewidth=2, markersize=8, color="#2563eb")
    for search_k, metrics in points:
        ax.annotate(
            f"search_k={search_k}",
            (metrics["avg_query_time_ms"], metrics["recall_at_10"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=9,
        )

    title = DATASET_TITLES.get(dataset, dataset)
    ax.set_xlabel("Average query time (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title(f"Annoy: Recall@10 vs Query Time — {title}")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f"annoy_{dataset}_tradeoff.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    with open(RESULTS_PATH) as f:
        all_results = json.load(f)

    for dataset, per_search_k in all_results.items():
        out_path = plot_tradeoff(dataset, per_search_k)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
