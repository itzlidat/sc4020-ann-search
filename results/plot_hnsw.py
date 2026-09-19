import json
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(SCRIPT_DIR, "hnsw_results.json")

with open(json_path, "r") as f:
    data = json.load(f)

for dataset, ef_dict in data.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    latencies = []
    recalls = []
    ef_values = []
    
    for ef_search, metrics in ef_dict.items():
        ef_values.append(ef_search)
        recalls.append(metrics["recall_at_10"])
        latencies.append(metrics["avg_query_time_ms"])

    ax.plot(latencies, recalls, marker="o", linewidth=2, color="#1f77b4", label="HNSW")
    
    for i, ef in enumerate(ef_values):
        ax.annotate(
            f"ef={ef}",
            (latencies[i], recalls[i]),
            textcoords="offset points",
            xytext=(6, -6),
            ha="left",
            fontsize=8,
        )

    ax.set_title(f"{dataset.upper()} - Recall vs Latency", fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Query Time (ms)", fontsize=10)
    ax.set_ylabel("Recall@10", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    output_path = os.path.join(SCRIPT_DIR, f"hnsw_{dataset}_tradeoff.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to {output_path}")
