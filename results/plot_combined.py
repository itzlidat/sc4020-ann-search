"""Combined comparison of all ANN methods across both datasets.

Reads every method's results JSON (each has its own layout), flattens them
into one table, prints a summary, and saves:
    combined_recall_vs_latency_<dataset>.png   (Figure 1, one per dataset)
    combined_index_size.png                    (Figure 2)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["sift1m", "wikipedia"]
FIELDS = ["recall_at_10", "avg_query_time_ms", "build_time_sec", "index_size_mb"]

# method label -> results file. PQ is split into ADC/SDC rows by distance_type.
FILES = {
    "Linear scan": "linear_scan_results.json",
    "LSH": "lsh_results.json",
    "PQ": "pq_results.json",
    "IVF": "ivf_results.json",
    "IVF+PQ": "ivf_pq_results.json",
    "HNSW": "hnsw_results.json",
    "Annoy": "annoy_results.json",
}

# Fixed categorical order from the validated reference palette (never cycled).
# Linear scan is a reference point, drawn in neutral ink rather than a hue.
STYLE = {
    "LSH": ("#2a78d6", "o"),
    "PQ-ADC": ("#eb6834", "s"),
    "PQ-SDC": ("#1baf7a", "D"),
    "IVF": ("#eda100", "^"),
    "IVF+PQ": ("#e87ba4", "v"),
    "HNSW": ("#008300", "P"),
    "Annoy": ("#4a3aa7", "X"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_rows():
    """Flatten every results file into rows of
    (method, dataset, param, recall, ms, build_sec, size_mb)."""
    rows = []
    for method, filename in FILES.items():
        with open(os.path.join(SCRIPT_DIR, filename), encoding="utf-8") as f:
            data = json.load(f)
        for dataset, per_param in data.items():
            if "results" in per_param:          # LSH: {"mode", "results": {...}}
                per_param = per_param["results"]
            if "recall_at_10" in per_param:      # linear scan: no param layer
                per_param = {"exact": per_param}
            for param, entry in per_param.items():
                name = method
                if method == "PQ":
                    name = "PQ-SDC" if entry["distance_type"] == "symmetric" else "PQ-ADC"
                rows.append({"method": name, "dataset": dataset, "param": param,
                             **{k: float(entry[k]) for k in FIELDS}})
    return rows


def pareto(points):
    """Points no faster point beats on recall, sorted by query time."""
    front, best = [], -1.0
    for p in sorted(points, key=lambda r: (r["avg_query_time_ms"], -r["recall_at_10"])):
        if p["recall_at_10"] > best:
            front.append(p)
            best = p["recall_at_10"]
    return front


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_recall_vs_latency(rows, dataset):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    style_axes(ax)
    data = [r for r in rows if r["dataset"] == dataset]

    for method, (color, marker) in STYLE.items():
        pts = [r for r in data if r["method"] == method]
        if not pts:
            continue
        front = pareto(pts)
        others = [p for p in pts if p not in front]
        # Dominated configs as faint markers, so nothing is hidden.
        ax.scatter([p["avg_query_time_ms"] for p in others], [p["recall_at_10"] for p in others],
                   s=18, marker=marker, color=color, alpha=0.25, linewidths=0)
        ax.plot([p["avg_query_time_ms"] for p in front], [p["recall_at_10"] for p in front],
                color=color, marker=marker, markersize=6, linewidth=2, label=method,
                markeredgecolor="#fcfcfb", markeredgewidth=1)

    for r in (r for r in data if r["method"] == "Linear scan"):
        ax.scatter(r["avg_query_time_ms"], r["recall_at_10"], s=140, marker="*",
                   color=INK, zorder=5, label="Linear scan (exact)")

    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Average query time (ms, log scale)", color=INK)
    ax.set_ylabel("Recall@10", color=INK)
    ax.set_title(f"Recall@10 vs query time: {dataset} (single-threaded)",
                 loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.01, "Lines: best configs per method (Pareto frontier). "
            "Faint markers: dominated configs.", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, va="bottom")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK)
    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, f"combined_recall_vs_latency_{dataset}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def plot_index_size(rows):
    # ADC/SDC share one PQ index, so size is reported once as "PQ".
    methods = ["Linear scan", "LSH", "PQ", "IVF", "IVF+PQ", "HNSW", "Annoy"]
    colors = {"sift1m": "#2a78d6", "wikipedia": "#eb6834"}
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    style_axes(ax)
    ax.grid(False, axis="x")
    width = 0.38

    for i, dataset in enumerate(DATASETS):
        for j, method in enumerate(methods):
            sizes = [r["index_size_mb"] for r in rows if r["dataset"] == dataset
                     and r["method"].split("-")[0] == method]
            if not sizes:
                continue
            x = j + (i - 0.5) * (width + 0.02)
            lo, hi = min(sizes), max(sizes)
            ax.bar(x, lo, width, color=colors[dataset],
                   label=dataset if j == 0 else None)
            fmt = lambda v: f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            label, top = fmt(lo), lo
            if hi > lo * 1.05:  # several configs with different sizes: show the span
                ax.plot([x, x], [lo, hi], color=INK, linewidth=1.2)
                ax.plot([x - 0.08, x + 0.08], [hi, hi], color=INK, linewidth=1.2)
                label, top = f"{fmt(lo)}–{fmt(hi)}", hi
            ax.annotate(label, (x, top), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=MUTED)

    ax.set_yscale("log")
    ax.set_xticks(range(len(methods)), methods, color=INK)
    ax.set_ylabel("Index size (MB, log scale)", color=INK)
    ax.set_title("Index size by method", loc="left", color=INK,
                 fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.01, "Bar: smallest config. Whisker: largest config, where configs differ.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    path = os.path.join(SCRIPT_DIR, "combined_index_size.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def print_summary(rows):
    print(f"\n{'dataset':<10} {'method':<12} {'n':>3} {'recall@10 range':>16} "
          f"{'query ms range':>17} {'size MB':>13} {'fastest at recall>=0.9':>26}")
    for dataset in DATASETS:
        for method in ["Linear scan", *STYLE]:
            pts = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            if not pts:
                continue
            rec = [p["recall_at_10"] for p in pts]
            ms = [p["avg_query_time_ms"] for p in pts]
            size = [p["index_size_mb"] for p in pts]
            ok = [p for p in pts if p["recall_at_10"] >= 0.9]
            best = min(ok, key=lambda p: p["avg_query_time_ms"]) if ok else None
            best_txt = (f"{best['avg_query_time_ms']:.3f} ms ({best['param']})"
                        if best else "never reaches 0.9")
            size_txt = (f"{min(size):.1f}" if max(size) <= min(size) * 1.05
                        else f"{min(size):.0f}-{max(size):.0f}")
            print(f"{dataset:<10} {method:<12} {len(pts):>3} "
                  f"{min(rec):>7.3f}-{max(rec):<8.3f} {min(ms):>8.3f}-{max(ms):<8.3f} "
                  f"{size_txt:>13} {best_txt:>26}")
        print()


if __name__ == "__main__":
    rows = load_rows()
    print_summary(rows)
    for dataset in DATASETS:
        plot_recall_vs_latency(rows, dataset)
    plot_index_size(rows)
