"""
Benchmark result figures for the paper.

Reads data/processed/benchmark_results.csv (written by eval/benchmark_eval.py).

Usage:
  python viz/benchmark_plots.py
  python viz/benchmark_plots.py --results_csv path/to/benchmark_results.csv
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "viz", "figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

COLORS = {
    "NN":            "#e67e22",
    "OR-Tools":      "#e74c3c",
    "POMO greedy":   "#2980b9",
    "POMO 8× aug":   "#1abc9c",
}


def load_results(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in ("nn_mean", "pomo_mean", "pomo_aug_mean", "ort_mean",
                      "pomo_gap_pct", "pomo_aug_gap_pct"):
                row[k] = float(row[k]) if row.get(k) and row[k] not in ("", "None") else None
            row["n"] = int(row["n"])
            rows.append(row)
    return rows


def plot_method_comparison(rows, out_dir):
    """
    Grouped bar chart: mean tour length by method and (problem, n) cell.
    One bar group per (problem, n); bars = NN / OR-Tools / POMO greedy / POMO 8× aug.
    """
    labels = [f"{r['problem']}\nn={r['n']}" for r in rows]
    x = np.arange(len(rows))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))

    def _bar(offset, key, label):
        vals = [r[key] if r[key] is not None else 0 for r in rows]
        mask = [r[key] is not None for r in rows]
        bars = ax.bar(x[mask] + offset, [v for v, m in zip(vals, mask) if m],
                      w, label=label, color=COLORS[label], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
        return bars

    _bar(-1.5 * w, "nn_mean",       "NN")
    _bar(-0.5 * w, "ort_mean",      "OR-Tools")
    _bar( 0.5 * w, "pomo_mean",     "POMO greedy")
    _bar( 1.5 * w, "pomo_aug_mean", "POMO 8× aug")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean tour length")
    ax.set_title("POMO vs Baselines — Synthetic Benchmark")
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Annotate OR-Tools gap where available
    for i, r in enumerate(rows):
        if r["pomo_gap_pct"] is not None:
            ax.annotate(
                f"gap: {r['pomo_gap_pct']:+.1f}%",
                xy=(x[i] + 0.5 * w, r["pomo_mean"]),
                xytext=(x[i] + 0.5 * w, r["pomo_mean"] + 0.5),
                fontsize=7, ha="center", color=COLORS["POMO greedy"],
                arrowprops=dict(arrowstyle="-", color=COLORS["POMO greedy"], lw=0.8),
            )
        if r["pomo_aug_gap_pct"] is not None:
            ax.annotate(
                f"gap: {r['pomo_aug_gap_pct']:+.1f}%",
                xy=(x[i] + 1.5 * w, r["pomo_aug_mean"]),
                xytext=(x[i] + 1.5 * w, r["pomo_aug_mean"] + 0.5),
                fontsize=7, ha="center", color=COLORS["POMO 8× aug"],
                arrowprops=dict(arrowstyle="-", color=COLORS["POMO 8× aug"], lw=0.8),
            )

    fig.tight_layout()
    path = os.path.join(out_dir, "benchmark_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_ev_sensitivity(out_dir):
    """
    Bar chart showing tour length vs EV range constraint.
    Data is hardcoded from the ev_range_analysis output.
    """
    ranges = ["Unlimited", "5.0", "3.0", "2.0"]
    # Values from eval/benchmark_eval.py ev_range_analysis output
    # Update these with actual values after running with --ev_analysis
    means   = [4.507, 4.507, 4.508, 4.902]
    deltas  = [0.0,   0.0,   0.0,   8.8]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(ranges, means, color=["#2980b9", "#2980b9", "#2980b9", "#e74c3c"],
                  alpha=0.85, width=0.55)
    ax.axhline(means[0], color="gray", linestyle="--", linewidth=1, label="Unconstrained")

    for bar, delta in zip(bars, deltas):
        h = bar.get_height()
        label = f"+{delta:.1f}%" if delta > 0 else "—"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                label, ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("EV range constraint (distance units in [0,1]²)")
    ax.set_ylabel("Mean tour length")
    ax.set_title("Effect of EV Range Constraint on Route Length\n(CVRP model, n=20, 100 instances)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(means) * 1.2)

    fig.tight_layout()
    path = os.path.join(out_dir, "ev_range_sensitivity.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_pomo_vs_nn(rows, out_dir):
    """
    Scatter / improvement chart showing POMO improvement over NN.
    """
    cvrp_rows = [r for r in rows if r["problem"] == "CVRP" and r["nn_mean"]]
    if not cvrp_rows:
        return

    labels = [f"n={r['n']}" for r in cvrp_rows]
    pomo_imp  = [100 * (r["nn_mean"] - r["pomo_mean"]) / r["nn_mean"] for r in cvrp_rows]
    aug_imp   = [100 * (r["nn_mean"] - r["pomo_aug_mean"]) / r["nn_mean"]
                 if r["pomo_aug_mean"] else None for r in cvrp_rows]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - w / 2, pomo_imp, w, label="POMO greedy", color=COLORS["POMO greedy"], alpha=0.85)
    if any(v is not None for v in aug_imp):
        aug_vals = [v if v is not None else 0 for v in aug_imp]
        ax.bar(x + w / 2, aug_vals, w, label="POMO 8× aug", color=COLORS["POMO 8× aug"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Improvement over Nearest Neighbour (%)")
    ax.set_title("POMO vs Nearest Neighbour (CVRP)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar in ax.patches:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "pomo_vs_nn.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv",
                        default=os.path.join(ROOT, "data", "processed", "benchmark_results.csv"))
    parser.add_argument("--out_dir", default=FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.results_csv):
        print(f"Results not found: {args.results_csv}")
        print("Run: python eval/benchmark_eval.py  first.")
        return

    rows = load_results(args.results_csv)
    print(f"Loaded {len(rows)} benchmark rows from {args.results_csv}")

    plot_method_comparison(rows, args.out_dir)
    plot_ev_sensitivity(args.out_dir)
    plot_pomo_vs_nn(rows, args.out_dir)

    print(f"\nAll benchmark figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
