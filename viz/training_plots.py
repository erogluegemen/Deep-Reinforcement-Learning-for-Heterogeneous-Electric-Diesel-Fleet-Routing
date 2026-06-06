"""
Evaluation result plots for the paper (300 DPI PNG).

Reads data/processed/pomo_results.csv (written by evaluate.py).

Usage:
  python viz/training_plots.py                         # all plots → viz/figures/
  python viz/training_plots.py --results_csv path.csv  # custom results file
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "viz", "figures")

DEPOT_COLORS = {"SAW": "#e74c3c", "IGA": "#2980b9", "CET": "#27ae60"}
VEHICLE_COLORS = {"SCV": "#f39c12", "MCV": "#8e44ad", "LCV": "#16a085", "LCV_BEV": "#2980b9"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})


def load_results(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["pomo_dist_km"] = float(row["pomo_dist_km"])
            row["n_nodes"] = int(row["n_nodes"])
            row["is_electric"] = row["is_electric"] in ("True", "true", "1")
            row["baseline_dist_km"] = float(row["baseline_dist_km"]) if row["baseline_dist_km"] else None
            row["reduction_pct"] = float(row["reduction_pct"]) if row["reduction_pct"] else None
            rows.append(row)
    return rows


def plot_depot_comparison(results, out_dir):
    """Bar chart: mean POMO vs DHL baseline distance per depot."""
    depots = ["SAW", "IGA", "CET"]
    pomo_means, base_means, counts = [], [], []

    for d in depots:
        sub = [r for r in results if r["depot"] == d]
        pomo_means.append(np.mean([r["pomo_dist_km"] for r in sub]))
        bm = [r["baseline_dist_km"] for r in sub if r["baseline_dist_km"]]
        base_means.append(np.mean(bm) if bm else 0)
        counts.append(len(sub))

    x = np.arange(len(depots))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - w / 2, base_means, w, label="DHL Baseline", color="#e67e22", alpha=0.85)
    bars2 = ax.bar(x + w / 2, pomo_means, w, label="POMO (fine-tuned)", color="#2980b9", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={c})" for d, c in zip(depots, counts)])
    ax.set_ylabel("Mean route distance (km)")
    ax.set_title("POMO vs DHL Baseline — by Depot")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "depot_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_reduction_histogram(results, out_dir):
    """Histogram of distance reduction % across all instances."""
    reductions = [r["reduction_pct"] for r in results if r["reduction_pct"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(reductions, bins=40, color="#2980b9", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="No change")
    ax.axvline(np.mean(reductions), color="#e74c3c", linewidth=1.5,
               linestyle="-", label=f"Mean: {np.mean(reductions):.1f}%")
    ax.set_xlabel("Distance reduction vs DHL baseline (%)")
    ax.set_ylabel("Number of instances")
    ax.set_title("Distribution of Route Distance Reduction")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, "reduction_histogram.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_n_nodes_scatter(results, out_dir):
    """Scatter: n_nodes vs POMO distance, coloured by depot."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for depot, color in DEPOT_COLORS.items():
        sub = [r for r in results if r["depot"] == depot]
        if not sub:
            continue
        ax.scatter(
            [r["n_nodes"] for r in sub],
            [r["pomo_dist_km"] for r in sub],
            c=color, alpha=0.5, s=18, label=depot,
        )
    ax.set_xlabel("Number of stops (n)")
    ax.set_ylabel("POMO route distance (km)")
    ax.set_title("Route Distance vs Instance Size")
    ax.legend(title="Depot")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(out_dir, "n_nodes_scatter.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_ev_vs_diesel(results, out_dir):
    """Box plot: POMO dist distribution for EV vs diesel LCV."""
    ev = [r["pomo_dist_km"] for r in results if r["is_electric"]]
    lcv = [r["pomo_dist_km"] for r in results
           if r["vehicle_type"] == "LCV" and not r["is_electric"]]
    mcv = [r["pomo_dist_km"] for r in results if r["vehicle_type"] == "MCV"]
    scv = [r["pomo_dist_km"] for r in results if r["vehicle_type"] == "SCV"]

    groups = [(ev, "LCV_BEV\n(EV)", "#2980b9"),
              (lcv, "LCV\n(diesel)", "#16a085"),
              (mcv, "MCV", "#8e44ad"),
              (scv, "SCV", "#f39c12")]
    groups = [(g, l, c) for g, l, c in groups if g]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot([g for g, _, _ in groups], patch_artist=True,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, (_, _, color) in zip(bp["boxes"], groups):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels([l for _, l, _ in groups])
    ax.set_ylabel("POMO route distance (km)")
    ax.set_title("Route Distance by Vehicle Type")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for i, (g, _, _) in enumerate(groups):
        ax.text(i + 1, ax.get_ylim()[1] * 0.97, f"n={len(g)}", ha="center",
                va="top", fontsize=9, color="gray")

    fig.tight_layout()
    path = os.path.join(out_dir, "ev_vs_diesel.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_summary_table(results, out_dir):
    """Text summary statistics figure for the paper appendix."""
    with_base = [r for r in results if r["reduction_pct"] is not None]
    reductions = [r["reduction_pct"] for r in with_base]
    ev = [r for r in results if r["is_electric"]]
    ev_over = [r for r in ev if r["pomo_dist_km"] > 200]

    lines = [
        ("Instances evaluated", f"{len(results)}"),
        ("POMO mean dist (km)", f"{np.mean([r['pomo_dist_km'] for r in results]):.2f}"),
        ("Baseline mean dist (km)", f"{np.mean([r['baseline_dist_km'] for r in with_base]):.2f}"),
        ("Mean reduction (%)", f"{np.mean(reductions):.2f}"),
        ("Reduction > 0 (better)", f"{sum(1 for x in reductions if x > 0)} / {len(reductions)}"),
        ("EV instances", f"{len(ev)}"),
        ("EV exceeding 200 km range", f"{len(ev_over)} / {len(ev)}"),
    ]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    table = ax.table(
        cellText=[[v] for _, v in lines],
        rowLabels=[k for k, _ in lines],
        colLabels=["Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    ax.set_title("POMO Evaluation Summary", pad=12, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_table.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv",
                        default=os.path.join(ROOT, "data", "processed", "pomo_results.csv"))
    parser.add_argument("--out_dir", default=FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.results_csv):
        print(f"Results CSV not found: {args.results_csv}")
        print("Run: python evaluate.py --mode real  first.")
        return

    results = load_results(args.results_csv)
    print(f"Loaded {len(results)} results from {args.results_csv}")

    plot_depot_comparison(results, args.out_dir)
    plot_reduction_histogram(results, args.out_dir)
    plot_n_nodes_scatter(results, args.out_dir)
    plot_ev_vs_diesel(results, args.out_dir)
    plot_summary_table(results, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
