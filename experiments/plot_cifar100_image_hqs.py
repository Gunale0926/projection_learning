from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("global_frequency", "global\nfreq.", "#9aa0a6"),
    ("online_logistic", "online\nlogistic", "#5b8bd9"),
    ("crossed_logistic", "crossed\nlogistic", "#62a87c"),
    ("dgm", "DGM\nposterior", "#c44e52"),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_metric(summary: dict, model: str, metric: str) -> float:
    return float(summary["aggregate"][model][metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("experiments/cifar100_image_hqs_results/summary.json"))
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("experiments/cifar100_image_hqs_results/posterior_ablation_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figs"))
    args = parser.parse_args()

    summary = load_json(args.summary)
    ablation = load_json(args.ablation)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.55), gridspec_kw={"width_ratios": [1.3, 1.05, 1.05]})
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.25, top=0.86, wspace=0.45)

    ax = axes[0]
    x = np.arange(len(METHODS))
    means = [get_metric(summary, key, "test_heldout_nll_mean") for key, _, _ in METHODS]
    stds = [get_metric(summary, key, "test_heldout_nll_std") for key, _, _ in METHODS]
    labels = [label for _, label, _ in METHODS]
    colors = [color for _, _, color in METHODS]
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.6, capsize=3)
    ax.axhline(np.log(2.0), color="#444444", linestyle="--", linewidth=0.9)
    ax.text(3.15, np.log(2.0) + 0.001, "random", fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("test NLL")
    ax.set_title("A. Non-oracle Image-HQS", loc="left", fontsize=9, fontweight="bold")
    ax.set_ylim(0.648, 0.698)
    ax.tick_params(axis="both", labelsize=8)

    ax = axes[1]
    diag = summary["aggregate"]["diagnostics"]
    diag_labels = ["kNN\nceiling", "soft-route\noracle", "DGM\nposterior"]
    diag_values = [
        float(diag["knn_coarse_posterior_ceiling_test_nll_mean"]),
        float(diag["oracle_soft_route_coarse_posterior_under_dgm_concepts_test_nll_mean"]),
        get_metric(summary, "dgm", "test_heldout_nll_mean"),
    ]
    diag_colors = ["#8d6cab", "#d49a3a", "#c44e52"]
    ax.bar(np.arange(3), diag_values, color=diag_colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(np.arange(3), diag_labels)
    ax.set_title("B. Routing ceiling", loc="left", fontsize=9, fontweight="bold")
    ax.set_ylim(0.648, 0.666)
    ax.tick_params(axis="both", labelsize=8)

    ax = axes[2]
    hit_labels = ["@1", "@3", "@5"]
    hit_values = [
        float(diag["posterior_true_bit_hit_at_1_mean"]),
        float(diag["posterior_true_bit_hit_at_3_mean"]),
        float(diag["posterior_true_bit_hit_at_5_mean"]),
    ]
    random_hits = [1.0 / 20.0, 3.0 / 20.0, 5.0 / 20.0]
    ax.bar(np.arange(3) - 0.18, random_hits, width=0.36, color="#bdbdbd", edgecolor="black", linewidth=0.5, label="random")
    ax.bar(np.arange(3) + 0.18, hit_values, width=0.36, color="#c44e52", edgecolor="black", linewidth=0.5, label="DGM")
    ax.set_xticks(np.arange(3), hit_labels)
    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("true coord. hit rate")
    ax.set_title("C. Coordinate recovery", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.tick_params(axis="both", labelsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)

    pdf_path = args.output_dir / "cifar100_image_hqs_evidence.pdf"
    png_path = args.output_dir / "cifar100_image_hqs_evidence.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
