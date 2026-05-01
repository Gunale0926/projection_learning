from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "bye_backprop_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bye_backprop_mplconfig"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
THEORY_RESULTS = ROOT / "experiments" / "refinement_theory_results" / "results_refinement_theory.json"
DPM_RESULTS = ROOT / "experiments" / "results_dpm.json"
ADAPTER_RESULTS = ROOT / "experiments" / "results_dgm_adaptation_quick.json"
OUT_DIR = ROOT / "experiments" / "publication_figures"

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "black": "#111111",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_vector(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def mean_std(row: dict[str, Any], key: str) -> tuple[float, float]:
    return float(row[f"{key}_mean"]), float(row.get(f"{key}_std", 0.0))


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=COLORS["light_gray"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#374151")
    ax.spines["left"].set_color("#374151")
    ax.spines["bottom"].set_color("#374151")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def plot_bayes_gap(theory: dict[str, Any]) -> None:
    by_delta = theory["known_bayes_gap"]["by_delta"]
    deltas = sorted(by_delta, key=float)
    x = np.array([float(by_delta[d]["theory_gap"]) for d in deltas])
    y = np.array([float(by_delta[d]["repair_vs_refined_slope_mean"]) for d in deltas])
    yerr = np.array([float(by_delta[d]["repair_vs_refined_slope_std"]) for d in deltas])
    bayes = np.array([float(by_delta[d]["bayes_oracle_slope_mean"]) for d in deltas])

    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        markersize=3.8,
        capsize=2.5,
        color=COLORS["blue"],
        linewidth=1.0,
        label="Repair vs. refined",
    )
    ax.plot(x, bayes, "s--", color=COLORS["orange"], markersize=3.2, linewidth=1.0, label="Bayes oracle")
    lim = max(float(x.max()), float(y.max())) * 1.08 + 1e-4
    ax.plot([0.0, lim], [0.0, lim], color=COLORS["black"], linewidth=0.7, alpha=0.75)
    ax.text(0.93 * lim, 0.80 * lim, r"$y=x$", fontsize=7, ha="right", color=COLORS["gray"])
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_xlabel(r"Theoretical gap $I(Y;U\mid Z)$")
    ax.set_ylabel("Estimated regret slope")
    ax.legend(frameon=False, loc="upper left")
    style_axis(ax)
    save_vector(fig, "bayes_gap_slope_vs_theory")


def plot_guarded_curve(theory: dict[str, Any]) -> None:
    by_m = theory["guarded_refinement"]["by_m"]
    m_values = np.array(sorted([int(m) for m in by_m]))
    recall = np.array([float(by_m[str(m)]["true_refinement_recall"]) for m in m_values])
    false_refine = np.array([float(by_m[str(m)]["false_refinement_rate"]) for m in m_values])
    spurious = np.array([float(by_m[str(m)]["spurious_distinction_acceptance"]) for m in m_values])
    repair_acc = np.array([float(by_m[str(m)]["repair_decision_accuracy"]) for m in m_values])

    fig, ax = plt.subplots(figsize=(3.35, 2.45), constrained_layout=True)
    ax.plot(m_values, recall, "o-", color=COLORS["blue"], linewidth=1.1, markersize=3.5, label="True recall")
    ax.plot(m_values, false_refine, "s-", color=COLORS["orange"], linewidth=1.1, markersize=3.2, label="False refine")
    ax.plot(m_values, spurious, "^-", color=COLORS["green"], linewidth=1.1, markersize=3.4, label="Spurious")
    ax.plot(m_values, repair_acc, "D-", color=COLORS["purple"], linewidth=1.1, markersize=3.0, label="Repair acc.")
    ax.set_xscale("log", base=2)
    ax.set_xticks(m_values)
    ax.set_xticklabels([str(m) for m in m_values])
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel("Scoring buffer size")
    ax.set_ylabel("Decision rate")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.48))
    style_axis(ax)
    save_vector(fig, "guarded_sample_size_curve")


def plot_contextual_xor(theory: dict[str, Any]) -> None:
    summary = theory["contextual_xor_ablation"]
    names = [
        "full_dgm_refine_repair_edges",
        "dgm_repair_only",
        "dgm_refine_only_no_repair",
        "no_edge_centroid_growth",
        "knn_cache_same_budget",
        "knn_cache_large_budget",
    ]
    labels = ["Full DGM", "Repair", "Refine", "Centroid", "kNN-8", "kNN-64"]
    x = np.arange(len(names))
    acc = np.array([float(summary[name]["prequential_accuracy_mean"]) for name in names])
    acc_err = np.array([float(summary[name]["prequential_accuracy_std"]) for name in names])
    nll = np.array([float(summary[name]["prequential_nll_mean"]) for name in names])
    purity = np.array([float(summary[name].get("concept_purity_mean", 0.0) or 0.0) for name in names])
    concepts = np.array([float(summary[name]["concepts_mean"]) for name in names])

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=True)
    bar_colors = [COLORS["blue"]] + [COLORS["gray"]] * (len(names) - 1)

    axes[0].bar(x, acc, yerr=acc_err, capsize=2, color=bar_colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel("Prequential accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=28, ha="right")
    style_axis(axes[0])
    add_panel_label(axes[0], "A")

    axes[1].bar(x, nll, color=bar_colors, edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Prequential NLL")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right")
    style_axis(axes[1])
    add_panel_label(axes[1], "B")

    ax2 = axes[2]
    width = 0.38
    ax2.bar(x - width / 2, purity, width=width, color=COLORS["green"], edgecolor="white", linewidth=0.5, label="Purity")
    ax2b = ax2.twinx()
    ax2b.bar(x + width / 2, concepts, width=width, color=COLORS["yellow"], edgecolor="white", linewidth=0.5, label="Concepts")
    ax2.set_ylim(0.0, 1.02)
    ax2b.set_ylim(0.0, max(70.0, float(concepts.max()) * 1.15))
    ax2.set_ylabel("Concept purity")
    ax2b.set_ylabel("Concepts")
    ax2.yaxis.label.set_color(COLORS["green"])
    ax2b.yaxis.label.set_color(COLORS["yellow"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=28, ha="right")
    style_axis(ax2)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_color("#374151")
    ax2b.tick_params(colors="#374151")
    add_panel_label(ax2, "C")
    save_vector(fig, "contextual_xor_ablation")


def plot_runtime_backprop(dpm_results: dict[str, Any], adapter: dict[str, Any]) -> None:
    online_tasks = [
        ("Contextual XOR", dpm_results["backprop_comparison"]),
        ("Sequence map", dpm_results["synthetic_lm_comparison"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.70), constrained_layout=True)

    labels = [task[0] for task in online_tasks]
    x = np.arange(len(labels))
    width = 0.34

    online_nll = np.array([mean_std(task[1]["backprop_mlp_online"], "prequential_log_loss")[0] for task in online_tasks])
    online_nll_err = np.array([mean_std(task[1]["backprop_mlp_online"], "prequential_log_loss")[1] for task in online_tasks])
    dgm_nll = np.array(
        [
            mean_std(online_tasks[0][1]["dpm_online"], "prequential_log_loss")[0],
            mean_std(online_tasks[1][1]["dgm_online"], "prequential_log_loss")[0],
        ]
    )
    dgm_nll_err = np.array(
        [
            mean_std(online_tasks[0][1]["dpm_online"], "prequential_log_loss")[1],
            mean_std(online_tasks[1][1]["dgm_online"], "prequential_log_loss")[1],
        ]
    )

    ax = axes[0]
    ax.bar(x - width / 2, online_nll, width, yerr=online_nll_err, capsize=2, color=COLORS["gray"], label="Online BP")
    ax.bar(x + width / 2, dgm_nll, width, yerr=dgm_nll_err, capsize=2, color=COLORS["blue"], label="DGM memory")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prequential NLL")
    style_axis(ax)
    add_panel_label(ax, "A")

    online_acc = np.array([mean_std(task[1]["backprop_mlp_online"], "test_accuracy")[0] for task in online_tasks])
    online_acc_err = np.array([mean_std(task[1]["backprop_mlp_online"], "test_accuracy")[1] for task in online_tasks])
    dgm_acc = np.array(
        [
            mean_std(online_tasks[0][1]["dpm_online"], "test_accuracy")[0],
            mean_std(online_tasks[1][1]["dgm_online"], "test_accuracy")[0],
        ]
    )
    dgm_acc_err = np.array(
        [
            mean_std(online_tasks[0][1]["dpm_online"], "test_accuracy")[1],
            mean_std(online_tasks[1][1]["dgm_online"], "test_accuracy")[1],
        ]
    )
    offline_acc = np.array([mean_std(task[1]["backprop_mlp_offline"], "test_accuracy")[0] for task in online_tasks])
    offline_acc_err = np.array([mean_std(task[1]["backprop_mlp_offline"], "test_accuracy")[1] for task in online_tasks])

    ax = axes[1]
    w = 0.26
    ax.bar(x - w, online_acc, w, yerr=online_acc_err, capsize=2, color=COLORS["gray"], label="Online BP")
    ax.bar(x, dgm_acc, w, yerr=dgm_acc_err, capsize=2, color=COLORS["blue"], label="DGM memory")
    ax.bar(x + w, offline_acc, w, yerr=offline_acc_err, capsize=2, color=COLORS["orange"], label="Offline BP")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy")
    style_axis(ax)
    add_panel_label(ax, "B")
    fig.legend(
        handles=[
            Patch(facecolor=COLORS["gray"], label="Online BP"),
            Patch(facecolor=COLORS["blue"], label="DGM memory"),
            Patch(facecolor=COLORS["orange"], label="Offline BP"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.42, 1.04),
        ncol=3,
        columnspacing=1.2,
    )

    ax = axes[2]
    method_specs = {
        "online_head": ("1-step BP", COLORS["gray"]),
        "tuned_head": ("200-step BP", COLORS["orange"]),
        "dgm": ("kNN memory", COLORS["green"]),
        "dgm_ridge": ("Ridge memory", COLORS["blue"]),
    }
    dataset_markers = {"MNIST": "o", "CIFAR10": "^"}
    for dataset, marker in dataset_markers.items():
        row = adapter[dataset]["results"]["5_shot"]
        for key, (label, color) in method_specs.items():
            acc_key = f"{key}_acc_mean"
            time_key = f"{key}_seconds_mean" if key != "dgm" else "dgm_adapter_seconds_mean"
            ax.scatter(
                float(row[time_key]),
                float(row[acc_key]),
                marker=marker,
                s=34,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
    for key, (label, color) in method_specs.items():
        ax.scatter([], [], marker="o", s=28, color=color, label=label)
    ax.scatter([], [], marker="o", s=28, color="white", edgecolor=COLORS["black"], label="MNIST")
    ax.scatter([], [], marker="^", s=32, color="white", edgecolor=COLORS["black"], label="CIFAR-10")
    ax.set_xscale("log")
    ax.set_xlim(1.5e-4, 6e-2)
    ax.set_ylim(0.06, 0.68)
    ax.set_xlabel("Adapter time (s, log scale)")
    ax.set_ylabel("5-shot accuracy")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    style_axis(ax, grid_axis="both")
    add_panel_label(ax, "C")

    save_vector(fig, "dgm_vs_backprop")


def main() -> None:
    configure_matplotlib()
    theory = load_json(THEORY_RESULTS)
    dpm_results = load_json(DPM_RESULTS)
    adapter = load_json(ADAPTER_RESULTS)
    plot_bayes_gap(theory)
    plot_guarded_curve(theory)
    plot_contextual_xor(theory)
    plot_runtime_backprop(dpm_results, adapter)
    print(f"wrote vector figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
