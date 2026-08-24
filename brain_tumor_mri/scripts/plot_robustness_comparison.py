"""Overlay ResNet50 robustness curves for all six attacks (brain tumor MRI)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attack_config import EPSILONS
from config import MODEL_COMPARISON_DIR, OUTPUT_DIR

ATTACK_SPECS = [
    {"key": "fgsm", "label": "FGSM", "color": "#1f77b4", "marker": "o"},
    {"key": "bim", "label": "BIM", "color": "#ff7f0e", "marker": "s"},
    {"key": "pgd", "label": "PGD", "color": "#2ca02c", "marker": "^"},
    {"key": "mifgsm", "label": "MI-FGSM", "color": "#9467bd", "marker": "D"},
    {"key": "deepfool", "label": "DeepFool", "color": "#d62728", "marker": "v"},
    {"key": "cw", "label": "C&W", "color": "#8c564b", "marker": "P"},
]

ADAPTED_ATTACK_SPECS = [
    {"key": "deepfool", "label": "DeepFool", "color": "#d62728", "marker": "v"},
    {"key": "cw", "label": "C&W", "color": "#8c564b", "marker": "P"},
]

METRICS = ("accuracy", "recall", "precision", "attack_success_rate")
METRIC_TITLES = {
    "accuracy": "Accuracy",
    "recall": "Recall",
    "precision": "Precision",
    "attack_success_rate": "Attack Success Rate",
}

MODEL_COMPARISON_ATTACKS = {"fgsm", "pgd", "deepfool"}


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _filter_resnet50(df: pd.DataFrame) -> pd.DataFrame:
    if "model" in df.columns:
        df = df[df["model"] == "ResNet50"]
    attacked = df[df["attack"] != "clean"].copy()
    return attacked


def load_attack_frame(output_dir: Path, key: str) -> pd.DataFrame | None:
    csv_path = output_dir / f"{key}_comparison.csv"
    if csv_path.exists():
        attacked = _filter_resnet50(pd.read_csv(csv_path))
        if not attacked.empty:
            return attacked

    if key in MODEL_COMPARISON_ATTACKS:
        mc_path = MODEL_COMPARISON_DIR / f"{key}_model_comparison.csv"
        if mc_path.exists():
            attacked = _filter_resnet50(pd.read_csv(mc_path))
            if not attacked.empty:
                return attacked

    metrics_path = output_dir / f"{key}_metrics.json"
    if not metrics_path.exists():
        return None

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    result_key = {
        "fgsm": "fgsm_results",
        "bim": "bim_results",
        "pgd": "pgd_results",
        "mifgsm": "mifgsm_results",
        "deepfool": "deepfool_results",
        "cw": "cw_results",
    }[key]
    rows = payload.get(result_key, [])
    if not rows:
        return None
    return pd.DataFrame(rows)


def load_clean_baseline(output_dir: Path) -> dict | None:
    for key in ("fgsm", "bim", "pgd", "mifgsm", "deepfool", "cw"):
        csv_path = output_dir / f"{key}_comparison.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            clean = df[df["attack"] == "clean"]
            if not clean.empty:
                row = clean.iloc[0]
                return {
                    "accuracy": float(row["accuracy"]),
                    "recall": float(row["recall"]),
                    "precision": float(row["precision"]),
                    "attack_success_rate": float(row["attack_success_rate"]),
                }

        if key in MODEL_COMPARISON_ATTACKS:
            mc_path = MODEL_COMPARISON_DIR / f"{key}_model_comparison.csv"
            if mc_path.exists():
                df = pd.read_csv(mc_path)
                clean = df[(df["model"] == "ResNet50") & (df["attack"] == "clean")]
                if not clean.empty:
                    row = clean.iloc[0]
                    return {
                        "accuracy": float(row["accuracy"]),
                        "recall": float(row["recall"]),
                        "precision": float(row["precision"]),
                        "attack_success_rate": float(row["attack_success_rate"]),
                    }

        metrics_path = output_dir / f"{key}_metrics.json"
        if metrics_path.exists():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            baseline = payload.get("baseline_clean_test")
            if baseline:
                return {
                    "accuracy": float(baseline["accuracy"]),
                    "recall": float(baseline["recall"]),
                    "precision": float(baseline["precision"]),
                    "attack_success_rate": float(baseline["attack_success_rate"]),
                }
    return None


def _epsilon_labels(epsilons: list[float]) -> list[str]:
    return [f"{int(round(e * 255))}/255" for e in epsilons]


def series_for_attack(
    df: pd.DataFrame,
    metric: str,
    baseline: dict | None,
    epsilons: list[float],
):
    """Map each attack to categorical x positions; match rows by epsilon_label."""
    labels = _epsilon_labels(epsilons)
    label_to_row = {}
    for _, row in df.iterrows():
        label = str(row.get("epsilon_label", "")).strip()
        if label in labels:
            label_to_row[label] = row

    xs = list(range(len(labels) + 1))
    if metric == "f1":
        ys = [f1_score(baseline["precision"], baseline["recall"])] if baseline else [np.nan]
        for label in labels:
            row = label_to_row.get(label)
            ys.append(
                f1_score(float(row["precision"]), float(row["recall"]))
                if row is not None
                else np.nan
            )
    else:
        ys = [baseline[metric]] if baseline else [np.nan]
        for label in labels:
            row = label_to_row.get(label)
            ys.append(float(row[metric]) if row is not None else np.nan)
    return xs, ys


def _eps_axis_labels(epsilons: list[float]):
    labels = _epsilon_labels(epsilons)
    xs = list(range(len(labels) + 1))
    xticklabels = ["0\n(clean)"] + labels
    return xs, xticklabels


def _plot_metric_lines(ax, output_dir: Path, metric: str, title: str, epsilons: list[float]):
    baseline = load_clean_baseline(output_dir)
    axis_xs, axis_labels = _eps_axis_labels(epsilons)

    for spec in ATTACK_SPECS:
        df = load_attack_frame(output_dir, spec["key"])
        if df is None:
            continue
        xs, ys = series_for_attack(df, metric, baseline, epsilons)
        ax.plot(
            xs,
            ys,
            color=spec["color"],
            marker=spec["marker"],
            linewidth=2,
            markersize=7,
            label=spec["label"],
        )

    ax.set_xticks(axis_xs)
    ax.set_xticklabels(axis_labels)
    ax.set_xlabel("Perturbation budget ε (L_inf)")
    ax.set_ylabel("F1 Score" if metric == "f1" else METRIC_TITLES.get(metric, metric))
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, loc="best")


def plot_combined(output_dir: Path, figures_dir: Path, epsilons: list[float]):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.ravel(), METRICS):
        _plot_metric_lines(
            ax,
            output_dir,
            metric,
            f"Brain Tumor ResNet50 — {METRIC_TITLES[metric]} vs ε",
            epsilons,
        )
    fig.suptitle(
        "Brain Tumor MRI — ResNet50 Robustness: FGSM / BIM / PGD / MI-FGSM / DeepFool / C&W",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = figures_dir / "robustness_all_attacks_combined.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_accuracy_only(output_dir: Path, figures_dir: Path, epsilons: list[float]):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric_lines(
        ax,
        output_dir,
        "accuracy",
        "Brain Tumor ResNet50 — Accuracy vs ε (All Attacks)",
        epsilons,
    )
    fig.tight_layout()
    out = figures_dir / "robustness_accuracy_all_attacks.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_f1_only(output_dir: Path, figures_dir: Path, epsilons: list[float]):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_metric_lines(
        ax,
        output_dir,
        "f1",
        "Brain Tumor ResNet50 — F1 vs ε (All Attacks)",
        epsilons,
    )
    fig.tight_layout()
    out = figures_dir / "robustness_f1_all_attacks.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def regenerate_all(output_dir: Path, figures_dir: Path | None = None):
    # FGSM/PGD/DeepFool model comparison uses 5 eps; standalone attacks use 7.
    comparison_eps = [i / 255 for i in [1, 2, 3, 4, 8]]
    figures_dir = figures_dir or output_dir / "figures"
    return {
        "combined": plot_combined(output_dir, figures_dir, comparison_eps),
        "accuracy": plot_accuracy_only(output_dir, figures_dir, comparison_eps),
        "f1": plot_f1_only(output_dir, figures_dir, comparison_eps),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Brain tumor ResNet50 six-attack robustness plots.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    paths = regenerate_all(args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
