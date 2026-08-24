"""Plot Gaussian defense (D0–D3) accuracy and gain vs Standard baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TASK_ROOT

DEFAULT_CSV = OUTPUT_DIR / "defense" / "gaussian" / "gaussian_defense_comparison.csv"
DEFAULT_FIG_DIR = OUTPUT_DIR / "defense" / "gaussian" / "figures"
PAPER_FIG_DIR = TASK_ROOT.parent / "paper_tables" / "brain_tumor_mri"

CONDITION_ORDER = ["D0", "D1", "D2", "D3"]
CONDITION_LABELS = {
    "D0": "D0\nNo defense",
    "D1": "D1\nAdv-only blur",
    "D2": "D2\nAll-input blur",
    "D3": "D3\nPGD-AT + blur",
}
CONDITION_COLORS = {
    "D0": "#7f7f7f",
    "D1": "#aec7e8",
    "D2": "#1f77b4",
    "D3": "#d62728",
}
ATTACKS = ("PGD", "BIM")
EPS_LABELS = ("2/255", "4/255")


def _attack_df(df: pd.DataFrame, attack: str, eps_label: str) -> pd.DataFrame:
    sub = df[(df["attack"] == attack) & (df["epsilon_label"] == eps_label)].copy()
    sub["acc_pct"] = sub["accuracy"] * 100.0
    return sub.set_index("condition").reindex(CONDITION_ORDER)


def _baseline_pct(df: pd.DataFrame, attack: str, eps_label: str) -> float:
    row = df[(df["condition"] == "D0") & (df["attack"] == attack) & (df["epsilon_label"] == eps_label)]
    if row.empty:
        raise ValueError(f"Missing D0 baseline for {attack} @ ε={eps_label}")
    return float(row.iloc[0]["accuracy"]) * 100.0


def plot_accuracy_comparison(df: pd.DataFrame, output_path: Path, steps: int, sigma: float):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    panel_specs = (
        ("PGD", "2/255", axes[0, 0]),
        ("PGD", "4/255", axes[0, 1]),
        ("BIM", "2/255", axes[1, 0]),
        ("BIM", "4/255", axes[1, 1]),
    )

    for attack, eps_label, ax in panel_specs:
        sub = _attack_df(df, attack, eps_label)
        xs = np.arange(len(CONDITION_ORDER))
        accs = sub["acc_pct"].tolist()
        bars = ax.bar(
            xs,
            accs,
            color=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
            edgecolor="#333333",
            linewidth=0.6,
        )
        baseline = _baseline_pct(df, attack, eps_label)
        ax.axhline(baseline, color="#444444", linestyle="--", linewidth=1.2, label="D0 baseline")

        for bar, acc, cond in zip(bars, accs, CONDITION_ORDER):
            delta = acc - baseline
            label = f"{acc:.1f}%"
            if cond != "D0":
                label += f"\n(+{delta:.1f})"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(xs)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Adversarial accuracy (%)")
        ax.set_title(f"{attack} @ ε={eps_label}", fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(
        f"Gaussian defense vs Standard baseline (ResNet50, steps={steps}, σ={sigma})",
        fontsize=13,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_vs_standard(df: pd.DataFrame, output_path: Path, steps: int, sigma: float):
    """Bar chart of accuracy gain (pp) over D0 for PGD and BIM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, attack in zip(axes, ATTACKS):
        gain_matrix = []
        for eps_label in EPS_LABELS:
            baseline = _baseline_pct(df, attack, eps_label)
            sub = _attack_df(df, attack, eps_label)
            gain_matrix.append(sub["acc_pct"] - baseline)
        gain_df = pd.DataFrame(gain_matrix, index=EPS_LABELS, columns=CONDITION_ORDER)

        xs = np.arange(len(EPS_LABELS))
        width = 0.2
        for idx, cond in enumerate(CONDITION_ORDER):
            offsets = xs + (idx - 1.5) * width
            values = gain_df[cond].tolist()
            bars = ax.bar(
                offsets,
                values,
                width=width,
                label=CONDITION_LABELS[cond].replace("\n", " "),
                color=CONDITION_COLORS[cond],
                edgecolor="#333333",
                linewidth=0.6,
            )
            for bar, val in zip(bars, values):
                if cond == "D0":
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"+{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"ε={label}" for label in EPS_LABELS])
        ax.set_ylabel("Gain over D0 (percentage points)")
        ax.set_title(f"{attack} — improvement vs Standard")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(bottom=-2)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Robustness gain over Standard ResNet50 (steps={steps}, σ={sigma})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pgd_bim_side_by_side(df: pd.DataFrame, output_path: Path, steps: int, sigma: float):
    """Single-panel view @ ε=4/255: PGD vs BIM accuracy by condition."""
    eps_label = "4/255"
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(CONDITION_ORDER))
    width = 0.35

    pgd = _attack_df(df, "PGD", eps_label)["acc_pct"].tolist()
    bim = _attack_df(df, "BIM", eps_label)["acc_pct"].tolist()
    d0_pgd, d0_bim = pgd[0], bim[0]

    ax.bar(xs - width / 2, pgd, width, label="PGD", color="#1f77b4", edgecolor="#333333", linewidth=0.6)
    ax.bar(xs + width / 2, bim, width, label="BIM", color="#ff7f0e", edgecolor="#333333", linewidth=0.6)

    for i, (p, b) in enumerate(zip(pgd, bim)):
        if i == 0:
            continue
        ax.text(i - width / 2, p + 1.2, f"+{p - d0_pgd:.1f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 1.2, f"+{b - d0_bim:.1f}", ha="center", fontsize=8)

    ax.axhline(d0_pgd, xmin=0, xmax=0.25, color="#1f77b4", linestyle="--", alpha=0.7)
    ax.axhline(d0_bim, xmin=0, xmax=0.25, color="#ff7f0e", linestyle="--", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], fontsize=9)
    ax.set_ylabel("Adversarial accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"PGD vs BIM @ ε=4/255 (steps={steps}, σ={sigma})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Gaussian defense comparison vs Standard.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--paper-dir", type=Path, default=PAPER_FIG_DIR)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(f"Missing CSV: {args.csv}. Run evaluate_gaussian_defense.py first.")

    df = pd.read_csv(args.csv)
    if "steps" in df.columns and df["steps"].notna().any():
        args.steps = int(df["steps"].dropna().iloc[0])
    if "sigma" in df.columns and df["sigma"].notna().any():
        args.sigma = float(df["sigma"].dropna().iloc[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "gaussian_defense_accuracy_vs_standard.png": plot_accuracy_comparison,
        "gaussian_defense_gain_vs_standard.png": plot_gain_vs_standard,
        "gaussian_defense_pgd_bim_eps4.png": plot_pgd_bim_side_by_side,
    }

    for name, plot_fn in outputs.items():
        out = args.output_dir / name
        plot_fn(df, out, args.steps, args.sigma)
        paper_out = args.paper_dir / name
        plot_fn(df, paper_out, args.steps, args.sigma)
        print(f"Wrote {out}", flush=True)
        print(f"Wrote {paper_out}", flush=True)


if __name__ == "__main__":
    main()
