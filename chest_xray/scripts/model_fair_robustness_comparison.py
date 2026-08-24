"""Fair cross-model robustness comparison: Adv AUC, F1, and clean-normalized relative drop."""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_comparison_common import EPSILONS, MODEL_STYLE, models_title


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_attack_df(path: Path, attack_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results: {path}")
    df = pd.read_csv(path)
    df = df[df["attack"].isin(["clean", attack_name])].copy()
    df["f1"] = df.apply(
        lambda row: f1_score(float(row["precision"]), float(row["recall"])), axis=1
    )
    return df


def enrich_with_relative_metrics(df: pd.DataFrame, attack_name: str) -> pd.DataFrame:
    """Add retention and relative drop vs each model's clean baseline."""
    metrics = ("accuracy", "adv_auc", "f1")
    rows = []
    for model_name, group in df.groupby("model"):
        clean = group[group["attack"] == "clean"].iloc[0]
        attacked = group[group["attack"] == attack_name].sort_values("epsilon")
        for _, row in attacked.iterrows():
            out = row.to_dict()
            for metric in metrics:
                clean_val = float(clean[metric]) if pd.notna(clean.get(metric)) else np.nan
                adv_val = float(row[metric]) if pd.notna(row.get(metric)) else np.nan
                if pd.notna(clean_val) and clean_val > 0 and pd.notna(adv_val):
                    out[f"{metric}_retention"] = adv_val / clean_val
                    out[f"{metric}_rel_drop"] = (clean_val - adv_val) / clean_val
                else:
                    out[f"{metric}_retention"] = np.nan
                    out[f"{metric}_rel_drop"] = np.nan
                out[f"clean_{metric}"] = clean_val
            rows.append(out)
    return pd.DataFrame(rows)


def plot_metric_curves(
    df: pd.DataFrame,
    attack_name: str,
    metric: str,
    output_path: Path,
    ylabel: str,
):
    if df[df["attack"] == attack_name][metric].notna().sum() == 0:
        print(f"Skipping {output_path.name}: no {metric} values for {attack_name}")
        return

    eps_values = [0.0] + EPSILONS
    eps_labels = ["0\n(clean)"] + [f"{int(round(e * 255))}/255" for e in EPSILONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in df.groupby("model"):
        style = MODEL_STYLE[model_name]
        clean = group[group["attack"] == "clean"].iloc[0]
        attacked = group[group["attack"] == attack_name].sort_values("epsilon")
        ys = [clean[metric]] + attacked[metric].tolist()
        if any(pd.isna(y) for y in ys):
            continue
        ax.plot(
            eps_values,
            ys,
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            label=f"{model_name} (clean={clean[metric]:.1%})",
        )

    ax.set_xticks(eps_values)
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("Perturbation budget ε")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{models_title(df)} — {attack_name} {ylabel} vs ε")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_relative_drop(
    rel_df: pd.DataFrame,
    attack_name: str,
    metric: str,
    output_path: Path,
):
    if rel_df[f"{metric}_rel_drop"].notna().sum() == 0:
        print(f"Skipping {output_path.name}: no {metric} relative-drop values for {attack_name}")
        return

    eps_values = EPSILONS
    eps_labels = [f"{int(round(e * 255))}/255" for e in EPSILONS]
    metric_label = {"accuracy": "Accuracy", "adv_auc": "Adv AUC", "f1": "F1"}[metric]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in rel_df.groupby("model"):
        style = MODEL_STYLE[model_name]
        group = group.sort_values("epsilon")
        ys = group[f"{metric}_rel_drop"].tolist()
        if any(pd.isna(y) for y in ys):
            continue
        ax.plot(
            eps_values,
            ys,
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            label=model_name,
        )

    ax.set_xticks(eps_values)
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("Perturbation budget ε")
    ax.set_ylabel(f"Relative drop in {metric_label}")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{models_title(rel_df)} — {attack_name} normalized {metric_label} drop vs ε\n"
        f"(drop = (clean − adv) / clean)"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_combined_relative_drop(rel_df: pd.DataFrame, attack_name: str, output_path: Path):
    """One panel per metric, all models overlaid."""
    metrics = ("accuracy", "adv_auc", "f1")
    labels = ("Accuracy", "Adv AUC", "F1")
    available = [
        (metric, label)
        for metric, label in zip(metrics, labels)
        if rel_df[f"{metric}_rel_drop"].notna().sum() > 0
    ]
    if not available:
        print(f"Skipping {output_path.name}: no relative-drop values for {attack_name}")
        return

    eps_values = EPSILONS
    eps_labels = [f"{int(round(e * 255))}/255" for e in EPSILONS]

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4.5), sharey=True)
    if len(available) == 1:
        axes = [axes]
    for ax, (metric, label) in zip(axes, available):
        for model_name, group in rel_df.groupby("model"):
            style = MODEL_STYLE[model_name]
            group = group.sort_values("epsilon")
            if group[f"{metric}_rel_drop"].isna().all():
                continue
            ax.plot(
                eps_values,
                group[f"{metric}_rel_drop"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                label=model_name,
            )
        ax.set_xticks(eps_values)
        ax.set_xticklabels(eps_labels, fontsize=8)
        ax.set_xlabel("ε")
        ax.set_title(label)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Relative drop (clean-normalized)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        f"{models_title(rel_df)} — {attack_name} fair comparison (higher drop = less robust)",
        y=1.12,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(attack_sections: List[Tuple[str, pd.DataFrame]], output_path: Path):
    lines = [
        "# Fair Cross-Model Robustness Comparison",
        "",
        "Metrics are normalized **per model** against its own clean baseline:",
        "",
        "- **F1** = 2·P·R / (P+R)",
        "- **Adv AUC** = ROC-AUC on adversarial probabilities",
        "- **Relative drop** = (clean − adv) / clean  →  0 = no degradation, 1 = total collapse",
        "",
        "Lower relative drop at the same ε ⇒ more robust **after accounting for clean performance**.",
        "",
    ]

    for attack_name, rel_df in attack_sections:
        lines.extend([f"## {attack_name}", ""])
        for model_name in rel_df["model"].unique():
            group = rel_df[rel_df["model"] == model_name].sort_values("epsilon")
            clean_acc = group["clean_accuracy"].iloc[0]
            clean_auc = group["clean_adv_auc"].iloc[0]
            clean_f1 = group["clean_f1"].iloc[0]
            lines.extend(
                [
                    f"### {model_name}",
                    "",
                    f"Clean — Acc: **{pct(clean_acc)}**, Adv AUC: **{clean_auc:.4f}**, F1: **{pct(clean_f1)}**",
                    "",
                    "| ε | Adv AUC | F1 | Acc rel. drop | AUC rel. drop | F1 rel. drop |",
                    "|---|---------|-----|---------------|---------------|--------------|",
                ]
            )
            for _, row in group.iterrows():
                adv_auc_cell = (
                    f"{row['adv_auc']:.4f}"
                    if pd.notna(row.get("adv_auc"))
                    else "—"
                )
                auc_drop_cell = (
                    pct(row["adv_auc_rel_drop"])
                    if pd.notna(row.get("adv_auc_rel_drop"))
                    else "—"
                )
                lines.append(
                    f"| {row['epsilon_label']} | {adv_auc_cell} | {pct(row['f1'])} | "
                    f"{pct(row['accuracy_rel_drop'])} | {auc_drop_cell} | "
                    f"{pct(row['f1_rel_drop'])} |"
                )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def process_attack(
    input_dir: Path,
    figures_dir: Path,
    attack_key: str,
    attack_name: str,
) -> pd.DataFrame:
    csv_path = input_dir / f"{attack_key}_model_comparison.csv"
    df = load_attack_df(csv_path, attack_name)
    rel_df = enrich_with_relative_metrics(df, attack_name)

    plot_metric_curves(
        df, attack_name, "adv_auc", figures_dir / f"fair_{attack_key}_adv_auc.png", "Adv AUC"
    )
    plot_metric_curves(df, attack_name, "f1", figures_dir / f"fair_{attack_key}_f1.png", "F1 Score")

    for metric in ("accuracy", "adv_auc", "f1"):
        plot_relative_drop(
            rel_df,
            attack_name,
            metric,
            figures_dir / f"fair_{attack_key}_{metric}_rel_drop.png",
        )

    plot_combined_relative_drop(
        rel_df, attack_name, figures_dir / f"fair_{attack_key}_combined_rel_drop.png"
    )

    rel_df["attack"] = attack_name
    return rel_df


def main():
    parser = argparse.ArgumentParser(description="Fair cross-model robustness comparison.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "model_comparison",
    )
    args = parser.parse_args()

    figures_dir = args.input_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rel_fgsm = process_attack(args.input_dir, figures_dir, "fgsm", "FGSM")
    rel_pgd = process_attack(args.input_dir, figures_dir, "pgd", "PGD")
    rel_deepfool = process_attack(args.input_dir, figures_dir, "deepfool", "DeepFool")

    combined = pd.concat([rel_fgsm, rel_pgd, rel_deepfool], ignore_index=True)
    csv_path = args.input_dir / "fair_robustness_comparison.csv"
    combined.to_csv(csv_path, index=False)

    summary = {
        "FGSM": rel_fgsm.drop(columns=["confusion_matrix"], errors="ignore").to_dict("records"),
        "PGD": rel_pgd.drop(columns=["confusion_matrix"], errors="ignore").to_dict("records"),
        "DeepFool": rel_deepfool.drop(columns=["confusion_matrix"], errors="ignore").to_dict("records"),
    }
    json_path = args.input_dir / "fair_robustness_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    md_path = args.input_dir / "fair_robustness_comparison.md"
    write_markdown(
        [("FGSM", rel_fgsm), ("PGD", rel_pgd), ("DeepFool", rel_deepfool)],
        md_path,
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved markdown: {md_path}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
