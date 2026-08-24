"""DeepFool robustness comparison: standard vs PGD-AT for supported architectures."""
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from attack_config import DEFAULT_STEPS
from comparison_common import EPSILONS, MODEL_STYLE, attack_input_config
from config import (
    DATA_DIR,
    adversarial_checkpoints,
    deepfool_standard_vs_at_figures_dir,
    deepfool_standard_vs_at_output_dir,
    model_checkpoints,
    resnet50_pgd_at_checkpoint_key,
)
from data_utils import build_dataframes, resolve_data_root
from fgsm_attack import arrays_from_generator
from model_deepfool_comparison import (
    ATTACK_NAME,
    evaluate_model,
    load_json_rows,
    pct,
    rows_for_model,
    write_progress,
)
from train_resnet50 import RANDOM_STATE, configure_devices, make_test_generator

ARCH_CONFIG: Dict[str, dict] = {
    "resnet50": {
        "standard_name": "ResNet50",
        "at_name": "ResNet50-AT",
        "file_prefix": "deepfool_resnet50_standard_vs_at",
    },
    "densenet121": {
        "standard_name": "DenseNet121",
        "at_name": "DenseNet121-AT",
        "file_prefix": "deepfool_densenet121_standard_vs_at",
    },
}


def resolve_at_checkpoint(arch: str, at_checkpoint: Optional[Path]) -> Path:
    if at_checkpoint is not None:
        return at_checkpoint
    checkpoints = adversarial_checkpoints()
    if arch == "resnet50":
        return checkpoints.get(
            resnet50_pgd_at_checkpoint_key(DEFAULT_STEPS),
            checkpoints["ResNet50-AT-steps10"],
        )
    return checkpoints["DenseNet121-AT"]


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    standard_name: str,
    at_name: str,
):
    eps_values = [0.0] + EPSILONS
    eps_labels = ["0\n(clean)"] + [f"{int(round(e * 255))}/255" for e in EPSILONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in (standard_name, at_name):
        group = df[df["model"] == model_name]
        if group.empty:
            continue
        style = MODEL_STYLE.get(model_name, {"color": "#333333", "marker": "o"})
        clean = group[group["attack"] == "clean"].iloc[0]
        attacked = group[group["attack"] == ATTACK_NAME].sort_values("epsilon")
        ys = [clean[metric]] + attacked[metric].tolist()
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
    ax.set_xlabel("Perturbation budget ε (L_inf cap)")
    ax.set_ylabel("AUC" if metric == "adv_auc" else metric.capitalize())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{standard_name} Standard vs PGD-AT — DeepFool "
        f"{metric.replace('_', ' ').title()} vs ε"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_comparison_outputs(
    all_rows: list,
    output_dir: Path,
    standard_name: str,
    at_name: str,
    file_prefix: str,
    model_key: str,
):
    figures_dir = deepfool_standard_vs_at_figures_dir(model=model_key)
    figures_dir.mkdir(parents=True, exist_ok=True)

    slim_rows = [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in all_rows]
    df = pd.DataFrame(slim_rows)

    csv_path = output_dir / f"{file_prefix}.csv"
    json_path = output_dir / f"{file_prefix}.json"
    md_path = output_dir / f"{file_prefix}.md"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2)

    eps_list = ", ".join(f"{int(round(e * 255))}/255" for e in EPSILONS)
    lines = [
        f"# {standard_name} Standard vs PGD-AT — DeepFool Comparison",
        "",
        f"Attack: **DeepFool** (L_inf capped, max_iter=15, overshoot=0.02) | ε ∈ {{{eps_list}}}",
        "",
    ]
    for model_name in (standard_name, at_name):
        group = df[df["model"] == model_name]
        if group.empty:
            continue
        clean = group[group["attack"] == "clean"].iloc[0]
        attacked = group[group["attack"] == ATTACK_NAME].sort_values("epsilon")
        lines.extend(
            [
                f"## {model_name}",
                "",
                f"Clean — Accuracy: **{pct(clean['accuracy'])}**, "
                f"Recall: **{pct(clean['recall'])}**, Precision: **{pct(clean['precision'])}**",
                "",
                "| ε | Accuracy | Recall | Precision | ASR (flip) | Adv AUC |",
                "|---|----------|--------|-----------|-----|",
            ]
        )
        for _, row in attacked.iterrows():
            lines.append(
                f"| {row['epsilon_label']} | {pct(row['accuracy'])} | {pct(row['recall'])} | "
                f"{pct(row['precision'])} | {pct(row['attack_success_rate'])} | {row.get('adv_auc', 0):.4f} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    for metric in ("accuracy", "recall", "precision", "attack_success_rate", "adv_auc"):
        plot_metric_comparison(
            df,
            metric,
            figures_dir / f"deepfool_{metric}_comparison.png",
            standard_name,
            at_name,
        )

    return csv_path, json_path, md_path, figures_dir


def run_comparison(args):
    arch_cfg = ARCH_CONFIG[args.arch]
    standard_name = arch_cfg["standard_name"]
    at_name = arch_cfg["at_name"]
    file_prefix = arch_cfg["file_prefix"]

    if args.output_dir is None:
        args.output_dir = deepfool_standard_vs_at_output_dir(model=args.arch)

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    gpus = configure_devices()
    if gpus:
        print(f"DeepFool standard vs AT ({standard_name}): running on {gpus[0].name}", flush=True)

    data_root = resolve_data_root(args.data_dir, skip_download=True)
    _, _, test_df = build_dataframes(data_root)
    y_true = test_df["label"].values

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "deepfool_model_comparison_progress.json"
    progress_rows = load_json_rows(progress_path) if args.resume else []

    at_path = resolve_at_checkpoint(args.arch, args.at_checkpoint)
    if not at_path.is_file():
        raise FileNotFoundError(f"Missing PGD-AT checkpoint: {at_path}")

    standard_path = model_checkpoints()[standard_name]
    input_cfg = attack_input_config(standard_name)
    test_gen = make_test_generator(test_df, rescale_input=input_cfg["rescale_input"])
    images, _ = arrays_from_generator(test_gen)

    evaluate_model(
        standard_name,
        standard_path,
        test_df,
        images,
        y_true,
        args.output_dir,
        max_iter=args.max_iter,
        overshoot=args.overshoot,
        existing_rows=rows_for_model(progress_rows, standard_name) if args.resume else None,
        progress_rows=progress_rows,
    )
    progress_rows = load_json_rows(progress_path)

    evaluate_model(
        at_name,
        at_path,
        test_df,
        images,
        y_true,
        args.output_dir,
        max_iter=args.max_iter,
        overshoot=args.overshoot,
        existing_rows=rows_for_model(progress_rows, at_name) if args.resume else None,
        progress_rows=progress_rows,
    )

    progress_rows = load_json_rows(progress_path)
    all_rows = [
        {**row, "confusion_matrix": row.get("confusion_matrix", [])}
        for row in rows_for_model(progress_rows, standard_name)
    ]
    all_rows.extend(
        {**row, "confusion_matrix": row.get("confusion_matrix", [])}
        for row in rows_for_model(progress_rows, at_name)
    )
    if not all_rows:
        raise RuntimeError(f"No DeepFool rows saved for {standard_name} / {at_name}")

    write_progress(
        args.output_dir,
        [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in all_rows],
    )

    csv_path, json_path, md_path, figures_dir = save_comparison_outputs(
        all_rows, args.output_dir, standard_name, at_name, file_prefix, args.arch
    )
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved markdown: {md_path}")
    print(f"Saved figures to: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepFool robustness: standard vs PGD-AT model."
    )
    parser.add_argument(
        "--arch",
        choices=sorted(ARCH_CONFIG),
        default="resnet50",
        help="Architecture to compare (default: resnet50).",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/adversarial/deepfool_standard_vs_at/[model/]",
    )
    parser.add_argument(
        "--at-checkpoint",
        type=Path,
        default=None,
        help="PGD-AT checkpoint (default: registry entry for --arch).",
    )
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--overshoot", type=float, default=0.02)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from existing comparison CSV.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    arch_cfg = ARCH_CONFIG[args.arch]
    if args.plot_only:
        if args.output_dir is None:
            args.output_dir = deepfool_standard_vs_at_output_dir(model=args.arch)
        csv_path = args.output_dir / f"{arch_cfg['file_prefix']}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No results CSV found: {csv_path}")
        df = pd.read_csv(csv_path)
        json_path = args.output_dir / f"{arch_cfg['file_prefix']}.json"
        all_rows = (
            json.loads(json_path.read_text(encoding="utf-8"))
            if json_path.exists()
            else df.to_dict("records")
        )
        _, _, md_path, figures_dir = save_comparison_outputs(
            all_rows,
            args.output_dir,
            arch_cfg["standard_name"],
            arch_cfg["at_name"],
            arch_cfg["file_prefix"],
            args.arch,
        )
        print(f"Regenerated plots from: {csv_path}")
        print(f"Saved markdown to: {md_path}")
        print(f"Saved figures to: {figures_dir}")
        return

    run_comparison(args)


if __name__ == "__main__":
    main()
