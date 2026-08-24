"""Compare multiple architectures under the same PGD attack budgets."""
import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from download_dataset import find_chest_xray_root
from fgsm_attack import (
    arrays_from_generator,
    load_model,
    predict_images,
)
from model_comparison_common import (
    DEFAULT_MODEL_NAMES,
    EPSILONS,
    MODEL_STYLE,
    attack_input_config,
    models_title,
    resolve_model_paths,
)
from pgd_attack import compute_metrics, generate_adversarial_batches
from train_resnet50 import (
    RANDOM_STATE,
    build_dataframes,
    configure_devices,
    make_test_generator,
    predict_generator,
)

ATTACK_NAME = "PGD"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def row_for_model(model_name: str, metrics: dict) -> dict:
    row = {
        "model": model_name,
        "attack": metrics["attack"],
        "epsilon": metrics["epsilon"],
        "epsilon_label": metrics["epsilon_label"],
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "attack_success_rate": metrics["attack_success_rate"],
        "adv_auc": metrics.get("adv_auc"),
        "confusion_matrix": metrics["confusion_matrix"],
    }
    if metrics.get("steps") is not None:
        row["steps"] = metrics["steps"]
    return row


def plot_metric_comparison(df: pd.DataFrame, metric: str, output_path: Path, steps: int):
    eps_values = [0.0] + EPSILONS
    eps_labels = ["0\n(clean)"] + [f"{int(round(e * 255))}/255" for e in EPSILONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, group in df.groupby("model"):
        style = MODEL_STYLE[model_name]
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
    ax.set_xlabel("Perturbation budget ε")
    ax.set_ylabel("AUC" if metric == "adv_auc" else metric.capitalize())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{models_title(df)} — PGD (steps={steps}) {metric.replace('_', ' ').title()} vs ε"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(df: pd.DataFrame, output_path: Path, steps: int):
    eps_list = ", ".join(f"{int(round(e * 255))}/255" for e in EPSILONS)
    lines = [
        "# PGD Cross-Model Comparison",
        "",
        f"Attack: **PGD** (random start, steps={steps}, α=ε/steps) | ε ∈ {{{eps_list}}}",
        "",
    ]
    for model_name in df["model"].unique():
        group = df[df["model"] == model_name]
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
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(all_rows, output_dir: Path, steps: int):
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    slim_rows = [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in all_rows]
    df = pd.DataFrame(slim_rows)

    csv_path = output_dir / "pgd_model_comparison.csv"
    json_path = output_dir / "pgd_model_comparison.json"
    md_path = output_dir / "pgd_model_comparison.md"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2)
    write_markdown(df, md_path, steps)

    for metric in ("accuracy", "recall", "precision", "attack_success_rate", "adv_auc"):
        plot_metric_comparison(df, metric, figures_dir / f"pgd_{metric}_comparison.png", steps)

    return csv_path, json_path, md_path, figures_dir


def load_json_rows(path: Path) -> list:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for_model(all_rows: list, model_name: str) -> list:
    return [row for row in all_rows if row.get("model") == model_name]


def write_progress(output_dir: Path, all_rows: list):
    progress_path = output_dir / "pgd_model_comparison_progress.json"
    serializable = [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in all_rows]
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def evaluate_model(
    model_name: str,
    model_path: Path,
    test_df,
    images,
    labels,
    y_true,
    output_dir: Path,
    steps: int,
    attack_batch_size: int,
    existing_rows: Optional[list] = None,
    progress_rows: Optional[list] = None,
):
    print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}", flush=True)
    existing_rows = existing_rows or []
    progress_rows = progress_rows if progress_rows is not None else []
    results = [{**row, "confusion_matrix": row.get("confusion_matrix", [])} for row in existing_rows]

    model = load_model(model_path)
    input_cfg = attack_input_config(model_name)

    clean_probs = predict_generator(
        model,
        make_test_generator(
            test_df,
            rescale_input=input_cfg.get("rescale_input_clean", input_cfg["rescale_input"]),
        ),
    )

    if not any(row.get("attack") == "clean" for row in existing_rows):
        clean_row = row_for_model(
            model_name, compute_metrics(y_true, clean_probs, epsilon=0.0, attack_name="clean")
        )
        results = [clean_row]
        progress_rows = [row for row in progress_rows if row.get("model") != model_name]
        progress_rows.extend([{k: v for k, v in clean_row.items() if k != "confusion_matrix"}])
        write_progress(output_dir, progress_rows)
        print(
            f"Clean — acc={clean_row['accuracy']:.4f}, auc={clean_row['adv_auc']:.4f}, "
            f"recall={clean_row['recall']:.4f}, precision={clean_row['precision']:.4f}",
            flush=True,
        )
    else:
        clean = next(row for row in existing_rows if row.get("attack") == "clean")
        print(
            f"Clean — skipped (resume) acc={clean['accuracy']:.4f}, auc={clean.get('adv_auc', 0):.4f}, "
            f"recall={clean['recall']:.4f}, precision={clean['precision']:.4f}",
            flush=True,
        )

    done_eps = {
        row["epsilon_label"]
        for row in existing_rows
        if row.get("attack") == ATTACK_NAME
    }

    for epsilon in EPSILONS:
        label = f"{int(round(epsilon * 255))}/255"
        if label in done_eps:
            cached = next(row for row in existing_rows if row.get("epsilon_label") == label)
            print(
                f"\nPGD ε={label} — skipped (resume) "
                f"acc={cached['accuracy']:.4f}, asr={cached['attack_success_rate']:.4f}",
                flush=True,
            )
            continue

        model = load_model(model_path)
        print(f"\nPGD ε={label} (steps={steps})", flush=True)
        attack_epsilon = epsilon * input_cfg.get("epsilon_scale", 1.0)
        adv_images = generate_adversarial_batches(
            model,
            images,
            labels,
            attack_epsilon,
            steps=steps,
            batch_size=attack_batch_size,
            clip_min=input_cfg["clip_min"],
            clip_max=input_cfg["clip_max"],
            forward_scale=input_cfg.get("forward_scale", 1.0),
        )
        infer_scale = input_cfg.get("infer_scale", 1.0)
        adv_for_predict = adv_images * infer_scale if infer_scale != 1.0 else adv_images
        adv_probs = predict_images(model, adv_for_predict, batch_size=attack_batch_size)
        metrics = compute_metrics(
            y_true,
            adv_probs,
            epsilon,
            attack_name=ATTACK_NAME,
            steps=steps,
            clean_probs=clean_probs,
        )
        metrics["steps"] = steps
        row = row_for_model(model_name, metrics)
        results.append(row)
        progress_rows = [
            r
            for r in progress_rows
            if not (r.get("model") == model_name and r.get("epsilon_label") == label)
        ]
        progress_rows.append({k: v for k, v in row.items() if k != "confusion_matrix"})
        write_progress(output_dir, progress_rows)
        print(
            f"  acc={row['accuracy']:.4f}, adv_auc={row['adv_auc']:.4f}, "
            f"asr={row['attack_success_rate']:.4f}",
            flush=True,
        )
        tf.keras.backend.clear_session()
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare architectures under PGD at ε=1,2,3,4,8/255."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "model_comparison",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model names (default: all available): {DEFAULT_MODEL_NAMES}",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="PGD iterations per sample (default: 10)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots and markdown from existing pgd_model_comparison.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip clean/PGD steps already present in pgd_model_comparison_progress.json",
    )
    parser.add_argument(
        "--attack-batch-size",
        type=int,
        default=32,
        help="Batch size for PGD generation and adversarial predict (default: 32)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU (useful when DirectML hangs on ConvNeXt PGD)",
    )
    parser.add_argument(
        "--merge-json",
        type=Path,
        default=None,
        help="Optional JSON rows to merge (e.g. completed results for other models)",
    )
    args = parser.parse_args()

    if args.plot_only:
        csv_path = args.output_dir / "pgd_model_comparison.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No results CSV found: {csv_path}")
        df = pd.read_csv(csv_path)
        json_path = args.output_dir / "pgd_model_comparison.json"
        all_rows = (
            json.loads(json_path.read_text(encoding="utf-8"))
            if json_path.exists()
            else df.to_dict("records")
        )
        steps = int(df["steps"].dropna().iloc[0]) if "steps" in df.columns and df["steps"].notna().any() else args.steps
        _, _, md_path, figures_dir = save_outputs(all_rows, args.output_dir, steps)
        print(f"Regenerated plots from: {csv_path}")
        print(f"Saved markdown to: {md_path}")
        print(f"Saved figures to: {figures_dir}")
        return

    project_root = Path(__file__).resolve().parent.parent
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if args.cpu_only:
        tf.config.set_visible_devices([], "GPU")
        print("model_pgd_comparison.py: CPU-only mode enabled", flush=True)
    gpus = configure_devices()
    if gpus:
        print(f"model_pgd_comparison.py: PGD will run on {gpus[0].name}", flush=True)
    else:
        print("WARNING: No GPU detected; PGD will run on CPU.", flush=True)

    data_root = find_chest_xray_root(args.data_dir)
    if data_root is None:
        raise FileNotFoundError("Dataset not found under data/.")

    _, _, test_df = build_dataframes(data_root)
    y_true = test_df["label"].values

    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = args.output_dir / "pgd_model_comparison_progress.json"
    merge_path = args.merge_json or args.output_dir / "pgd_model_comparison.json"
    progress_rows = load_json_rows(progress_path) if args.resume else []
    merged_rows = load_json_rows(merge_path) if args.resume and merge_path.exists() else []
    if args.resume and merged_rows:
        progress_rows = merged_rows + [
            row
            for row in progress_rows
            if row.get("model") not in {r.get("model") for r in merged_rows}
        ]
        write_progress(args.output_dir, progress_rows)

    models_explicit = "--models" in sys.argv
    model_names = args.models if models_explicit else (args.models or DEFAULT_MODEL_NAMES)
    model_paths = resolve_model_paths(model_names, project_root, models_explicit=models_explicit)

    all_rows = []
    for model_name, model_path in model_paths.items():
        existing_rows = rows_for_model(progress_rows, model_name) if args.resume else []
        test_gen = make_test_generator(
            test_df, rescale_input=attack_input_config(model_name)["rescale_input"]
        )
        images, labels = arrays_from_generator(test_gen)
        all_rows.extend(
            evaluate_model(
                model_name,
                model_path,
                test_df,
                images,
                labels,
                y_true,
                args.output_dir,
                steps=args.steps,
                attack_batch_size=args.attack_batch_size,
                existing_rows=existing_rows if args.resume else None,
                progress_rows=progress_rows,
            )
        )
        progress_rows = load_json_rows(progress_path)

    if args.resume:
        models_run = set(model_paths)
        existing_json = load_json_rows(args.output_dir / "pgd_model_comparison.json")
        preserved = [row for row in existing_json if row.get("model") not in models_run]
        if not preserved and merged_rows:
            preserved = [row for row in merged_rows if row.get("model") not in models_run]
        final_rows = preserved + all_rows
    else:
        final_rows = all_rows

    csv_path, json_path, md_path, figures_dir = save_outputs(final_rows, args.output_dir, args.steps)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved markdown: {md_path}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
