"""Compare multiple architectures under the same DeepFool attack budgets."""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from deepfool_attack import compute_metrics, generate_deepfool_batches
from download_dataset import find_chest_xray_root
from fgsm_attack import arrays_from_generator, load_model, predict_images
from model_comparison_common import (
    DEFAULT_MODEL_NAMES,
    EPSILONS,
    MODEL_STYLE,
    models_title,
    resolve_model_paths,
    attack_input_config,
)
from train_resnet50 import (
    RANDOM_STATE,
    build_dataframes,
    configure_devices,
    make_test_generator,
    predict_generator,
)

ATTACK_NAME = "DeepFool"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def row_for_model(model_name: str, metrics: dict) -> dict:
    return {
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


def plot_metric_comparison(df: pd.DataFrame, metric: str, output_path: Path):
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
    ax.set_xlabel("Perturbation budget ε (L_inf cap)")
    ax.set_ylabel("AUC" if metric == "adv_auc" else metric.capitalize())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{models_title(df)} — DeepFool {metric.capitalize()} vs ε")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(df: pd.DataFrame, output_path: Path):
    eps_list = ", ".join(f"{int(round(e * 255))}/255" for e in EPSILONS)
    lines = [
        "# DeepFool Cross-Model Comparison",
        "",
        f"Attack: **DeepFool** (L_inf capped) | ε ∈ {{{eps_list}}}",
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
                "| ε | Accuracy | Recall | Precision | ASR | Adv AUC |",
                "|---|----------|--------|-----------|-----|---------|",
            ]
        )
        for _, row in attacked.iterrows():
            lines.append(
                f"| {row['epsilon_label']} | {pct(row['accuracy'])} | {pct(row['recall'])} | "
                f"{pct(row['precision'])} | {pct(row['attack_success_rate'])} | "
                f"{row.get('adv_auc', 0):.4f} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_outputs(all_rows, output_dir: Path):
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    slim_rows = [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in all_rows]
    df = pd.DataFrame(slim_rows)

    csv_path = output_dir / "deepfool_model_comparison.csv"
    json_path = output_dir / "deepfool_model_comparison.json"
    md_path = output_dir / "deepfool_model_comparison.md"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2)
    write_markdown(df, md_path)

    for metric in ("accuracy", "recall", "precision", "attack_success_rate", "adv_auc"):
        plot_metric_comparison(df, metric, figures_dir / f"deepfool_{metric}_comparison.png")

    return csv_path, json_path, md_path, figures_dir


def load_json_rows(path: Path) -> list:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for_model(all_rows: list, model_name: str) -> list:
    return [row for row in all_rows if row.get("model") == model_name]


def write_progress(output_dir: Path, all_rows: list):
    progress_path = output_dir / "deepfool_model_comparison_progress.json"
    serializable = [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in all_rows]
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def row_missing_adv_auc(row: dict) -> bool:
    value = row.get("adv_auc")
    return value is None or (isinstance(value, float) and np.isnan(value))


def evaluate_model(
    model_name: str,
    model_path: Path,
    test_df,
    images,
    y_true,
    output_dir: Path,
    max_iter: int,
    overshoot: float,
    existing_rows: Optional[list] = None,
    progress_rows: Optional[list] = None,
    backfill_adv_auc: bool = False,
):
    print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}", flush=True)
    existing_rows = existing_rows or []
    progress_rows = progress_rows if progress_rows is not None else []
    results = [{**row, "confusion_matrix": row.get("confusion_matrix", [])} for row in existing_rows]

    model = load_model(model_path)

    clean_probs = predict_generator(
        model,
        make_test_generator(
            test_df,
            rescale_input=attack_input_config(model_name)["rescale_input"],
        ),
    )

    clean_existing = next((row for row in existing_rows if row.get("attack") == "clean"), None)
    need_clean = clean_existing is None or (backfill_adv_auc and row_missing_adv_auc(clean_existing))

    if need_clean:
        clean_row = row_for_model(
            model_name, compute_metrics(y_true, clean_probs, epsilon=0.0, attack_name="clean")
        )
        results = [row for row in results if row.get("attack") != "clean"]
        results.insert(0, clean_row)
        progress_rows = [row for row in progress_rows if row.get("model") != model_name]
        progress_rows.extend(
            [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in results]
        )
        write_progress(output_dir, progress_rows)
        print(
            f"Clean — acc={clean_row['accuracy']:.4f}, adv_auc={clean_row['adv_auc']:.4f}, "
            f"recall={clean_row['recall']:.4f}, precision={clean_row['precision']:.4f}",
            flush=True,
        )
    else:
        clean = clean_existing
        print(
            f"Clean — skipped (resume) acc={clean['accuracy']:.4f}, "
            f"adv_auc={clean.get('adv_auc', 0):.4f}, "
            f"recall={clean['recall']:.4f}, precision={clean['precision']:.4f}",
            flush=True,
        )

    done_eps = set()
    for row in existing_rows:
        if row.get("attack") != ATTACK_NAME:
            continue
        if backfill_adv_auc and row_missing_adv_auc(row):
            continue
        done_eps.add(row["epsilon_label"])

    for epsilon in EPSILONS:
        label = f"{int(round(epsilon * 255))}/255"
        if label in done_eps:
            cached = next(row for row in existing_rows if row.get("epsilon_label") == label)
            print(
                f"\nDeepFool ε={label} — skipped (resume) "
                f"acc={cached['accuracy']:.4f}, adv_auc={cached.get('adv_auc', 0):.4f}, "
                f"asr={cached['attack_success_rate']:.4f}",
                flush=True,
            )
            continue

        print(f"\nDeepFool ε={label} (max_iter={max_iter})", flush=True)
        adv_images = generate_deepfool_batches(
            model,
            images,
            epsilon,
            max_iter=max_iter,
            overshoot=overshoot,
        )
        adv_probs = predict_images(model, adv_images)
        row = row_for_model(model_name, compute_metrics(y_true, adv_probs, epsilon, attack_name=ATTACK_NAME))
        results = [r for r in results if not (r.get("epsilon_label") == label and r.get("attack") == ATTACK_NAME)]
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
            f"recall={row['recall']:.4f}, precision={row['precision']:.4f}, "
            f"asr={row['attack_success_rate']:.4f}",
            flush=True,
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare architectures under DeepFool at ε=1,2,3,4,8/255."
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
    parser.add_argument("--max-iter", type=int, default=15)
    parser.add_argument("--overshoot", type=float, default=0.02)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots and markdown from existing deepfool_model_comparison.csv",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip clean/DeepFool steps already present in deepfool_model_comparison_progress.json",
    )
    parser.add_argument(
        "--backfill-adv-auc",
        action="store_true",
        help="Re-run only rows missing adv_auc (uses deepfool_model_comparison.json as resume source)",
    )
    parser.add_argument(
        "--merge-json",
        type=Path,
        default=None,
        help="Optional JSON rows to merge (e.g. completed results for other models)",
    )
    args = parser.parse_args()

    if args.plot_only:
        csv_path = args.output_dir / "deepfool_model_comparison.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No results CSV found: {csv_path}")
        df = pd.read_csv(csv_path)
        json_path = args.output_dir / "deepfool_model_comparison.json"
        all_rows = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else df.to_dict("records")
        _, _, md_path, figures_dir = save_outputs(all_rows, args.output_dir)
        print(f"Regenerated plots from: {csv_path}")
        print(f"Saved markdown to: {md_path}")
        print(f"Saved figures to: {figures_dir}")
        return

    project_root = Path(__file__).resolve().parent.parent
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    gpus = configure_devices()
    if gpus:
        print(f"model_deepfool_comparison.py: DeepFool will run on {gpus[0].name}", flush=True)
    else:
        print("WARNING: No GPU detected; DeepFool will run on CPU.", flush=True)

    data_root = find_chest_xray_root(args.data_dir)
    if data_root is None:
        raise FileNotFoundError("Dataset not found under data/.")

    _, _, test_df = build_dataframes(data_root)
    y_true = test_df["label"].values

    args.output_dir.mkdir(parents=True, exist_ok=True)

    progress_path = args.output_dir / "deepfool_model_comparison_progress.json"
    merge_path = args.merge_json
    progress_rows = load_json_rows(progress_path)
    merged_rows = load_json_rows(merge_path) if merge_path and merge_path.exists() else []

    json_path = args.output_dir / "deepfool_model_comparison.json"
    if args.backfill_adv_auc and json_path.exists():
        progress_rows = load_json_rows(json_path)
    elif args.resume and merged_rows:
        progress_rows = merged_rows + [
            row for row in progress_rows if row.get("model") not in {r.get("model") for r in merged_rows}
        ]
        write_progress(args.output_dir, progress_rows)

    models_explicit = "--models" in sys.argv
    model_names = args.models if models_explicit else (args.models or DEFAULT_MODEL_NAMES)
    model_paths = resolve_model_paths(model_names, project_root, models_explicit=models_explicit)

    all_rows = []
    for model_name, model_path in model_paths.items():
        existing_rows = rows_for_model(progress_rows, model_name) if (args.resume or args.backfill_adv_auc) else []
        test_gen = make_test_generator(
            test_df, rescale_input=attack_input_config(model_name)["rescale_input"]
        )
        images, _ = arrays_from_generator(test_gen)
        all_rows.extend(
            evaluate_model(
                model_name,
                model_path,
                test_df,
                images,
                y_true,
                args.output_dir,
                max_iter=args.max_iter,
                overshoot=args.overshoot,
                existing_rows=existing_rows if (args.resume or args.backfill_adv_auc) else None,
                progress_rows=progress_rows,
                backfill_adv_auc=args.backfill_adv_auc,
            )
        )
        progress_rows = load_json_rows(progress_path)

    if args.resume or args.backfill_adv_auc:
        models_run = set(model_paths)
        existing_json = load_json_rows(args.output_dir / "deepfool_model_comparison.json")
        preserved = [row for row in existing_json if row.get("model") not in models_run]
        if not preserved and merged_rows:
            preserved = [row for row in merged_rows if row.get("model") not in models_run]
        final_rows = preserved + all_rows
    else:
        final_rows = all_rows

    csv_path, json_path, md_path, figures_dir = save_outputs(final_rows, args.output_dir)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved markdown: {md_path}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
