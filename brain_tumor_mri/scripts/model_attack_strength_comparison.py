"""Compare L_inf attack strength (FGSM, BIM, PGD, MIM) at fixed epsilon on Brain MRI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from attack_config import DEFAULT_MU, DEFAULT_STEPS
from comparison_common import attack_input_config, resolve_model_paths
from config import DATA_DIR, MODEL_COMPARISON_DIR, TASK_ROOT
from data_utils import build_dataframes, resolve_data_root
from fgsm_attack import (
    arrays_from_generator,
    compute_metrics,
    generate_adversarial_batches as generate_fgsm_batches,
    generate_bim_batches,
    load_model,
    predict_images,
)
from mifgsm_attack import generate_adversarial_batches as generate_mim_batches
from pgd_attack import generate_adversarial_batches as generate_pgd_batches
from train_resnet50 import RANDOM_STATE, configure_devices, make_test_generator, predict_generator

STRENGTH_EPSILONS = [2 / 255, 4 / 255]
ATTACK_ORDER = ["FGSM", "BIM", "PGD", "MIM"]

BG = "#1e1e1e"
FG = "#e8e8e8"
GRID = "#3a3a3a"
HEADER_BG = "#2d2d2d"


def mean_linf_norm(clean_images: np.ndarray, adv_images: np.ndarray) -> float:
    delta = np.abs(adv_images - clean_images)
    per_sample = delta.reshape(len(delta), -1).max(axis=1)
    return float(np.mean(per_sample))


def run_attack(
    attack: str,
    model,
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    steps: int,
    mu: float,
):
    if attack == "FGSM":
        adv = generate_fgsm_batches(model, images, labels, epsilon)
    elif attack == "BIM":
        adv = generate_bim_batches(model, images, labels, epsilon, steps=steps)
    elif attack == "PGD":
        adv = generate_pgd_batches(model, images, labels, epsilon, steps=steps)
    elif attack == "MIM":
        adv = generate_mim_batches(model, images, labels, epsilon, steps=steps, mu=mu)
    else:
        raise ValueError(f"Unknown attack: {attack}")
    return adv


def load_progress(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(path: Path, rows: list[dict]):
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def evaluate_attack_strength(
    model_name: str,
    model_path: Path,
    test_df,
    epsilons: list[float],
    steps: int,
    mu: float,
    progress_path: Path,
    resume: bool,
) -> list[dict]:
    input_cfg = attack_input_config(model_name)
    test_gen = make_test_generator(test_df, rescale_input=input_cfg["rescale_input"])
    images, labels = arrays_from_generator(test_gen)
    y_true = test_df["label"].values

    model = load_model(model_path)
    clean_probs = predict_generator(
        model,
        make_test_generator(test_df, rescale_input=input_cfg["rescale_input"]),
    )
    clean_metrics = compute_metrics(y_true, clean_probs, epsilon=0.0, attack_name="clean")
    clean_acc = clean_metrics["accuracy"]

    rows = []
    done_keys = set()
    if resume:
        for row in load_progress(progress_path):
            if row.get("model") == model_name:
                rows.append(row)
                done_keys.add((row["attack"], row["epsilon_label"]))

    for epsilon in epsilons:
        eps_label = f"{int(round(epsilon * 255))}/255"
        for attack in ATTACK_ORDER:
            key = (attack, eps_label)
            if key in done_keys:
                print(f"  skip {model_name} | {attack} | eps={eps_label} (resume)", flush=True)
                continue
            print(
                f"  {model_name} | {attack} | eps={eps_label} | steps={steps if attack != 'FGSM' else 1}",
                flush=True,
            )
            adv_images = run_attack(attack, model, images, labels, epsilon, steps, mu)
            adv_probs = predict_images(model, adv_images)
            metrics = compute_metrics(
                y_true, adv_probs, epsilon, attack_name=attack, clean_probs=clean_probs
            )
            row = {
                "task": "Brain MRI",
                "model": model_name,
                "attack": attack,
                "epsilon": epsilon,
                "epsilon_label": eps_label,
                "steps": 1 if attack == "FGSM" else steps,
                "robust_acc": metrics["accuracy"],
                "accuracy_drop": clean_acc - metrics["accuracy"],
                "asr": metrics["attack_success_rate"],
                "sensitivity": metrics["recall"],
                "precision": metrics["precision"],
                "adv_auc": metrics["adv_auc"],
                "mean_linf": mean_linf_norm(images, adv_images),
                "clean_acc": clean_acc,
            }
            rows.append(row)
            progress_rows = load_progress(progress_path)
            progress_rows = [
                r
                for r in progress_rows
                if not (r.get("model") == model_name and r.get("attack") == attack and r.get("epsilon_label") == eps_label)
            ]
            progress_rows.append(row)
            save_progress(progress_path, progress_rows)
    return rows


def write_markdown(df: pd.DataFrame, output_path: Path, steps: int):
    lines = [
        "# Brain MRI — Attack Strength Comparison (L∞)",
        "",
        f"Iterative attacks use **steps={steps}**, α=ε/steps. MIM uses μ={DEFAULT_MU}.",
        "ε ∈ {2/255, 4/255}. Higher ASR / accuracy drop / lower robust acc ⇒ stronger attack.",
        "",
    ]
    for (model_name, eps_label), group in df.groupby(["model", "epsilon_label"]):
        clean_acc = group["clean_acc"].iloc[0]
        lines.extend(
            [
                f"## {model_name} @ ε={eps_label} (clean acc={clean_acc:.2%})",
                "",
                "| Attack | Robust Acc | Acc Drop | ASR | Sensitivity | Mean L∞ |",
                "|--------|------------|----------|-----|-------------|---------|",
            ]
        )
        for _, row in group.sort_values("attack", key=lambda s: s.map({a: i for i, a in enumerate(ATTACK_ORDER)})).iterrows():
            lines.append(
                f"| {row['attack']} | {row['robust_acc']:.2%} | {row['accuracy_drop']:.2%} | "
                f"{row['asr']:.2%} | {row['sensitivity']:.2%} | {row['mean_linf']:.6f} |"
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def render_table_png(df: pd.DataFrame, output_path: Path, model_name: str, eps_label: str, clean_acc: float):
    sub = df[(df["model"] == model_name) & (df["epsilon_label"] == eps_label)].copy()
    sub = sub.set_index("attack").reindex(ATTACK_ORDER).reset_index()

    headers = ["Attack", "Robust Acc", "Acc Drop", "ASR", "Sensitivity", "Mean L∞"]
    table_rows = []
    for _, row in sub.iterrows():
        table_rows.append(
            [
                row["attack"],
                f"{row['robust_acc'] * 100:.2f}",
                f"{row['accuracy_drop'] * 100:.2f}",
                f"{row['asr'] * 100:.2f}",
                f"{row['sensitivity'] * 100:.2f}",
                f"{row['mean_linf']:.4f}",
            ]
        )

    data = [headers] + table_rows
    fig, ax = plt.subplots(figsize=(11, 2.6))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(
        f"Brain MRI Attack Strength — {model_name} @ ε={eps_label} (clean={clean_acc * 100:.2f}%)",
        color=FG,
        fontsize=12,
        pad=14,
    )
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=FG, weight="bold")
        else:
            cell.set_facecolor(BG)
            cell.set_text_props(color=FG)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare FGSM/BIM/PGD/MIM attack strength at ε=2/255 and 4/255."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_COMPARISON_DIR)
    parser.add_argument("--models", nargs="+", default=["ResNet50"])
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument(
        "--paper-tables-dir",
        type=Path,
        default=TASK_ROOT.parent / "paper_tables" / "brain_tumor_mri",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()

    data_root = resolve_data_root(args.data_dir, skip_download=True)
    _, _, test_df = build_dataframes(data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_tables_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "attack_strength_comparison_progress.json"

    model_paths = resolve_model_paths(args.models, models_explicit=True)
    all_rows = load_progress(progress_path) if args.resume else []
    for model_name, model_path in model_paths.items():
        print(f"\n{'=' * 60}\nAttack strength: {model_name}\n{'=' * 60}", flush=True)
        model_rows = evaluate_attack_strength(
            model_name,
            model_path,
            test_df,
            STRENGTH_EPSILONS,
            args.steps,
            args.mu,
            progress_path,
            args.resume,
        )
        if not args.resume:
            all_rows.extend(model_rows)
        else:
            all_rows = load_progress(progress_path)

    if not all_rows:
        raise RuntimeError("No attack strength rows produced.")

    df = pd.DataFrame(all_rows)
    csv_path = args.output_dir / "attack_strength_comparison.csv"
    json_path = args.output_dir / "attack_strength_comparison.json"
    md_path = args.output_dir / "attack_strength_comparison.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    write_markdown(df, md_path, args.steps)

    for model_name in df["model"].unique():
        for eps_label in df["epsilon_label"].unique():
            clean_acc = float(
                df[(df["model"] == model_name) & (df["epsilon_label"] == eps_label)]["clean_acc"].iloc[0]
            )
            slug = model_name.lower().replace("-", "_")
            eps_slug = eps_label.replace("/", "_")
            png_path = args.paper_tables_dir / f"table_attack_strength_{slug}_eps{eps_slug}.png"
            render_table_png(df, png_path, model_name, eps_label, clean_acc)
            print(f"Saved {png_path}", flush=True)

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved markdown: {md_path}")


if __name__ == "__main__":
    main()
