"""Evaluate Gaussian smoothing defense (D0–D3) on Brain MRI ResNet50.

Experimental conditions
-----------------------
D0  No defense (Standard ResNet50, inputs unchanged)
D1  Blur adversarial samples only (Standard)
D2  Blur all inputs (Standard)
D3  Blur all inputs (PGD-AT ResNet50, steps=10 checkpoint)

Attacks: PGD + BIM, steps=10, ε ∈ {2/255, 4/255} (default).
Outputs: CSV + paper table10 LaTeX.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from attack_config import DEFAULT_STEPS
from config import DATA_DIR, OUTPUT_DIR, adversarial_checkpoints, model_checkpoints, resnet50_pgd_at_checkpoint_key
from data_utils import build_dataframes, resolve_data_root
from defense_common import apply_gaussian_blur
from fgsm_attack import arrays_from_generator, compute_metrics, generate_bim_batches, load_model, predict_images
from pgd_attack import generate_adversarial_batches as generate_pgd_batches
from train_resnet50 import RANDOM_STATE, configure_devices, make_test_generator

TASK_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = TASK_ROOT.parent / "paper_tables" / "brain_tumor_mri"

DEFAULT_EPSILONS = [2 / 255, 4 / 255]
ATTACK_GENERATORS: Dict[str, Callable] = {
    "PGD": generate_pgd_batches,
    "BIM": generate_bim_batches,
}


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    label: str
    model_variant: str  # "standard" | "pgd_at"
    blur_clean: bool
    blur_adv: bool


CONDITIONS: Dict[str, ConditionSpec] = {
    "D0": ConditionSpec("D0", "No defense", "standard", False, False),
    "D1": ConditionSpec("D1", "Adv-only Gaussian", "standard", False, True),
    "D2": ConditionSpec("D2", "All-input Gaussian", "standard", True, True),
    "D3": ConditionSpec("D3", "PGD-AT + all-input Gaussian", "pgd_at", True, True),
}


def maybe_blur(images: np.ndarray, enabled: bool, sigma: float) -> np.ndarray:
    if not enabled:
        return images
    return apply_gaussian_blur(images, radius=sigma)


def evaluate_clean(model, images: np.ndarray, y_true: np.ndarray, spec: ConditionSpec, sigma: float) -> dict:
    inputs = maybe_blur(images, spec.blur_clean, sigma)
    probs = predict_images(model, inputs)
    metrics = compute_metrics(y_true, probs, 0.0, attack_name="clean")
    metrics.update(
        {
            "condition": spec.key,
            "condition_label": spec.label,
            "model_variant": spec.model_variant,
            "blur_clean": spec.blur_clean,
            "blur_adv": spec.blur_adv,
            "sigma": sigma,
            "steps": DEFAULT_STEPS,
        }
    )
    return metrics


def evaluate_adversarial(
    model,
    images: np.ndarray,
    y_true: np.ndarray,
    adv_images: np.ndarray,
    spec: ConditionSpec,
    sigma: float,
    epsilon: float,
    attack_name: str,
    clean_probs: np.ndarray,
) -> dict:
    model_inputs = maybe_blur(adv_images, spec.blur_adv, sigma)
    probs = predict_images(model, model_inputs)
    metrics = compute_metrics(
        y_true,
        probs,
        epsilon,
        attack_name=attack_name,
        clean_probs=clean_probs,
    )
    metrics.update(
        {
            "condition": spec.key,
            "condition_label": spec.label,
            "model_variant": spec.model_variant,
            "blur_clean": spec.blur_clean,
            "blur_adv": spec.blur_adv,
            "sigma": sigma,
            "steps": DEFAULT_STEPS,
        }
    )
    return metrics


def generate_attack(
    attack_name: str,
    model,
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    steps: int,
) -> np.ndarray:
    generate_fn = ATTACK_GENERATORS[attack_name]
    return generate_fn(model, images, labels, epsilon, steps=steps)


def plot_gaussian_defense(df: pd.DataFrame, output_path: Path, sigma: float, steps: int):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    eps_labels = sorted(df[df["attack"] != "clean"]["epsilon_label"].unique(), key=lambda s: int(s.split("/")[0]))
    xs = list(range(len(eps_labels)))

    for condition, group in df[df["attack"] != "clean"].groupby("condition"):
        sub = group.sort_values("epsilon")
        for attack_name, attack_group in sub.groupby("attack"):
            accs = attack_group["accuracy"].tolist()
            asrs = attack_group["attack_success_rate"].tolist()
            label = f"{condition} · {attack_name}"
            axes[0].plot(xs, accs, marker="o", linewidth=2, label=label)
            axes[1].plot(xs, asrs, marker="s", linewidth=2, label=label)

    for ax, metric, title in zip(
        axes,
        ["Accuracy", "Attack Success Rate"],
        [
            f"Robust accuracy vs ε (Gaussian σ={sigma})",
            f"ASR vs ε (Gaussian σ={sigma})",
        ],
    ):
        ax.set_xticks(xs)
        ax.set_xticklabels(eps_labels)
        ax.set_xlabel("Perturbation budget ε (L∞)")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"ResNet50 Gaussian defense (D0–D3) — PGD/BIM steps={steps}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_table10_tex(df: pd.DataFrame, output_path: Path, sigma: float, steps: int):
    """Build table10: one row per D0–D3, columns for clean + PGD/BIM @ ε=2,4."""
    clean = df[df["attack"] == "clean"].drop_duplicates(subset=["condition"])
    clean_map = {row["condition"]: float(row["accuracy"]) * 100 for _, row in clean.iterrows()}

    def cell(condition: str, attack: str, eps_label: str) -> str:
        row = df[
            (df["condition"] == condition)
            & (df["attack"] == attack)
            & (df["epsilon_label"] == eps_label)
        ]
        if row.empty:
            return "--"
        return f"{float(row.iloc[0]['accuracy']) * 100:5.2f}"

    tex = f"""% Table 10: Gaussian smoothing defense (ResNet50, Brain MRI)
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Gaussian smoothing defense on ResNet50 (Brain MRI). Adversarial accuracy (\\%) under PGD and BIM with steps={steps}; blur radius $\\sigma={sigma}$.}}
  \\label{{tab:gaussian_defense_resnet50}}
  \\begin{{tabular}}{{llrrrrr}}
    \\toprule
    Cond. & Setting & Clean & PGD ($\\epsilon$=2/255) & BIM ($\\epsilon$=2/255) & PGD ($\\epsilon$=4/255) & BIM ($\\epsilon$=4/255) \\\\
    \\midrule
"""
    for key, spec in CONDITIONS.items():
        if key not in clean_map and key != "D3":
            continue
        clean_val = clean_map.get(key)
        clean_cell = f"{clean_val:5.2f}" if clean_val is not None else "--"
        tex += (
            f"    {key} & {spec.label} & {clean_cell} "
            f"& {cell(key, 'PGD', '2/255')} & {cell(key, 'BIM', '2/255')} "
            f"& {cell(key, 'PGD', '4/255')} & {cell(key, 'BIM', '4/255')} \\\\\n"
        )

    tex += f"""    \\bottomrule
  \\end{{tabular}}
  \\vspace{{0.25em}}
  \\begin{{minipage}}{{0.92\\linewidth}}
    \\footnotesize
    D0: no preprocessing. D1: Gaussian blur on adversarial inputs only.
    D2: blur on all inputs (Standard ResNet50).
    D3: blur on all inputs with PGD-AT ResNet50 (trained at $\\epsilon_{{\\train}}=4/255$, steps={steps}).
    PGD uses random start; BIM has no random start; $\\alpha=\\epsilon/\\text{{steps}}$.
  \\end{{minipage}}
\\end{{table}}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")


def run_evaluation(
    standard_model,
    at_model,
    images: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    sigma: float,
    epsilons: Iterable[float],
    steps: int,
    conditions: Iterable[str],
) -> List[dict]:
    rows: List[dict] = []
    clean_probs_cache: Dict[str, np.ndarray] = {}
    attack_model_map = {"standard": standard_model, "pgd_at": at_model}

    for cond_key in conditions:
        spec = CONDITIONS[cond_key]
        model = attack_model_map[spec.model_variant]
        if model is None:
            print(f"Skipping {cond_key}: missing {spec.model_variant} checkpoint.", flush=True)
            continue

        clean_metrics = evaluate_clean(model, images, y_true, spec, sigma)
        rows.append(clean_metrics)
        clean_probs_cache[cond_key] = predict_images(
            model, maybe_blur(images, spec.blur_clean, sigma)
        )
        print(
            f"{cond_key} clean: acc={clean_metrics['accuracy']:.3f}",
            flush=True,
        )

    for attack_name in ("PGD", "BIM"):
        for epsilon in epsilons:
            eps_label = f"{int(round(epsilon * 255))}/255"
            adv_standard = generate_attack(attack_name, standard_model, images, labels, epsilon, steps)
            adv_at = None
            if at_model is not None and "D3" in conditions:
                adv_at = generate_attack(attack_name, at_model, images, labels, epsilon, steps)

            for cond_key in conditions:
                spec = CONDITIONS[cond_key]
                model = attack_model_map[spec.model_variant]
                if model is None:
                    continue
                adv_images = adv_at if spec.model_variant == "pgd_at" else adv_standard
                if adv_images is None:
                    continue
                metrics = evaluate_adversarial(
                    model,
                    images,
                    y_true,
                    adv_images,
                    spec,
                    sigma,
                    epsilon,
                    attack_name,
                    clean_probs_cache[cond_key],
                )
                rows.append(metrics)
                print(
                    f"{cond_key} {attack_name} ε={eps_label}: "
                    f"acc={metrics['accuracy']:.3f} asr={metrics['attack_success_rate']:.3f}",
                    flush=True,
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gaussian smoothing defense (D0–D3).")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "defense" / "gaussian",
    )
    parser.add_argument(
        "--standard-model-path",
        type=Path,
        default=model_checkpoints()["ResNet50"],
    )
    parser.add_argument(
        "--pgd-at-model-path",
        type=Path,
        default=None,
        help="PGD-AT checkpoint for D3 (default: registry ResNet50-AT-steps10).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=DEFAULT_EPSILONS,
        help="L_inf budgets as floats, e.g. 0.007843 0.015686 for 2/255 and 4/255.",
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur radius (PIL radius).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--paper-table-path",
        type=Path,
        default=PAPER_DIR / "table10_gaussian_defense_resnet50.tex",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    at_path = args.pgd_at_model_path
    if at_path is None:
        at_key = resnet50_pgd_at_checkpoint_key(args.steps)
        at_path = adversarial_checkpoints().get(at_key)
    need_at = "D3" in args.conditions
    if need_at and (at_path is None or not at_path.is_file()):
        raise FileNotFoundError(
            f"PGD-AT checkpoint required for D3 but not found: {at_path}. "
            "Train with train_adversarial.py or pass --pgd-at-model-path."
        )

    if not args.standard_model_path.is_file():
        raise FileNotFoundError(f"Missing Standard ResNet50 checkpoint: {args.standard_model_path}")

    data_root = resolve_data_root(args.data_dir, skip_download=args.skip_download)
    _, _, test_df = build_dataframes(data_root)
    test_gen = make_test_generator(test_df)
    images, labels = arrays_from_generator(test_gen)
    y_true = test_df["label"].values

    print(f"Loading Standard ResNet50: {args.standard_model_path}", flush=True)
    standard_model = load_model(args.standard_model_path)
    at_model = None
    if need_at:
        print(f"Loading PGD-AT ResNet50: {at_path}", flush=True)
        at_model = load_model(at_path)

    rows = run_evaluation(
        standard_model,
        at_model,
        images,
        labels,
        y_true,
        args.sigma,
        args.epsilons,
        args.steps,
        args.conditions,
    )

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "gaussian_defense_comparison.csv"
    json_path = args.output_dir / "gaussian_defense_metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_table10_tex(df, args.paper_table_path, args.sigma, args.steps)

    if not args.no_plot:
        plot_path = figures_dir / "gaussian_defense_comparison.png"
        plot_gaussian_defense(df, plot_path, args.sigma, args.steps)
        print(f"Saved figure: {plot_path}", flush=True)
        from plot_gaussian_defense_comparison import (
            plot_accuracy_comparison,
            plot_gain_vs_standard,
            plot_pgd_bim_side_by_side,
        )

        for name, plot_fn in (
            ("gaussian_defense_accuracy_vs_standard.png", plot_accuracy_comparison),
            ("gaussian_defense_gain_vs_standard.png", plot_gain_vs_standard),
            ("gaussian_defense_pgd_bim_eps4.png", plot_pgd_bim_side_by_side),
        ):
            out = figures_dir / name
            plot_fn(df, out, args.steps, args.sigma)
            paper_out = PAPER_DIR / name
            plot_fn(df, paper_out, args.steps, args.sigma)
            print(f"Saved figure: {out}", flush=True)

    print(f"\nSaved CSV: {csv_path}", flush=True)
    print(f"Saved table: {args.paper_table_path}", flush=True)


if __name__ == "__main__":
    main()
