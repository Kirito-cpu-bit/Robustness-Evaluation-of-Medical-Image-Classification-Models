"""Evaluate preprocessing defenses against FGSM/PGD on brain tumor ResNet50."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from attack_config import DEFAULT_STEPS
from config import DATA_DIR, OUTPUT_DIR
from data_utils import build_dataframes, resolve_data_root
from defense_common import DEFENSE_LABELS, DEFENSE_REGISTRY
from fgsm_attack import arrays_from_generator, compute_metrics, generate_adversarial_batches, load_model
from pgd_attack import generate_adversarial_batches as generate_pgd_batches
from train_resnet50 import RANDOM_STATE, configure_devices, make_test_generator, predict_images

COMPARISON_EPS = [i / 255 for i in [1, 2, 3, 4, 8]]
ATTACK_FNS = {
    "FGSM": ("fgsm", generate_adversarial_batches),
    "PGD": ("pgd", generate_pgd_batches),
}


def evaluate_defense(model, images, labels, adv_images, defense_key: str, epsilon: float, attack_name: str):
    defended = DEFENSE_REGISTRY[defense_key](adv_images)
    probs = predict_images(model, defended)
    metrics = compute_metrics(labels, probs, epsilon, attack_name=attack_name)
    metrics["defense"] = defense_key
    metrics["defense_label"] = DEFENSE_LABELS[defense_key]
    return metrics


def plot_defense_comparison(df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    eps_labels = ["0\n(clean)"] + [f"{int(round(e * 255))}/255" for e in COMPARISON_EPS]
    xs = list(range(len(eps_labels)))

    for attack_name, group in df.groupby("attack"):
        for defense_key, sub in group.groupby("defense"):
            clean = sub[sub["epsilon_label"] == "0/255"]
            attacked = sub[sub["epsilon_label"] != "0/255"].sort_values("epsilon")
            if clean.empty:
                continue
            accs = [float(clean.iloc[0]["accuracy"])] + attacked["accuracy"].tolist()
            asrs = [float(clean.iloc[0]["attack_success_rate"])] + attacked["attack_success_rate"].tolist()
            label = f"{attack_name} + {DEFENSE_LABELS[defense_key]}"
            axes[0].plot(xs, accs, marker="o", linewidth=2, label=label)
            axes[1].plot(xs, asrs, marker="s", linewidth=2, label=label)

    for ax, metric, title in zip(
        axes,
        ["Accuracy", "Attack Success Rate"],
        ["Accuracy vs ε (with preprocessing defense)", "ASR vs ε (with preprocessing defense)"],
    ):
        ax.set_xticks(xs)
        ax.set_xticklabels(eps_labels)
        ax.set_xlabel("Perturbation budget ε (L_inf)")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Brain Tumor ResNet50 — Preprocessing Defense Evaluation", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate preprocessing defenses.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "defense")
    parser.add_argument("--model-path", type=Path, default=OUTPUT_DIR / "models" / "cnn_4_final.h5")
    parser.add_argument("--attacks", nargs="+", default=["FGSM", "PGD"], choices=list(ATTACK_FNS))
    parser.add_argument(
        "--defenses",
        nargs="+",
        default=["none", "gaussian", "jpeg75", "median3"],
        choices=list(DEFENSE_REGISTRY),
    )
    parser.add_argument("--pgd-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_root = resolve_data_root(args.data_dir, skip_download=args.skip_download)
    _, _, test_df = build_dataframes(data_root)
    test_gen = make_test_generator(test_df)
    images, labels = arrays_from_generator(test_gen)
    y_true = test_df["label"].values
    model = load_model(args.model_path)

    rows = []
    for attack_name in args.attacks:
        _, generate_fn = ATTACK_FNS[attack_name]
        for defense_key in args.defenses:
            clean_defended = DEFENSE_REGISTRY[defense_key](images)
            clean_probs_def = predict_images(model, clean_defended)
            clean_metrics = compute_metrics(y_true, clean_probs_def, 0.0, attack_name="clean")
            clean_metrics["attack"] = attack_name
            clean_metrics["defense"] = defense_key
            clean_metrics["defense_label"] = DEFENSE_LABELS[defense_key]
            rows.append(clean_metrics)

            for epsilon in COMPARISON_EPS:
                if attack_name == "PGD":
                    adv_images = generate_fn(
                        model, images, labels, epsilon, steps=args.pgd_steps
                    )
                else:
                    adv_images = generate_fn(model, images, epsilon)
                metrics = evaluate_defense(
                    model, images, y_true, adv_images, defense_key, epsilon, attack_name
                )
                metrics["attack"] = attack_name
                rows.append(metrics)
                print(
                    f"{attack_name} ε={metrics['epsilon_label']} "
                    f"{DEFENSE_LABELS[defense_key]}: "
                    f"acc={metrics['accuracy']:.3f} asr={metrics['attack_success_rate']:.3f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "preprocessing_defense_comparison.csv"
    json_path = args.output_dir / "preprocessing_defense_metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_defense_comparison(df, figures_dir / "preprocessing_defense_comparison.png")
    print(f"\nSaved CSV: {csv_path}", flush=True)
    print(f"Saved figure: {figures_dir / 'preprocessing_defense_comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
