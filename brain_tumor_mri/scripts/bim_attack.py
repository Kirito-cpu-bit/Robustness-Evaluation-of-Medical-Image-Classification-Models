"""BIM adversarial attack evaluation and robustness curve plotting."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tensorflow.keras.metrics import Precision, Recall

from attack_config import DEFAULT_STEPS, EPSILONS
from config import CM_LABELS, DATA_DIR, OUTPUT_DIR
from data_utils import build_dataframes, resolve_data_root
from fgsm_attack import (
    arrays_from_generator,
    generate_bim_batches,
    load_model,
    predict_images,
)
from train_resnet50 import (
    RANDOM_STATE,
    configure_devices,
    make_test_generator,
    predict_generator,
)


def compute_metrics(y_true, y_prob, epsilon, attack_name="BIM", steps=DEFAULT_STEPS):
    y_pred = np.rint(y_prob).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "attack": attack_name,
        "epsilon": float(epsilon),
        "epsilon_label": f"{int(round(epsilon * 255))}/255",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "attack_success_rate": float(np.mean(y_pred != y_true)),
        "confusion_matrix": cm.tolist(),
    }
    if attack_name == "BIM":
        metrics["steps"] = steps
        metrics["alpha"] = float(epsilon / steps)
    return metrics


def plot_confusion_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="Reds")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.xaxis.set_ticklabels(CM_LABELS)
    ax.yaxis.set_ticklabels(CM_LABELS)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_curve(results, baseline_metrics, output_path, steps):
    eps_labels = ["0\n(clean)"] + [item["epsilon_label"] for item in results]
    eps_values = [0.0] + [item["epsilon"] for item in results]

    accuracies = [baseline_metrics["accuracy"]] + [item["accuracy"] for item in results]
    recalls = [baseline_metrics["recall"]] + [item["recall"] for item in results]
    precisions = [baseline_metrics["precision"]] + [item["precision"] for item in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_values, accuracies, marker="o", linewidth=2, label="Accuracy")
    ax.plot(eps_values, recalls, marker="s", linewidth=2, label="Recall")
    ax.plot(eps_values, precisions, marker="^", linewidth=2, label="Precision")
    ax.set_xticks(eps_values)
    ax.set_xticklabels(eps_labels)
    ax.set_xlabel("Perturbation budget ε")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"BIM Robustness Curve on Test Set (steps={steps})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=OUTPUT_DIR / "models" / "cnn_4_final.h5",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of BIM iterations per image (default: 10)",
    )
    args = parser.parse_args()

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()

    data_root = resolve_data_root(args.data_dir, skip_download=True)
    _, _, test_df = build_dataframes(data_root)
    test_gen = make_test_generator(test_df)
    images, labels = arrays_from_generator(test_gen)
    y_true = test_df["label"].values
    model = load_model(args.model_path)

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating clean test images ...")
    clean_probs = predict_generator(model, make_test_generator(test_df))
    baseline = compute_metrics(y_true, clean_probs, epsilon=0.0, attack_name="clean")
    baseline["split"] = "test"
    plot_confusion_matrix(
        np.array(baseline["confusion_matrix"]),
        "Confusion Matrix - TEST SET (Clean)",
        figures_dir / "confusion_matrix_test_clean_bim.png",
    )
    print(json.dumps({k: v for k, v in baseline.items() if k != "confusion_matrix"}, indent=2))

    attack_results = []
    for epsilon in EPSILONS:
        alpha = epsilon / args.steps
        print(
            f"\nRunning BIM with epsilon={epsilon:.6f} ({int(round(epsilon * 255))}/255), "
            f"steps={args.steps}, alpha={alpha:.6f} ..."
        )
        adv_images = generate_bim_batches(model, images, labels, epsilon, steps=args.steps)
        adv_probs = predict_images(model, adv_images)
        metrics = compute_metrics(
            y_true, adv_probs, epsilon, attack_name="BIM", steps=args.steps
        )
        attack_results.append(metrics)

        epsilon_slug = metrics["epsilon_label"].replace("/", "_")
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            f"Confusion Matrix - TEST SET (BIM ε={metrics['epsilon_label']}, steps={args.steps})",
            figures_dir / f"confusion_matrix_test_bim_{epsilon_slug}.png",
        )
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    robustness_path = figures_dir / "robustness_curve_bim.png"
    plot_robustness_curve(attack_results, baseline, robustness_path, args.steps)

    summary = {
        "bim_config": {"steps": args.steps, "random_start": False, "norm": "L_inf"},
        "baseline_clean_test": {k: v for k, v in baseline.items() if k != "confusion_matrix"},
        "bim_results": [
            {k: v for k, v in item.items() if k != "confusion_matrix"} for item in attack_results
        ],
    }
    metrics_path = args.output_dir / "bim_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    comparison_df = pd.DataFrame(
        [{"condition": "clean", **{k: v for k, v in baseline.items() if k != "confusion_matrix"}}]
        + [
            {
                "condition": f"BIM ε={item['epsilon_label']}",
                **{k: v for k, v in item.items() if k != "confusion_matrix"},
            }
            for item in attack_results
        ]
    )
    comparison_df.to_csv(args.output_dir / "bim_comparison.csv", index=False)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved comparison table to: {args.output_dir / 'bim_comparison.csv'}")
    print(f"Saved robustness curve to: {robustness_path}")


if __name__ == "__main__":
    main()
