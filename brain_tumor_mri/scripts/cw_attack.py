"""Carlini & Wagner (C&W) L_inf adversarial attack evaluation and robustness curve plotting."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from attack_config import DEFAULT_STEPS, EPSILONS
from config import CM_LABELS, DATA_DIR, OUTPUT_DIR
from data_utils import build_dataframes, resolve_data_root
from fgsm_attack import arrays_from_generator, load_model, predict_images
from train_resnet50 import (
    BATCH_SIZE,
    RANDOM_STATE,
    configure_devices,
    make_test_generator,
    predict_generator,
)

DEFAULT_KAPPA = 0.0


def cw_margin_loss(prob, labels, kappa=DEFAULT_KAPPA):
    """C&W untargeted margin in logit space: relu(Z_true - Z_wrong + kappa)."""
    labels = tf.reshape(labels, (-1, 1))
    prob = tf.clip_by_value(prob, 1e-6, 1.0 - 1e-6)
    logit = tf.math.log(prob / (1.0 - prob))
    logit_true = tf.where(labels < 0.5, -logit, logit)
    logit_wrong = tf.where(labels < 0.5, logit, -logit)
    return tf.reduce_mean(tf.nn.relu(logit_true - logit_wrong + kappa))


def cw_attack_batch(model, images, labels, epsilon, alpha, steps, kappa):
    """L_inf C&W: minimize margin in logit space, same projection pattern as BIM."""
    adv_images = tf.identity(images)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            prob = model(adv_images, training=False)
            loss = cw_margin_loss(prob, labels, kappa)
        gradient = tape.gradient(loss, adv_images)
        adv_images = adv_images - alpha * tf.sign(gradient)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)
    return adv_images


def generate_cw_batches(
    model,
    images,
    labels,
    epsilon,
    steps=DEFAULT_STEPS,
    kappa=DEFAULT_KAPPA,
    batch_size=BATCH_SIZE,
):
    alpha = epsilon / steps
    adv_images = []
    total_batches = int(np.ceil(len(images) / batch_size))
    for batch_idx, start in enumerate(range(0, len(images), batch_size)):
        batch_x = tf.convert_to_tensor(images[start : start + batch_size])
        batch_y = tf.convert_to_tensor(labels[start : start + batch_size])
        adv_batch = cw_attack_batch(
            model,
            batch_x,
            batch_y,
            tf.constant(epsilon, tf.float32),
            tf.constant(alpha, tf.float32),
            steps,
            float(kappa),
        )
        adv_images.append(adv_batch.numpy())
        print(f"    C&W batch {batch_idx + 1}/{total_batches}", flush=True)
    return np.concatenate(adv_images, axis=0)


def compute_metrics(y_true, y_prob, epsilon, attack_name="C&W", steps=DEFAULT_STEPS, kappa=DEFAULT_KAPPA):
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
    if attack_name == "C&W":
        metrics["steps"] = steps
        metrics["alpha"] = float(epsilon / steps)
        metrics["kappa"] = float(kappa)
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


def plot_robustness_curve(results, baseline_metrics, output_path, steps, kappa):
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
    ax.set_xlabel("Perturbation budget ε (L_inf)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"C&W Robustness Curve (steps={steps}, κ={kappa}, α=ε/steps)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_progress_rows(progress_path: Path) -> list:
    if not progress_path.exists():
        return []
    return json.loads(progress_path.read_text(encoding="utf-8"))


def metrics_from_progress_row(row: dict) -> dict:
    return {**row, "confusion_matrix": row.get("confusion_matrix", [])}


def main():
    parser = argparse.ArgumentParser(
        description="L_inf C&W attack using logit-margin objective (aligned with BIM/PGD steps)."
    )
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
        help=f"Number of C&W iterations (default: {DEFAULT_STEPS}, same as BIM/PGD)",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=DEFAULT_KAPPA,
        help="C&W confidence margin κ (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for attack generation (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip clean/epsilon steps already saved in cw_progress.json",
    )
    parser.add_argument(
        "--skip-confusion-matrices",
        action="store_true",
        help="Skip per-epsilon confusion matrix PNGs to save time",
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
    progress_path = args.output_dir / "cw_progress.json"
    progress_rows = load_progress_rows(progress_path) if args.resume else []
    done_eps = {row["epsilon_label"] for row in progress_rows if row.get("attack") == "C&W"}

    if args.resume and progress_rows and any(row.get("attack") == "clean" for row in progress_rows):
        clean_row = next(row for row in progress_rows if row.get("attack") == "clean")
        baseline = metrics_from_progress_row(clean_row)
        print(
            "Clean — skipped (resume) "
            f"acc={baseline['accuracy']:.4f}, recall={baseline['recall']:.4f}, "
            f"precision={baseline['precision']:.4f}",
            flush=True,
        )
    else:
        print("Evaluating clean test images ...")
        clean_probs = predict_generator(model, make_test_generator(test_df))
        baseline = compute_metrics(y_true, clean_probs, epsilon=0.0, attack_name="clean")
        baseline["split"] = "test"
        if not args.skip_confusion_matrices:
            plot_confusion_matrix(
                np.array(baseline["confusion_matrix"]),
                "Confusion Matrix - TEST SET (Clean)",
                figures_dir / "confusion_matrix_test_clean_cw.png",
            )
        print(json.dumps({k: v for k, v in baseline.items() if k != "confusion_matrix"}, indent=2))
        progress_rows = [{k: v for k, v in baseline.items() if k != "confusion_matrix"}]

    attack_results = [
        metrics_from_progress_row(row)
        for row in progress_rows
        if row.get("attack") == "C&W"
    ]

    for epsilon in EPSILONS:
        label = f"{int(round(epsilon * 255))}/255"
        if label in done_eps:
            cached = next(row for row in progress_rows if row.get("epsilon_label") == label)
            print(
                f"\nC&W ε={label} — skipped (resume) "
                f"acc={cached['accuracy']:.4f}, asr={cached['attack_success_rate']:.4f}",
                flush=True,
            )
            continue

        alpha = epsilon / args.steps
        print(
            f"\nRunning C&W with L_inf cap ε={label} "
            f"(steps={args.steps}, alpha={alpha:.6f}, kappa={args.kappa}) ...",
            flush=True,
        )
        adv_images = generate_cw_batches(
            model,
            images,
            labels,
            epsilon,
            steps=args.steps,
            kappa=args.kappa,
            batch_size=args.batch_size,
        )
        adv_probs = predict_images(model, adv_images)
        metrics = compute_metrics(
            y_true,
            adv_probs,
            epsilon,
            steps=args.steps,
            kappa=args.kappa,
        )
        attack_results.append(metrics)
        progress_rows.append({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        with progress_path.open("w", encoding="utf-8") as handle:
            json.dump(progress_rows, handle, indent=2)

        if not args.skip_confusion_matrices:
            plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                f"Confusion Matrix - TEST SET (C&W ε={metrics['epsilon_label']})",
                figures_dir / f"confusion_matrix_test_cw_{metrics['epsilon_label'].replace('/', '_')}.png",
            )
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    attack_results = sorted(attack_results, key=lambda item: item["epsilon"])

    robustness_path = figures_dir / "robustness_curve_cw.png"
    plot_robustness_curve(attack_results, baseline, robustness_path, args.steps, args.kappa)

    summary = {
        "cw_config": {
            "steps": args.steps,
            "kappa": args.kappa,
            "alpha": "epsilon / steps",
            "batch_size": args.batch_size,
            "objective": "relu(Z_true - Z_wrong + kappa) in logit space",
            "norm": "L_inf",
        },
        "baseline_clean_test": {k: v for k, v in baseline.items() if k != "confusion_matrix"},
        "cw_results": [
            {k: v for k, v in item.items() if k != "confusion_matrix"} for item in attack_results
        ],
    }
    metrics_path = args.output_dir / "cw_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    comparison_df = pd.DataFrame(
        [{"condition": "clean", **{k: v for k, v in baseline.items() if k != "confusion_matrix"}}]
        + [
            {
                "condition": f"C&W ε={item['epsilon_label']}",
                **{k: v for k, v in item.items() if k != "confusion_matrix"},
            }
            for item in attack_results
        ]
    )
    comparison_df.to_csv(args.output_dir / "cw_comparison.csv", index=False)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved progress to: {progress_path}")
    print(f"Saved comparison table to: {args.output_dir / 'cw_comparison.csv'}")
    print(f"Saved robustness curve to: {robustness_path}")


if __name__ == "__main__":
    main()
