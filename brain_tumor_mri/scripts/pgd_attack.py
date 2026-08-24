"""PGD adversarial attack evaluation and robustness curve plotting."""
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
from config import CM_LABELS
from data_utils import build_dataframes, resolve_data_root
from train_resnet50 import (
    BATCH_SIZE,
    RANDOM_STATE,
    configure_devices,
    make_test_generator,
    predict_generator,
)


def load_model(model_path: Path):
    path_str = str(model_path)
    if "convnext" in path_str.lower():
        from train_model import build_model, compile_model_for_architecture

        model, _ = build_model("convnext_tiny")
        compile_model_for_architecture(model, "convnext_tiny", phase="finetune")
        model.load_weights(path_str)
        return model
    return tf.keras.models.load_model(
        path_str,
        custom_objects={"Recall": Recall, "Precision": Precision},
    )


def arrays_from_generator(generator):
    """Materialize generator batches in the same order used during training."""
    images = []
    labels = []
    batch_count = int(np.ceil(generator.n / BATCH_SIZE))
    for batch_idx in range(batch_count):
        batch_x, batch_y = generator[batch_idx]
        images.append(batch_x)
        labels.append(batch_y.reshape(-1))
    images = np.concatenate(images, axis=0)[: generator.n]
    labels = np.concatenate(labels, axis=0)[: generator.n]
    return images.astype(np.float32), labels.astype(np.float32)


def pgd_attack_batch(model, images, labels, epsilon, alpha, steps):
    """Madry PGD: random start + iterative signed gradients with L_inf projection."""
    labels = tf.reshape(labels, (-1, 1))
    random_start = tf.random.uniform(tf.shape(images), -epsilon, epsilon, dtype=tf.float32)
    adv_images = tf.clip_by_value(images + random_start, 0.0, 1.0)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_images)
            predictions = model(adv_images, training=False)
            loss = tf.keras.losses.binary_crossentropy(labels, predictions)
        gradient = tape.gradient(loss, adv_images)
        adv_images = adv_images + alpha * tf.sign(gradient)
        adv_images = tf.clip_by_value(adv_images, images - epsilon, images + epsilon)
        adv_images = tf.clip_by_value(adv_images, 0.0, 1.0)
    return adv_images


def generate_adversarial_batches(
    model, images, labels, epsilon, steps=DEFAULT_STEPS, batch_size=BATCH_SIZE
):
    alpha = epsilon / steps
    adv_images = []
    for start in range(0, len(images), batch_size):
        batch_x = tf.convert_to_tensor(images[start : start + batch_size])
        batch_y = tf.convert_to_tensor(labels[start : start + batch_size])
        adv_batch = pgd_attack_batch(
            model,
            batch_x,
            batch_y,
            tf.constant(epsilon, tf.float32),
            tf.constant(alpha, tf.float32),
            steps,
        )
        adv_images.append(adv_batch.numpy())
    return np.concatenate(adv_images, axis=0)


def predict_images(model, images, batch_size=BATCH_SIZE):
    probabilities = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        batch_probs = model.predict(batch, verbose=0)
        probabilities.append(batch_probs.reshape(-1))
    return np.concatenate(probabilities, axis=0)


def compute_metrics(y_true, y_prob, epsilon, attack_name="PGD", steps=DEFAULT_STEPS):
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
    if attack_name == "PGD":
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
    ax.set_title(f"PGD Robustness Curve on Test Set (steps={steps})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "models" / "cnn_4_final.h5",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help="Number of PGD iterations per image (default: 10)",
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

    print("Evaluating clean test images (ImageDataGenerator, same as training) ...")
    clean_probs = predict_generator(model, make_test_generator(test_df))
    baseline = compute_metrics(y_true, clean_probs, epsilon=0.0, attack_name="clean")
    baseline["split"] = "test"
    plot_confusion_matrix(
        np.array(baseline["confusion_matrix"]),
        "Confusion Matrix - TEST SET (Clean)",
        figures_dir / "confusion_matrix_test_clean_pgd.png",
    )
    print(json.dumps({k: v for k, v in baseline.items() if k != "confusion_matrix"}, indent=2))

    attack_results = []
    for epsilon in EPSILONS:
        alpha = epsilon / args.steps
        print(
            f"\nRunning PGD with epsilon={epsilon:.6f} ({int(round(epsilon * 255))}/255), "
            f"steps={args.steps}, alpha={alpha:.6f} ..."
        )
        adv_images = generate_adversarial_batches(
            model, images, labels, epsilon, steps=args.steps
        )
        adv_probs = predict_images(model, adv_images)
        metrics = compute_metrics(
            y_true, adv_probs, epsilon, attack_name="PGD", steps=args.steps
        )
        attack_results.append(metrics)

        epsilon_slug = metrics["epsilon_label"].replace("/", "_")
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            f"Confusion Matrix - TEST SET (PGD ε={metrics['epsilon_label']}, steps={args.steps})",
            figures_dir / f"confusion_matrix_test_pgd_{epsilon_slug}.png",
        )
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    robustness_path = figures_dir / "robustness_curve_pgd.png"
    plot_robustness_curve(attack_results, baseline, robustness_path, args.steps)

    summary = {
        "pgd_config": {
            "steps": args.steps,
            "random_start": True,
            "norm": "L_inf",
        },
        "baseline_clean_test": {k: v for k, v in baseline.items() if k != "confusion_matrix"},
        "pgd_results": [
            {k: v for k, v in item.items() if k != "confusion_matrix"} for item in attack_results
        ],
    }
    metrics_path = args.output_dir / "pgd_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    comparison_df = pd.DataFrame(
        [{"condition": "clean", **{k: v for k, v in baseline.items() if k != "confusion_matrix"}}]
        + [
            {
                "condition": f"PGD ε={item['epsilon_label']}",
                **{k: v for k, v in item.items() if k != "confusion_matrix"},
            }
            for item in attack_results
        ]
    )
    comparison_df.to_csv(args.output_dir / "pgd_comparison.csv", index=False)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved comparison table to: {args.output_dir / 'pgd_comparison.csv'}")
    print(f"Saved robustness curve to: {robustness_path}")


if __name__ == "__main__":
    main()
