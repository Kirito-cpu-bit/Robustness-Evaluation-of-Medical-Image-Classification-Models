"""FGSM adversarial attack evaluation and robustness curve plotting."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from tensorflow.keras.metrics import Precision, Recall
from train_resnet50 import (
    BATCH_SIZE,
    build_dataframes,
    configure_devices,
    make_test_generator,
    predict_generator,
)
from attack_config import DEFAULT_STEPS, EPSILONS
from download_dataset import find_chest_xray_root



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


def fgsm_step_eager(model, images, labels, epsilon, clip_min, clip_max, forward_scale):
    labels = tf.reshape(tf.convert_to_tensor(labels, dtype=tf.float32), (-1, 1))
    images_tf = tf.convert_to_tensor(images, dtype=tf.float32)
    scale = tf.cast(forward_scale, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(images_tf)
        predictions = model(images_tf * scale, training=False)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    gradient = tape.gradient(loss, images_tf)
    if gradient is None:
        raise ValueError("FGSM gradient is None; check model input scaling.")
    signed_grad = tf.sign(gradient)
    adv_images = images_tf + tf.cast(epsilon, tf.float32) * signed_grad
    return tf.clip_by_value(adv_images, clip_min, clip_max).numpy()


@tf.function
def fgsm_step(model, images, labels, epsilon, clip_min=0.0, clip_max=1.0, forward_scale=1.0):
    labels = tf.reshape(labels, (-1, 1))
    scale = tf.cast(forward_scale, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images * scale, training=False)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    adv_images = images + epsilon * signed_grad
    return tf.clip_by_value(adv_images, clip_min, clip_max)


def generate_adversarial_batches(
    model,
    images,
    labels,
    epsilon,
    batch_size=BATCH_SIZE,
    clip_min=0.0,
    clip_max=1.0,
    input_scale=1.0,
    forward_scale=None,
    use_eager=False,
):
    """input_scale kept for backward compatibility; forward_scale sets model input multiplier."""
    scale = 1.0 if forward_scale is None else forward_scale
    adv_images = []
    for start in range(0, len(images), batch_size):
        batch_x = images[start : start + batch_size]
        batch_y = labels[start : start + batch_size]
        if use_eager or scale != 1.0:
            adv_images.append(
                fgsm_step_eager(
                    model,
                    batch_x,
                    batch_y,
                    epsilon,
                    clip_min,
                    clip_max,
                    float(scale),
                )
            )
        else:
            adv_images.append(
                fgsm_step(
                    model,
                    tf.convert_to_tensor(batch_x),
                    tf.convert_to_tensor(batch_y),
                    tf.constant(epsilon, tf.float32),
                    clip_min,
                    clip_max,
                    float(scale),
                ).numpy()
            )
    return np.concatenate(adv_images, axis=0)


@tf.function
def bim_attack_batch(model, images, labels, epsilon, alpha, steps):
    """BIM: iterative FGSM with L_inf projection (monotonic robustness curves)."""
    labels = tf.reshape(labels, (-1, 1))
    adv_images = tf.identity(images)
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


def generate_bim_batches(
    model,
    images,
    labels,
    epsilon,
    steps=DEFAULT_STEPS,
    batch_size=BATCH_SIZE,
):
    alpha = epsilon / steps
    adv_images = []
    for start in range(0, len(images), batch_size):
        batch_x = tf.convert_to_tensor(images[start : start + batch_size])
        batch_y = tf.convert_to_tensor(labels[start : start + batch_size])
        adv_images.append(
            bim_attack_batch(
                model,
                batch_x,
                batch_y,
                tf.constant(epsilon, tf.float32),
                tf.constant(alpha, tf.float32),
                steps,
            ).numpy()
        )
    return np.concatenate(adv_images, axis=0)


def predict_images(model, images, batch_size=BATCH_SIZE):
    probabilities = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        batch_probs = model.predict(batch, verbose=0)
        probabilities.append(batch_probs.reshape(-1))
    return np.concatenate(probabilities, axis=0)


def compute_metrics(y_true, y_prob, epsilon, attack_name="FGSM", clean_probs=None):
    y_pred = np.rint(y_prob).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "attack": attack_name,
        "epsilon": float(epsilon),
        "epsilon_label": f"{int(round(epsilon * 255))}/255",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "adv_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
    }
    if clean_probs is not None:
        clean_pred = np.rint(clean_probs).astype(int)
        metrics["attack_success_rate"] = float(np.mean(y_pred != clean_pred))
    else:
        metrics["attack_success_rate"] = float(np.mean(y_pred != y_true))
    return metrics


def plot_confusion_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="Reds")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.xaxis.set_ticklabels(["NORMAL", "PNEUMONIA"])
    ax.yaxis.set_ticklabels(["NORMAL", "PNEUMONIA"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_robustness_curve(results, baseline_metrics, output_path):
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
    ax.set_title("FGSM Robustness Curve on Test Set")
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
    args = parser.parse_args()

    configure_devices()

    data_root = find_chest_xray_root(args.data_dir)
    if data_root is None:
        raise FileNotFoundError("Dataset not found under data/chest_xray")

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
        figures_dir / "confusion_matrix_test_clean.png",
    )
    print(json.dumps({k: v for k, v in baseline.items() if k != "confusion_matrix"}, indent=2))

    attack_results = []
    for epsilon in EPSILONS:
        print(f"\nRunning FGSM with epsilon={epsilon:.6f} ({int(round(epsilon * 255))}/255) ...")
        adv_images = generate_adversarial_batches(model, images, labels, epsilon)
        adv_probs = predict_images(model, adv_images)
        metrics = compute_metrics(y_true, adv_probs, epsilon)
        attack_results.append(metrics)

        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            f"Confusion Matrix - TEST SET (FGSM ε={metrics['epsilon_label']})",
            figures_dir / f"confusion_matrix_test_fgsm_{metrics['epsilon_label'].replace('/', '_')}.png",
        )
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    robustness_path = figures_dir / "robustness_curve_fgsm.png"
    plot_robustness_curve(attack_results, baseline, robustness_path)

    summary = {
        "baseline_clean_test": {k: v for k, v in baseline.items() if k != "confusion_matrix"},
        "fgsm_results": [
            {k: v for k, v in item.items() if k != "confusion_matrix"} for item in attack_results
        ],
    }
    metrics_path = args.output_dir / "fgsm_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    comparison_df = pd.DataFrame(
        [{"condition": "clean", **{k: v for k, v in baseline.items() if k != "confusion_matrix"}}]
        + [
            {"condition": f"FGSM ε={item['epsilon_label']}", **{k: v for k, v in item.items() if k != "confusion_matrix"}}
            for item in attack_results
        ]
    )
    comparison_df.to_csv(args.output_dir / "fgsm_comparison.csv", index=False)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved robustness curve to: {robustness_path}")


if __name__ == "__main__":
    main()
