"""DeepFool adversarial attack evaluation and robustness curve plotting."""
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

from attack_config import EPSILONS
from download_dataset import find_chest_xray_root
from fgsm_attack import arrays_from_generator, load_model, predict_images
from train_resnet50 import BATCH_SIZE, build_dataframes, configure_devices, make_test_generator, predict_generator

DEFAULT_MAX_ITER = 15
DEFAULT_OVERSHOOT = 0.02


def deepfool_perturbation(model, image: np.ndarray, max_iter: int = DEFAULT_MAX_ITER, overshoot: float = DEFAULT_OVERSHOOT):
    """Compute minimal DeepFool perturbation for one image (binary sigmoid classifier)."""
    x = image.astype(np.float32).copy()
    x_tf = tf.convert_to_tensor(x[np.newaxis, ...])
    prob = model(x_tf, training=False)[0, 0]
    orig_pred = float(prob.numpy())
    orig_class = int(orig_pred >= 0.5)

    for _ in range(max_iter):
        x_tf = tf.convert_to_tensor(x[np.newaxis, ...])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_tf)
            prob = model(x_tf, training=False)[0, 0]
            score_normal = 1.0 - prob
            score_pneumonia = prob

        grad_normal = tape.gradient(score_normal, x_tf)[0].numpy()
        grad_pneumonia = tape.gradient(score_pneumonia, x_tf)[0].numpy()
        del tape

        scores = np.array([float(score_normal.numpy()), float(score_pneumonia.numpy())], dtype=np.float32)
        grads = [grad_normal, grad_pneumonia]
        pred_class = int(scores.argmax())

        best_pert = None
        best_norm = float("inf")
        for alt_class in range(2):
            if alt_class == pred_class:
                continue
            w = grads[alt_class] - grads[pred_class]
            f_diff = scores[alt_class] - scores[pred_class]
            w_norm_sq = float(np.dot(w.flatten(), w.flatten())) + 1e-12
            pert = (abs(f_diff) / w_norm_sq) * w
            pert_norm = float(np.linalg.norm(pert))
            if pert_norm < best_norm:
                best_norm = pert_norm
                best_pert = pert

        if best_pert is None:
            break

        x = x + (1.0 + overshoot) * best_pert
        x = np.clip(x, 0.0, 1.0)

        new_pred = float(model(tf.convert_to_tensor(x[np.newaxis, ...]), training=False)[0, 0].numpy())
        if int(new_pred >= 0.5) != orig_class:
            break

    return (x - image).astype(np.float32)


def generate_deepfool_batches(
    model,
    images,
    epsilon,
    max_iter: int = DEFAULT_MAX_ITER,
    overshoot: float = DEFAULT_OVERSHOOT,
    log_every: int = 50,
):
    """DeepFool perturbation capped by L_inf budget epsilon."""
    adv_images = np.empty_like(images)
    total = len(images)
    for idx in range(total):
        delta = deepfool_perturbation(model, images[idx], max_iter=max_iter, overshoot=overshoot)
        delta = np.clip(delta, -epsilon, epsilon)
        adv_images[idx] = np.clip(images[idx] + delta, 0.0, 1.0)
        if log_every and (idx + 1) % log_every == 0:
            print(f"    DeepFool images {idx + 1}/{total}", flush=True)
    return adv_images


def compute_metrics(y_true, y_prob, epsilon, attack_name="DeepFool"):
    y_pred = np.rint(y_prob).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "attack": attack_name,
        "epsilon": float(epsilon),
        "epsilon_label": f"{int(round(epsilon * 255))}/255",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "attack_success_rate": float(np.mean(y_pred != y_true)),
        "adv_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": cm.tolist(),
    }


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
    ax.set_xlabel("Perturbation budget ε (L_inf cap)")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("DeepFool Robustness Curve on Test Set")
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
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--overshoot", type=float, default=DEFAULT_OVERSHOOT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip clean/epsilon steps already saved in deepfool_progress.json",
    )
    parser.add_argument(
        "--skip-confusion-matrices",
        action="store_true",
        help="Skip per-epsilon confusion matrix PNGs to save time",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print progress every N images (0 disables)",
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
    progress_path = args.output_dir / "deepfool_progress.json"
    progress_rows = load_progress_rows(progress_path) if args.resume else []
    done_eps = {
        row["epsilon_label"] for row in progress_rows if row.get("attack") == "DeepFool"
    }

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
                figures_dir / "confusion_matrix_test_clean.png",
            )
        print(json.dumps({k: v for k, v in baseline.items() if k != "confusion_matrix"}, indent=2))
        progress_rows = [{k: v for k, v in baseline.items() if k != "confusion_matrix"}]

    attack_results = [
        metrics_from_progress_row(row)
        for row in progress_rows
        if row.get("attack") == "DeepFool"
    ]

    for epsilon in EPSILONS:
        label = f"{int(round(epsilon * 255))}/255"
        if label in done_eps:
            cached = next(row for row in progress_rows if row.get("epsilon_label") == label)
            print(
                f"\nDeepFool ε={label} — skipped (resume) "
                f"acc={cached['accuracy']:.4f}, asr={cached['attack_success_rate']:.4f}",
                flush=True,
            )
            continue

        print(f"\nRunning DeepFool with L_inf cap ε={label} (max_iter={args.max_iter}) ...", flush=True)
        adv_images = generate_deepfool_batches(
            model,
            images,
            epsilon,
            max_iter=args.max_iter,
            overshoot=args.overshoot,
            log_every=args.log_every,
        )
        adv_probs = predict_images(model, adv_images)
        metrics = compute_metrics(y_true, adv_probs, epsilon)
        attack_results.append(metrics)
        progress_rows.append({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        with progress_path.open("w", encoding="utf-8") as handle:
            json.dump(progress_rows, handle, indent=2)

        if not args.skip_confusion_matrices:
            plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                f"Confusion Matrix - TEST SET (DeepFool ε={metrics['epsilon_label']})",
                figures_dir / f"confusion_matrix_test_deepfool_{metrics['epsilon_label'].replace('/', '_')}.png",
            )
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))

    attack_results = sorted(attack_results, key=lambda item: item["epsilon"])

    robustness_path = figures_dir / "robustness_curve_deepfool.png"
    plot_robustness_curve(attack_results, baseline, robustness_path)

    summary = {
        "baseline_clean_test": {k: v for k, v in baseline.items() if k != "confusion_matrix"},
        "deepfool_results": [
            {k: v for k, v in item.items() if k != "confusion_matrix"} for item in attack_results
        ],
    }
    metrics_path = args.output_dir / "deepfool_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    comparison_df = pd.DataFrame(
        [{"condition": "clean", **{k: v for k, v in baseline.items() if k != "confusion_matrix"}}]
        + [
            {
                "condition": f"DeepFool ε={item['epsilon_label']}",
                **{k: v for k, v in item.items() if k != "confusion_matrix"},
            }
            for item in attack_results
        ]
    )
    comparison_df.to_csv(args.output_dir / "deepfool_comparison.csv", index=False)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved progress to: {progress_path}")
    print(f"Saved robustness curve to: {robustness_path}")


if __name__ == "__main__":
    main()
