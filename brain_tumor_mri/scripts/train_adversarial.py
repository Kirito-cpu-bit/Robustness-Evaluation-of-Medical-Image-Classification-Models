"""PGD adversarial training (PGD-AT) for brain tumor MRI classifiers."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from attack_config import DEFAULT_STEPS
from config import DATA_DIR, OUTPUT_DIR
from data_utils import build_dataframes, resolve_data_root
from fgsm_attack import load_model
from pgd_attack import pgd_attack_batch
from train_resnet50 import (
    RANDOM_STATE,
    compile_model,
    configure_devices,
    evaluate_split,
    make_generators,
    plot_confusion_matrix,
    predict_generator,
    steps,
)

DEFAULT_TRAIN_EPS = 4 / 255
DEFAULT_TRAIN_STEPS = DEFAULT_STEPS
DEFAULT_EPOCHS = 10
DEFAULT_MIX = 0.5

ARCH_CONFIG: Dict[str, dict] = {
    "resnet50": {
        "display_name": "ResNet50",
        "init_model": OUTPUT_DIR / "models" / "cnn_4_final.h5",
        "output_dir": OUTPUT_DIR / "adversarial",
        "best_name": "resnet50_pgd_at_best.h5",
        "final_name": "resnet50_pgd_at_final.h5",
    },
    "densenet121": {
        "display_name": "DenseNet121",
        "init_model": OUTPUT_DIR / "densenet121" / "models" / "densenet121_final.h5",
        "output_dir": OUTPUT_DIR / "densenet121" / "adversarial",
        "best_name": "densenet121_pgd_at_best.h5",
        "final_name": "densenet121_pgd_at_final.h5",
    },
}


def build_arch_model(architecture: str):
    if architecture == "resnet50":
        from train_resnet50 import build_model

        model, _ = build_model()
        return compile_model(model, learning_rate=0.01, decay=0.001)

    from train_model import build_model, compile_model_for_architecture

    model, _ = build_model(architecture)
    return compile_model_for_architecture(model, architecture, phase="finetune")


@tf.function
def pgd_adversarial_examples(model, images, labels, epsilon, pgd_steps):
    alpha = epsilon / tf.cast(pgd_steps, tf.float32)
    return pgd_attack_batch(
        model,
        images,
        labels,
        tf.constant(epsilon, tf.float32),
        alpha,
        pgd_steps,
    )


@tf.function
def train_batch(model, optimizer, images, labels, sample_weight, epsilon, pgd_steps, mix_clean):
    adv_images = pgd_adversarial_examples(model, images, labels, epsilon, pgd_steps)

    with tf.GradientTape() as tape:
        pred_clean = model(images, training=True)
        pred_adv = model(adv_images, training=True)
        loss_clean = tf.keras.losses.binary_crossentropy(labels, pred_clean)
        loss_adv = tf.keras.losses.binary_crossentropy(labels, pred_adv)
        if mix_clean > 0.0:
            loss = mix_clean * loss_clean + (1.0 - mix_clean) * loss_adv
        else:
            loss = loss_adv
        loss = loss * sample_weight
        loss_value = tf.reduce_mean(loss)

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value


def run_epoch(model, optimizer, generator, class_weight_dict, epsilon, pgd_steps, mix_clean):
    losses = []
    step_count = steps(generator)
    for step_idx in range(step_count):
        batch_x, batch_y = generator[step_idx]
        x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        y = tf.reshape(tf.convert_to_tensor(batch_y, dtype=tf.float32), (-1, 1))
        weights = tf.constant(
            [class_weight_dict[int(label)] for label in batch_y.reshape(-1)],
            dtype=tf.float32,
        )
        loss = train_batch(model, optimizer, x, y, weights, epsilon, pgd_steps, mix_clean)
        losses.append(float(loss.numpy()))
        if (step_idx + 1) % 50 == 0 or step_idx + 1 == step_count:
            print(f"  step {step_idx + 1}/{step_count} loss={losses[-1]:.4f}", flush=True)
    return float(np.mean(losses))


def evaluate_model(model, val_gen, test_gen, val_df, test_df, figures_dir: Path, display_name: str):
    results = []
    for split_name, df_split, gen_split in (
        ("validation", val_df, val_gen),
        ("test", test_df, test_gen),
    ):
        probs = predict_generator(model, gen_split)
        metrics, cm, _ = evaluate_split(df_split["label"].values, probs, split_name)
        results.append(metrics)
        plot_confusion_matrix(
            cm,
            f"PGD-AT {display_name} — {split_name.upper()}",
            figures_dir / f"confusion_matrix_{split_name}.png",
        )
        print(json.dumps(metrics, indent=2), flush=True)
    return results


def load_progress(output_dir: Path) -> dict:
    progress_path = output_dir / "pgd_at_progress.json"
    if not progress_path.exists():
        return {}
    return json.loads(progress_path.read_text(encoding="utf-8"))


def save_progress(output_dir: Path, payload: dict):
    progress_path = output_dir / "pgd_at_progress.json"
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_architecture(architecture: str, args):
    if architecture not in ARCH_CONFIG:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(ARCH_CONFIG)}")

    cfg = ARCH_CONFIG[architecture]
    display_name = cfg["display_name"]
    output_dir = args.output_dir or cfg["output_dir"]
    init_model = args.init_model or cfg["init_model"]

    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best_path = models_dir / cfg["best_name"]
    final_path = models_dir / cfg["final_name"]

    data_root = resolve_data_root(args.data_dir, skip_download=args.skip_download)
    train_df, val_df, test_df = build_dataframes(data_root)
    train_gen, val_gen, test_gen = make_generators(train_df, val_df, test_df)

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    history = []
    best_val_acc = -1.0
    start_epoch = 1

    if args.resume:
        progress = load_progress(output_dir)
        if progress.get("architecture") not in (None, architecture):
            raise ValueError(
                f"Progress architecture mismatch: {progress.get('architecture')} vs {architecture}"
            )
        history = progress.get("history", [])
        best_val_acc = float(progress.get("best_val_acc", -1.0))
        start_epoch = int(progress.get("next_epoch", len(history) + 1))
        resume_path = best_path if best_path.is_file() else init_model
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model = load_model(resume_path)
        print(
            f"Resumed from {resume_path} at epoch {start_epoch}/{args.epochs} "
            f"(best_val_acc={best_val_acc:.4f})",
            flush=True,
        )
    elif init_model.is_file():
        model = load_model(init_model)
        print(f"Loaded init weights from: {init_model}", flush=True)
    else:
        model = build_arch_model(architecture)
        print(f"Training from scratch (init model not found: {init_model}).", flush=True)

    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)

    print(
        f"PGD-AT {display_name}: ε_train={args.train_epsilon:.6f} "
        f"({int(round(args.train_epsilon * 255))}/255), "
        f"steps={args.pgd_steps}, epochs={args.epochs}, mix_clean={args.mix_clean}",
        flush=True,
    )

    if start_epoch > args.epochs:
        print("All epochs already completed; loading best checkpoint for evaluation.", flush=True)
    else:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
            train_loss = run_epoch(
                model,
                optimizer,
                train_gen,
                class_weight_dict,
                args.train_epsilon,
                args.pgd_steps,
                args.mix_clean,
            )
            val_probs = predict_generator(model, val_gen)
            val_metrics, _, _ = evaluate_split(val_df["label"].values, val_probs, "validation")
            val_acc = val_metrics["accuracy"]
            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
            print(
                f"  val_acc={val_acc:.4f} recall={val_metrics['recall']:.4f} "
                f"precision={val_metrics['precision']:.4f}",
                flush=True,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save(best_path, include_optimizer=False)
                print(f"  saved best to {best_path}", flush=True)

            save_progress(
                output_dir,
                {
                    "architecture": architecture,
                    "best_val_acc": best_val_acc,
                    "next_epoch": epoch + 1,
                    "history": history,
                },
            )

    if best_path.is_file():
        model = load_model(best_path)
    model.save(final_path, include_optimizer=False)

    results = evaluate_model(model, val_gen, test_gen, val_df, test_df, figures_dir, display_name)
    config = {
        "architecture": architecture,
        "display_name": display_name,
        "train_epsilon": args.train_epsilon,
        "train_epsilon_label": f"{int(round(args.train_epsilon * 255))}/255",
        "pgd_steps": args.pgd_steps,
        "epochs": args.epochs,
        "mix_clean": args.mix_clean,
        "init_model": str(init_model),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "resumed": bool(args.resume),
    }
    metrics_path = output_dir / "pgd_at_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump({"config": config, "history": history, "evaluation": results}, handle, indent=2)
    print(f"\nSaved PGD-AT {display_name} model to: {final_path}", flush=True)
    return final_path


def main():
    parser = argparse.ArgumentParser(description="PGD adversarial training for brain tumor MRI models.")
    parser.add_argument(
        "--arch",
        choices=sorted(ARCH_CONFIG),
        default="resnet50",
        help="Architecture to adversarially fine-tune (default: resnet50).",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Warm-start checkpoint (default: standard final model for --arch).",
    )
    parser.add_argument("--train-epsilon", type=float, default=DEFAULT_TRAIN_EPS)
    parser.add_argument("--pgd-steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--mix-clean",
        type=float,
        default=DEFAULT_MIX,
        help="Weight on clean loss; remainder on adversarial loss.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from pgd_at_progress.json and the best PGD-AT checkpoint.",
    )
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()
    train_architecture(args.arch, args)


if __name__ == "__main__":
    main()
