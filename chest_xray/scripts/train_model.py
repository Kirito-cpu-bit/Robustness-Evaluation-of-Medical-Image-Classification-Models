"""Train transfer-learning baselines (DenseNet121, ResNet18, etc.)."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from convnext_gpu_patch import patch_convnext_for_gpu
from download_dataset import download_and_extract, find_chest_xray_root
from resnet18_model import ResNet18
from train_resnet50 import (
    BATCH_SIZE,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FREEZE_EPOCHS,
    IMG_SIZE,
    INPUT_SHAPE,
    RANDOM_STATE,
    build_dataframes,
    configure_devices,
    evaluate_split,
    fit_model,
    make_generators,
    plot_confusion_matrix,
    predict_generator,
    steps,
)
try:
    from tensorflow.keras.optimizers.legacy import SGD
except ImportError:
    from tensorflow.keras.optimizers import SGD


ARCHITECTURES = {
    "resnet50": ResNet50,
    "densenet121": DenseNet121,
    "resnet18": ResNet18,
    "mobilenetv2": MobileNetV2,
    "convnext_tiny": ConvNeXtTiny,
}

CONVNEXT_TRAINING = {
    "freeze_lr": 1e-3,
    "finetune_lr": 1e-4,
    "use_class_weight_phase1": True,
    "use_class_weight_phase2": False,
    "checkpoint_monitor": "val_accuracy",
    "checkpoint_mode": "max",
}


def create_backbone(architecture: str):
    """Instantiate ImageNet backbone with architecture-specific options."""
    base_ctor = ARCHITECTURES[architecture]
    if architecture == "convnext_tiny":
        base = base_ctor(
            include_top=False,
            include_preprocessing=False,
            weights="imagenet",
            input_shape=INPUT_SHAPE,
        )
        return patch_convnext_for_gpu(base)
    return base_ctor(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)


def rescale_input_for_architecture(architecture: str) -> bool:
    """All comparison models use generator rescale to [0, 1]."""
    return True


def compile_model_for_architecture(model, architecture: str, phase: str = "freeze"):
    """Compile with architecture-specific optimizer settings."""
    if architecture == "convnext_tiny":
        learning_rate = (
            CONVNEXT_TRAINING["freeze_lr"]
            if phase == "freeze"
            else CONVNEXT_TRAINING["finetune_lr"]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        learning_rate, decay = (0.02, 0.01) if phase == "freeze" else (0.01, 0.001)
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9, decay=decay)

    recall = Recall(name="recall")
    precision = Precision(name="precision")
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", recall, precision],
    )
    return model


def class_weights_for_phase(architecture: str, class_weight_dict: dict, phase: str):
    if architecture != "convnext_tiny":
        return class_weight_dict
    if phase == "freeze" and CONVNEXT_TRAINING["use_class_weight_phase1"]:
        return class_weight_dict
    if phase == "finetune" and CONVNEXT_TRAINING["use_class_weight_phase2"]:
        return class_weight_dict
    return None


def checkpoint_settings_for_architecture(architecture: str):
    if architecture == "convnext_tiny":
        return CONVNEXT_TRAINING["checkpoint_monitor"], CONVNEXT_TRAINING["checkpoint_mode"]
    return "val_loss", "min"


def build_model(architecture: str):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(ARCHITECTURES)}")

    base = create_backbone(architecture)
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(base.input, outputs=output)

    for layer in base.layers:
        layer.trainable = False
    return model, base


def run_evaluation(
    model,
    architecture,
    args,
    val_df,
    val_gen,
    test_df,
    test_gen,
    output_dir,
    figures_dir,
    rescale_input=True,
):
    results = []
    split_map = {
        "validation": (val_df, val_gen),
        "test": (test_df, test_gen),
    }
    eval_rescale = 1.0 / 255 if rescale_input else None

    for split_name, (df_split, gen_split) in split_map.items():
        eval_gen = ImageDataGenerator(rescale=eval_rescale).flow_from_dataframe(
            dataframe=df_split,
            x_col="image",
            y_col="label",
            color_mode="rgb",
            class_mode="raw",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        eval_results = model.evaluate(eval_gen, steps=steps(eval_gen), verbose=1)
        loss = float(eval_results[0])
        accuracy = float(eval_results[1])
        recall = float(eval_results[2]) if len(eval_results) > 2 else None
        precision = float(eval_results[3]) if len(eval_results) > 3 else None
        probs = predict_generator(model, gen_split)
        metrics, cm, _ = evaluate_split(df_split["label"].values, probs, split_name)
        metrics["architecture"] = architecture
        metrics["loss"] = float(loss)
        metrics["accuracy_keras"] = float(accuracy)
        metrics["recall_keras"] = float(recall) if recall is not None else metrics["recall"]
        metrics["precision_keras"] = float(precision) if precision is not None else metrics["precision"]
        results.append(metrics)

        plot_confusion_matrix(
            cm,
            f"Confusion Matrix - {split_name.upper()} SET ({architecture})",
            figures_dir / f"confusion_matrix_{split_name}.png",
        )
        print(f"\n{split_name.upper()} metrics ({architecture}):")
        print(json.dumps(metrics, indent=2))
        print("Confusion matrix:\n", cm)

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    preds_df = test_df.copy()
    preds_df["probability"] = predict_generator(model, test_gen)
    preds_df["prediction"] = np.rint(preds_df["probability"]).astype(int)
    preds_df.to_csv(output_dir / "test_predictions.csv", index=False)
    return results, metrics_path


def train_and_evaluate(architecture: str, args):
    output_dir = args.output_dir / architecture
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = build_dataframes(args.data_root)
    rescale_input = rescale_input_for_architecture(architecture)
    train_gen, val_gen, test_gen = make_generators(
        train_df, val_df, test_df, rescale_input=rescale_input
    )
    if args.eval_only:
        final_path = args.eval_only
        model, _ = build_model(architecture)
        model = compile_model_for_architecture(model, architecture, phase="finetune")
        model.load_weights(str(final_path))
        results, metrics_path = run_evaluation(
            model,
            architecture,
            args,
            val_df,
            val_gen,
            test_df,
            test_gen,
            output_dir,
            figures_dir,
            rescale_input=rescale_input,
        )
        print(f"\nLoaded weights from: {final_path}")
        print(f"Saved metrics to: {metrics_path}")
        print(f"Saved figures to: {figures_dir}")
        return results

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print("class_weight:", class_weight_dict)

    model, base = build_model(architecture)
    checkpoint_path = models_dir / f"{architecture}_best.h5"

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        model.load_weights(str(resume_path))
        print(f"Resumed weights from: {resume_path} (skipping Phase 1)")
    else:
        model = compile_model_for_architecture(model, architecture, phase="freeze")
        monitor, mode = checkpoint_settings_for_architecture(architecture)
        if architecture == "convnext_tiny":
            print(
                "ConvNeXt config: Adam freeze_lr="
                f"{CONVNEXT_TRAINING['freeze_lr']}, finetune_lr={CONVNEXT_TRAINING['finetune_lr']}, "
                f"checkpoint={monitor} ({mode})"
            )
        print(f"Phase 1: train with frozen {architecture} backbone")
        fit_model(
            model,
            train_gen,
            val_gen,
            class_weights_for_phase(architecture, class_weight_dict, "freeze"),
            epochs=args.freeze_epochs,
            patience=3,
            model_path=checkpoint_path,
            monitor=monitor,
            mode=mode,
        )

    for layer in base.layers:
        layer.trainable = True
    model = compile_model_for_architecture(model, architecture, phase="finetune")
    monitor, mode = checkpoint_settings_for_architecture(architecture)

    print(f"Phase 2: fine-tune all {architecture} layers")
    fit_model(
        model,
        train_gen,
        val_gen,
        class_weights_for_phase(architecture, class_weight_dict, "finetune"),
        epochs=args.finetune_epochs,
        patience=10,
        model_path=checkpoint_path,
        monitor=monitor,
        mode=mode,
    )

    final_path = models_dir / f"{architecture}_final.h5"
    model.save(final_path, include_optimizer=False)

    results, metrics_path = run_evaluation(
        model,
        architecture,
        args,
        val_df,
        val_gen,
        test_df,
        test_gen,
        output_dir,
        figures_dir,
        rescale_input=rescale_input,
    )

    print(f"\nSaved checkpoint to: {checkpoint_path}")
    print(f"Saved final model to: {final_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved figures to: {figures_dir}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=sorted(ARCHITECTURES),
        required=True,
        help="Model architecture to train",
    )
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
    parser.add_argument("--freeze-epochs", type=int, default=DEFAULT_FREEZE_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--eval-only",
        type=Path,
        default=None,
        help="Skip training; load weights from this checkpoint and run evaluation only.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Load weights and skip Phase 1; continue Phase 2 fine-tuning only.",
    )
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()

    if args.skip_download:
        data_root = find_chest_xray_root(args.data_dir)
        if data_root is None:
            raise FileNotFoundError("Dataset not found. Run without --skip-download.")
    else:
        data_root = download_and_extract(args.data_dir)

    args.data_root = data_root
    train_and_evaluate(args.arch, args)


if __name__ == "__main__":
    main()
