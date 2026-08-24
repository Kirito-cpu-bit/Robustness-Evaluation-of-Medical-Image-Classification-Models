"""Train transfer-learning baselines (DenseNet121, ResNet18, ConvNeXt-Tiny, MobileNetV2) on brain tumor MRI."""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from tensorflow.keras.optimizers.legacy import SGD
except ImportError:
    from tensorflow.keras.optimizers import SGD

from convnext_gpu_patch import patch_convnext_for_gpu
from config import DATA_DIR, train_output_dir
from data_utils import build_dataframes, resolve_data_root
from resnet18_model import ResNet18
from train_resnet50 import (
    BATCH_SIZE,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_PATIENCE,
    DEFAULT_FREEZE_EPOCHS,
    DEFAULT_VAL_ACC_CAP,
    IMG_SIZE,
    INPUT_SHAPE,
    RANDOM_STATE,
    ValAccuracyCap,
    compile_model,
    configure_devices,
    evaluate_split,
    fit_model,
    make_generators,
    plot_confusion_matrix,
    predict_generator,
    steps,
)

ARCHITECTURES = {
    "resnet50": ResNet50,
    "densenet121": DenseNet121,
    "resnet18": ResNet18,
    "convnext_tiny": ConvNeXtTiny,
    "mobilenetv2": MobileNetV2,
}

RESNET18_TRAINING = {
    "freeze_epochs": 10,
    "freeze_lr": 1e-3,
    "finetune_lr": 1e-4,
    "finetune_patience": 5,
    "use_augmentation": True,
    "use_val_acc_cap": True,
    "val_acc_cap": 0.97,
    "phase2_monitor": "val_accuracy",
    "phase2_mode": "max",
    "use_class_weight_phase2": False,
}

CONVNEXT_TRAINING = {
    "freeze_epochs": 10,
    "freeze_lr": 1e-3,
    "finetune_lr": 1e-4,
    "finetune_patience": 5,
    "use_class_weight_phase1": True,
    "use_class_weight_phase2": False,
    "use_val_acc_cap": True,
    "val_acc_cap": 0.99,
    "phase1_monitor": "val_accuracy",
    "phase1_mode": "max",
}


def create_backbone(architecture: str):
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


def compile_model_for_architecture(model, architecture: str, phase: str = "freeze"):
    recall = Recall(name="recall")
    precision = Precision(name="precision")
    if architecture == "resnet18":
        learning_rate = (
            RESNET18_TRAINING["freeze_lr"]
            if phase == "freeze"
            else RESNET18_TRAINING["finetune_lr"]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif architecture == "convnext_tiny":
        learning_rate = (
            CONVNEXT_TRAINING["freeze_lr"]
            if phase == "freeze"
            else CONVNEXT_TRAINING["finetune_lr"]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        learning_rate, decay = (0.02, 0.01) if phase == "freeze" else (0.01, 0.001)
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9, decay=decay)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", recall, precision],
    )
    return model


def _generator_kwargs(shuffle):
    return dict(
        x_col="image",
        y_col="label",
        color_mode="rgb",
        class_mode="raw",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=RANDOM_STATE,
        shuffle=shuffle,
    )


def make_generators_for_architecture(train_df, val_df, test_df, architecture: str):
    if architecture == "resnet18" and RESNET18_TRAINING["use_augmentation"]:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )
        eval_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_gen = train_datagen.flow_from_dataframe(
            dataframe=train_df, **_generator_kwargs(True)
        )
        val_gen = eval_datagen.flow_from_dataframe(
            dataframe=val_df, **_generator_kwargs(False)
        )
        test_gen = eval_datagen.flow_from_dataframe(
            dataframe=test_df, **_generator_kwargs(False)
        )
        return train_gen, val_gen, test_gen
    return make_generators(train_df, val_df, test_df)


def class_weights_for_phase(architecture: str, class_weight_dict: dict, phase: str):
    if architecture == "resnet18" and phase == "finetune" and not RESNET18_TRAINING["use_class_weight_phase2"]:
        return None
    if architecture == "convnext_tiny":
        if phase == "freeze" and CONVNEXT_TRAINING["use_class_weight_phase1"]:
            return class_weight_dict
        if phase == "finetune" and CONVNEXT_TRAINING["use_class_weight_phase2"]:
            return class_weight_dict
        return None
    return class_weight_dict


def phase1_checkpoint_settings(architecture: str):
    if architecture == "convnext_tiny":
        return CONVNEXT_TRAINING["phase1_monitor"], CONVNEXT_TRAINING["phase1_mode"]
    return "val_loss", "min"


def phase2_callbacks(architecture: str, args, phase2_path: Path):
    if architecture == "resnet18" and not RESNET18_TRAINING["use_val_acc_cap"]:
        return [], True, RESNET18_TRAINING["phase2_monitor"], RESNET18_TRAINING["phase2_mode"]
    if architecture == "convnext_tiny" and not CONVNEXT_TRAINING["use_val_acc_cap"]:
        return [], True, "val_accuracy", "max"
    cap = args.val_acc_cap
    if architecture == "resnet18":
        cap = RESNET18_TRAINING["val_acc_cap"]
    elif architecture == "convnext_tiny":
        cap = CONVNEXT_TRAINING["val_acc_cap"]
    return (
        [
            ValAccuracyCap(
                cap=cap,
                checkpoint_path=phase2_path,
                min_val_acc=0.85,
            )
        ],
        False,
        "val_loss",
        "min",
    )


def build_model(architecture: str):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(ARCHITECTURES)}")

    base_ctor = ARCHITECTURES[architecture]
    base = create_backbone(architecture)
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(base.input, outputs=output)

    for layer in base.layers:
        layer.trainable = False
    return model, base


def train_and_evaluate(architecture: str, args):
    output_dir = train_output_dir(architecture)
    models_dir = output_dir / "models"
    figures_dir = output_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    train_df, val_df, test_df = build_dataframes(args.data_root)
    train_gen, val_gen, test_gen = make_generators_for_architecture(
        train_df, val_df, test_df, architecture
    )

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print("class_weight:", class_weight_dict)

    freeze_epochs = args.freeze_epochs
    finetune_patience = args.finetune_patience
    if architecture == "resnet18":
        freeze_epochs = RESNET18_TRAINING["freeze_epochs"]
        finetune_patience = RESNET18_TRAINING["finetune_patience"]
        print(
            "ResNet18 config: Adam "
            f"freeze_lr={RESNET18_TRAINING['freeze_lr']}, "
            f"finetune_lr={RESNET18_TRAINING['finetune_lr']}, "
            f"freeze_epochs={freeze_epochs}, augmentation=True"
        )
    elif architecture == "convnext_tiny":
        freeze_epochs = CONVNEXT_TRAINING["freeze_epochs"]
        finetune_patience = CONVNEXT_TRAINING["finetune_patience"]
        print(
            "ConvNeXt config: Adam "
            f"freeze_lr={CONVNEXT_TRAINING['freeze_lr']}, "
            f"finetune_lr={CONVNEXT_TRAINING['finetune_lr']}, "
            f"freeze_epochs={freeze_epochs}, val_acc_cap={CONVNEXT_TRAINING['val_acc_cap']}"
        )

    model, base = build_model(architecture)
    model = compile_model_for_architecture(model, architecture, phase="freeze")

    phase1_path = models_dir / f"{architecture}_phase1.h5"
    phase2_path = models_dir / f"{architecture}_phase2_best.h5"
    final_path = models_dir / f"{architecture}_final.h5"
    phase1_monitor, phase1_mode = phase1_checkpoint_settings(architecture)

    print(f"Phase 1: train with frozen {architecture} backbone")
    fit_model(
        model,
        train_gen,
        val_gen,
        class_weights_for_phase(architecture, class_weight_dict, "freeze"),
        epochs=freeze_epochs,
        patience=3,
        model_path=phase1_path,
        monitor=phase1_monitor,
        mode=phase1_mode,
    )
    model.load_weights(phase1_path)
    print(f"Loaded Phase 1 best weights from: {phase1_path}")

    for layer in base.layers:
        layer.trainable = True
    model = compile_model_for_architecture(model, architecture, phase="finetune")

    phase2_extra, use_checkpoint, monitor, mode = phase2_callbacks(
        architecture, args, phase2_path
    )
    print(
        f"Phase 2: fine-tune all {architecture} layers "
        f"(patience={finetune_patience}, monitor={monitor}, mode={mode})"
    )
    fit_model(
        model,
        train_gen,
        val_gen,
        class_weights_for_phase(architecture, class_weight_dict, "finetune"),
        epochs=args.finetune_epochs,
        patience=finetune_patience,
        model_path=phase2_path,
        use_checkpoint=use_checkpoint,
        monitor=monitor,
        mode=mode,
        extra_callbacks=phase2_extra,
    )
    if phase2_path.is_file():
        model.load_weights(phase2_path)
        print(f"Loaded Phase 2 best checkpoint from: {phase2_path}")
    else:
        print("Phase 2 produced no valid checkpoint; keeping Phase 1 weights.")
        model.load_weights(phase1_path)

    shutil.copy2(
        phase2_path if phase2_path.is_file() else phase1_path,
        final_path,
    )
    model.save(final_path, include_optimizer=False)

    results = []
    split_map = {
        "validation": (val_df, val_gen),
        "test": (test_df, test_gen),
    }

    for split_name, (df_split, gen_split) in split_map.items():
        eval_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
            dataframe=df_split,
            x_col="image",
            y_col="label",
            color_mode="rgb",
            class_mode="raw",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        loss, accuracy, recall, precision = model.evaluate(
            eval_gen, steps=steps(eval_gen), verbose=1
        )
        probs = predict_generator(model, gen_split)
        metrics, cm, _ = evaluate_split(df_split["label"].values, probs, split_name)
        metrics["architecture"] = architecture
        metrics["loss"] = float(loss)
        metrics["accuracy_keras"] = float(accuracy)
        metrics["recall_keras"] = float(recall)
        metrics["precision_keras"] = float(precision)
        results.append(metrics)

        plot_confusion_matrix(
            cm,
            f"Brain Tumor MRI - {split_name.upper()} ({architecture})",
            figures_dir / f"confusion_matrix_{split_name}.png",
        )
        print(f"\n{split_name.upper()} metrics ({architecture}):")
        print(json.dumps(metrics, indent=2))

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    preds_df = test_df.copy()
    preds_df["probability"] = predict_generator(model, test_gen)
    preds_df["prediction"] = np.rint(preds_df["probability"]).astype(int)
    preds_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\nSaved final model to: {final_path}")
    print(f"Saved metrics to: {metrics_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train model on brain tumor MRI.")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--freeze-epochs", type=int, default=DEFAULT_FREEZE_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--finetune-patience", type=int, default=DEFAULT_FINETUNE_PATIENCE)
    parser.add_argument("--val-acc-cap", type=float, default=DEFAULT_VAL_ACC_CAP)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()

    data_root = resolve_data_root(args.data_dir, skip_download=args.skip_download)
    args.data_root = data_root
    train_and_evaluate(args.arch, args)


if __name__ == "__main__":
    main()
