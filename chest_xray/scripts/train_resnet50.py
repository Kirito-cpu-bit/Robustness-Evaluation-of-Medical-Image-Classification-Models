"""Train ResNet50 transfer-learning baseline and export evaluation metrics."""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
try:
    from tensorflow.keras.optimizers.legacy import SGD
except ImportError:
    from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall

from download_dataset import download_and_extract, find_chest_xray_root


BATCH_SIZE = 32
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
RANDOM_STATE = 42
DEFAULT_FREEZE_EPOCHS = 5
DEFAULT_FINETUNE_EPOCHS = 30


def build_dataframes(data_root: Path):
    path_train = data_root / "train"
    path_test = data_root / "test"

    frames = []
    for split_name, split_path in [("original_train", path_train), ("original_test", path_test)]:
        rows = []
        for label_name, label_value in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            class_dir = split_path / label_name
            for image_path in sorted(class_dir.glob("*.jpeg")):
                rows.append((str(image_path), label_value))
        frames.append(pd.DataFrame(rows, columns=["image", "label"]))
        print(f"{split_name}: {len(rows)} images")

    train_df, test_df = frames
    master_df = pd.concat([train_df, test_df], ignore_index=True)
    master_train_df, test_df = train_test_split(
        master_df, test_size=0.15, random_state=RANDOM_STATE, stratify=master_df["label"]
    )
    train_df, val_df = train_test_split(
        master_train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=master_train_df["label"],
    )
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


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


def make_test_generator(test_df, shuffle=False, rescale_input=True):
    datagen = ImageDataGenerator(rescale=1.0 / 255 if rescale_input else None)
    return datagen.flow_from_dataframe(
        dataframe=test_df,
        **_generator_kwargs(shuffle),
    )


def make_generators(train_df, val_df, test_df, shuffle_train=True, rescale_input=True):
    datagen = ImageDataGenerator(rescale=1.0 / 255 if rescale_input else None)
    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df, **_generator_kwargs(shuffle_train)
    )
    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df, **_generator_kwargs(False)
    )
    test_gen = make_test_generator(test_df, shuffle=False, rescale_input=rescale_input)
    return train_gen, val_gen, test_gen


def build_model():
    base = ResNet50(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(base.input, outputs=output)

    for layer in base.layers:
        layer.trainable = False
    return model, base


def compile_model(model, learning_rate, decay):
    recall = Recall(name="recall")
    precision = Precision(name="precision")
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9, decay=decay)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", recall, precision],
    )
    return model


def steps(generator):
    return max(1, generator.n // BATCH_SIZE)


def fit_model(
    model,
    train_gen,
    val_gen,
    class_weight_dict,
    epochs,
    patience,
    model_path,
    save_weights_only=False,
    monitor="val_loss",
    mode="min",
):
    callbacks = [
        ModelCheckpoint(
            str(model_path),
            save_best_only=True,
            monitor=monitor,
            mode=mode,
            save_weights_only=save_weights_only,
        ),
        EarlyStopping(
            patience=patience,
            restore_best_weights=True,
            monitor=monitor,
            mode=mode,
        ),
    ]
    fit_kwargs = {
        "validation_data": val_gen,
        "epochs": epochs,
        "steps_per_epoch": steps(train_gen),
        "validation_steps": steps(val_gen),
        "callbacks": callbacks,
        "verbose": 1,
    }
    if class_weight_dict is not None:
        fit_kwargs["class_weight"] = class_weight_dict
    return model.fit(train_gen, **fit_kwargs)


def predict_generator(model, generator):
    preds = model.predict(generator, steps=steps(generator) + 1, verbose=1)
    return preds[: generator.n].reshape(-1)


def evaluate_split(y_true, y_prob, split_name):
    y_pred = np.rint(y_prob).astype(int)
    metrics = {
        "split": split_name,
        "loss": None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm, y_pred


def plot_confusion_matrix(cm, title, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="g", ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.xaxis.set_ticklabels(["NORMAL", "PNEUMONIA"])
    ax.yaxis.set_ticklabels(["NORMAL", "PNEUMONIA"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def configure_devices():
    import sys

    gpus = tf.config.list_physical_devices("GPU")
    print(f"Python: {sys.executable}")
    print(f"TensorFlow: {tf.__version__} (CUDA build: {tf.test.is_built_with_cuda()})")
    print("Available GPUs:", gpus)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not set memory growth for {gpu}: {exc}")
    if gpus:
        print(f"Training will use GPU: {gpus[0].name}")
        print("Note: Windows uses DirectML backend; 'CUDA build: False' is expected.")
    else:
        print("No GPU detected. Training will run on CPU.")
        print("Use .venv-gpu\\Scripts\\python.exe to enable DirectML GPU.")
    return gpus


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
    parser.add_argument("--freeze-epochs", type=int, default=DEFAULT_FREEZE_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    configure_devices()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = args.output_dir / "models"
    figures_dir = args.output_dir / "figures"
    models_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    if args.skip_download:
        data_root = find_chest_xray_root(args.data_dir)
        if data_root is None:
            raise FileNotFoundError("Dataset not found. Run without --skip-download.")
    else:
        data_root = download_and_extract(args.data_dir)

    train_df, val_df, test_df = build_dataframes(data_root)
    train_gen, val_gen, test_gen = make_generators(train_df, val_df, test_df)

    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print("class_weight:", class_weight_dict)

    model, base = build_model()
    model = compile_model(model, learning_rate=0.02, decay=0.01)

    model_path = models_dir / "cnn_4.h5"
    print("Phase 1: train with frozen ResNet50 backbone")
    fit_model(
        model,
        train_gen,
        val_gen,
        class_weight_dict,
        epochs=args.freeze_epochs,
        patience=3,
        model_path=model_path,
    )

    for layer in base.layers:
        layer.trainable = True
    model = compile_model(model, learning_rate=0.01, decay=0.001)

    print("Phase 2: fine-tune all layers")
    fit_model(
        model,
        train_gen,
        val_gen,
        class_weight_dict,
        epochs=args.finetune_epochs,
        patience=10,
        model_path=model_path,
    )

    model.save(models_dir / "cnn_4_final.h5")

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
        metrics["loss"] = float(loss)
        metrics["accuracy_keras"] = float(accuracy)
        metrics["recall_keras"] = float(recall)
        metrics["precision_keras"] = float(precision)
        results.append(metrics)

        plot_confusion_matrix(
            cm,
            f"Confusion Matrix - {split_name.upper()} SET",
            figures_dir / f"confusion_matrix_{split_name}.png",
        )
        print(f"\n{split_name.upper()} metrics:")
        print(json.dumps(metrics, indent=2))
        print("Confusion matrix:\n", cm)

    metrics_path = models_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    preds_df = test_df.copy()
    preds_df["probability"] = predict_generator(model, test_gen)
    preds_df["prediction"] = np.rint(preds_df["probability"]).astype(int)
    preds_df.to_csv(models_dir / "test_predictions.csv", index=False)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
