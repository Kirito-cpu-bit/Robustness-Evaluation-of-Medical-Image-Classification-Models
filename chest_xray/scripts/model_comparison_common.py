"""Shared constants and helpers for cross-model attack comparison scripts."""
from pathlib import Path

import pandas as pd

from attack_config import DEFAULT_STEPS

EPSILONS = [i / 255 for i in [1, 2, 3, 4, 8]]

DEFAULT_MODELS = {
    "ResNet50": "outputs/models/cnn_4_final.h5",
    "DenseNet121": "outputs/densenet121/models/densenet121_final.h5",
    "ResNet18": "outputs/resnet18/models/resnet18_final.h5",
    "MobileNetV2": "outputs/mobilenetv2/models/mobilenetv2_final.h5",
    "ConvNeXt-Tiny": "outputs/convnext_tiny/models/convnext_tiny_best.h5",
}

MODEL_STYLE = {
    "ResNet50": {"color": "#1f77b4", "marker": "o"},
    "DenseNet121": {"color": "#ff7f0e", "marker": "s"},
    "ResNet18": {"color": "#2ca02c", "marker": "^"},
    "MobileNetV2": {"color": "#e377c2", "marker": "D"},
    "ConvNeXt-Tiny": {"color": "#9467bd", "marker": "v"},
}

DEFAULT_MODEL_NAMES = list(DEFAULT_MODELS)


def uses_rescale_input(model_name: str) -> bool:
    return True


def attack_input_config(model_name: str) -> dict:
    return {
        "rescale_input": True,
        "rescale_input_clean": True,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "epsilon_scale": 1.0,
        "forward_scale": 1.0,
        "infer_scale": 1.0,
        "bim_steps": DEFAULT_STEPS,
    }


def models_title(df: pd.DataFrame) -> str:
    """Build a dynamic plot title from the models present in the dataframe."""
    names = list(df["model"].unique())
    if len(names) <= 1:
        return names[0] if names else "Models"
    return " vs ".join(names)


def resolve_model_paths(
    model_names: list,
    project_root: Path,
    models_explicit: bool,
) -> dict:
    """
    Map model names to existing checkpoint paths.

    When models_explicit is False (default --models), skip missing checkpoints with a warning.
    When models_explicit is True (--models was passed on the CLI), raise FileNotFoundError.
    """
    resolved = {}
    for name in model_names:
        if name not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model {name}. Choose from {DEFAULT_MODEL_NAMES}")
        path = project_root / DEFAULT_MODELS[name]
        if not path.exists():
            if models_explicit:
                raise FileNotFoundError(
                    f"Model checkpoint not found for {name}: {path}\n"
                    f"Train it first, e.g. scripts/train_model.py --arch {name.lower()}"
                )
            print(f"WARNING: Skipping {name} — checkpoint not found: {path}", flush=True)
            continue
        resolved[name] = path
    if not resolved:
        raise FileNotFoundError("No model checkpoints found for the requested comparison.")
    return resolved
