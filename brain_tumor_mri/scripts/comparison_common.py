"""Shared helpers for brain tumor MRI model comparison."""
from pathlib import Path

from attack_config import DEFAULT_STEPS
from config import MODEL_COMPARISON_DIR, TASK_ROOT, model_checkpoints

EPSILONS = [i / 255 for i in [1, 2, 3, 4, 8]]

DEFAULT_MODELS = {
    name: str(path.relative_to(TASK_ROOT)).replace("\\", "/")
    for name, path in model_checkpoints().items()
}
DEFAULT_MODEL_NAMES = list(DEFAULT_MODELS)

MODEL_STYLE = {
    "ResNet50": {"color": "#1f77b4", "marker": "o"},
    "ResNet50-AT": {"color": "#d62728", "marker": "s"},
    "DenseNet121": {"color": "#ff7f0e", "marker": "s"},
    "DenseNet121-AT": {"color": "#d62728", "marker": "s"},
    "ResNet18": {"color": "#2ca02c", "marker": "^"},
    "ConvNeXt-Tiny": {"color": "#9467bd", "marker": "v"},
    "MobileNetV2": {"color": "#e377c2", "marker": "D"},
}


def attack_input_config(model_name: str) -> dict:
    return {"rescale_input": True, "clip_min": 0.0, "clip_max": 1.0, "bim_steps": DEFAULT_STEPS}


def models_title(df) -> str:
    names = list(df["model"].unique())
    return names[0] if len(names) == 1 else " vs ".join(names)


def resolve_model_paths(model_names, models_explicit: bool) -> dict:
    resolved = {}
    for name in model_names:
        if name not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model: {name}")
        path = TASK_ROOT / DEFAULT_MODELS[name]
        if not path.exists():
            if models_explicit:
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            print(f"WARNING: skip {name}, not found: {path}", flush=True)
            continue
        resolved[name] = path
    if not resolved:
        raise FileNotFoundError("No checkpoints found.")
    return resolved
