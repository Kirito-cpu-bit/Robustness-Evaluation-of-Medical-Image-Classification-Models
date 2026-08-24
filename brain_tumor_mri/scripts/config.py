"""Constants and paths for brain tumor MRI task."""
from pathlib import Path
from typing import Dict, Optional

TASK_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_ROOT / "data"
OUTPUT_DIR = TASK_ROOT / "outputs"
MODEL_COMPARISON_DIR = OUTPUT_DIR / "model_comparison"

TASK_TITLE = "Brain Tumor MRI Detection"
HF_REPO = "usamaJabar/brain-tumor-mri-classification-merged"

NEGATIVE_CLASS_NAMES = ["no_tumor", "notumor", "no", "normal"]
POSITIVE_CLASS_NAMES = ["glioma", "meningioma", "pituitary", "tumor", "yes"]
CM_LABELS = ["NO_TUMOR", "TUMOR"]
IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

# Merge all on-disk splits (train+val+test) then stratified resplit for evaluation stability.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def model_checkpoints() -> Dict[str, Path]:
    return {
        "ResNet50": OUTPUT_DIR / "models" / "cnn_4_final.h5",
        "DenseNet121": OUTPUT_DIR / "densenet121" / "models" / "densenet121_final.h5",
        "ResNet18": OUTPUT_DIR / "resnet18" / "models" / "resnet18_final.h5",
        "ConvNeXt-Tiny": OUTPUT_DIR / "convnext_tiny" / "models" / "convnext_tiny_final.h5",
        "MobileNetV2": OUTPUT_DIR / "mobilenetv2" / "models" / "mobilenetv2_final.h5",
    }


def adversarial_checkpoints() -> Dict[str, Path]:
    return {
        # Legacy ResNet50 PGD-AT (train pgd_steps=7, eval steps=5).
        "ResNet50-AT": OUTPUT_DIR / "adversarial" / "models" / "resnet50_pgd_at_final.h5",
        # ResNet50 PGD-AT trained with pgd_steps=10 (default train/eval).
        "ResNet50-AT-steps10": OUTPUT_DIR
        / "adversarial_steps10"
        / "models"
        / "resnet50_pgd_at_final.h5",
        "DenseNet121-AT": OUTPUT_DIR
        / "densenet121"
        / "adversarial"
        / "models"
        / "densenet121_pgd_at_final.h5",
    }


def resnet50_pgd_at_checkpoint_key(eval_steps: int) -> str:
    """Map PGD evaluation step count to a ResNet50-AT checkpoint registry key."""
    if eval_steps == 10:
        return "ResNet50-AT-steps10"
    if eval_steps == 5:
        return "ResNet50-AT"
    return f"ResNet50-AT-steps{eval_steps}"


def pgd_standard_vs_at_dir() -> Path:
    return OUTPUT_DIR / "adversarial" / "pgd_standard_vs_at"


def pgd_standard_vs_at_steps_dir(steps: int, model: str = "resnet50") -> Path:
    base = pgd_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base / f"steps{steps}"
    return base / model.lower() / f"steps{steps}"


def pgd_standard_vs_at_figures_dir(steps: int, model: str = "resnet50") -> Path:
    base = pgd_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base / "figures" / f"steps{steps}"
    return base / "figures" / f"{model.lower()}_steps{steps}"


def bim_standard_vs_at_dir() -> Path:
    return OUTPUT_DIR / "adversarial" / "bim_standard_vs_at"


def bim_standard_vs_at_steps_dir(steps: int, model: str = "resnet50") -> Path:
    base = bim_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base / f"steps{steps}"
    return base / model.lower() / f"steps{steps}"


def bim_standard_vs_at_figures_dir(steps: int, model: str = "resnet50") -> Path:
    base = bim_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base / "figures" / f"steps{steps}"
    return base / "figures" / f"{model.lower()}_steps{steps}"


def fgsm_standard_vs_at_dir() -> Path:
    return OUTPUT_DIR / "adversarial" / "fgsm_standard_vs_at"


def fgsm_standard_vs_at_output_dir(model: str = "resnet50") -> Path:
    base = fgsm_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base
    return base / model.lower()


def fgsm_standard_vs_at_figures_dir(model: str = "resnet50") -> Path:
    base = fgsm_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base / "figures"
    return base / "figures" / model.lower()


def deepfool_standard_vs_at_dir() -> Path:
    return OUTPUT_DIR / "adversarial" / "deepfool_standard_vs_at"


def deepfool_standard_vs_at_output_dir(model: str = "resnet50") -> Path:
    base = deepfool_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base
    return base / model.lower()


def deepfool_standard_vs_at_figures_dir(model: str = "resnet50") -> Path:
    base = deepfool_standard_vs_at_dir()
    if model.lower() == "resnet50":
        return base / "figures"
    return base / "figures" / model.lower()


def train_output_dir(architecture: str) -> Path:
    if architecture == "resnet50":
        return OUTPUT_DIR
    return OUTPUT_DIR / architecture
