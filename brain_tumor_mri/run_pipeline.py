"""Brain tumor MRI task — full pipeline launcher."""
import argparse
import subprocess
import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_ROOT.parent
PYTHON = PROJECT_ROOT / ".venv-gpu" / "Scripts" / "python.exe"
SCRIPTS = TASK_ROOT / "scripts"

STEPS = {
    "download": [PYTHON, str(SCRIPTS / "download_dataset.py"), "--data-dir", str(TASK_ROOT / "data")],
    "train-resnet50": [
        PYTHON,
        str(SCRIPTS / "train_resnet50.py"),
        "--data-dir",
        str(TASK_ROOT / "data"),
        "--output-dir",
        str(TASK_ROOT / "outputs"),
    ],
    "train-densenet121": [
        PYTHON,
        str(SCRIPTS / "train_model.py"),
        "--arch",
        "densenet121",
        "--data-dir",
        str(TASK_ROOT / "data"),
    ],
    "train-resnet18": [
        PYTHON,
        str(SCRIPTS / "train_model.py"),
        "--arch",
        "resnet18",
        "--data-dir",
        str(TASK_ROOT / "data"),
    ],
    "train-convnext-tiny": [
        PYTHON,
        str(SCRIPTS / "train_model.py"),
        "--arch",
        "convnext_tiny",
        "--data-dir",
        str(TASK_ROOT / "data"),
    ],
    "train-mobilenetv2": [
        PYTHON,
        str(SCRIPTS / "train_model.py"),
        "--arch",
        "mobilenetv2",
        "--data-dir",
        str(TASK_ROOT / "data"),
    ],
    "fgsm": [
        PYTHON,
        str(SCRIPTS / "model_fgsm_comparison.py"),
        "--data-dir",
        str(TASK_ROOT / "data"),
        "--output-dir",
        str(TASK_ROOT / "outputs" / "model_comparison"),
    ],
    "pgd": [
        PYTHON,
        str(SCRIPTS / "model_pgd_comparison.py"),
        "--data-dir",
        str(TASK_ROOT / "data"),
        "--output-dir",
        str(TASK_ROOT / "outputs" / "model_comparison"),
    ],
    "deepfool": [
        PYTHON,
        str(SCRIPTS / "model_deepfool_comparison.py"),
        "--data-dir",
        str(TASK_ROOT / "data"),
        "--output-dir",
        str(TASK_ROOT / "outputs" / "model_comparison"),
    ],
    "fair": [
        PYTHON,
        str(SCRIPTS / "model_fair_robustness_comparison.py"),
        "--input-dir",
        str(TASK_ROOT / "outputs" / "model_comparison"),
    ],
    "train-pgd-at-resnet50": [
        PYTHON,
        str(SCRIPTS / "train_adversarial.py"),
        "--arch",
        "resnet50",
        "--data-dir",
        str(TASK_ROOT / "data"),
    ],
    "train-pgd-at-densenet121": [
        PYTHON,
        str(SCRIPTS / "train_adversarial.py"),
        "--arch",
        "densenet121",
        "--data-dir",
        str(TASK_ROOT / "data"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Brain tumor MRI task pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=[
            "download",
            "train-resnet50",
            "train-densenet121",
            "train-resnet18",
            "fgsm",
            "pgd",
            "deepfool",
            "fair",
        ],
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"Python not found: {PYTHON}", file=sys.stderr)
        sys.exit(1)

    extra = ["--skip-download"] if args.skip_download else []
    for step in args.steps:
        cmd = STEPS[step] + (extra if step.startswith("train") else [])
        print(">>>", " ".join(str(c) for c in cmd))
        subprocess.run(cmd, cwd=str(SCRIPTS), check=True)


if __name__ == "__main__":
    main()
