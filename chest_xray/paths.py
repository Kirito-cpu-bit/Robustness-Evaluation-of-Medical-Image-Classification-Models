"""Root paths for the chest X-ray pneumonia task."""
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
DATA_DIR = TASK_ROOT / "data"
OUTPUT_DIR = TASK_ROOT / "outputs"
MODEL_COMPARISON_DIR = OUTPUT_DIR / "model_comparison"
