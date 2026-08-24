"""Train ResNet50 PGD-AT (steps=10) then run standard vs AT PGD comparison."""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
TASK_ROOT = SCRIPTS.parent
PYTHON = TASK_ROOT.parent / ".venv-gpu" / "Scripts" / "python.exe"
AT_OUT = TASK_ROOT / "outputs" / "adversarial_steps10"

sys.path.insert(0, str(SCRIPTS))
from config import pgd_standard_vs_at_steps_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"Python not found: {PYTHON}", file=sys.stderr)
        sys.exit(1)

    AT_OUT.mkdir(parents=True, exist_ok=True)
    train_log = AT_OUT / f"pgd_at_resnet50_steps{args.pgd_steps}.log"

    if not args.skip_train:
        cmd = [
            str(PYTHON),
            str(SCRIPTS / "train_adversarial.py"),
            "--arch",
            "resnet50",
            "--data-dir",
            str(TASK_ROOT / "data"),
            "--output-dir",
            str(AT_OUT),
            "--pgd-steps",
            str(args.pgd_steps),
            "--skip-download",
        ]
        print(">>>", " ".join(cmd), flush=True)
        with train_log.open("w", encoding="utf-8") as handle:
            subprocess.run(cmd, cwd=str(SCRIPTS), check=True, stdout=handle, stderr=subprocess.STDOUT)

    at_ckpt = AT_OUT / "models" / "resnet50_pgd_at_final.h5"
    if not at_ckpt.is_file():
        raise FileNotFoundError(f"Missing AT checkpoint: {at_ckpt}")

    if not args.skip_compare:
        compare_out = pgd_standard_vs_at_steps_dir(args.pgd_steps)
        compare_log = compare_out / f"pgd_compare_steps{args.pgd_steps}.log"
        compare_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PYTHON),
            str(SCRIPTS / "model_pgd_standard_vs_at.py"),
            "--data-dir",
            str(TASK_ROOT / "data"),
            "--output-dir",
            str(compare_out),
            "--at-checkpoint",
            str(at_ckpt),
            "--steps",
            str(args.pgd_steps),
        ]
        print(">>>", " ".join(cmd), flush=True)
        with compare_log.open("w", encoding="utf-8") as handle:
            subprocess.run(cmd, cwd=str(SCRIPTS), check=True, stdout=handle, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
