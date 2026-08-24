"""Run five-model PGD/BIM @ steps=10 and refresh derived tables/figures."""
from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TASK_ROOT / "scripts"
MC = TASK_ROOT / "outputs" / "model_comparison"
ROOT = TASK_ROOT.parent
PY = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
LOG = MC / f"run_pgd_bim_steps10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def run_step(label: str, cmd: list[str], log_handle):
    line = f"\n{'=' * 72}\n>>> {label}\n{'=' * 72}\n"
    print(line, flush=True)
    log_handle.write(line)
    log_handle.flush()
    subprocess.run(cmd, cwd=str(SCRIPTS), check=True, stdout=log_handle, stderr=subprocess.STDOUT)


def archive_steps5():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name in ("pgd_model_comparison.csv", "bim_model_comparison.csv"):
        src = MC / name
        if src.is_file():
            dst = MC / f"{src.stem}_steps5_backup_{ts}{src.suffix}"
            shutil.copy2(src, dst)
            print(f"Backed up {src.name} -> {dst.name}", flush=True)


def clear_progress():
    for pattern in (
        "pgd_model_comparison_progress.json",
        "bim_model_comparison_progress.json",
        "pgd_model_comparison_progress_steps10.json",
    ):
        p = MC / pattern
        if p.is_file():
            p.unlink()
            print(f"Removed {p.name}", flush=True)


def main():
    if not PY.is_file():
        print(f"Missing venv python: {PY}", file=sys.stderr)
        sys.exit(1)

    MC.mkdir(parents=True, exist_ok=True)
    archive_steps5()
    clear_progress()

    steps = "10"
    with LOG.open("w", encoding="utf-8") as log:
        run_step(
            f"PGD five-model comparison (steps={steps})",
            [str(PY), str(SCRIPTS / "model_pgd_comparison.py"), "--steps", steps],
            log,
        )
        run_step(
            f"BIM five-model comparison (steps={steps})",
            [str(PY), str(SCRIPTS / "model_bim_comparison.py"), "--steps", steps],
            log,
        )
        run_step(
            "Fair robustness comparison",
            [str(PY), str(SCRIPTS / "model_fair_robustness_comparison.py")],
            log,
        )
        run_step(
            "Attack strength summary",
            [str(PY), str(SCRIPTS / "summarize_attack_strength.py")],
            log,
        )
        run_step(
            "PGD robustness five-model table",
            [str(PY), str(SCRIPTS / "summarize_pgd_robustness_table.py")],
            log,
        )
        run_step(
            "Table04 tex",
            [str(PY), str(SCRIPTS / "update_table04_robustness_eps4.py")],
            log,
        )
        run_step(
            "Paper table PNGs (table04 + table09)",
            [
                str(PY),
                str(ROOT / "paper_tables" / "scripts" / "render_brain_mri_table_images.py"),
            ],
            log,
        )

    print(f"\nDone. Log: {LOG}", flush=True)


if __name__ == "__main__":
    main()
