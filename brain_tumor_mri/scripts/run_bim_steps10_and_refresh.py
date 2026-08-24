"""Run five-model BIM @ steps=10 and refresh derived tables/figures (steps=10 only)."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TASK_ROOT / "scripts"
MC = TASK_ROOT / "outputs" / "model_comparison"
ROOT = TASK_ROOT.parent
PY = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
LOG = MC / f"run_bim_steps10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

STEPS = "10"
ATTACK_BATCH = "4"


def run_step(label: str, cmd: list[str], log_handle):
    line = f"\n{'=' * 72}\n>>> {label}\n{'=' * 72}\n"
    print(line, flush=True)
    log_handle.write(line)
    log_handle.flush()
    subprocess.run(cmd, cwd=str(SCRIPTS), check=True, stdout=log_handle, stderr=subprocess.STDOUT)


def main():
    if not PY.is_file():
        print(f"Missing venv python: {PY}", file=sys.stderr)
        sys.exit(1)

    MC.mkdir(parents=True, exist_ok=True)

    with LOG.open("w", encoding="utf-8") as log:
        run_step(
            f"BIM five-model comparison (steps={STEPS}, attack-batch={ATTACK_BATCH})",
            [
                str(PY),
                str(SCRIPTS / "model_bim_comparison.py"),
                "--steps",
                STEPS,
                "--attack-batch-size",
                ATTACK_BATCH,
            ],
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
            "PGD robustness five-model table (steps from CSV)",
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
