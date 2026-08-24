"""Resume PGD @ steps=10 from pgd_model_comparison_progress.json, then run BIM @ steps=10."""

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

LOG = MC / f"run_pgd_bim_steps10_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"



# Smaller batches reduce DirectML TDR risk on long PGD loops (ConvNeXt).

PGD_ATTACK_BATCH = "8"





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



    progress = MC / "pgd_model_comparison_progress.json"

    if not progress.is_file():

        print(f"Missing PGD progress file (need --resume source): {progress}", file=sys.stderr)

        sys.exit(1)



    MC.mkdir(parents=True, exist_ok=True)

    steps = "10"



    with LOG.open("w", encoding="utf-8") as log:

        run_step(

            f"PGD five-model comparison (steps={steps}, --resume, attack-batch={PGD_ATTACK_BATCH})",

            [

                str(PY),

                str(SCRIPTS / "model_pgd_comparison.py"),

                "--steps",

                steps,

                "--resume",

                "--attack-batch-size",

                PGD_ATTACK_BATCH,

            ],

            log,

        )

        run_step(

            f"BIM five-model comparison (steps={steps})",

            [str(PY), str(SCRIPTS / "model_bim_comparison.py"), "--steps", steps],

            log,

        )



    print(f"\nDone. Log: {LOG}", flush=True)





if __name__ == "__main__":

    main()

