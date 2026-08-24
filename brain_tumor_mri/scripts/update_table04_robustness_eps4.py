"""Regenerate table04 (FGSM/PGD/BIM @ eps=4/255) from model_comparison CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TASK_ROOT = Path(__file__).resolve().parents[1]
MC = TASK_ROOT / "outputs" / "model_comparison"
PAPER = TASK_ROOT.parent / "paper_tables" / "brain_tumor_mri"

MODEL_ORDER = ["ResNet50", "DenseNet121", "ResNet18", "ConvNeXt-Tiny", "MobileNetV2"]
EPS = "4/255"


def _acc_pct(csv_path: Path, attack: str, model: str) -> float:
    df = pd.read_csv(csv_path)
    row = df[
        (df["model"] == model) & (df["attack"] == attack) & (df["epsilon_label"] == EPS)
    ]
    if row.empty:
        raise ValueError(f"Missing {attack} @ {EPS} for {model} in {csv_path}")
    return round(float(row.iloc[0]["accuracy"]) * 100, 2)


def pgd_steps() -> int:
    df = pd.read_csv(MC / "pgd_model_comparison.csv")
    sub = df[df["attack"] == "PGD"]["steps"].dropna()
    return int(sub.iloc[0]) if len(sub) else 10


def bim_steps() -> int:
    path = MC / "bim_model_comparison.csv"
    if not path.is_file():
        return 10
    df = pd.read_csv(path)
    sub = df[df["attack"] == "BIM"]["steps"].dropna()
    return int(sub.iloc[0]) if len(sub) else 10


def bim_acc_pct(model: str) -> str:
    path = MC / "bim_model_comparison.csv"
    if not path.is_file():
        return "--"
    return str(_acc_pct(path, "BIM", model))


def main():
    fgsm_csv = MC / "fgsm_model_comparison.csv"
    pgd_csv = MC / "pgd_model_comparison.csv"
    ps, bs = pgd_steps(), bim_steps()

    rows = []
    for model in MODEL_ORDER:
        rows.append(
            {
                "model": model,
                "fgsm": _acc_pct(fgsm_csv, "FGSM", model),
                "pgd": _acc_pct(pgd_csv, "PGD", model),
                "bim": bim_acc_pct(model),
            }
        )

    bim_note = "" if (MC / "bim_model_comparison.csv").is_file() else " (BIM column pending rerun)."

    tex_path = PAPER / "table04_robustness_five_models_eps4.tex"
    tex = f"""% Table 4: Standard-model robustness at epsilon = 4/255 (Brain MRI)
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Adversarial accuracy (\\%) of five standard models on Brain MRI at $\\epsilon=4/255$.}}
  \\label{{tab:brain_mri_robustness_eps4}}
  \\begin{{tabular}}{{lrrr}}
    \\toprule
    Model & FGSM & PGD (steps={ps}) & BIM (steps={bs}) \\\\
    \\midrule
"""
    for r in rows:
        bim_cell = r["bim"] if r["bim"] == "--" else f"{float(r['bim']):5.2f}"
        tex += (
            f"    {r['model']:<15} & {float(r['fgsm']):5.2f} & {float(r['pgd']):5.2f} "
            f"& {bim_cell} \\\\\n"
        )
    tex += f"""    \\bottomrule
  \\end{{tabular}}
  \\vspace{{0.25em}}
  \\begin{{minipage}}{{0.92\\linewidth}}
    \\footnotesize
    PGD: random start, $\\alpha=\\epsilon/\\text{{steps}}$, steps={ps}.
    BIM: no random start, $\\alpha=\\epsilon/\\text{{steps}}$, steps={bs}.{bim_note}
  \\end{{minipage}}
\\end{{table}}
"""
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Wrote {tex_path} (PGD steps={ps}, BIM steps={bs})")


if __name__ == "__main__":
    main()
