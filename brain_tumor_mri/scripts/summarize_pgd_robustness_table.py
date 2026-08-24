"""Build Brain MRI five-model PGD robustness table from existing comparison CSV."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TASK_ROOT = Path(__file__).resolve().parents[1]
PGD_CSV = TASK_ROOT / "outputs" / "model_comparison" / "pgd_model_comparison.csv"
OUT_DIR = TASK_ROOT / "outputs" / "model_comparison"
PAPER_DIR = TASK_ROOT.parent / "paper_tables" / "brain_tumor_mri"

MODEL_ORDER = ["ResNet50", "DenseNet121", "ResNet18", "ConvNeXt-Tiny", "MobileNetV2"]
EPS_LABELS = ["1/255", "2/255", "4/255"]

BG = "#1e1e1e"
FG = "#e8e8e8"
GRID = "#3a3a3a"
HEADER_BG = "#2d2d2d"


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        g = df[df["model"] == model]
        clean = g[g["attack"] == "clean"].iloc[0]
        clean_acc = float(clean["accuracy"])
        clean_recall = float(clean["recall"])
        clean_auc = float(clean["adv_auc"])

        eps_rows = {}
        norm_auc_parts = []
        for eps_label in EPS_LABELS:
            r = g[(g["attack"] == "PGD") & (g["epsilon_label"] == eps_label)].iloc[0]
            eps_rows[eps_label] = r
            adv_auc = float(r["adv_auc"]) if pd.notna(r["adv_auc"]) else 0.0
            if clean_auc > 0:
                norm_auc_parts.append(adv_auc / clean_auc)

        r2 = eps_rows["2/255"]
        robust_f1_2 = f1(float(r2["precision"]), float(r2["recall"]))
        sens_ret_2 = float(r2["recall"]) / clean_recall if clean_recall > 0 else 0.0

        rows.append(
            {
                "task": "Brain MRI",
                "attack": "PGD",
                "model": model,
                "clean_acc_pct": round(clean_acc * 100, 2),
                "robust_acc_1_255_pct": round(float(eps_rows["1/255"]["accuracy"]) * 100, 2),
                "robust_acc_2_255_pct": round(float(eps_rows["2/255"]["accuracy"]) * 100, 2),
                "robust_acc_4_255_pct": round(float(eps_rows["4/255"]["accuracy"]) * 100, 2),
                "normalized_r_auc": round(sum(norm_auc_parts) / len(norm_auc_parts), 4),
                "asr_2_255_pct": round(float(r2["attack_success_rate"]) * 100, 2),
                "sens_retention_2_255_pct": round(sens_ret_2 * 100, 2),
                "robust_f1_2_255_pct": round(robust_f1_2 * 100, 2),
                "pgd_steps": int(float(eps_rows["1/255"]["steps"])) if pd.notna(eps_rows["1/255"]["steps"]) else 5,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, path: Path, steps: int):
    lines = [
        "# Brain MRI — PGD Robustness (Five Models)",
        "",
        f"Attack: **PGD** (random start, steps={steps}, α=ε/steps). Test set n=1356.",
        "",
        "**Normalized R-AUC** = mean(Adv AUC / Clean AUC) @ ε ∈ {1/255, 2/255, 4/255}.",
        "**Sens. Retention @2/255** = Recall@PGD(ε=2/255) / Recall@Clean.",
        "**Robust F1 @2/255** = F1 under PGD @ ε=2/255.",
        "",
        "| Model | Clean Acc. | Robust Acc. @1/255 | Robust Acc. @2/255 | Robust Acc. @4/255 | Normalized R-AUC | ASR @2/255 | Sens. Retention @2/255 | Robust F1 @2/255 |",
        "| ----- | ---------: | -----------------: | -----------------: | -----------------: | ---------------: | ---------: | ---------------------: | ---------------: |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['model']} | {row['clean_acc_pct']:.2f} | {row['robust_acc_1_255_pct']:.2f} | "
            f"{row['robust_acc_2_255_pct']:.2f} | {row['robust_acc_4_255_pct']:.2f} | "
            f"{row['normalized_r_auc']:.4f} | {row['asr_2_255_pct']:.2f} | "
            f"{row['sens_retention_2_255_pct']:.2f} | {row['robust_f1_2_255_pct']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(table: pd.DataFrame, path: Path, steps: int):
    lines = [
        "% Table: Brain MRI PGD robustness (five models)",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{PGD robustness comparison on Brain MRI ($n=1356$, steps={steps}). "
        "Normalized R-AUC is the mean Adv-AUC retention @ $\\epsilon\\in\\{1,2,4\\}/255$.}}",
        "  \\label{tab:brain_mri_pgd_robustness_five_models}",
        "  \\begin{tabular}{lrrrrrrrr}",
        "    \\toprule",
        "    Model & Clean Acc. & Rob.@1/255 & Rob.@2/255 & Rob.@4/255 & Norm R-AUC & ASR@2/255 & Sens.Ret.@2/255 & Rob.F1@2/255 \\\\",
        "    \\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"    {row['model']} & {row['clean_acc_pct']:.2f} & {row['robust_acc_1_255_pct']:.2f} & "
            f"{row['robust_acc_2_255_pct']:.2f} & {row['robust_acc_4_255_pct']:.2f} & "
            f"{row['normalized_r_auc']:.4f} & {row['asr_2_255_pct']:.2f} & "
            f"{row['sens_retention_2_255_pct']:.2f} & {row['robust_f1_2_255_pct']:.2f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def render_png(table: pd.DataFrame, path: Path, steps: int):
    headers = [
        "Model",
        "Clean Acc.",
        "Rob.@1/255",
        "Rob.@2/255",
        "Rob.@4/255",
        "Norm R-AUC",
        "ASR@2/255",
        "Sens.Ret.@2/255",
        "Rob.F1@2/255",
    ]
    data = [headers]
    for _, row in table.iterrows():
        data.append(
            [
                row["model"],
                f"{row['clean_acc_pct']:.2f}",
                f"{row['robust_acc_1_255_pct']:.2f}",
                f"{row['robust_acc_2_255_pct']:.2f}",
                f"{row['robust_acc_4_255_pct']:.2f}",
                f"{row['normalized_r_auc']:.4f}",
                f"{row['asr_2_255_pct']:.2f}",
                f"{row['sens_retention_2_255_pct']:.2f}",
                f"{row['robust_f1_2_255_pct']:.2f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(14, 3.2))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(
        f"Brain MRI PGD Robustness — Five Models (steps={steps}, ε=1/255, 2/255, 4/255)",
        color=FG,
        fontsize=12,
        pad=14,
    )
    tbl = ax.table(cellText=data, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=FG, weight="bold")
        else:
            cell.set_facecolor(BG)
            cell.set_text_props(color=FG)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PGD_CSV)
    table = build_table(df)
    steps = int(table["pgd_steps"].iloc[0])

    csv_path = OUT_DIR / "pgd_robustness_five_models.csv"
    md_path = OUT_DIR / "pgd_robustness_five_models.md"
    tex_path = PAPER_DIR / "table09_pgd_robustness_five_models.tex"
    png_path = PAPER_DIR / "table09_pgd_robustness_five_models.png"

    table.drop(columns=["pgd_steps"]).to_csv(csv_path, index=False)
    write_markdown(table, md_path, steps)
    write_latex(table, tex_path, steps)
    render_png(table, png_path, steps)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
