"""Render Brain MRI paper tables as dark-theme PNG images."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

OUT_DIR = Path(__file__).resolve().parents[1] / "brain_tumor_mri"

BG = "#1e1e1e"
FG = "#e8e8e8"
GRID = "#3a3a3a"
HEADER_BG = "#2d2d2d"
BEST = "#ffffff"


def _style_table(table, n_rows: int, n_cols: int, bold_cells: set[tuple[int, int]] | None = None):
    bold_cells = bold_cells or set()
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=FG, weight="bold", ha="center", va="center")
        else:
            cell.set_facecolor(BG)
            weight = "bold" if (row, col) in bold_cells else "normal"
            color = BEST if (row, col) in bold_cells else FG
            cell.set_text_props(color=color, weight=weight, ha="center", va="center")


def _save(fig, filename: str):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG, edgecolor=BG)
    plt.close(fig)
    print(f"Wrote {path}")


def render_table03():
    headers = ["Task", "Model", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "ROC-AUC"]
    rows = [
        ["Brain MRI", "ResNet50", "97.71", "97.43", "99.61", "91.77", "98.51", "0.9944"],
        ["Brain MRI", "DenseNet121", "97.57", "98.35", "98.44", "94.82", "98.40", "0.9959"],
        ["Brain MRI", "ResNet18", "91.00", "92.10", "96.40", "74.09", "94.20", "0.9588"],
        ["Brain MRI", "ConvNeXt-Tiny", "98.30", "99.61", "98.15", "98.78", "98.87", "0.9973"],
        ["Brain MRI", "MobileNetV2", "85.77", "95.93", "84.82", "88.72", "90.04", "0.9400"],
    ]
    bold = {(4, c) for c in (2, 3, 5, 6, 7)}  # ConvNeXt-Tiny best cols (not sensitivity)
    data = [headers] + rows
    fig, ax = plt.subplots(figsize=(14, 2.8))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    _style_table(table, len(data), len(headers), bold)
    _save(fig, "table03_clean_brain_mri_five_models.png")


def render_table04():
    mc = Path(__file__).resolve().parents[2] / "brain_tumor_mri" / "outputs" / "model_comparison"
    eps = "4/255"
    pgd_df = pd.read_csv(mc / "pgd_model_comparison.csv")
    fgsm_df = pd.read_csv(mc / "fgsm_model_comparison.csv")
    ps = int(pgd_df[pgd_df["attack"] == "PGD"]["steps"].dropna().iloc[0])
    bim_path = mc / "bim_model_comparison.csv"
    if bim_path.is_file():
        bim_df = pd.read_csv(bim_path)
        bs = int(bim_df[bim_df["attack"] == "BIM"]["steps"].dropna().iloc[0])
    else:
        bim_df = None
        bs = 10

    def acc(model, attack, df):
        r = df[(df["model"] == model) & (df["attack"] == attack) & (df["epsilon_label"] == eps)]
        return f"{float(r.iloc[0]['accuracy']) * 100:.2f}"

    headers = ["Model", "FGSM", f"PGD (steps={ps})", f"BIM (steps={bs})"]
    models = ["ResNet50", "DenseNet121", "ResNet18", "ConvNeXt-Tiny", "MobileNetV2"]
    rows = []
    for m in models:
        bim_val = acc(m, "BIM", bim_df) if bim_df is not None else "--"
        rows.append([m, acc(m, "FGSM", fgsm_df), acc(m, "PGD", pgd_df), bim_val])
    data = [headers] + rows
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    _style_table(table, len(data), len(headers))
    _save(fig, "table04_robustness_five_models_eps4.png")


def _render_at_curve_table(filename: str, attack_label: str, resnet_std, resnet_at, densenet_std, densenet_at):
    headers = ["Model", "Variant", "ε=1/255", "ε=2/255", "ε=3/255", "ε=4/255", "ε=8/255"]
    rows = [
        ["ResNet50", "Standard", *resnet_std],
        ["", "PGD-AT", *resnet_at],
        ["DenseNet121", "Standard", *densenet_std],
        ["", "PGD-AT", *densenet_at],
    ]
    data = [headers] + rows
    fig, ax = plt.subplots(figsize=(12, 3.2))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    _style_table(table, len(data), len(headers))
    ax.set_title(f"Standard vs PGD-AT under {attack_label}", color=FG, fontsize=12, pad=12)
    _save(fig, filename)


def render_table05():
    _render_at_curve_table(
        "table05_pgd_at_pgd_attack_steps10.png",
        "PGD (steps=10)",
        ["86.95", "61.28", "27.21", "7.60", "0.00"],
        ["98.53", "97.35", "94.76", "89.38", "34.07"],
        ["0.81", "0.00", "0.00", "0.00", "0.00"],
        ["98.16", "97.64", "94.76", "88.42", "30.16"],
    )


def render_table06():
    headers = ["Model", "Variant", "PGD", "BIM", "FGSM"]
    rows = [
        ["ResNet50", "Standard", "7.60", "42.70", "53.39"],
        ["", "PGD-AT", "89.38", "80.24", "90.71"],
        ["DenseNet121", "Standard", "0.00", "1.33", "22.27"],
        ["", "PGD-AT", "88.42", "77.58", "89.53"],
    ]
    data = [headers] + rows
    fig, ax = plt.subplots(figsize=(9, 3.0))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    _style_table(table, len(data), len(headers))
    ax.set_title("Cross-attack @ ε=4/255 (PGD/BIM steps=10)", color=FG, fontsize=12, pad=12)
    _save(fig, "table06_pgd_at_cross_attack_eps4.png")


def render_table07():
    _render_at_curve_table(
        "table07_pgd_at_bim_steps10.png",
        "BIM (steps=10)",
        ["82.30", "52.95", "44.10", "42.70", "42.04"],
        ["98.16", "96.31", "90.49", "80.24", "19.54"],
        ["68.66", "21.53", "4.94", "1.33", "0.07"],
        ["98.08", "96.31", "89.16", "77.58", "16.67"],
    )


def render_table08():
    _render_at_curve_table(
        "table08_pgd_at_fgsm.png",
        "FGSM",
        ["86.43", "71.39", "58.85", "53.39", "44.62"],
        ["98.45", "97.27", "94.47", "90.71", "66.22"],
        ["78.61", "53.83", "35.40", "22.27", "5.01"],
        ["98.08", "97.12", "94.03", "89.53", "58.41"],
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    render_table03()
    render_table04()
    render_table05()
    render_table06()
    render_table07()
    render_table08()


if __name__ == "__main__":
    main()
