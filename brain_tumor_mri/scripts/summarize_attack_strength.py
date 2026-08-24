"""Summarize existing Brain MRI attack results into attack-strength tables (no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TASK_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = TASK_ROOT / "outputs" / "model_comparison"
PAPER_DIR = TASK_ROOT.parent / "paper_tables" / "brain_tumor_mri"

EPS_LABELS = ["2/255", "4/255"]
ATTACK_SOURCES = {
    "FGSM": OUT_DIR / "fgsm_model_comparison.csv",
    "BIM": OUT_DIR / "bim_model_comparison.csv",
    "PGD": OUT_DIR / "pgd_model_comparison.csv",
}
MIM_CSV = TASK_ROOT / "outputs" / "mifgsm_comparison.csv"
MODELS = ["ResNet50", "DenseNet121", "ResNet18", "ConvNeXt-Tiny", "MobileNetV2"]

BG = "#1e1e1e"
FG = "#e8e8e8"
GRID = "#3a3a3a"
HEADER_BG = "#2d2d2d"


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def load_attack_rows() -> pd.DataFrame:
    fgsm = pd.read_csv(ATTACK_SOURCES["FGSM"])
    fgsm = fgsm[fgsm["attack"].isin(["clean", "FGSM"])]

    bim = pd.read_csv(ATTACK_SOURCES["BIM"])
    bim = bim[bim["attack"] == "BIM"]

    pgd = pd.read_csv(ATTACK_SOURCES["PGD"])
    pgd = pgd[pgd["attack"] == "PGD"]

    mim = pd.read_csv(MIM_CSV)
    mim = mim[mim["attack"] == "MI-FGSM"].copy()
    mim["model"] = "ResNet50"
    mim["attack"] = "MIM"

    return pd.concat([fgsm, bim, pgd, mim], ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        clean_rows = df[(df["model"] == model) & (df["attack"] == "clean")]
        if clean_rows.empty:
            continue
        clean_acc = float(clean_rows.iloc[0]["accuracy"])
        clean_recall = float(clean_rows.iloc[0]["recall"])

        for eps_label in EPS_LABELS:
            eps_val = int(eps_label.split("/")[0]) / 255
            for attack in ["FGSM", "BIM", "PGD", "MIM"]:
                sub = df[
                    (df["model"] == model)
                    & (df["attack"] == attack)
                    & (df["epsilon_label"] == eps_label)
                ]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                robust = float(r["accuracy"])
                rows.append(
                    {
                        "task": "Brain MRI",
                        "model": model,
                        "epsilon_label": eps_label,
                        "epsilon": eps_val,
                        "attack": attack,
                        "robust_acc_pct": round(robust * 100, 2),
                        "accuracy_drop_pct": round((clean_acc - robust) * 100, 2),
                        "asr_pct": round(float(r["attack_success_rate"]) * 100, 2),
                        "sensitivity_pct": round(float(r["recall"]) * 100, 2),
                        "mean_linf": round(eps_val, 6),
                        "clean_acc_pct": round(clean_acc * 100, 2),
                        "steps": int(r["steps"]) if pd.notna(r.get("steps")) else 1,
                    }
                )
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, path: Path):
    lines = [
        "# Brain MRI — Attack Strength Summary (from existing results)",
        "",
        "Sources: `fgsm_model_comparison.csv`, `bim_model_comparison.csv`, "
        "`pgd_model_comparison.csv` (PGD/BIM steps=10); `mifgsm_comparison.csv` "
        "(MIM/MI-FGSM, ResNet50 only, steps=10).",
        "",
        "ε ∈ {2/255, 4/255}. Mean L∞ = attack budget ε (L∞ projection; not separately logged).",
        "Lower Robust Acc / higher ASR / Acc Drop ⇒ stronger attack.",
        "",
    ]
    for (model, eps_label), group in summary.groupby(["model", "epsilon_label"]):
        clean = group["clean_acc_pct"].iloc[0]
        lines.extend(
            [
                f"## {model} @ ε={eps_label} (clean acc={clean:.2f}%)",
                "",
                "| Attack | Robust Acc (%) | Acc Drop (%) | ASR (%) | Sensitivity (%) | Mean L∞ |",
                "|--------|----------------|--------------|---------|-----------------|---------|",
            ]
        )
        order = ["FGSM", "BIM", "PGD", "MIM"]
        group = group.set_index("attack").reindex([a for a in order if a in group["attack"].values]).reset_index()
        for _, row in group.iterrows():
            lines.append(
                f"| {row['attack']} | {row['robust_acc_pct']:.2f} | {row['accuracy_drop_pct']:.2f} | "
                f"{row['asr_pct']:.2f} | {row['sensitivity_pct']:.2f} | {row['mean_linf']:.6f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_png(group: pd.DataFrame, path: Path, model: str, eps_label: str, clean_acc: float):
    headers = ["Attack", "Robust Acc", "Acc Drop", "ASR", "Sensitivity", "Mean L∞"]
    order = ["FGSM", "BIM", "PGD", "MIM"]
    g = group.set_index("attack").reindex([a for a in order if a in group["attack"].values]).reset_index()
    data = [headers]
    for _, row in g.iterrows():
        data.append(
            [
                row["attack"],
                f"{row['robust_acc_pct']:.2f}",
                f"{row['accuracy_drop_pct']:.2f}",
                f"{row['asr_pct']:.2f}",
                f"{row['sensitivity_pct']:.2f}",
                f"{row['mean_linf']:.4f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 2.2 + 0.35 * len(g)))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(
        f"Brain MRI Attack Strength — {model} @ ε={eps_label} (clean={clean_acc:.2f}%)",
        color=FG,
        fontsize=12,
        pad=12,
    )
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
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

    raw = load_attack_rows()
    summary = build_summary(raw)

    csv_path = OUT_DIR / "attack_strength_summary.csv"
    md_path = OUT_DIR / "attack_strength_summary.md"
    summary.to_csv(csv_path, index=False)
    write_markdown(summary, md_path)

    for (model, eps_label), group in summary.groupby(["model", "epsilon_label"]):
        slug = model.lower().replace("-", "_")
        eps_slug = eps_label.replace("/", "_")
        png = PAPER_DIR / f"table_attack_strength_{slug}_eps{eps_slug}.png"
        render_png(group, png, model, eps_label, float(group["clean_acc_pct"].iloc[0]))
        print(f"Wrote {png}")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
