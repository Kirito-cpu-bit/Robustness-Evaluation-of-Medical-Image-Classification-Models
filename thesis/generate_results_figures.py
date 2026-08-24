"""Generate dissertation figures directly from saved experimental outputs."""
from pathlib import Path
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thesis" / "dissertation_overleaf" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["ResNet50", "DenseNet121", "ResNet18", "ConvNeXt-Tiny", "MobileNetV2"]
ATTACKS = ["FGSM", "PGD", "BIM"]
COLORS = dict(zip(MODELS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]))

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "figure.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def read(task, name):
    return pd.read_csv(ROOT / task / "outputs" / "model_comparison" / name)


def heatmap(ax, values, rows, cols, title, fmt=".1f", vmin=0, vmax=100):
    im = ax.imshow(values, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = values[i, j]
            colour = "white" if val > 58 else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", color=colour, fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.5); spine.set_color("0.5")
    return im


# 1. Clean metrics across both tasks.
brain_clean = read("brain_tumor_mri", "brain_mri_clean_five_models.csv")
chest_rows = []
chest_dirs = {"ResNet50": "models", "DenseNet121": "densenet121", "ResNet18": "resnet18",
              "ConvNeXt-Tiny": "convnext_tiny", "MobileNetV2": "mobilenetv2"}
for model, folder in chest_dirs.items():
    data = pd.read_csv(ROOT / "chest_xray" / "outputs" / folder / "test_predictions.csv")
    y, p, score = data.label.to_numpy(), data.prediction.to_numpy(), data.probability.to_numpy()
    tp, tn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 0)).sum()
    fp, fn = ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
    acc = (tp + tn) / len(y); precision = tp / (tp + fp); recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    chest_rows.append([model, acc, f1])
chest_clean = pd.DataFrame(chest_rows, columns=["Model", "Accuracy", "F1"])

fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), sharey=True)
for ax, data, title in [(axes[0], brain_clean, "Brain MRI"), (axes[1], chest_clean, "Chest X-ray")]:
    x = np.arange(len(MODELS)); width = .36
    ordered = data.set_index("Model").loc[MODELS]
    ax.bar(x-width/2, ordered.Accuracy*100, width, label="Accuracy", color="#0072B2")
    ax.bar(x+width/2, ordered.F1*100, width, label="F1", color="#E69F00")
    ax.set_xticks(x, [m.replace("-Tiny", "\nTiny") for m in MODELS], rotation=24, ha="right")
    ax.set_ylim(75, 101); ax.grid(axis="y", alpha=.25); ax.set_title(title, fontweight="bold")
axes[0].set_ylabel("Score (%)")
axes[1].legend(frameon=False, ncol=2, loc="lower right")
fig.suptitle("Clean classification performance", fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "clean_performance_overview.pdf")


# 2. Clean confusion matrices, recomputed from saved test predictions.
fig, axes = plt.subplots(2, 5, figsize=(7.15, 3.25))
task_specs = [
    ("Brain MRI", ROOT / "brain_tumor_mri" / "outputs",
     {"ResNet50":"", "DenseNet121":"densenet121", "ResNet18":"resnet18", "ConvNeXt-Tiny":"convnext_tiny", "MobileNetV2":"mobilenetv2"},
     ["NO_TUMOR", "TUMOR"]),
    ("Chest X-ray", ROOT / "chest_xray" / "outputs", chest_dirs, ["NORMAL", "PNEUMONIA"]),
]
for row, (task, base, folders, labels) in enumerate(task_specs):
    for col, model in enumerate(MODELS):
        folder = folders[model]
        path = base / folder / "test_predictions.csv" if folder else base / "test_predictions.csv"
        d = pd.read_csv(path); y, p = d.label.to_numpy(), d.prediction.to_numpy()
        cm = np.array([[((y==0)&(p==0)).sum(), ((y==0)&(p==1)).sum()],
                       [((y==1)&(p==0)).sum(), ((y==1)&(p==1)).sum()]])
        norm = cm / cm.sum(axis=1, keepdims=True) * 100
        ax = axes[row, col]; ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n{norm[i,j]:.1f}%", ha="center", va="center",
                        fontsize=6.8, color="white" if norm[i,j] > 55 else "black")
        ax.set_xticks([0,1], labels, rotation=35, ha="right", fontsize=6.5)
        ax.set_yticks([0,1], labels if col == 0 else [], fontsize=6.5)
        ax.set_title(model.replace("-Tiny", "-Tiny"), fontsize=8, fontweight="bold")
        if col == 0: ax.set_ylabel(f"{task}\nTrue class", fontsize=8)
        if row == 1: ax.set_xlabel("Predicted class", fontsize=7)
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(.4)
fig.suptitle("Clean-test confusion matrices (count and row percentage)", fontweight="bold", y=1.015)
fig.tight_layout(pad=.55)
save(fig, "clean_confusion_matrices.pdf")


# 2b. Clinical sensitivity: clean recall and retention under attack.
fig = plt.figure(figsize=(7.15, 5.7))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=.48, wspace=.38)
for c, (task_dir, title) in enumerate([("brain_tumor_mri", "Brain MRI"), ("chest_xray", "Chest X-ray")]):
    ax = fig.add_subplot(gs[0, c])
    fgsm = read(task_dir, "fgsm_model_comparison.csv")
    clean_recall = np.array([
        fgsm[(fgsm.model == model) & (fgsm.epsilon_label == "0/255")].recall.iloc[0] * 100
        for model in MODELS
    ])
    bars = ax.bar(np.arange(len(MODELS)), clean_recall, color=[COLORS[m] for m in MODELS])
    ax.set_xticks(np.arange(len(MODELS)), ["R50", "Dense121", "R18", "ConvNeXt", "MobileV2"])
    ax.set_ylim(75, 101); ax.grid(axis="y", alpha=.25); ax.set_title(f"{title}: clean", fontweight="bold")
    if c == 0: ax.set_ylabel("Sensitivity / Recall (%)")
    for bar, value in zip(bars, clean_recall):
        ax.text(bar.get_x()+bar.get_width()/2, value+.35, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5)

for c, (task_dir, title) in enumerate([("brain_tumor_mri", "Brain MRI"), ("chest_xray", "Chest X-ray")]):
    ax = fig.add_subplot(gs[1, c])
    vals = np.zeros((len(MODELS), len(ATTACKS)))
    for i, model in enumerate(MODELS):
        for j, attack in enumerate(ATTACKS):
            d = read(task_dir, f"{attack.lower()}_model_comparison.csv")
            clean = d[(d.model==model) & (d.epsilon_label=="0/255")].recall.iloc[0]
            attacked = d[(d.model==model) & (d.epsilon_label=="4/255")].recall.iloc[0]
            vals[i,j] = attacked / clean * 100 if clean else 0
    im = heatmap(ax, vals, MODELS, ATTACKS, f"{title}: attacked")
fig.colorbar(im, ax=fig.axes, label="Clean sensitivity retained (%)", fraction=.022, pad=.025)
fig.suptitle(r"Sensitivity and retention under attack ($\epsilon=4/255$)", fontweight="bold", y=.995)
fig.subplots_adjust(left=.12, right=.90, bottom=.10, top=.92)
save(fig, "clinical_sensitivity_summary.pdf")


# 3. Full robustness curves: three attacks by two tasks.
fig, axes = plt.subplots(3, 2, figsize=(7.15, 7.0), sharex=True, sharey=True)
for c, (task_dir, task_title) in enumerate([("brain_tumor_mri", "Brain MRI"), ("chest_xray", "Chest X-ray")]):
    for r, attack in enumerate(ATTACKS):
        d = read(task_dir, f"{attack.lower()}_model_comparison.csv")
        ax = axes[r, c]
        for model in MODELS:
            z = d[d.model == model].sort_values("epsilon")
            ax.plot(z.epsilon*255, z.accuracy*100, marker="o", ms=3, lw=1.3,
                    color=COLORS[model], label=model)
        ax.set_title(f"{task_title}: {attack}", fontweight="bold")
        ax.grid(alpha=.25); ax.set_xlim(-.15, 8.15); ax.set_ylim(-2, 102)
        if c == 0: ax.set_ylabel("Robust accuracy (%)")
        if r == 2: ax.set_xlabel(r"Perturbation budget $\epsilon$ (/255)")
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, -.008))
fig.suptitle("Robustness degradation across models, attacks, and tasks", fontweight="bold", y=1.005)
fig.tight_layout(rect=[0, .045, 1, .98])
save(fig, "adversarial_robustness_curves.pdf")


# 4. Robust accuracy heatmaps at epsilon=4/255.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
for ax, (task_dir, title) in zip(axes, [("brain_tumor_mri", "Brain MRI"), ("chest_xray", "Chest X-ray")]):
    vals = np.zeros((len(MODELS), len(ATTACKS)))
    for j, attack in enumerate(ATTACKS):
        d = read(task_dir, f"{attack.lower()}_model_comparison.csv")
        for i, model in enumerate(MODELS):
            vals[i,j] = d[(d.model==model) & (d.epsilon_label=="4/255")].accuracy.iloc[0]*100
    im = heatmap(ax, vals, MODELS, ATTACKS, title)
fig.colorbar(im, ax=axes, label="Robust accuracy (%)", fraction=.03, pad=.025)
fig.suptitle(r"Architecture--attack interaction at $\epsilon=4/255$", fontweight="bold", y=1.02)
fig.subplots_adjust(left=.14, right=.91, bottom=.18, top=.80, wspace=.35)
save(fig, "robustness_heatmaps_eps4.pdf")


# 5. Accuracy retention heatmaps at epsilon=4/255.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
for ax, (task_dir, title) in zip(axes, [("brain_tumor_mri", "Brain MRI"), ("chest_xray", "Chest X-ray")]):
    vals = np.zeros((len(MODELS), len(ATTACKS)))
    for i, model in enumerate(MODELS):
        for j, attack in enumerate(ATTACKS):
            d = read(task_dir, f"{attack.lower()}_model_comparison.csv")
            clean = d[(d.model==model) & (d.epsilon_label=="0/255")].accuracy.iloc[0]
            robust = d[(d.model==model) & (d.epsilon_label=="4/255")].accuracy.iloc[0]
            vals[i,j] = robust / clean * 100
    im = heatmap(ax, vals, MODELS, ATTACKS, title)
fig.colorbar(im, ax=axes, label="Clean accuracy retained (%)", fraction=.03, pad=.025)
fig.suptitle(r"Fair robustness at $\epsilon=4/255$", fontweight="bold", y=1.02)
fig.subplots_adjust(left=.14, right=.91, bottom=.18, top=.80, wspace=.35)
save(fig, "fair_robustness_retention_eps4.pdf")


# 6. ResNet50 attack-strength comparison across tasks and budgets.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), sharey=True)
for ax, (task_dir, title) in zip(axes, [("brain_tumor_mri", "Brain MRI"), ("chest_xray", "Chest X-ray")]):
    d = read(task_dir, "attack_strength_summary.csv")
    d = d[(d.model=="ResNet50") & d.attack.isin(["FGSM","PGD","BIM","MIM"])]
    attacks = ["FGSM","BIM","PGD","MIM"]; x=np.arange(4); w=.36
    for k, eps in enumerate(["2/255","4/255"]):
        vals=[d[(d.attack==a)&(d.epsilon_label==eps)].robust_acc_pct.iloc[0] for a in attacks]
        ax.bar(x+(k-.5)*w, vals, w, label=eps, color=["#56B4E9","#D55E00"][k])
    ax.set_xticks(x, attacks); ax.set_ylim(0,100); ax.grid(axis="y", alpha=.25)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Attack")
axes[0].set_ylabel("Robust accuracy (%)"); axes[1].legend(title=r"$\epsilon$", frameon=False)
fig.suptitle("ResNet50 attack-strength comparison", fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "resnet50_attack_strength_comparison.pdf")


# 7. Cross-attack generalisation of adversarial training at epsilon=4/255.
std_files = {"FGSM":"fgsm", "PGD":"pgd", "BIM":"bim"}
values = {}
for model, folder in [("ResNet50", ""), ("DenseNet121", "densenet121")]:
    values[model] = {}
    for attack, stem in std_files.items():
        if attack == "FGSM":
            path = ROOT/"brain_tumor_mri"/"outputs"/"adversarial"/"fgsm_standard_vs_at"/folder/f"fgsm_{model.lower()}_standard_vs_at.csv"
        else:
            path = ROOT/"brain_tumor_mri"/"outputs"/"adversarial"/f"{stem}_standard_vs_at"/folder/"steps10"/f"{stem}_{model.lower()}_standard_vs_at.csv"
        d = pd.read_csv(path)
        std_main = read("brain_tumor_mri", f"{stem}_model_comparison.csv")
        standard = std_main[(std_main.model==model)&(std_main.epsilon_label=="4/255")].accuracy.iloc[0]*100
        at = d[(d.model.str.endswith("-AT"))&(d.epsilon_label=="4/255")].accuracy.iloc[0]*100
        values[model][attack] = (standard, at)
fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), sharey=True)
for ax, model in zip(axes, ["ResNet50","DenseNet121"]):
    x=np.arange(3); w=.36
    ax.bar(x-w/2,[values[model][a][0] for a in ATTACKS],w,label="Standard",color="#999999")
    ax.bar(x+w/2,[values[model][a][1] for a in ATTACKS],w,label="PGD-AT",color="#009E73")
    ax.set_xticks(x,ATTACKS); ax.set_ylim(0,100); ax.grid(axis="y",alpha=.25); ax.set_title(model,fontweight="bold")
axes[0].set_ylabel("Robust accuracy (%)"); axes[1].legend(frameon=False)
fig.suptitle(r"Cross-attack effect of PGD adversarial training at $\epsilon=4/255$",fontweight="bold",y=1.01)
fig.tight_layout(); save(fig,"pgd_at_cross_attack_eps4.pdf")


# 8. Gaussian defence confusion matrices at epsilon=4/255.
g = pd.read_csv(ROOT/"brain_tumor_mri"/"outputs"/"defense"/"gaussian"/"gaussian_defense_comparison.csv")
fig, axes = plt.subplots(2,4,figsize=(7.15,3.25))
for r, attack in enumerate(["PGD","BIM"]):
    for c, cond in enumerate(["D0","D1","D2","D3"]):
        q=g[(g.attack==attack)&(g.condition==cond)&(g.epsilon_label=="4/255")].iloc[0]
        cm=np.array(ast.literal_eval(q.confusion_matrix)); norm=cm/cm.sum(axis=1,keepdims=True)*100
        ax=axes[r,c]; ax.imshow(norm,cmap="Blues",vmin=0,vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{cm[i,j]}\n{norm[i,j]:.1f}%",ha="center",va="center",fontsize=7,
                        color="white" if norm[i,j]>55 else "black")
        ax.set_xticks([0,1],["NO_TUMOR","TUMOR"],rotation=30,ha="right",fontsize=6.5)
        ax.set_yticks([0,1],["NO_TUMOR","TUMOR"] if c==0 else [],fontsize=6.5)
        ax.set_title(cond,fontsize=8,fontweight="bold")
        if c==0: ax.set_ylabel(f"{attack}\nTrue class",fontsize=8)
        if r==1: ax.set_xlabel("Predicted class",fontsize=7)
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(.4)
fig.suptitle(r"Gaussian defence error patterns at $\epsilon=4/255$",fontweight="bold",y=1.015)
fig.tight_layout(pad=.55); save(fig,"gaussian_defence_confusion_matrices_eps4.pdf")


# 9. DeepFool standard versus adversarially trained curve.
d=pd.read_csv(ROOT/"brain_tumor_mri"/"outputs"/"adversarial"/"deepfool_standard_vs_at"/"deepfool_resnet50_standard_vs_at.csv")
fig,ax=plt.subplots(figsize=(5.3,3.0))
for model,color in [("ResNet50","#999999"),("ResNet50-AT","#009E73")]:
    q=d[d.model==model].sort_values("epsilon")
    ax.plot(q.epsilon*255,q.accuracy*100,marker="o",lw=1.7,label=model.replace("-AT"," PGD-AT"),color=color)
ax.set(xlabel=r"Perturbation cap $\epsilon$ (/255)",ylabel="Robust accuracy (%)",xlim=(-.15,8.15),ylim=(0,102))
ax.grid(alpha=.25); ax.legend(frameon=False); ax.set_title("DeepFool robustness on Brain MRI",fontweight="bold")
fig.tight_layout(); save(fig,"deepfool_standard_vs_at.pdf")

print(f"Generated figures in {OUT}")
