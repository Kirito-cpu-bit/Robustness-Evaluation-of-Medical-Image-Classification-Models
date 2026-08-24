# Dual-Dataset Adversarial Robustness for Medical Image Classification

**Author:** Xihao Wang (UCL MSc AI for Biomedicine and Healthcare)  
**Dissertation:** *Robustness Evaluation of Medical Image Classification Models: A Study on Brain MRI and Chest X-Ray*

This repository implements the full experimental pipeline for the dissertation: fine-tuning five ImageNet-pretrained CNNs on two medical imaging tasks, evaluating white-box **L∞** adversarial attacks, comparing robustness across modalities, and running **PGD adversarial training (PGD-AT)** plus **Gaussian smoothing** defences on brain MRI.

---

## Repository layout

```
.
├── brain_tumor_mri/          # Brain MRI (NO_TUMOR vs TUMOR)
│   ├── scripts/              # Training, attacks, PGD-AT, Gaussian D0–D3
│   ├── data/                 # Downloaded / extracted dataset (not in git)
│   ├── outputs/              # Checkpoints and CSV/JSON results
│   └── run_pipeline.py       # Task orchestrator
├── chest_xray/               # Chest X-ray (NORMAL vs PNEUMONIA)
│   ├── scripts/              # Training and cross-model attack comparison
│   ├── data/
│   ├── outputs/
│   └── run_pipeline.py
├── paper_tables/             # LaTeX tables exported from experiments
├── thesis/                   # Dissertation source (LaTeX)
├── docs/
│   └── DEVELOPER_MANUAL.md   # Setup, dependencies, script map (see Appendix)
├── requirements.txt
└── README.md                 # This file
```

Each task directory is **self-contained**: run scripts from `brain_tumor_mri/` or `chest_xray/` with that task’s `data/` and `outputs/`.

---

## How the code works (high level)

1. **Data** — Datasets are downloaded (Hugging Face or Mendeley mirror), merged, and stratified into 70/15/15 train/val/test splits (`data_utils.py`, `download_dataset.py`).
2. **Training** — Five architectures (ResNet50, DenseNet121, ResNet18, MobileNetV2, ConvNeXt-Tiny) are fine-tuned with a two-phase transfer-learning schedule (`train_resnet50.py`, `train_model.py`).
3. **Attacks** — FGSM, PGD, BIM, DeepFool, MIM, and C&W generate adversarial examples on the test set; metrics are written to `outputs/model_comparison/` (`fgsm_attack.py`, `pgd_attack.py`, `model_*_comparison.py`).
4. **Fair comparison** — Relative degradation (Adv AUC/F1, relative drop vs clean baseline) is computed across models (`model_fair_robustness_comparison.py`).
5. **Defences (brain MRI only)** — PGD-AT (`train_adversarial.py`) and four Gaussian conditions D0–D3 (`evaluate_gaussian_defense.py`).

The entry point for most workflows is **`run_pipeline.py`** in each task folder, which chains the above steps via subprocess calls.

---

## Quick start

### 1. Environment

```powershell
# Windows (tested with Python 3.8 + DirectML on RTX 4060 Laptop GPU)
python -m venv .venv-gpu
.\.venv-gpu\Scripts\pip install -r requirements.txt
.\.venv-gpu\Scripts\pip install datasets huggingface_hub tensorflow-directml-plugin
```

On Linux with CUDA, install `tensorflow` matching your CUDA version instead of DirectML.

### 2. Brain MRI pipeline

```powershell
cd brain_tumor_mri
..\.venv-gpu\Scripts\python.exe run_pipeline.py --steps download
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download --steps train-resnet50 train-densenet121 fgsm pgd fair
```

PGD-AT and Gaussian defence:

```powershell
..\.venv-gpu\Scripts\python.exe scripts\train_adversarial.py --arch resnet50 --data-dir data
..\.venv-gpu\Scripts\python.exe scripts\evaluate_gaussian_defense.py --data-dir data
```

### 3. Chest X-ray pipeline

```powershell
cd chest_xray
..\.venv-gpu\Scripts\python.exe run_pipeline.py --steps download
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download --steps train-resnet50 train-densenet121 fgsm pgd bim deepfool fair
```

---

## Outputs

| Location | Contents |
|----------|----------|
| `*/outputs/models/` | ResNet50 checkpoint (`cnn_4_final.h5`) |
| `*/outputs/<arch>/models/` | Other architecture checkpoints |
| `*/outputs/model_comparison/` | Per-attack CSV/JSON, robustness curves |
| `brain_tumor_mri/outputs/defense/gaussian/` | D0–D3 Gaussian results |
| `brain_tumor_mri/outputs/adversarial/` | PGD-AT checkpoints and standard-vs-AT tables |
| `paper_tables/` | LaTeX fragments for the dissertation |

---

## Reproducibility

- Random seed **42** for stratified splits (`data_utils.py`).
- Attack budgets: **ε ∈ {1,2,3,4,5,6,8}/255** (brain MRI); cross-task summaries at **2/255** and **4/255**.
- PGD/BIM: **10 steps**, **α = ε/10**, PGD uses random start inside the L∞ ball.

See **`docs/DEVELOPER_MANUAL.md`** for the full script index, dependency list, and server setup instructions (also summarised in dissertation Appendix B).

---

## Anonymous review repository

For blind review, upload this repository to [anonymous.4open.science](https://anonymous.4open.science/) and replace the placeholder link in the dissertation appendix with the generated URL.

---

## Licence

See [LICENSE.md](LICENSE.md). Dataset terms follow the respective Hugging Face / Mendeley sources.
