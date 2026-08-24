# Developer Manual

**Project:** Dual-dataset adversarial robustness for medical image classification  
**Author:** Xihao Wang  
**Last updated:** August 2026

This document explains how to set up the codebase on a new machine, which scripts implement each capability, and where results are written. It accompanies the dissertation appendix.

---

## 1. System requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 (primary), or Linux with NVIDIA GPU |
| Python | 3.8–3.10 recommended |
| RAM | ≥ 16 GB |
| GPU | ≥ 8 GB VRAM (batch size 32 at 224×224) |
| Disk | ~5 GB code + ~2 GB brain MRI data + ~1.2 GB chest X-ray zip |

**Tested configuration:** Windows 11, Python 3.8, TensorFlow 2.12 with `tensorflow-directml-plugin`, RTX 4060 Laptop GPU (8 GB).

---

## 2. Dependencies

Core libraries (`requirements.txt`):

```
tensorflow==2.12.0
numpy==1.23.5
pandas==2.0.3
scikit-learn==1.3.2
matplotlib==3.7.5
seaborn==0.13.2
Pillow>=9.0.0
huggingface_hub>=0.20.0
```

Additional packages used at runtime:

| Package | Purpose |
|---------|---------|
| `datasets` | Hugging Face brain MRI download / export |
| `tensorflow-directml-plugin` | GPU on Windows via DirectML (optional) |

### Fresh install (Windows)

```powershell
git clone <repository-url>
cd pneumonia_detection_resnet50-master
python -m venv .venv-gpu
.\.venv-gpu\Scripts\pip install --upgrade pip
.\.venv-gpu\Scripts\pip install -r requirements.txt
.\.venv-gpu\Scripts\pip install datasets huggingface_hub tensorflow-directml-plugin
```

### Fresh install (Linux + CUDA)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install datasets huggingface_hub
# Install tensorflow matching your CUDA version, e.g.:
pip install tensorflow==2.12.0
```

Update `run_pipeline.py` to point `PYTHON` at your venv interpreter if not using `.venv-gpu/Scripts/python.exe`.

---

## 3. Directory map

```
brain_tumor_mri/scripts/     # Primary development for MRI + defences
chest_xray/scripts/          # Chest X-ray mirror of attack/training scripts
paper_tables/                # Generated LaTeX tables
thesis/                      # Dissertation LaTeX source
```

Scripts are **not** installed as a Python package. Run them with `cwd` set to the task’s `scripts/` directory (as `run_pipeline.py` does) so local imports (`config`, `data_utils`, `attack_config`) resolve correctly.

---

## 4. Functionality → script index

### 4.1 Data download and preprocessing

| Task | Script | Output |
|------|--------|--------|
| Brain MRI download | `brain_tumor_mri/scripts/download_dataset.py` | `brain_tumor_mri/data/{train,val,test}/` |
| Brain MRI splits / DataFrames | `brain_tumor_mri/scripts/data_utils.py` | Used by all training/attack scripts |
| Chest X-ray download | `chest_xray/scripts/download_dataset.py` | `chest_xray/data/chest_xray/` |

Key constants: `config.py` (paths, HF repo ID, class names), `attack_config.py` (ε list, PGD steps).

### 4.2 Model training (standard fine-tuning)

| Architecture | Brain MRI | Chest X-ray |
|--------------|-----------|-------------|
| ResNet50 | `train_resnet50.py` | `train_resnet50.py` |
| DenseNet121, ResNet18, MobileNetV2, ConvNeXt-Tiny | `train_model.py --arch <name>` | `train_model.py --arch <name>` |

Checkpoints:

- ResNet50 → `outputs/models/cnn_4_final.h5`
- Others → `outputs/<arch>/models/<arch>_final.h5` (ConvNeXt: `convnext_tiny_best.h5`)

Orchestration: `run_pipeline.py --steps train-resnet50 train-densenet121 ...`

### 4.3 Adversarial attacks (white-box L∞)

| Attack | Core implementation | Multi-model comparison |
|--------|---------------------|------------------------|
| FGSM | `fgsm_attack.py` | `model_fgsm_comparison.py` |
| PGD (10-step, random start) | `pgd_attack.py` | `model_pgd_comparison.py` |
| BIM | `bim_attack.py` | `model_bim_comparison.py` |
| DeepFool | `deepfool_attack.py` | `model_deepfool_comparison.py` |
| MIM | `mifgsm_attack.py` | `model_attack_strength_comparison.py` (ResNet50) |
| C&W (L∞ variant) | `cw_attack.py` | ResNet50 only |

Shared helpers: `comparison_common.py` / `model_comparison_common.py` (model paths, input scaling, ε list).

Example (brain MRI, all available checkpoints):

```powershell
cd brain_tumor_mri
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download --steps fgsm pgd deepfool fair
```

Results → `outputs/model_comparison/<attack>/`.

### 4.4 Fair robustness aggregation

| Script | Input | Output |
|--------|-------|--------|
| `model_fair_robustness_comparison.py` | CSVs in `outputs/model_comparison/` | Relative drop, Adv AUC/F1 curves, summary Markdown |

Run after attack comparisons: `run_pipeline.py --steps fair`.

### 4.5 PGD adversarial training (brain MRI only)

| Script | Description |
|--------|-------------|
| `train_adversarial.py --arch resnet50` | PGD-AT from standard checkpoint, ε_train=4/255 |
| `train_adversarial.py --arch densenet121` | Same protocol for DenseNet121 |
| `model_pgd_standard_vs_at.py` | Compare standard vs PGD-AT under PGD/BIM |
| `run_pgd_at_steps10.py` | Batch launcher for steps=10 evaluation |

Checkpoints → `outputs/adversarial/models/` or `outputs/adversarial_steps10/models/`.

### 4.6 Gaussian smoothing defence (brain MRI, ResNet50)

| Script | Description |
|--------|-------------|
| `evaluate_gaussian_defense.py` | D0–D3 factorial evaluation (PGD + BIM) |
| `defense_common.py` | `apply_gaussian_blur()` |
| `plot_gaussian_defense_comparison.py` | Figures for paper |

Conditions:

| ID | Clean blur | Adv blur | Model |
|----|------------|----------|-------|
| D0 | No | No | Standard |
| D1 | No | Yes | Standard |
| D2 | Yes | Yes | Standard |
| D3 | Yes | Yes | PGD-AT |

Results → `outputs/defense/gaussian/gaussian_defense_comparison.csv`.

### 4.7 Paper / thesis artefacts

| Script | Output |
|--------|--------|
| `summarize_attack_strength.py` | Attack-strength tables |
| `summarize_pgd_robustness_table.py` | PGD robustness LaTeX |
| `update_table04_robustness_eps4.py` | Table fragments |
| `thesis/generate_report_figures.py` | Dissertation PDF figures |

LaTeX tables → `paper_tables/brain_tumor_mri/`, `paper_tables/chest_xray/`.

---

## 5. Running on a new server

### Step-by-step

1. **Clone** the repository (or upload to the server).
2. **Create venv** and install dependencies (Section 2).
3. **Verify GPU** (optional):
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices())
   ```
4. **Download data** (one task at a time):
   ```powershell
   cd brain_tumor_mri
   python run_pipeline.py --steps download
   cd ../chest_xray
   python run_pipeline.py --steps download
   ```
5. **Train models** (long-running; use `--steps` to run incrementally):
   ```powershell
   python run_pipeline.py --skip-download --steps train-resnet50 train-densenet121 train-resnet18 train-mobilenetv2 train-convnext-tiny
   ```
6. **Run attacks** after all required checkpoints exist:
   ```powershell
   python run_pipeline.py --skip-download --steps fgsm pgd bim deepfool fair
   ```
7. **Brain MRI defences** (optional):
   ```powershell
   cd brain_tumor_mri
   python scripts/train_adversarial.py --arch resnet50 --data-dir data
   python scripts/evaluate_gaussian_defense.py --data-dir data
   ```

### Environment variables / paths

- Edit `PYTHON` in `run_pipeline.py` if the venv path differs.
- Large artefacts (`data/`, `outputs/`, `*.h5`) are typically **gitignored**; copy checkpoints separately when migrating servers.
- Hugging Face downloads may require `HF_TOKEN` for rate limits (public datasets usually work without a token).

### Common issues

| Problem | Fix |
|---------|-----|
| `ImportError: config` | Run script via `run_pipeline.py` or set `cwd` to `scripts/` |
| ConvNeXt GPU error on DirectML | `convnext_gpu_patch.py` is applied automatically in `train_model.py` |
| Missing checkpoint in comparison | Train that architecture first; comparison skips missing models with a warning |
| OOM on GPU | Reduce batch size in `train_resnet50.py` (`BATCH_SIZE = 32` → 16) |

---

## 6. Reproducibility checklist

- [ ] Python 3.8+ and pinned packages from `requirements.txt`
- [ ] Random seed 42 (splits in `data_utils.py`)
- [ ] Same checkpoint paths as `comparison_common.py` / `model_comparison_common.py`
- [ ] PGD/BIM: 10 steps, α = ε/10
- [ ] Full test set evaluated (no subsampling)

---

## 7. Contact / citation

Dissertation author: **Xihao Wang**, University College London, MSc Artificial Intelligence for Biomedicine and Healthcare, 2026.

For anonymous peer review, use the link published in dissertation Appendix A (hosted on [anonymous.4open.science](https://anonymous.4open.science/)).
