# Brain Tumor MRI Detection

Binary classification task: **NO_TUMOR** (0) vs **TUMOR** (1). Original multi-class labels (glioma, meningioma, pituitary) are merged into TUMOR; `no_tumor` maps to NO_TUMOR.

Dataset: [usamaJabar/brain-tumor-mri-classification-merged](https://huggingface.co/datasets/usamaJabar/brain-tumor-mri-classification-merged) on Hugging Face.

> **Project overview, dependencies, and full script index:** see [../README.md](../README.md) and [../docs/DEVELOPER_MANUAL.md](../docs/DEVELOPER_MANUAL.md).

## Layout

```
brain_tumor_mri/
├── scripts/
│   ├── config.py, data_utils.py, comparison_common.py
│   ├── download_dataset.py, train_resnet50.py, train_model.py
│   ├── fgsm_attack.py, pgd_attack.py, deepfool_attack.py
│   ├── model_*_comparison.py, model_fair_robustness_comparison.py
│   └── run_pipeline.py          # delegates to ../run_pipeline.py
├── data/                        # train/val/test with NO_TUMOR and TUMOR folders
├── outputs/                     # models and model_comparison results
├── paths.py
└── run_pipeline.py
```

## Prerequisites

```powershell
.\.venv-gpu\Scripts\pip.exe install datasets huggingface_hub
```

## Quick start

```powershell
cd brain_tumor_mri
..\.venv-gpu\Scripts\python.exe run_pipeline.py --steps download
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download --steps train-resnet50 train-densenet121 train-resnet18
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download --steps fgsm pgd deepfool fair
```

Full pipeline:

```powershell
cd brain_tumor_mri
..\.venv-gpu\Scripts\python.exe run_pipeline.py
```

## Download only

```powershell
cd brain_tumor_mri
..\.venv-gpu\Scripts\python.exe scripts\download_dataset.py --help
..\.venv-gpu\Scripts\python.exe scripts\download_dataset.py --data-dir data
```

## Notes

- Scripts import `config`, `data_utils`, and `comparison_common` locally (no `brain_tumor.*` package).
- Root-level `scripts/*.py` pneumonia files are not modified.
