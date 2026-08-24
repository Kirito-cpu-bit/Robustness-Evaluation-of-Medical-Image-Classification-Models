# Chest X-Ray Pneumonia Detection

胸部 X 光肺炎二分类（Normal vs Pneumonia），五模型对抗鲁棒性对比：ResNet50、DenseNet121、ResNet18、MobileNetV2、ConvNeXt-Tiny。

攻击：FGSM、PGD、BIM（steps=10）、DeepFool；含公平鲁棒性（Fair Robustness）汇总。

> **项目总览、依赖与完整脚本索引：** 见 [../README.md](../README.md) 与 [../docs/DEVELOPER_MANUAL.md](../docs/DEVELOPER_MANUAL.md)。

## 目录结构

```
chest_xray/
├── data/
│   ├── chest_xray/          # train/test（NORMAL, PNEUMONIA）
│   └── hf_chest_xray/       # Hugging Face 数据集元数据
├── scripts/
│   ├── download_dataset.py, train_resnet50.py, train_model.py
│   ├── fgsm_attack.py, pgd_attack.py, deepfool_attack.py   # 攻击实现
│   ├── model_*_comparison.py, model_fair_robustness_comparison.py
│   └── summarize_attack_strength.py
├── outputs/
│   ├── models/              # ResNet50 权重 (cnn_4_final.h5) + metrics
│   ├── densenet121/, resnet18/, mobilenetv2/, convnext_tiny/
│   ├── model_comparison/    # 五模型 FGSM/PGD/BIM/DeepFool 对比 CSV
│   └── legacy/resnet50/     # 早期单模型 ResNet50 攻击结果（归档）
├── paths.py
├── run_pipeline.py
└── run_train_gpu.bat
```

## 运行方式

在项目根目录使用 GPU 虚拟环境：

```powershell
cd chest_xray
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download
```

分步示例：

```powershell
# 训练五模型
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download --steps train-resnet50 train-densenet121 train-resnet18 train-mobilenetv2 train-convnext-tiny

# 攻击对比 + 公平鲁棒性
..\.venv-gpu\Scripts\python.exe run_pipeline.py --skip-download --steps fgsm pgd bim deepfool fair

# 汇总攻击强度表（无需 GPU）
..\.venv-gpu\Scripts\python.exe scripts\summarize_attack_strength.py
```

主要结果位于 `outputs/model_comparison/`；论文用 PNG 表输出到 `paper_tables/chest_xray/`。

## 说明

- Chest X-ray 章节**不包含** PGD-AT、Gaussian 防御、LaTeX table03–10（见 `brain_tumor_mri/`）。
- ConvNeXt-Tiny 的 PGD 高 ε 结果因 GPU 稳定性问题以 0 填充；BIM 已完整跑完五模型。
- `outputs/legacy/resnet50/` 保留早期单模型脚本产物（含 MIM），新实验请使用 `model_*_comparison.py`。
