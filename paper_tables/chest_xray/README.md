# Chest X-ray Paper Tables

Chest X-ray 论文图表目录（与 `paper_tables/brain_tumor_mri/` 对称）。

## 攻击强度对比（ε = 2/255, 4/255）

汇总脚本（从已有 CSV 生成，不重新跑 GPU）：

```powershell
.venv-gpu\Scripts\python.exe chest_xray\scripts\summarize_attack_strength.py
```

输出：

- `chest_xray/outputs/model_comparison/attack_strength_summary.csv`
- `chest_xray/outputs/model_comparison/attack_strength_summary.md`
- `paper_tables/chest_xray/table_attack_strength_*_eps*.png`

数据来源：

- FGSM / PGD / BIM：五模型（`outputs/model_comparison/*_model_comparison.csv`）
- MIM：仅 ResNet50（`outputs/legacy/resnet50/mifgsm_comparison.csv`，可选）
- PGD 的 ConvNeXt-Tiny 高 ε 可能为填充值
