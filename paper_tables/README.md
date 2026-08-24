# Paper Tables (LaTeX)

与 `brain_tumor_mri/`、`chest_xray/` 同级的论文表格目录。

```
paper_tables/
├── README.md
├── scripts/
│   └── render_brain_mri_table_images.py   # 重新生成 Brain MRI 图片表格
├── all_tables.tex
├── brain_tumor_mri/     # Brain MRI 实验表格
│   ├── all_tables.tex
│   ├── table03_clean_brain_mri_five_models.tex
│   ├── table03_clean_brain_mri_five_models.png
│   ├── table04_robustness_five_models_eps4.tex
│   ├── table04_robustness_five_models_eps4.png
│   └── ... (table05–08 均有 .tex + .png)
└── chest_xray/          # Chest X-ray 实验表格（待补充）
```

## LaTeX 导言区

```latex
\usepackage{booktabs}
\usepackage{multirow}
```

## Brain MRI

单表（表3 干净五模型）：

```latex
\input{paper_tables/brain_tumor_mri/table03_clean_brain_mri_five_models.tex}
```

或直接插入图片：

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=\linewidth]{paper_tables/brain_tumor_mri/table03_clean_brain_mri_five_models.png}
  \caption{Clean image classification performance on Brain MRI ($n=1356$).}
  \label{fig:brain_mri_clean_five_models}
\end{figure}
```

全部 Brain MRI 表格：

```latex
\input{paper_tables/brain_tumor_mri/all_tables.tex}
```

全部 Brain MRI **图片版**表格（深色主题 PNG）：

```latex
\usepackage{graphicx}
\input{paper_tables/brain_tumor_mri/all_tables_figures.tex}
```

重新生成图片：

```bat
.venv-gpu\Scripts\python.exe paper_tables\scripts\render_brain_mri_table_images.py
```

## Chest X-ray

表格文件将放在 `paper_tables/chest_xray/`，结构与 Brain MRI 对称。
