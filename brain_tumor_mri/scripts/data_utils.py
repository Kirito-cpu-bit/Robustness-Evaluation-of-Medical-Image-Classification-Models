"""Brain tumor MRI dataset download and dataframe builders."""
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR,
    HF_REPO,
    IMAGE_GLOBS,
    NEGATIVE_CLASS_NAMES,
    POSITIVE_CLASS_NAMES,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

RANDOM_STATE = 42


def _normalize_class(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _label_from_class_folder(class_name: str) -> Optional[int]:
    norm = _normalize_class(class_name)
    negatives = {_normalize_class(n) for n in NEGATIVE_CLASS_NAMES}
    positives = {_normalize_class(n) for n in POSITIVE_CLASS_NAMES}
    if norm in negatives:
        return 0
    if norm in positives:
        return 1
    return None


def _collect_images(split_dir: Path) -> list:
    rows = []
    seen = set()
    if not split_dir.is_dir():
        return rows
    class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    if class_dirs:
        for class_dir in sorted(class_dirs):
            label = _label_from_class_folder(class_dir.name)
            if label is None:
                continue
            for pattern in IMAGE_GLOBS:
                for image_path in sorted(class_dir.glob(pattern)):
                    key = str(image_path.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append((str(image_path), label))
        if rows:
            return rows
    for pattern in IMAGE_GLOBS:
        for image_path in sorted(split_dir.rglob(pattern)):
            key = str(image_path.resolve()).lower()
            if key in seen:
                continue
            label = _label_from_class_folder(image_path.parent.name)
            if label is None:
                continue
            seen.add(key)
            rows.append((str(image_path), label))
    return rows


def find_dataset_root(root: Optional[Path] = None) -> Optional[Path]:
    base = root or DATA_DIR
    for candidate in [base, base / "brain_tumor_mri"]:
        if not candidate.is_dir():
            continue
        train_dir = next((candidate / n for n in ("train", "Training") if (candidate / n).is_dir()), None)
        test_dir = next((candidate / n for n in ("test", "Testing") if (candidate / n).is_dir()), None)
        if train_dir and test_dir and _collect_images(train_dir):
            return candidate
        for split_name in ("train", "val", "test"):
            if (candidate / split_name).is_dir() and _collect_images(candidate / split_name):
                return candidate
    return None


def _scan_split_dirs(data_root: Path) -> Tuple[list, list, Optional[list]]:
    val_dir = next((data_root / n for n in ("val", "validation") if (data_root / n).is_dir()), None)
    train_dir = next((data_root / n for n in ("train", "Training") if (data_root / n).is_dir()), None)
    test_dir = next((data_root / n for n in ("test", "Testing") if (data_root / n).is_dir()), None)
    if not train_dir or not test_dir:
        raise FileNotFoundError(f"Could not locate train/test under {data_root}")
    return _collect_images(train_dir), _collect_images(test_dir), _collect_images(val_dir) if val_dir else None


def build_dataframes(data_root: Path):
    train_rows, test_rows, val_rows = _scan_split_dirs(data_root)
    all_rows = list(train_rows) + list(test_rows)
    if val_rows:
        all_rows.extend(val_rows)
    master = pd.DataFrame(all_rows, columns=["image", "label"])
    master["image_key"] = master["image"].apply(
        lambda path: str(Path(path).resolve()).lower()
    )
    master = master.drop_duplicates(subset=["image_key"]).drop(columns=["image_key"])

    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    trainval_df, test_df = train_test_split(
        master,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=master["label"],
    )
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=trainval_df["label"],
    )
    print(
        f"Merged resplit ({int(TRAIN_RATIO * 100)}/{int(VAL_RATIO * 100)}/{int(TEST_RATIO * 100)}): "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
        f"(total={len(master)})"
    )
    return train_df, val_df, test_df


def _export_hf_split(split_dataset, split_name: str, target_root: Path) -> int:
    count = 0
    for idx, example in enumerate(split_dataset):
        image = example.get("image")
        label_name = example.get("label")
        if image is None:
            continue
        if isinstance(label_name, int):
            features = split_dataset.features.get("label")
            if features is not None and hasattr(features, "int2str"):
                label_name = features.int2str(label_name)
            else:
                continue
        label = _label_from_class_folder(str(label_name))
        if label is None:
            continue
        folder = "NO_TUMOR" if label == 0 else "TUMOR"
        out_dir = target_root / split_name / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(str(example.get("image_path", f"{idx}.jpg"))).suffix or ".jpg"
        image.save(out_dir / f"{split_name}_{idx:05d}{suffix}")
        count += 1
    return count


def _extract_from_cached_zip(target: Path) -> int:
    """Extract train/val/test splits from the HF zip if already cached locally."""
    import shutil
    import zipfile

    from huggingface_hub import hf_hub_download

    cache_dir = target / "hf_cache"
    zip_path = cache_dir / "brain-tumor-mri-classification.zip"
    if not zip_path.is_file():
        try:
            zip_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename="brain-tumor-mri-classification.zip",
                    local_dir=str(cache_dir),
                )
            )
        except Exception:
            return 0

    count = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            lower = member.lower().replace("\\", "/")
            if not lower.endswith((".jpg", ".jpeg", ".png")):
                continue
            if "/images/" not in lower:
                continue
            parts = Path(member).parts
            try:
                idx = [p.lower() for p in parts].index("images")
                split_name = parts[idx + 1]
                class_name = parts[idx + 2]
                filename = parts[idx + 3]
            except (ValueError, IndexError):
                continue
            if split_name not in {"train", "val", "test"}:
                continue
            if _label_from_class_folder(class_name) is None:
                continue
            out_dir = target / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            if out_path.is_file():
                count += 1
                continue
            with archive.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def download_dataset(root: Optional[Path] = None) -> Path:
    target = root or DATA_DIR
    existing = find_dataset_root(target)
    if existing is not None:
        print(f"Dataset already at: {existing}")
        return existing
    target.mkdir(parents=True, exist_ok=True)

    print("Extracting from cached zip (or downloading zip first) ...")
    total = _extract_from_cached_zip(target)
    if total > 0:
        print(f"Extracted {total} images to {target}")
        data_root = find_dataset_root(target)
        if data_root is not None:
            return data_root

    print(f"Falling back to Hugging Face dataset API: {HF_REPO}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets huggingface_hub") from exc
    dataset_dict = load_dataset(HF_REPO)
    total = 0
    for split_name, split_dataset in dataset_dict.items():
        norm = "val" if split_name == "validation" else split_name
        if norm in {"train", "val", "test"} and "image" in split_dataset.column_names:
            total += _export_hf_split(split_dataset, norm, target)
    if total == 0:
        raise RuntimeError("No images exported.")
    print(f"Exported {total} images to {target}")
    return target


def resolve_data_root(root: Optional[Path] = None, skip_download: bool = False) -> Path:
    if not skip_download:
        return download_dataset(root)
    data_root = find_dataset_root(root)
    if data_root is None:
        raise FileNotFoundError(f"No dataset under {root or DATA_DIR}")
    return data_root
