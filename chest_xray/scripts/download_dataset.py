"""Download Chest X-Ray Pneumonia dataset if not present."""
import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Optional

HF_REPO_ID = "ahulikal/chest-xray-pneumonia-mirror"
MENDELEY_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/rscbjbr9sj-3.zip"
)


def find_chest_xray_root(data_dir: Path) -> Optional[Path]:
    candidates = list(data_dir.rglob("chest_xray"))
    for candidate in candidates:
        train_dir = candidate / "train"
        test_dir = candidate / "test"
        if train_dir.is_dir() and test_dir.is_dir():
            normal_count = len(list((train_dir / "NORMAL").glob("*.jpeg")))
            pneumonia_count = len(list((train_dir / "PNEUMONIA").glob("*.jpeg")))
            if normal_count > 0 and pneumonia_count > 0:
                return candidate
    return None


def count_images(data_root: Path) -> int:
    return len(list(data_root.rglob("*.jpeg")))


def download_from_huggingface(data_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = data_dir / "hf_chest_xray"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "chest_xray"

    print(f"Downloading full dataset from Hugging Face: {HF_REPO_ID}")
    print("File: chest_xray.zip (~1.2 GB). Supports resume if interrupted.")

    zip_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename="chest_xray.zip",
            local_dir=str(cache_dir),
            resume_download=True,
        )
    )
    print(f"Downloaded zip to: {zip_path}")

    print("Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    extracted = find_chest_xray_root(data_dir)
    if extracted is None:
        raise RuntimeError("Extraction finished but chest_xray folder was not found.")

    if extracted != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
        extracted = target

    image_count = count_images(extracted)
    print(f"Dataset ready at: {extracted}")
    print(f"Total images: {image_count}")
    return extracted


def download_from_mendeley(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "chest_xray_pneumonia.zip"
    print(f"Downloading dataset to {zip_path} ...")
    import urllib.request

    urllib.request.urlretrieve(MENDELEY_URL, zip_path)

    print("Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)

    zip_path.unlink(missing_ok=True)

    extracted = find_chest_xray_root(data_dir)
    if extracted is None:
        raise RuntimeError("Download finished but chest_xray folder was not found.")

    target = data_dir / "chest_xray"
    if extracted != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
        extracted = target

    print(f"Dataset ready at: {extracted}")
    print(f"Total images: {count_images(extracted)}")
    return extracted


def download_and_extract(data_dir: Path, source: str = "huggingface") -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = find_chest_xray_root(data_dir)
    if existing is not None:
        print(f"Dataset already exists at: {existing}")
        print(f"Total images: {count_images(existing)}")
        return existing

    if source == "huggingface":
        return download_from_huggingface(data_dir)
    if source == "mendeley":
        return download_from_mendeley(data_dir)
    raise ValueError(f"Unknown source: {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "mendeley"],
        default="huggingface",
    )
    args = parser.parse_args()
    download_and_extract(args.data_dir, source=args.source)


if __name__ == "__main__":
    main()
