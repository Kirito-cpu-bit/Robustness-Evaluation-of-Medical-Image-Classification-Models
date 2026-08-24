"""Download brain tumor MRI dataset from Hugging Face."""
import argparse
from pathlib import Path

from config import DATA_DIR
from data_utils import download_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download brain tumor MRI dataset (usamaJabar/brain-tumor-mri-classification-merged)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory to store train/val/test splits",
    )
    args = parser.parse_args()
    path = download_dataset(args.data_dir)
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
