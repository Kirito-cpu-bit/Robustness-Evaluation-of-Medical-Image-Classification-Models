"""Shared preprocessing defenses and metrics helpers."""
from __future__ import annotations

import io
from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageFilter


def apply_gaussian_blur(images: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Gaussian blur on RGB images in [0, 1], shape (N, H, W, 3)."""
    out = np.empty_like(images)
    for idx in range(len(images)):
        arr = (np.clip(images[idx], 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        out[idx] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def apply_jpeg_compression(images: np.ndarray, quality: int = 75) -> np.ndarray:
    """JPEG recompression defense on RGB images in [0, 1]."""
    out = np.empty_like(images)
    for idx in range(len(images)):
        arr = (np.clip(images[idx], 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        restored = Image.open(buffer).convert("RGB")
        out[idx] = np.asarray(restored, dtype=np.float32) / 255.0
    return out


def apply_median_filter(images: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter via PIL (size must be odd)."""
    out = np.empty_like(images)
    for idx in range(len(images)):
        arr = (np.clip(images[idx], 0.0, 1.0) * 255.0).astype(np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.ModeFilter(size=size))
        out[idx] = np.asarray(img, dtype=np.float32) / 255.0
    return out


DEFENSE_REGISTRY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "none": lambda x: x,
    "gaussian": lambda x: apply_gaussian_blur(x, radius=1.0),
    "jpeg75": lambda x: apply_jpeg_compression(x, quality=75),
    "median3": lambda x: apply_median_filter(x, size=3),
}

DEFENSE_LABELS = {
    "none": "No defense",
    "gaussian": "Gaussian blur (σ≈1)",
    "jpeg75": "JPEG Q=75",
    "median3": "Median filter 3×3",
}
