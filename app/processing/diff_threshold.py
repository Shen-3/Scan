from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class DiffThresholdParams:
    use_adaptive: bool = True
    gaussian_sigma: float = 1.0
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    adaptive_block_size: int = 11
    adaptive_c: float = 0.0


def normalize_image(gray: np.ndarray, clahe_clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray)


def diff_and_threshold(
    aligned_gray: np.ndarray,
    template_gray: np.ndarray,
    params: DiffThresholdParams,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute binary mask of bullet holes by subtracting template and thresholding."""
    norm_aligned = normalize_image(aligned_gray, params.clahe_clip_limit, params.clahe_tile_grid_size)
    norm_template = normalize_image(template_gray, params.clahe_clip_limit, params.clahe_tile_grid_size)
    diff = cv2.absdiff(norm_aligned, norm_template)

    if params.gaussian_sigma > 0:
        ksize = max(3, int(params.gaussian_sigma * 6 + 1) // 2 * 2 + 1)
        diff = cv2.GaussianBlur(diff, (ksize, ksize), params.gaussian_sigma)

    if params.use_adaptive:
        block_size = max(3, params.adaptive_block_size | 1)
        binary = cv2.adaptiveThreshold(
            diff,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            params.adaptive_c,
        )
    else:
        _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if mask is not None:
        binary = cv2.bitwise_and(binary, mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.morph_kernel_size, params.morph_kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=params.morph_iterations)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=params.morph_iterations)
    return binary
