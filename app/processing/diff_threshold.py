from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pathlib import Path
from app.utils.image_io import imwrite

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
    debug_dir: Optional[Path] = None,
    target_id: Optional[str] = None,
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

    def _post_process(src: np.ndarray) -> np.ndarray:
        """Apply mask and morphology for cleaner blobs."""
        out = src
        if mask is not None:
            out = cv2.bitwise_and(out, mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.morph_kernel_size, params.morph_kernel_size))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=params.morph_iterations)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=params.morph_iterations)
        return out

    binary = _post_process(binary)
    current_nz = int((binary > 0).sum())

    # If nothing (or almost nothing) survived, retry with more permissive thresholds.
    if current_nz < 10:
        # Try a simple fixed threshold based on image statistics.
        mean = float(diff.mean())
        std = float(diff.std())
        fallback_thresh = max(5.0, mean + 0.5 * std)
        _, binary_loose = cv2.threshold(diff, fallback_thresh, 255, cv2.THRESH_BINARY)
        binary_loose = _post_process(binary_loose)
        loose_nz = int((binary_loose > 0).sum())
        if loose_nz > current_nz:
            binary = binary_loose
            current_nz = loose_nz

    if current_nz < 10:
        # Final fallback: adaptive threshold with a slightly more aggressive offset.
        block_size = max(3, params.adaptive_block_size | 1)
        binary_adapt = cv2.adaptiveThreshold(
            diff,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            params.adaptive_c - 2.0,
        )
        binary_adapt = _post_process(binary_adapt)
        adapt_nz = int((binary_adapt > 0).sum())
        if adapt_nz > current_nz:
            binary = binary_adapt

    if debug_dir is not None and target_id:
        debug_dir.mkdir(parents=True, exist_ok=True)
        diff_path = debug_dir / f"{target_id}_diff.png"
        morph_path = debug_dir / f"{target_id}_morph.png"
        if not imwrite(diff_path, diff):
            # Log via OpenCV fallback to avoid introducing logger dependency here.
            print(f"Failed to save diff image: {diff_path}")
        if not imwrite(morph_path, binary):
            print(f"Failed to save morph image: {morph_path}")

    
    return binary
