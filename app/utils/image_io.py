from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def imread(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read image from disk handling non-ASCII paths."""
    file_path = Path(path)
    data = np.fromfile(str(file_path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, flags)
    return image


def imwrite(
    path: Path | str,
    image: np.ndarray,
    params: Optional[Sequence[int]] = None,
) -> bool:
    """Write image to disk handling non-ASCII paths."""
    file_path = Path(path)
    suffix = file_path.suffix or ".png"
    encode_params = [] if params is None else list(params)
    success, buffer = cv2.imencode(suffix, image, encode_params)
    if not success:
        return False
    buffer.tofile(str(file_path))
    return True


def resize_to_max_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Resize keeping aspect ratio so the longest edge does not exceed max_edge."""
    if max_edge <= 0:
        return image
    height, width = image.shape[:2]
    max_dim = max(height, width)
    if max_dim <= max_edge:
        return image
    scale = max_edge / float(max_dim)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


__all__ = ["imread", "imwrite", "resize_to_max_edge"]
