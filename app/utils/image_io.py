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


__all__ = ["imread", "imwrite"]
