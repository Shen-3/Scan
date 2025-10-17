from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when camera interaction fails."""


@dataclass
class CameraFrame:
    data: np.ndarray
    timestamp: float


class CameraManager:
    """Wraps OpenCV capture to provide predictable camera interaction."""

    def __init__(
        self,
        device_index: int = 0,
        target_resolution: Optional[Tuple[int, int]] = None,
        warmup_frames: int = 5,
        backend: Optional[int] = None,
    ) -> None:
        self.device_index = device_index
        self.target_resolution = target_resolution
        self.warmup_frames = warmup_frames
        self.backend = backend
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if self._capture and self._capture.isOpened():
            return
        logger.debug("Opening camera %s", self.device_index)
        if self.backend is not None:
            capture = cv2.VideoCapture(self.device_index, self.backend)
        else:
            capture = cv2.VideoCapture(self.device_index)
        if not capture or not capture.isOpened():
            raise CameraError(f"Unable to open camera {self.device_index}")
        if self.target_resolution:
            width, height = self.target_resolution
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture = capture
        self._warmup()

    def _warmup(self) -> None:
        if not self._capture:
            return
        for idx in range(self.warmup_frames):
            ok, _ = self._capture.read()
            if not ok:
                logger.warning("Warmup frame %d failed", idx)
                break
            time.sleep(0.02)

    def close(self) -> None:
        if self._capture:
            logger.debug("Releasing camera %s", self.device_index)
            self._capture.release()
        self._capture = None

    def capture_frame(self) -> CameraFrame:
        self._ensure_open()
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Failed to capture frame")
        timestamp = time.time()
        return CameraFrame(data=frame, timestamp=timestamp)

    def _ensure_open(self) -> None:
        if not self._capture or not self._capture.isOpened():
            raise CameraError("Camera is not opened")

    def get_resolution(self) -> Tuple[int, int]:
        self._ensure_open()
        assert self._capture is not None
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    @staticmethod
    def list_devices(max_devices: int = 10) -> List[int]:
        """Best-effort listing across first N indices."""
        found: List[int] = []
        for idx in range(max_devices):
            capture = cv2.VideoCapture(idx)
            if capture is not None and capture.isOpened():
                found.append(idx)
                capture.release()
        return found

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

