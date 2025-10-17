from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ScaleModel:
    """Stores conversion between pixels and millimetres."""

    mm_per_pixel: float
    reference_name: str = "default"
    metadata: Optional[dict] = None

    def pixels_to_mm(self, value: float) -> float:
        return value * self.mm_per_pixel

    def mm_to_pixels(self, value_mm: float) -> float:
        if self.mm_per_pixel == 0:
            raise ValueError("mm_per_pixel must be non-zero")
        return value_mm / self.mm_per_pixel

    def to_dict(self) -> dict:
        return {
            "mm_per_pixel": self.mm_per_pixel,
            "reference_name": self.reference_name,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScaleModel":
        return cls(
            mm_per_pixel=float(data["mm_per_pixel"]),
            reference_name=data.get("reference_name", "default"),
            metadata=data.get("metadata"),
        )


def mm_per_pixel_from_grid(step_mm: float, pixel_distances: Sequence[float]) -> float:
    """Estimate mm per pixel using grid spacing from multiple measurements."""
    if not pixel_distances:
        raise ValueError("pixel_distances must be non-empty")
    distances = np.array(pixel_distances, dtype=np.float64)
    mean_pixels = float(distances.mean())
    if mean_pixels <= 0:
        raise ValueError("mean pixel distance must be positive")
    return step_mm / mean_pixels


def mm_per_pixel_from_points(
    point_pairs: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]],
    known_distance_mm: float,
) -> float:
    """Estimate scale from arbitrary point pairs separated by known distance."""
    distances_px = []
    for (x1, y1), (x2, y2) in point_pairs:
        dx = x1 - x2
        dy = y1 - y2
        dist = np.hypot(dx, dy)
        if dist > 0:
            distances_px.append(dist)
    return mm_per_pixel_from_grid(known_distance_mm, distances_px)


def save_scale_model(model: ScaleModel, path: Path) -> None:
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_scale_model(path: Path) -> ScaleModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ScaleModel.from_dict(data)

