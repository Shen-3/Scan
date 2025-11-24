from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ShotPoint:
    """Represents a single bullet hole in millimetres relative to target origin."""

    id: int
    x_mm: float
    y_mm: float
    radius_mm: float
    confidence: float = 1.0
    source: str = "auto"  # auto|manual|adjusted

    def as_tuple(self) -> Tuple[float, float]:
        return self.x_mm, self.y_mm


@dataclass
class DetectionDebug:
    """Auxiliary data for manual quality control."""

    rejected: List[Tuple[float, float]]
    segments: List[np.ndarray]


@dataclass
class ShotMetrics:
    """Computed ballistic metrics for a target."""

    shot_count: int
    mean_x_mm: float
    mean_y_mm: float
    std_x_mm: float
    std_y_mm: float
    mean_radius_mm: float
    extreme_spread_mm: float
    r50_mm: float
    displacement_mm: float
    azimuth_deg: float
    covariance: Tuple[float, float, float]  # var_x, var_y, cov_xy


@dataclass
class ProcessingStats:
    """Timing diagnostics for each stage."""

    capture_ms: float = 0.0
    align_ms: float = 0.0
    diff_ms: float = 0.0
    detect_ms: float = 0.0
    metrics_ms: float = 0.0


@dataclass
class ProcessingResult:
    """Full description of a processed target."""

    target_id: str
    timestamp: datetime
    profile_name: Optional[str] = None
    debug_info: Optional[DetectionDebug] = None
    points: List[ShotPoint] = field(default_factory=list)
    metrics: Optional[ShotMetrics] = None
    stats: ProcessingStats = field(default_factory=ProcessingStats)
    image_path: Optional[str] = None
    overlay_path: Optional[str] = None
    mm_per_pixel: Optional[float] = None
    homography: Optional[Sequence[float]] = None  # flattened 3x3
    aligned_gray: Optional[np.ndarray] = None
    overlay_image: Optional[np.ndarray] = None
    binary_mask: Optional[np.ndarray] = None
    origin_px: Optional[Tuple[float, float]] = None

    def to_summary_dict(self) -> dict:
        metrics_dict = {}
        if self.metrics:
            metrics_dict = {
                "shot_count": self.metrics.shot_count,
                "mean_x_mm": self.metrics.mean_x_mm,
                "mean_y_mm": self.metrics.mean_y_mm,
                "std_x_mm": self.metrics.std_x_mm,
                "std_y_mm": self.metrics.std_y_mm,
                "mean_radius_mm": self.metrics.mean_radius_mm,
                "extreme_spread_mm": self.metrics.extreme_spread_mm,
                "r50_mm": self.metrics.r50_mm,
                "displacement_mm": self.metrics.displacement_mm,
                "azimuth_deg": self.metrics.azimuth_deg,
                "covariance_xx": self.metrics.covariance[0],
                "covariance_yy": self.metrics.covariance[1],
                "covariance_xy": self.metrics.covariance[2],
            }
        return {
            "target_id": self.target_id,
            "timestamp": self.timestamp.isoformat(),
            "point_count": len(self.points),
            "mm_per_pixel": self.mm_per_pixel,
            **metrics_dict,
        }

    def to_points_table(self) -> List[dict]:
        return [
            {
                "id": point.id,
                "x_mm": point.x_mm,
                "y_mm": point.y_mm,
                "radius_mm": point.radius_mm,
                "confidence": point.confidence,
                "source": point.source,
            }
            for point in self.points
        ]

    def recompute_metrics(self) -> None:
        if not self.points:
            self.metrics = None
            return
        from app.processing.metrics import compute_metrics

        self.metrics = compute_metrics(self.points)

    def to_csv_summary(self) -> dict:
        metrics = self.metrics
        summary = {
            "target_id": self.target_id,
            "date_time": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "N": len(self.points),
            "cx": metrics.mean_x_mm if metrics else 0.0,
            "cy": metrics.mean_y_mm if metrics else 0.0,
            "d": metrics.displacement_mm if metrics else 0.0,
            "azimuth_deg": metrics.azimuth_deg if metrics else 0.0,
            "extreme_spread": metrics.extreme_spread_mm if metrics else 0.0,
            "mean_radius": metrics.mean_radius_mm if metrics else 0.0,
            "R50": metrics.r50_mm if metrics else 0.0,
            "sigma_x": metrics.std_x_mm if metrics else 0.0,
            "sigma_y": metrics.std_y_mm if metrics else 0.0,
            "profile_name": self.profile_name or "",
        }
        return summary
