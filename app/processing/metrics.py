from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

from app.models import ShotMetrics, ShotPoint


def compute_metrics(points: Iterable[ShotPoint]) -> ShotMetrics:
    pts: List[ShotPoint] = list(points)
    if not pts:
        raise ValueError("At least one point required for metrics")

    xs = np.array([p.x_mm for p in pts], dtype=np.float64)
    ys = np.array([p.y_mm for p in pts], dtype=np.float64)
    count = len(pts)
    mean_x = float(xs.mean())
    mean_y = float(ys.mean())
    std_x = float(xs.std(ddof=1)) if count > 1 else 0.0
    std_y = float(ys.std(ddof=1)) if count > 1 else 0.0
    displacement = math.hypot(mean_x, mean_y)
    azimuth = math.degrees(math.atan2(mean_y, mean_x))
    radii = np.sqrt((xs - mean_x) ** 2 + (ys - mean_y) ** 2)
    mean_radius = float(radii.mean()) if count > 0 else 0.0
    r50 = float(np.median(radii)) if count > 0 else 0.0
    extreme_spread = float(_extreme_spread(xs, ys))
    cov = np.cov(xs, ys) if count > 1 else np.zeros((2, 2))
    covariance = (float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1]))
    return ShotMetrics(
        shot_count=count,
        mean_x_mm=mean_x,
        mean_y_mm=mean_y,
        std_x_mm=std_x,
        std_y_mm=std_y,
        mean_radius_mm=mean_radius,
        extreme_spread_mm=extreme_spread,
        r50_mm=r50,
        displacement_mm=displacement,
        azimuth_deg=azimuth,
        covariance=covariance,
    )


def _extreme_spread(xs: np.ndarray, ys: np.ndarray) -> float:
    count = len(xs)
    if count < 2:
        return 0.0
    max_distance = 0.0
    for i in range(count - 1):
        dx = xs[i] - xs[i + 1 :]
        dy = ys[i] - ys[i + 1 :]
        distances = np.hypot(dx, dy)
        local_max = float(distances.max(initial=0.0))
        if local_max > max_distance:
            max_distance = local_max
    return max_distance

