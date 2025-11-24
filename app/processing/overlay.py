from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from app.models import ShotMetrics, ShotPoint, DetectionDebug


def render_overlay(
    base_bgr: np.ndarray,
    points: Iterable[ShotPoint],
    metrics: Optional[ShotMetrics],
    mm_per_pixel: Optional[float],
    origin_px: Tuple[float, float],
    show_r50: bool = True,
    show_r90: bool = False,
    color_center: Tuple[int, int, int] = (0, 255, 0),
    show_debug: bool = False,
    debug_info: Optional[DetectionDebug] = None,
) -> np.ndarray:
    """Return annotated copy of base image with shot overlay."""
    annotated = base_bgr.copy()
    if annotated.ndim != 3 or annotated.shape[2] != 3:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    height, width = annotated.shape[:2]
    scale_factor = max(1.0, min(5.0, min(height, width) / 1200.0))

    def scaled(value: float, minimum: int = 1) -> int:
        return max(minimum, int(round(value * scale_factor)))

    text_scale = 0.5 * scale_factor
    text_thickness = scaled(1)

    center = (int(round(origin_px[0])), int(round(origin_px[1])))
    cv2.drawMarker(
        annotated,
        center,
        color_center,
        markerType=cv2.MARKER_CROSS,
        markerSize=scaled(24),
        thickness=scaled(2),
    )

    for point in points:
        x_px = int(round((point.x_mm / (mm_per_pixel or 1.0)) + origin_px[0]))
        y_px = int(round((point.y_mm / (mm_per_pixel or 1.0)) + origin_px[1]))
        radius_px = int(max(scaled(3), round(point.radius_mm / (mm_per_pixel or 1.0))))
        cv2.circle(annotated, (x_px, y_px), radius_px, (0, 0, 255), scaled(2))
        cv2.putText(
            annotated,
            str(point.id),
            (x_px + 4, y_px - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    if metrics and mm_per_pixel:
        center_px = (
            int(round(metrics.mean_x_mm / mm_per_pixel + origin_px[0])),
            int(round(metrics.mean_y_mm / mm_per_pixel + origin_px[1])),
        )
        cv2.circle(annotated, center_px, scaled(6), (255, 200, 0), -1)
        cv2.putText(
            annotated,
            "STP",
            (center_px[0] + 8, center_px[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 200, 0),
            text_thickness,
            cv2.LINE_AA,
        )
        if show_r50:
            radius_px = int(round(metrics.r50_mm / mm_per_pixel))
            if radius_px > 0:
                cv2.circle(annotated, center_px, radius_px, (0, 200, 255), scaled(2))
        if show_r90:
            r90_mm = 1.2816 * max(metrics.std_x_mm, metrics.std_y_mm)
            radius_px = int(round(r90_mm / mm_per_pixel))
            if radius_px > 0:
                cv2.circle(annotated, center_px, radius_px, (128, 0, 255), scaled(2))

    if show_debug and debug_info:
        for cx, cy in debug_info.rejected:
                cv2.drawMarker(
                    annotated,
                    (int(round(cx)), int(round(cy))),
                    (0, 255, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=scaled(18),
                    thickness=scaled(2),
                )
        for mask in debug_info.segments:
            if mask.shape[:2] != annotated.shape[:2]:
                continue
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (255, 0, 255), scaled(1))

    return annotated
