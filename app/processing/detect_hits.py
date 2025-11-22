from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from app.models import ShotPoint, DetectionDebug


@dataclass
class DetectionParams:
    min_diameter_mm: float = 4.0
    max_diameter_mm: float = 14.0
    min_circularity: float = 0.6
    min_intensity_drop: float = 8.0
    split_large_components: bool = True
    split_min_distance_mm: float = 5.0
    min_diameter_relaxation: float = 0.5


def detect_hits(
    binary_mask: np.ndarray,
    aligned_gray: np.ndarray,
    mm_per_pixel: float,
    params: DetectionParams,
    origin_px: Tuple[float, float] = (0.0, 0.0),
    bullet_diameter_mm: Optional[float] = None,
    debug: bool = False,
    template_gray: Optional[np.ndarray] = None,
) -> Tuple[List[ShotPoint], Optional[DetectionDebug]]:
    """Extract bullet hole centers from binary mask."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points: List[ShotPoint] = []
    rejected: List[Tuple[float, float]] = []
    segments: List[np.ndarray] = []
    next_id = 1
    min_radius_px = params.min_diameter_mm / 2.0 / mm_per_pixel
    max_radius_px = params.max_diameter_mm / 2.0 / mm_per_pixel
    min_distance_px = params.split_min_distance_mm / mm_per_pixel
    relaxation = min(max(params.min_diameter_relaxation, 0.0), 1.0)
    relaxed_radius_px = max(0.0, min_radius_px * relaxation)
    display_radius_mm = bullet_diameter_mm / 2.0 if bullet_diameter_mm and bullet_diameter_mm > 0 else None

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px <= 0:
            continue
        perimeter = max(cv2.arcLength(contour, True), 1e-6)
        circularity = 4.0 * math.pi * area_px / (perimeter * perimeter)
        center_px, radius_px = cv2.minEnclosingCircle(contour)
        eq_radius_px = math.sqrt(area_px / math.pi)
        local_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        cv2.drawContours(local_mask, [contour], -1, (255, 255, 255), -1)
        mean_inside = float(cv2.mean(aligned_gray, mask=local_mask)[0])
        dilated = cv2.dilate(local_mask, np.ones((5, 5), np.uint8))
        ring_mask = cv2.subtract(dilated, local_mask)
        mean_outside = float(cv2.mean(aligned_gray, mask=ring_mask)[0])
        intensity_drop = mean_outside - mean_inside
        intensity_contrast = abs(intensity_drop)
        if template_gray is not None:
            template_inside = float(cv2.mean(template_gray, mask=local_mask)[0])
            intensity_contrast = max(intensity_contrast, abs(template_inside - mean_inside))
        center_tuple = (float(center_px[0]), float(center_px[1]))
        if eq_radius_px < relaxed_radius_px or eq_radius_px > max_radius_px or circularity < params.min_circularity:
            if debug:
                rejected.append(center_tuple)
            continue
        small_component = eq_radius_px < min_radius_px
        threshold_drop = float(params.min_intensity_drop)
        if small_component:
            threshold_drop = max(threshold_drop * 1.3, threshold_drop + 5.0)
        if intensity_contrast < threshold_drop:
            if debug:
                rejected.append(center_tuple)
            continue

        needs_split = params.split_large_components and radius_px > max_radius_px * 1.2
        if needs_split:
            sub_points = _split_component(local_mask, min_distance_px)
            if len(sub_points) > 1:
                for cx, cy in sub_points:
                    points.append(
                        ShotPoint(
                            id=next_id,
                            x_mm=(cx - origin_px[0]) * mm_per_pixel,
                            y_mm=(cy - origin_px[1]) * mm_per_pixel,
                            radius_mm=display_radius_mm or (eq_radius_px * mm_per_pixel),
                            confidence=0.7,
                        )
                    )
                    next_id += 1
                if debug:
                    segments.append(local_mask.copy())
                continue

        cx, cy = _contour_centroid(contour)
        if cx is None or cy is None:
            if debug:
                rejected.append(center_tuple)
            continue
        cx_val = float(cx)
        cy_val = float(cy)
        confidence = 0.7 if small_component else 1.0
        points.append(
            ShotPoint(
                id=next_id,
                x_mm=(cx_val - origin_px[0]) * mm_per_pixel,
                y_mm=(cy_val - origin_px[1]) * mm_per_pixel,
                radius_mm=display_radius_mm or (eq_radius_px * mm_per_pixel),
                confidence=confidence,
            )
        )
        next_id += 1

    # If nothing was accepted but contours were present, run a relaxed pass that only checks size and shape.
    if not points and contours:
        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px <= 0:
                continue
            perimeter = max(cv2.arcLength(contour, True), 1e-6)
            circularity = 4.0 * math.pi * area_px / (perimeter * perimeter)
            eq_radius_px = math.sqrt(area_px / math.pi)
            if eq_radius_px < relaxed_radius_px or eq_radius_px > max_radius_px:
                continue
            if circularity < params.min_circularity * 0.8:
                continue
            cx, cy = _contour_centroid(contour)
            if cx is None or cy is None:
                continue
            points.append(
                ShotPoint(
                    id=next_id,
                    x_mm=(float(cx) - origin_px[0]) * mm_per_pixel,
                    y_mm=(float(cy) - origin_px[1]) * mm_per_pixel,
                    radius_mm=display_radius_mm or (eq_radius_px * mm_per_pixel),
                    confidence=0.4,
                    source="auto",
                )
            )
            next_id += 1

    debug_info = DetectionDebug(rejected=rejected, segments=segments) if debug else None
    return points, debug_info


def _contour_centroid(contour: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None, None
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    return cx, cy


def _split_component(component_mask: np.ndarray, min_distance_px: float) -> List[Tuple[float, float]]:
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 2:
        return []
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    roi = component_mask[y_min : y_max + 1, x_min : x_max + 1]
    distance = np.asarray(ndi.distance_transform_edt(roi), dtype=np.float32)
    if float(distance.max()) < min_distance_px * 0.5:
        return []

    ksize = max(3, int(min_distance_px) * 2 + 1)
    ksize = min(ksize, max(roi.shape[0], roi.shape[1]))
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(distance, kernel)
    max_distance = float(distance.max())
    peak_mask = (distance == dilated) & (distance > 0.5 * max_distance)
    peak_mask = peak_mask.astype(np.uint8)
    num_labels, markers = cv2.connectedComponents(peak_mask)
    if num_labels <= 1:
        return []

    markers = markers.astype(np.int32)
    markers = markers + 1
    markers[roi == 0] = 0

    color_roi = cv2.cvtColor((roi > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(color_roi, markers)
    centers: List[Tuple[float, float]] = []
    for label in np.unique(labels):
        if label <= 1:
            continue
        mask = labels == label
        if mask.sum() == 0:
            continue
        cy, cx = ndi.center_of_mass(mask)
        centers.append((float(x_min + cx), float(y_min + cy)))
    return centers


def split_roi_components(
    binary_mask: np.ndarray,
    mm_per_pixel: float,
    params: DetectionParams,
    origin_px: Tuple[float, float],
    roi: Tuple[int, int, int, int],
    bullet_diameter_mm: Optional[float] = None,
) -> List[ShotPoint]:
    """Attempt to split oversized components within selected ROI using watershed."""
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return []
    y1 = min(binary_mask.shape[0], y + h)
    x1 = min(binary_mask.shape[1], x + w)
    roi_mask = binary_mask[y:y1, x:x1]
    if roi_mask.size == 0 or not np.any(roi_mask):
        return []

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_radius_px = params.min_diameter_mm / 2.0 / mm_per_pixel
    max_radius_px = params.max_diameter_mm / 2.0 / mm_per_pixel
    min_distance_px = params.split_min_distance_mm / mm_per_pixel
    display_radius_mm = bullet_diameter_mm / 2.0 if bullet_diameter_mm and bullet_diameter_mm > 0 else None
    new_points: List[ShotPoint] = []
    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px <= 0:
            continue
        eq_radius_px = math.sqrt(area_px / math.pi)
        if eq_radius_px < min_radius_px:
            continue
        if eq_radius_px < max_radius_px * 1.1:
            # Component already within expected size; no splitting needed.
            continue
        local_mask = np.zeros_like(roi_mask, dtype=np.uint8)
        cv2.drawContours(local_mask, [contour], -1, (255, 255, 255), -1)
        centers = _split_component(local_mask, min_distance_px)
        if len(centers) <= 1:
            continue
        for cx, cy in centers:
            global_x = float(x + cx)
            global_y = float(y + cy)
            new_points.append(
                ShotPoint(
                    id=0,
                    x_mm=(global_x - origin_px[0]) * mm_per_pixel,
                    y_mm=(global_y - origin_px[1]) * mm_per_pixel,
                    radius_mm=display_radius_mm or (eq_radius_px * mm_per_pixel),
                    confidence=0.6,
                    source="split",
                )
            )
    return new_points
