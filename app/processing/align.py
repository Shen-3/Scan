from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, cast

import cv2
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cv2 import DMatch, KeyPoint  # pragma: no cover
else:  # pragma: no cover
    DMatch = Any
    KeyPoint = Any


@dataclass
class AlignmentResult:
    aligned: np.ndarray
    homography: Optional[np.ndarray]
    inliers: int
    total_matches: int
    method: str = "orb"
    quality: Optional[float] = None


_TEMPLATE_CORNER_CACHE: Dict[int, Tuple[np.ndarray, Tuple[int, int]]] = {}


def estimate_target_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect outer rectangular corners of the target if possible."""
    return _detect_target_corners(image)


def align_to_template(
    frame_gray: np.ndarray,
    template_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_features: int = 1500,
    good_match_ratio: float = 0.75,
    ransac_reproj_threshold: float = 3.0,
    template_corners: Optional[np.ndarray] = None,
) -> AlignmentResult:
    """Align frame to template using sheet corners with ECC refinement, falling back to ORB."""
    corner_result = _align_via_corners(
        frame_gray,
        template_gray,
        template_corners=template_corners,
    )
    if corner_result is not None:
        return corner_result

    # Fallback: original ORB-based alignment
    orb = cv2.ORB.create(max_features)
    mask_arg: Any = mask if mask is not None else None
    kp1_raw, des1_raw = orb.detectAndCompute(frame_gray, mask_arg)
    kp2_raw, des2_raw = orb.detectAndCompute(template_gray, mask_arg)

    kp1 = list(cast(Sequence[KeyPoint], kp1_raw))
    kp2 = list(cast(Sequence[KeyPoint], kp2_raw))
    des1 = cast(Optional[np.ndarray], des1_raw)
    des2 = cast(Optional[np.ndarray], des2_raw)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning("Not enough features for alignment")
        return AlignmentResult(frame_gray.copy(), None, 0, 0)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    matches: List[Sequence[DMatch]] = list(raw_matches)
    good_matches: List[DMatch] = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair[0], pair[1]
        if m.distance < good_match_ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        logger.warning("Insufficient good matches: %s", len(good_matches))
        return AlignmentResult(frame_gray.copy(), None, len(good_matches), len(matches))

    src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

    H_raw, status_raw = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    H_mat = cast(Optional[np.ndarray], H_raw)
    status_mat = cast(Optional[np.ndarray], status_raw)
    inliers = int(status_mat.sum()) if isinstance(status_mat, np.ndarray) else 0
    H = H_mat
    if H is None:
        logger.warning("Homography estimation failed or unstable")
        return AlignmentResult(frame_gray.copy(), None, inliers, len(good_matches))
    if np.linalg.cond(H) > 1e6:
        logger.warning("Homography estimation failed or unstable")
        return AlignmentResult(frame_gray.copy(), None, inliers, len(good_matches))

    height, width = template_gray.shape[:2]
    aligned = cv2.warpPerspective(frame_gray, H, (width, height))
    return AlignmentResult(
        aligned=aligned,
        homography=H,
        inliers=inliers,
        total_matches=len(good_matches),
        method="orb",
        quality=None,
    )


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Transform Nx2 array of points using homography."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be Nx2 array")
    pts_h = cv2.convertPointsToHomogeneous(points).reshape(-1, 3).T
    transformed = homography @ pts_h
    transformed = transformed[:2] / transformed[2]
    return transformed.T


def _align_via_corners(
    frame_gray: np.ndarray,
    template_gray: np.ndarray,
    template_corners: Optional[np.ndarray],
) -> Optional[AlignmentResult]:
    template_corners = _get_template_corners(template_gray, template_corners)
    if template_corners is None:
        return None

    frame_corners = _detect_target_corners(frame_gray)
    if frame_corners is None:
        logger.debug("Corner-based detection failed for frame; falling back to ORB")
        return None

    try:
        H_corner = cv2.getPerspectiveTransform(frame_corners.astype(np.float32), template_corners.astype(np.float32))
    except cv2.error as exc:
        logger.debug("Perspective transform from detected corners failed: %s", exc)
        return None

    height, width = template_gray.shape[:2]
    prealigned = cv2.warpPerspective(frame_gray, H_corner, (width, height))
    ecc_warp, ecc_score = _refine_with_ecc(prealigned, template_gray)
    if ecc_warp is not None:
        H_total = ecc_warp @ H_corner
        aligned = cv2.warpPerspective(frame_gray, H_total, (width, height))
        quality = ecc_score
    else:
        H_total = H_corner
        aligned = prealigned
        quality = _compute_similarity(aligned, template_gray)

    if not np.isfinite(H_total).all():
        logger.debug("Corner-based homography is not finite; fallback to ORB")
        return None

    return AlignmentResult(
        aligned=aligned,
        homography=H_total,
        inliers=0,
        total_matches=0,
        method="corners",
        quality=quality,
    )


def _get_template_corners(template_gray: np.ndarray, override: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if override is not None:
        return override
    key = id(template_gray)
    shape = cast(Tuple[int, int], template_gray.shape)
    cached = _TEMPLATE_CORNER_CACHE.get(key)
    if cached and cached[1] == shape:
        return cached[0]
    corners = _detect_target_corners(template_gray)
    if corners is not None:
        _TEMPLATE_CORNER_CACHE[key] = (corners.copy(), shape)
    return corners


def _detect_target_corners(image: np.ndarray) -> Optional[np.ndarray]:
    if image.ndim != 2:
        raise ValueError("Corner detection expects a grayscale image")
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    blur_array = np.asarray(blur, dtype=np.float32)
    median = float(np.median(blur_array))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if lower >= upper:
        lower, upper = 50, 150

    edges = cv2.Canny(blur, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    image_area = float(image.shape[0] * image.shape[1])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.2:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
        if approx.shape[0] < 4:
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if approx.shape[0] < 4:
            rect = cv2.minAreaRect(contour)
            approx = cv2.boxPoints(rect)
        pts = np.array(approx, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4:
            continue
        ordered = _order_points(pts)
        corners = cv2.cornerSubPix(
            image,
            ordered.reshape(-1, 1, 2),
            winSize=(25, 25),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 0.1),
        )
        return corners.reshape(-1, 2)
    return None


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _refine_with_ecc(aligned: np.ndarray, template: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if aligned.shape != template.shape:
        return None, None
    height, width = template.shape
    max_dim = float(max(height, width))
    ecc_scale = 1.0
    max_ecc_dim = 1600.0
    if max_dim > max_ecc_dim:
        ecc_scale = max_ecc_dim / max_dim
        new_size = (max(int(width * ecc_scale), 32), max(int(height * ecc_scale), 32))
        aligned_small = cv2.resize(aligned, new_size, interpolation=cv2.INTER_AREA)
        template_small = cv2.resize(template, new_size, interpolation=cv2.INTER_AREA)
    else:
        aligned_small = aligned
        template_small = template

    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        template_norm = template_small.astype(np.float32) / 255.0
        aligned_norm = aligned_small.astype(np.float32) / 255.0
        ecc_score, warp = cv2.findTransformECC(
            template_norm,
            aligned_norm,
            warp,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
        )
    except cv2.error as exc:
        logger.debug("ECC refinement failed: %s", exc)
        return None, None
    if not np.isfinite(warp).all():
        return None, None
    if ecc_scale != 1.0:
        scale = np.array([[ecc_scale, 0.0, 0.0], [0.0, ecc_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        scale_inv = np.array([[1.0 / ecc_scale, 0.0, 0.0], [0.0, 1.0 / ecc_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        warp = scale_inv.dot(warp).dot(scale)
    return warp, float(ecc_score)


def _compute_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 1e-6:
        return 0.0
    # map correlation coefficient (-1..1) to 0..1 range
    corr = float(np.clip((a * b).sum() / denom, -1.0, 1.0))
    return 0.5 * (corr + 1.0)


__all__ = [
    "AlignmentResult",
    "align_to_template",
    "apply_homography",
    "estimate_target_corners",
]
