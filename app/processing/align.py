from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING, cast

import cv2
import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cv2 import DMatch, KeyPoint  # pragma: no cover
else:  # pragma: no cover
    DMatch = Any
    KeyPoint = Any


class AlignmentResult:
    def __init__(
        self,
        aligned: np.ndarray,
        homography: Optional[np.ndarray],
        inliers: int,
        total_matches: int,
        origin_px: Tuple[float, float],
    ) -> None:
        self.aligned = aligned
        self.homography = homography
        self.inliers = inliers
        self.total_matches = total_matches
        self.origin_px = origin_px

    def __repr__(self) -> str:  # helpful for debugging
        return (
            f"AlignmentResult(inliers={self.inliers}, total_matches={self.total_matches},"
            f" origin_px={self.origin_px})"
        )


def _order_points(points: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _quad_aspect_ratio(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    width = (width_top + width_bottom) / 2.0
    height = (height_left + height_right) / 2.0
    if height == 0:
        return 0.0
    return width / height


def detect_border_quad(
    gray: np.ndarray,
    aspect_ratio: float,
    min_area_ratio: float = 0.4,
    aspect_tolerance: float = 0.2,
) -> Optional[np.ndarray]:
    """
    Find the outer rectangular border of the target.

    Returns 4x2 float array of corner points ordered TL, TR, BR, BL or None.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    min_area = min_area_ratio * w * h
    best = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        quad = _order_points(approx.reshape(4, 2))
        ar = _quad_aspect_ratio(quad)
        lower = aspect_ratio * (1 - aspect_tolerance)
        upper = aspect_ratio * (1 + aspect_tolerance)
        if lower <= ar <= upper:
            best = quad
            break
    return best


def align_with_border(
    frame_gray: np.ndarray,
    template_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_features: int = 1500,
    good_match_ratio: float = 0.75,
    ransac_reproj_threshold: float = 3.0,
    aspect_tolerance: float = 0.2,
    min_area_ratio: float = 0.4,
    refine_with_features: bool = True,
) -> AlignmentResult:
    """
    Align using the outer black border as primary cue; optionally refine with ORB.
    Falls back to pure ORB if border not detected.
    """
    h, w = template_gray.shape[:2]
    aspect = w / float(h)
    frame_quad = detect_border_quad(
        frame_gray,
        aspect_ratio=aspect,
        min_area_ratio=min_area_ratio,
        aspect_tolerance=aspect_tolerance,
    )
    template_quad = detect_border_quad(
        template_gray,
        aspect_ratio=aspect,
        min_area_ratio=min_area_ratio,
        aspect_tolerance=aspect_tolerance,
    )

    template_origin = (
        float(template_gray.shape[1] / 2.0),
        float(template_gray.shape[0] / 2.0),
    )
    if template_quad is not None:
        template_origin = (float(template_quad[:, 0].mean()), float(template_quad[:, 1].mean()))

    if frame_quad is None:
        logger.info("Border not found in frame; falling back to feature alignment")
        return align_to_template(
            frame_gray,
            template_gray,
            mask=mask,
            max_features=max_features,
            good_match_ratio=good_match_ratio,
            ransac_reproj_threshold=ransac_reproj_threshold,
            origin_px=template_origin,
        )

    if template_quad is None:
        logger.warning("Border not found in template; falling back to feature alignment")
        return align_to_template(
            frame_gray,
            template_gray,
            mask=mask,
            max_features=max_features,
            good_match_ratio=good_match_ratio,
            ransac_reproj_threshold=ransac_reproj_threshold,
            origin_px=template_origin,
        )

    H_border = cv2.getPerspectiveTransform(frame_quad.astype(np.float32), template_quad.astype(np.float32))
    if np.linalg.cond(H_border) > 1e6:
        logger.warning("Border homography unstable; falling back to feature alignment")
        return align_to_template(
            frame_gray,
            template_gray,
            mask=mask,
            max_features=max_features,
            good_match_ratio=good_match_ratio,
            ransac_reproj_threshold=ransac_reproj_threshold,
            origin_px=template_origin,
        )

    warped = cv2.warpPerspective(frame_gray, H_border, (w, h))
    if not refine_with_features:
        return AlignmentResult(
            aligned=warped,
            homography=H_border,
            inliers=0,
            total_matches=0,
            origin_px=template_origin,
        )

    refine = align_to_template(
        warped,
        template_gray,
        mask=mask,
        max_features=max_features,
        good_match_ratio=good_match_ratio,
        ransac_reproj_threshold=ransac_reproj_threshold,
        origin_px=template_origin,
    )
    if refine.homography is None:
        return AlignmentResult(
            aligned=warped,
            homography=H_border,
            inliers=0,
            total_matches=0,
            origin_px=template_origin,
        )

    combined = refine.homography @ H_border
    return AlignmentResult(
        aligned=refine.aligned,
        homography=combined,
        inliers=refine.inliers,
        total_matches=refine.total_matches,
        origin_px=template_origin,
    )


def align_to_template(
    frame_gray: np.ndarray,
    template_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_features: int = 1500,
    good_match_ratio: float = 0.75,
    ransac_reproj_threshold: float = 3.0,
    origin_px: Optional[Tuple[float, float]] = None,
) -> AlignmentResult:
    """Align frame to template via ORB feature matching with RANSAC."""
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
        return AlignmentResult(frame_gray.copy(), None, 0, 0, origin_px or (
            float(template_gray.shape[1] / 2.0),
            float(template_gray.shape[0] / 2.0),
        ))

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
        return AlignmentResult(frame_gray.copy(), None, len(good_matches), len(matches), origin_px or (
            float(template_gray.shape[1] / 2.0),
            float(template_gray.shape[0] / 2.0),
        ))

    src_pts = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)

    H_raw, status_raw = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
    H_mat = cast(Optional[np.ndarray], H_raw)
    status_mat = cast(Optional[np.ndarray], status_raw)
    inliers = int(status_mat.sum()) if isinstance(status_mat, np.ndarray) else 0
    H = H_mat
    if H is None:
        logger.warning("Homography estimation failed or unstable")
        return AlignmentResult(frame_gray.copy(), None, inliers, len(good_matches), origin_px or (
            float(template_gray.shape[1] / 2.0),
            float(template_gray.shape[0] / 2.0),
        ))
    if np.linalg.cond(H) > 1e6:
        logger.warning("Homography estimation failed or unstable")
        return AlignmentResult(frame_gray.copy(), None, inliers, len(good_matches), origin_px or (
            float(template_gray.shape[1] / 2.0),
            float(template_gray.shape[0] / 2.0),
        ))

    height, width = template_gray.shape[:2]
    aligned = cv2.warpPerspective(frame_gray, H, (width, height))
    return AlignmentResult(
        aligned=aligned,
        homography=H,
        inliers=inliers,
        total_matches=len(good_matches),
        origin_px=origin_px or (
            float(template_gray.shape[1] / 2.0),
            float(template_gray.shape[0] / 2.0),
        ),
    )



def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Transform Nx2 array of points using homography."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be Nx2 array")
    pts_h = cv2.convertPointsToHomogeneous(points).reshape(-1, 3).T
    transformed = homography @ pts_h
    transformed = transformed[:2] / transformed[2]
    return transformed.T
