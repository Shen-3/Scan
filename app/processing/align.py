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


@dataclass
class AlignmentResult:
    aligned: np.ndarray
    homography: Optional[np.ndarray]
    inliers: int
    total_matches: int


def align_to_template(
    frame_gray: np.ndarray,
    template_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    max_features: int = 1500,
    good_match_ratio: float = 0.75,
    ransac_reproj_threshold: float = 3.0,
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
    return AlignmentResult(aligned=aligned, homography=H, inliers=inliers, total_matches=len(good_matches))


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Transform Nx2 array of points using homography."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be Nx2 array")
    pts_h = cv2.convertPointsToHomogeneous(points).reshape(-1, 3).T
    transformed = homography @ pts_h
    transformed = transformed[:2] / transformed[2]
    return transformed.T