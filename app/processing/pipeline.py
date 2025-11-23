from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from app.models import ProcessingResult, ProcessingStats, ShotPoint
from app.processing.align import AlignmentResult, align_with_border
from app.processing.detect_hits import DetectionParams, detect_hits, split_roi_components
from app.processing.diff_threshold import DiffThresholdParams, diff_and_threshold
from app.processing.metrics import compute_metrics
from app.processing.overlay import render_overlay
from app.processing.scale import ScaleModel
from app.utils.image_io import imread, imwrite

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    diff_params: DiffThresholdParams
    detection_params: DetectionParams
    bullet_diameter_mm: float
    downscale_factor: float
    mask_path: Optional[Path]
    template_path: Path
    output_dir: Path
    show_r50: bool = True
    show_r90: bool = False
    collect_debug: bool = False


class ProcessingPipeline:
    def __init__(self, scale_model: ScaleModel, config: PipelineConfig) -> None:
        self.scale_model = scale_model
        self.config = config
        self.downscale_factor = max(config.downscale_factor, 0.1)
        self.template_full = self._load_template(config.template_path)
        self.template_gray = self._resize_image(self.template_full, self.downscale_factor)
        base_mask = self._load_mask(config.mask_path) if config.mask_path else None
        self.mask_full = base_mask
        self.mask = self._resize_image(base_mask, self.downscale_factor) if base_mask is not None else None
        self.origin_px = (
            self.template_gray.shape[1] / 2.0,
            self.template_gray.shape[0] / 2.0,
        )

    def process(self, frame_bgr: np.ndarray, target_id: str) -> ProcessingResult:
        stats = ProcessingStats()
        start = time.perf_counter()

        frame_gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_bgr_scaled = self._resize_image(frame_bgr, self.downscale_factor)
        frame_gray = cv2.cvtColor(frame_bgr_scaled, cv2.COLOR_BGR2GRAY)

        align_start = time.perf_counter()
        alignment = self._align_scaled(frame_gray, frame_gray_full)
        stats.align_ms = (time.perf_counter() - align_start) * 1000

        origin_px = alignment.origin_px or self.origin_px
        mm_per_pixel = self.scale_model.mm_per_pixel * self.downscale_factor

        diff_start = time.perf_counter()
        binary = diff_and_threshold(
            alignment.aligned,
            self.template_gray,
            params=self.config.diff_params,
            mask=self.mask,
            debug_dir=self.config.output_dir if self.config.collect_debug else None,
            target_id=target_id,
        )
        stats.diff_ms = (time.perf_counter() - diff_start) * 1000

        detect_start = time.perf_counter()
        points, debug_info = detect_hits(
            binary,
            alignment.aligned,
            mm_per_pixel=mm_per_pixel,
            params=self.config.detection_params,
            origin_px=origin_px,
            bullet_diameter_mm=self.config.bullet_diameter_mm,
            debug=self.config.collect_debug,
            template_gray=self.template_gray,
        )
        stats.detect_ms = (time.perf_counter() - detect_start) * 1000

        metrics = None
        if points:
            metrics_start = time.perf_counter()
            metrics = compute_metrics(points)
            stats.metrics_ms = (time.perf_counter() - metrics_start) * 1000

        overlay = render_overlay(
            cv2.cvtColor(alignment.aligned, cv2.COLOR_GRAY2BGR),
            points,
            metrics,
            mm_per_pixel,
            origin_px=origin_px,
            show_r50=self.config.show_r50,
            show_r90=self.config.show_r90,
            show_debug=False,
            debug_info=debug_info,
        )

        total_ms = (time.perf_counter() - start) * 1000
        accounted = stats.align_ms + stats.diff_ms + stats.detect_ms + stats.metrics_ms
        stats.capture_ms = max(0.0, total_ms - accounted)
        result = ProcessingResult(
            target_id=target_id,
            timestamp=datetime.now(),
            profile_name=self.scale_model.reference_name,
            points=points,
            metrics=metrics,
            stats=stats,
            mm_per_pixel=mm_per_pixel,
            homography=alignment.homography.flatten().tolist() if alignment.homography is not None else None,
            aligned_gray=alignment.aligned,
            overlay_image=overlay,
            binary_mask=binary,
            origin_px=origin_px,
            debug_info=debug_info,
        )
        self._store_intermediate(result, frame_bgr, overlay)
        return result

    def _store_intermediate(self, result: ProcessingResult, original_bgr: np.ndarray, overlay: np.ndarray) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{result.target_id}.png"
        overlay_name = f"{result.target_id}_overlay.png"
        original_path = self.config.output_dir / base_name
        overlay_path = self.config.output_dir / overlay_name
        if not imwrite(original_path, original_bgr):
            logger.warning("Failed to save original frame: %s", original_path)
        if not imwrite(overlay_path, overlay):
            logger.warning("Failed to save overlay: %s", overlay_path)
        result.image_path = str(original_path)
        result.overlay_path = str(overlay_path)

    def split_roi(self, result: ProcessingResult, roi: Tuple[int, int, int, int]) -> List[ShotPoint]:
        if result.binary_mask is None:
            raise RuntimeError("Бинарная маска недоступна для текущего результата")
        mm_per_pixel = result.mm_per_pixel or self.scale_model.mm_per_pixel
        origin = result.origin_px or self.origin_px
        new_points = split_roi_components(
            result.binary_mask,
            mm_per_pixel,
            self.config.detection_params,
            origin,
            roi,
            bullet_diameter_mm=self.config.bullet_diameter_mm,
        )
        if not new_points:
            return []
        filtered: List[ShotPoint] = []
        for candidate in new_points:
            duplicate = False
            for existing in result.points:
                if math.hypot(candidate.x_mm - existing.x_mm, candidate.y_mm - existing.y_mm) < 1.0:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(candidate)
        return filtered

    @staticmethod
    def _resize_image(image: Optional[np.ndarray], downscale_factor: float) -> Optional[np.ndarray]:
        if image is None:
            return None
        if downscale_factor == 1.0:
            return image
        h, w = image.shape[:2]
        new_w = max(1, int(round(w / downscale_factor)))
        new_h = max(1, int(round(h / downscale_factor)))
        interpolation = cv2.INTER_AREA if downscale_factor >= 1.0 else cv2.INTER_LINEAR
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    @staticmethod
    def _load_template(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        template = imread(path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Failed to load template: {path}")
        return template

    @staticmethod
    def _load_mask(path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        if not path.exists():
            logger.warning("Mask path does not exist: %s", path)
            return None
        if path.is_dir():
            logger.warning("Mask path points to a directory; ignoring: %s", path)
            return None
        mask = imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning("Failed to load mask: %s", path)
            return None
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def _align_scaled(self, frame_gray_scaled: np.ndarray, frame_gray_full: np.ndarray) -> "AlignmentResult":
        if self.downscale_factor == 1.0:
            return align_with_border(
                frame_gray_scaled,
                self.template_gray,
                mask=self.mask,
                max_features=1500,
                good_match_ratio=0.75,
                ransac_reproj_threshold=3.0,
            )

        alignment_full = align_with_border(
            frame_gray_full,
            self.template_full,
            mask=self.mask_full,
            max_features=1500,
            good_match_ratio=0.75,
            ransac_reproj_threshold=3.0,
        )

        origin_scaled = (
            alignment_full.origin_px[0] / self.downscale_factor,
            alignment_full.origin_px[1] / self.downscale_factor,
        )

        if alignment_full.homography is None:
            aligned_resized = self._resize_image(alignment_full.aligned, self.downscale_factor)
            return AlignmentResult(
                aligned=aligned_resized if aligned_resized is not None else frame_gray_scaled,
                homography=None,
                inliers=alignment_full.inliers,
                total_matches=alignment_full.total_matches,
                origin_px=origin_scaled,
            )

        homography_scaled = self._scale_homography(alignment_full.homography, self.downscale_factor)
        aligned = cv2.warpPerspective(
            frame_gray_scaled,
            homography_scaled,
            (self.template_gray.shape[1], self.template_gray.shape[0]),
        )
        return AlignmentResult(
            aligned=aligned,
            homography=homography_scaled,
            inliers=alignment_full.inliers,
            total_matches=alignment_full.total_matches,
            origin_px=origin_scaled,
        )

    @staticmethod
    def _scale_homography(homography: np.ndarray, factor: float) -> np.ndarray:
        if factor == 1.0:
            return homography
        scale_down = np.array(
            [[1.0 / factor, 0.0, 0.0], [0.0, 1.0 / factor, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        scale_up = np.array(
            [[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return scale_down @ homography @ scale_up