from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.models import ShotPoint
from app.processing.align import align_to_template
from app.processing.diff_threshold import DiffThresholdParams, diff_and_threshold
from app.processing.detect_hits import DetectionParams, detect_hits, split_roi_components
from app.processing.metrics import compute_metrics
from app.processing.overlay import render_overlay
from app.processing.pipeline import PipelineConfig, ProcessingPipeline
from app.processing.scale import ScaleModel


def _make_feature_rich_template(size: int = 200) -> np.ndarray:
    template = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(template, (size // 2, size // 2), size // 3, 200, 3)
    cv2.line(template, (20, 20), (size - 20, size - 20), 255, 2)
    cv2.line(template, (20, size - 20), (size - 20, 20), 180, 2)
    cv2.rectangle(template, (40, 80), (80, 120), 220, -1)
    cv2.rectangle(template, (size - 90, size - 130), (size - 40, size - 80), 150, -1)
    return template


def test_align_to_template_recovers_translation():
    template = _make_feature_rich_template()
    h_true = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    shifted = cv2.warpPerspective(template, h_true, (template.shape[1], template.shape[0]))
    result = align_to_template(shifted, template)
    assert result.homography is not None
    diff = cv2.absdiff(result.aligned, template)
    assert float(diff.mean()) < 5.0


def test_align_to_template_handles_featureless_images():
    blank = np.zeros((100, 100), dtype=np.uint8)
    result = align_to_template(blank, blank)
    assert result.homography is None
    assert np.array_equal(result.aligned, blank)


def test_diff_and_threshold_detects_dark_region():
    template = np.full((120, 120), 180, dtype=np.uint8)
    aligned = template.copy()
    cv2.circle(aligned, (60, 60), 12, 70, -1)
    params = DiffThresholdParams(use_adaptive=False, gaussian_sigma=0.0, morph_kernel_size=5, morph_iterations=1)
    binary = diff_and_threshold(aligned, template, params)
    assert binary[60, 60] == 255
    assert int((binary > 0).sum()) > 250


def test_diff_and_threshold_adaptive_uses_c_offset():
    template = np.full((160, 160), 200, dtype=np.uint8)
    aligned = template.copy()
    cv2.circle(aligned, (80, 80), 18, 40, -1)
    params = DiffThresholdParams(
        use_adaptive=True,
        gaussian_sigma=0.0,
        morph_kernel_size=3,
        morph_iterations=1,
        adaptive_block_size=11,
        adaptive_c=0.0,
    )
    binary = diff_and_threshold(aligned, template, params)
    ratio = float((binary > 0).sum()) / binary.size
    assert ratio < 0.2
    assert int((binary[60:100, 60:100] > 0).sum()) > 0


def test_detect_hits_returns_points_with_debug():
    binary = np.zeros((160, 160), dtype=np.uint8)
    centers = [(50, 70), (110, 90)]
    for cx, cy in centers:
        cv2.circle(binary, (cx, cy), 10, 255, -1)
    aligned = np.full((160, 160), 210, dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(aligned, (cx, cy), 10, 60, -1)
    params = DetectionParams(
        min_diameter_mm=6.0,
        max_diameter_mm=22.0,
        min_circularity=0.5,
        min_intensity_drop=5.0,
        split_large_components=False,
    )
    mm_per_pixel = 0.5
    points, debug = detect_hits(binary, aligned, mm_per_pixel=mm_per_pixel, params=params, origin_px=(0.0, 0.0), debug=True)
    assert len(points) == 2
    xs = sorted([round(p.x_mm, 1) for p in points])
    ys = sorted([round(p.y_mm, 1) for p in points])
    assert xs == sorted([round(cx * mm_per_pixel, 1) for cx, _ in centers])
    assert ys == sorted([round(cy * mm_per_pixel, 1) for _, cy in centers])
    assert debug is not None
    assert not debug.rejected


def test_detect_hits_handles_small_but_contrasty_component():
    size = 400
    template = np.full((size, size), 200, dtype=np.uint8)
    cv2.circle(template, (size // 2, size // 2), 120, 80, -1)
    frame = template.copy()
    cv2.circle(frame, (size // 2, size // 2), 18, 20, -1)
    params = DiffThresholdParams(use_adaptive=True, gaussian_sigma=1.0, adaptive_c=0.0)
    binary = diff_and_threshold(frame, template, params)
    detection_params = DetectionParams(
        min_diameter_mm=4.5,
        max_diameter_mm=18.0,
        min_intensity_drop=5.0,
    )
    mm_per_pixel = 0.0833
    points, debug = detect_hits(
        binary,
        frame,
        mm_per_pixel=mm_per_pixel,
        params=detection_params,
        origin_px=(size / 2, size / 2),
        debug=True,
        template_gray=template,
    )
    assert len(points) == 1
    assert debug is not None
    assert not debug.rejected


def test_split_roi_components_separates_overlapping_blobs():
    binary = np.zeros((180, 180), dtype=np.uint8)
    cv2.circle(binary, (90, 80), 26, 255, -1)
    cv2.circle(binary, (120, 80), 26, 255, -1)
    params = DetectionParams(
        min_diameter_mm=4.0,
        max_diameter_mm=18.0,
        split_large_components=True,
        split_min_distance_mm=6.0,
        min_intensity_drop=0.0,
    )
    roi = (70, 50, 100, 80)
    points = split_roi_components(
        binary,
        mm_per_pixel=0.5,
        params=params,
        origin_px=(0.0, 0.0),
        roi=roi,
    )
    assert len(points) >= 2
    xs = sorted(round(p.x_mm, 1) for p in points)
    assert xs[0] < xs[-1]


def test_render_overlay_draws_annotations():
    base = np.zeros((100, 100, 3), dtype=np.uint8)
    points = [
        ShotPoint(id=1, x_mm=-10.0, y_mm=0.0, radius_mm=4.0),
        ShotPoint(id=2, x_mm=12.0, y_mm=5.0, radius_mm=4.0),
    ]
    metrics = compute_metrics(points)
    overlay = render_overlay(
        base,
        points,
        metrics,
        mm_per_pixel=1.0,
        origin_px=(50.0, 50.0),
        show_r50=True,
        show_r90=True,
        show_debug=False,
    )
    # center marker should alter origin pixel
    assert not np.array_equal(overlay[50, 50], base[50, 50])
    # shot annotations should draw non-black pixels near expected positions
    pt1 = overlay[50, 40]  # approximate location of first shot
    region = overlay[52:58, 60:66]
    assert pt1.any()
    assert region.any()


def test_processing_pipeline_end_to_end(tmp_path: Path):
    template = np.full((160, 160), 200, dtype=np.uint8)
    template_path = tmp_path / "template.png"
    cv2.imwrite(str(template_path), template)
    scale_model = ScaleModel(mm_per_pixel=0.5, reference_name="test-profile")
    config = PipelineConfig(
        diff_params=DiffThresholdParams(use_adaptive=False, gaussian_sigma=0.0, morph_kernel_size=5, morph_iterations=1),
        detection_params=DetectionParams(
            min_diameter_mm=6.0,
            max_diameter_mm=18.0,
            min_circularity=0.5,
            min_intensity_drop=5.0,
            split_large_components=False,
        ),
        mask_path=None,
        template_path=template_path,
        output_dir=tmp_path / "results",
        show_r50=True,
        show_r90=False,
        collect_debug=True,
    )
    pipeline = ProcessingPipeline(scale_model, config)
    frame = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    cv2.circle(frame, (80, 90), 12, (30, 30, 30), -1)
    result = pipeline.process(frame, target_id="sample")
    assert result.metrics is not None
    assert result.points
    assert result.image_path is not None
    assert Path(result.image_path).exists()
    assert result.overlay_path is not None
    assert Path(result.overlay_path).exists()
    assert result.stats.align_ms >= 0.0
    assert result.mm_per_pixel == pytest.approx(scale_model.mm_per_pixel) 
