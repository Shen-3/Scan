import pytest

from app.models import ShotPoint
from app.processing.metrics import compute_metrics


def test_compute_metrics_symmetry():
    points = [
        ShotPoint(id=1, x_mm=0.0, y_mm=0.0, radius_mm=5.0),
        ShotPoint(id=2, x_mm=10.0, y_mm=0.0, radius_mm=5.0),
        ShotPoint(id=3, x_mm=0.0, y_mm=10.0, radius_mm=5.0),
    ]
    metrics = compute_metrics(points)
    assert metrics.shot_count == 3
    assert metrics.mean_x_mm == pytest.approx(10.0 / 3.0)
    assert metrics.mean_y_mm == pytest.approx(10.0 / 3.0)
    assert metrics.extreme_spread_mm == pytest.approx(14.142, rel=1e-3)
    assert metrics.mean_radius_mm == pytest.approx((0.0 + 10.0 + 10.0) / 3.0)
    assert metrics.r50_mm == pytest.approx(10.0)


def test_compute_metrics_single_point():
    point = ShotPoint(id=1, x_mm=5.0, y_mm=-2.0, radius_mm=5.0)
    metrics = compute_metrics([point])
    assert metrics.mean_x_mm == 5.0
    assert metrics.displacement_mm == pytest.approx((5.0**2 + (-2.0) ** 2) ** 0.5)
    assert metrics.extreme_spread_mm == 0.0
    assert metrics.mean_radius_mm == pytest.approx((5.0**2 + (-2.0) ** 2) ** 0.5)
    assert metrics.r50_mm == pytest.approx((5.0**2 + (-2.0) ** 2) ** 0.5)
