import math

import pytest

from app.processing.scale import mm_per_pixel_from_grid, mm_per_pixel_from_points, ScaleModel


def test_mm_per_pixel_from_grid():
    distances = [100.0, 102.0, 98.0]
    mm_per_pixel = mm_per_pixel_from_grid(10.0, distances)
    assert mm_per_pixel == pytest.approx(0.1, rel=1e-2)


def test_scale_model_conversion():
    model = ScaleModel(mm_per_pixel=0.5)
    assert model.pixels_to_mm(10.0) == 5.0
    assert math.isclose(model.mm_to_pixels(5.0), 10.0)


def test_mm_per_pixel_from_points():
    pairs = [((0.0, 0.0), (10.0, 0.0)), ((0.0, 0.0), (0.0, 10.0))]
    mm_per_pixel = mm_per_pixel_from_points(pairs, 5.0)
    assert mm_per_pixel == pytest.approx(0.5)
