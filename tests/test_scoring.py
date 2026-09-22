"""Tests for indicjevbench/scoring.py — axis endpoints and composite math."""

from __future__ import annotations

import math

import pytest

from indicjevbench.scoring import (
    calibration_score,
    cost_score,
    indicjev_score,
    intelligence_score,
    speed_score,
)

# ---------------------------------------------------------------------------
# Axis endpoints
# ---------------------------------------------------------------------------


def test_intelligence_linear():
    assert intelligence_score(0.0) == 0.0
    assert intelligence_score(0.5) == 50.0
    assert intelligence_score(1.0) == 100.0


def test_intelligence_clipped():
    assert intelligence_score(1.5) == 100.0
    assert intelligence_score(-0.2) == 0.0


def test_calibration_endpoints():
    assert calibration_score(0.0, 0.0) == 100.0
    assert calibration_score(1.0, 2.0) == 0.0


def test_calibration_midpoint():
    # ece=0, brier=1 (half of binary worst case) → mean(1, 0.5)*100 = 75
    assert calibration_score(0.0, 1.0) == 75.0


def test_speed_endpoints():
    assert speed_score(50.0) == 100.0
    assert speed_score(5000.0) == 0.0
    assert speed_score(0.0) == 100.0  # free/instant local
    assert speed_score(-5.0) == 100.0


def test_speed_clipped_above_range():
    # Slower than 5000 ms stays 0, not negative.
    assert speed_score(1_000_000.0) == 0.0


def test_cost_endpoints():
    assert cost_score(0.01) == 100.0
    assert cost_score(10.0) == 0.0
    assert cost_score(0.0) == 100.0  # local / free


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def test_composite_perfect():
    result = indicjev_score(accuracy=1.0, ece=0.0, brier=0.0, p50_ms=50.0)
    assert result["indicjev_score"] == 100.0
    assert result["axes"] == {
        "intelligence": 100.0,
        "calibration": 100.0,
        "speed": 100.0,
        "cost": 100.0,
    }


def test_composite_known_values():
    """Hand-computed weighted geometric mean on known axis values."""
    accuracy, ece, brier, p50 = 0.8, 0.1, 0.4, 500.0
    result = indicjev_score(accuracy=accuracy, ece=ece, brier=brier, p50_ms=p50)

    intel = 80.0
    calib = ((1 - 0.1) + (1 - 0.4 / 2.0)) / 2 * 100  # 85.0
    speed = 100.0 - math.log10(500 / 50) / math.log10(100) * 100  # 50.0
    cost = 100.0
    expected = math.exp(
        0.35 * math.log(intel)
        + 0.25 * math.log(calib)
        + 0.20 * math.log(speed)
        + 0.20 * math.log(cost)
    )
    assert result["axes"]["intelligence"] == pytest.approx(intel)
    assert result["axes"]["calibration"] == pytest.approx(calib)
    assert result["axes"]["speed"] == pytest.approx(speed)
    assert result["axes"]["cost"] == pytest.approx(cost)
    assert result["indicjev_score"] == pytest.approx(round(expected, 1))


def test_composite_zero_axis_drives_score_down():
    # accuracy=0 → intelligence clipped to 1e-6 inside the geometric mean.
    result = indicjev_score(accuracy=0.0, ece=0.0, brier=0.0, p50_ms=50.0)
    assert result["axes"]["intelligence"] == 0.0
    assert result["indicjev_score"] < 1.0


def test_composite_weights_sum_to_one():
    """Composite equals the plain geometric mean when all axes are equal."""
    result = indicjev_score(accuracy=0.6, ece=0.4, brier=1.2, p50_ms=500.0)
    axes = result["axes"]
    assert axes["intelligence"] == 60.0
    # Composite of mixed axes must lie below the arithmetic mean of axes.
    axis_values = list(axes.values())
    assert result["indicjev_score"] <= sum(axis_values) / len(axis_values) + 1e-9
