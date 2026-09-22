from __future__ import annotations

import math


def intelligence_score(accuracy: float) -> float:
    """0-100 linear from accuracy."""
    return max(0.0, min(100.0, accuracy * 100))

def calibration_score(ece: float, brier: float) -> float:
    """0-100: mean(1-ECE, 1-brier/2) scaled."""
    brier_norm = min(brier / 2.0, 1.0)  # max brier for binary=2.0
    return max(0.0, min(100.0, ((1 - ece) + (1 - brier_norm)) / 2 * 100))

def speed_score(p50_ms: float) -> float:
    """0-100: 50ms=100, 5000ms=0, log scale."""
    if p50_ms <= 0:
        return 100.0
    score = 100.0 - max(0.0, math.log10(p50_ms / 50) / math.log10(100)) * 100
    return max(0.0, min(100.0, score))

def cost_score(cost_per_1k: float) -> float:
    """0-100: $0.01/1k=100, $10/1k=0, log scale. 0.0 for local (free) = 100."""
    if cost_per_1k <= 0:
        return 100.0
    score = 100.0 - max(0.0, math.log10(cost_per_1k / 0.01) / math.log10(1000)) * 100
    return max(0.0, min(100.0, score))

def indicjev_score(
    accuracy: float,
    ece: float,
    brier: float,
    p50_ms: float,
    cost_per_1k: float = 0.0,
) -> dict:
    """Compute composite IndicJevScore.

    Returns dict with axes (0-100 each) and composite score (geometric mean).
    Weights: intelligence=35%, calibration=25%, speed=20%, cost=20%.
    """
    intel = intelligence_score(accuracy)
    calib = calibration_score(ece, brier)
    speed = speed_score(p50_ms)
    cost  = cost_score(cost_per_1k)

    # Weighted geometric mean (all must be > 0)
    weights = [0.35, 0.25, 0.20, 0.20]
    axes = [intel, calib, speed, cost]
    # Clip to 1e-6 to avoid log(0)
    composite = math.exp(sum(w * math.log(max(a, 1e-6)) for w, a in zip(weights, axes)))

    return {
        "indicjev_score": round(composite, 1),
        "axes": {
            "intelligence": round(intel, 1),
            "calibration": round(calib, 1),
            "speed": round(speed, 1),
            "cost": round(cost, 1),
        },
    }
