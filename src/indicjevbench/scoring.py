"""IndicJevScore: composite scoring from accuracy, calibration, speed, cost.

Each axis maps to a 0-100 score:

- **intelligence** — linear in accuracy: ``accuracy * 100``.
- **calibration** — mean of ``1 - ECE`` and ``1 - brier/2``, scaled to 100.
  Brier is normalised by its worst-case value of 2.0 (binary, fully wrong).
- **speed** — log scale: 50 ms -> 100, 5000 ms -> 0.
- **cost** — log scale: $0.01/1k tokens -> 100, $10/1k -> 0; free (local)
  models score 100.

The composite is a weighted geometric mean of the four axes with weights
0.35 (intelligence), 0.25 (calibration), 0.20 (speed), 0.20 (cost). Every
axis is clipped to ``[1e-6, 100]`` before the geometric mean so a zero axis
drives the composite toward 0 without raising ``log(0)``.
"""

import math
from typing import Any


def intelligence_score(accuracy: float) -> float:
    """Map accuracy to a 0-100 intelligence axis score.

    Args:
        accuracy: Fraction correct in ``[0, 1]`` (values outside are clipped).

    Returns:
        Linear score ``accuracy * 100`` clipped to ``[0, 100]``.
    """
    return max(0.0, min(100.0, accuracy * 100))


def calibration_score(ece: float, brier: float) -> float:
    """Map calibration error to a 0-100 calibration axis score.

    Args:
        ece: Expected Calibration Error in ``[0, 1]``.
        brier: Mean Brier score (worst case 2.0 for binary).

    Returns:
        ``mean(1 - ECE, 1 - brier/2) * 100`` clipped to ``[0, 100]``.
    """
    brier_norm = min(brier / 2.0, 1.0)  # max brier for binary=2.0
    return max(0.0, min(100.0, ((1 - ece) + (1 - brier_norm)) / 2 * 100))


def speed_score(p50_ms: float) -> float:
    """Map p50 latency to a 0-100 speed axis score (log scale).

    Args:
        p50_ms: Median decision latency in milliseconds. Non-positive values
            (instant/local) score 100.

    Returns:
        Score with 50 ms -> 100 and 5000 ms -> 0, clipped to ``[0, 100]``.
    """
    if p50_ms <= 0:
        return 100.0
    score = 100.0 - max(0.0, math.log10(p50_ms / 50) / math.log10(100)) * 100
    return max(0.0, min(100.0, score))


def cost_score(cost_per_1k: float) -> float:
    """Map cost to a 0-100 cost axis score (log scale).

    Args:
        cost_per_1k: Cost in USD per 1k tokens. Non-positive values (local,
            free) score 100.

    Returns:
        Score with $0.01/1k -> 100 and $10/1k -> 0, clipped to ``[0, 100]``.
    """
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
) -> dict[str, Any]:
    """Compute the composite IndicJevScore.

    Args:
        accuracy: Fraction correct in ``[0, 1]``.
        ece: Expected Calibration Error in ``[0, 1]``.
        brier: Mean Brier score (worst case 2.0 for binary).
        p50_ms: Median decision latency in milliseconds.
        cost_per_1k: Cost in USD per 1k tokens; 0.0 for local/free models.

    Returns:
        Dict with ``indicjev_score`` (weighted geometric mean, rounded to one
        decimal) and ``axes`` — the four axis scores, each rounded to one
        decimal. Axis weights: intelligence 0.35, calibration 0.25, speed
        0.20, cost 0.20.
    """
    intel = intelligence_score(accuracy)
    calib = calibration_score(ece, brier)
    speed = speed_score(p50_ms)
    cost = cost_score(cost_per_1k)

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
