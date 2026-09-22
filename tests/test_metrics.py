"""Tests for indicjevbench/metrics.py — pure Python, no GPU, no network."""

from __future__ import annotations

import json
import math

import pytest

from indicjevbench.metrics import (
    accuracy,
    append_to_leaderboard,
    automatable_share,
    breakdown,
    brier,
    compute_all,
    ece,
    macro_f1,
    mae_expected_level,
    nll,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _uniform(k: int) -> list[float]:
    return [1.0 / k] * k


def _one_hot(k: int, idx: int) -> list[float]:
    v = [0.0] * k
    v[idx] = 1.0
    return v


def _noul_probs(p_true: float) -> list[float]:
    return [1.0 - p_true, p_true]


# Tiny synthetic batch: 5 choice (K=3), 5 score (K=5), 5 noul
_EXAMPLES = [
    # choice
    {
        "id": "e0",
        "lang": "hi-Deva",
        "source": "massive",
        "questions": [{"qid": "q", "type": "choice", "label": 0}],
    },
    {
        "id": "e1",
        "lang": "hi-Deva",
        "source": "massive",
        "questions": [{"qid": "q", "type": "choice", "label": 1}],
    },
    {
        "id": "e2",
        "lang": "bn-Beng",
        "source": "massive",
        "questions": [{"qid": "q", "type": "choice", "label": 2}],
    },
    {
        "id": "e3",
        "lang": "bn-Beng",
        "source": "massive",
        "questions": [{"qid": "q", "type": "choice", "label": 0}],
    },
    {
        "id": "e4",
        "lang": "ta-Taml",
        "source": "massive",
        "questions": [{"qid": "q", "type": "choice", "label": 1}],
    },
    # score
    {
        "id": "e5",
        "lang": "hi-Latn",
        "source": "synthetic",
        "questions": [{"qid": "q", "type": "score", "label": 0}],
    },
    {
        "id": "e6",
        "lang": "hi-Latn",
        "source": "synthetic",
        "questions": [{"qid": "q", "type": "score", "label": 2}],
    },
    {
        "id": "e7",
        "lang": "hi-Deva",
        "source": "synthetic",
        "questions": [{"qid": "q", "type": "score", "label": 4}],
    },
    {
        "id": "e8",
        "lang": "hi-Deva",
        "source": "synthetic",
        "questions": [{"qid": "q", "type": "score", "label": 1}],
    },
    {
        "id": "e9",
        "lang": "hi-Deva",
        "source": "synthetic",
        "questions": [{"qid": "q", "type": "score", "label": 3}],
    },
    # noul
    {
        "id": "e10",
        "lang": "hi-Latn",
        "source": "comilingua",
        "questions": [{"qid": "q", "type": "noul", "label": True}],
    },
    {
        "id": "e11",
        "lang": "hi-Latn",
        "source": "comilingua",
        "questions": [{"qid": "q", "type": "noul", "label": False}],
    },
    {
        "id": "e12",
        "lang": "hi-Latn",
        "source": "comilingua",
        "questions": [{"qid": "q", "type": "noul", "label": True}],
    },
    {
        "id": "e13",
        "lang": "bn-Beng",
        "source": "comilingua",
        "questions": [{"qid": "q", "type": "noul", "label": False}],
    },
    {
        "id": "e14",
        "lang": "bn-Beng",
        "source": "comilingua",
        "questions": [{"qid": "q", "type": "noul", "label": True}],
    },
]

_ANSWERS_PERFECT = [
    # choice — argmax matches label
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 0)},
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 1)},
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 2)},
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 0)},
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 1)},
    # score — argmax matches label, expected = label+1
    {"id": "q", "type": "score", "probabilities": _one_hot(5, 0), "expected": 1.0},
    {"id": "q", "type": "score", "probabilities": _one_hot(5, 2), "expected": 3.0},
    {"id": "q", "type": "score", "probabilities": _one_hot(5, 4), "expected": 5.0},
    {"id": "q", "type": "score", "probabilities": _one_hot(5, 1), "expected": 2.0},
    {"id": "q", "type": "score", "probabilities": _one_hot(5, 3), "expected": 4.0},
    # noul
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.9)},  # true, correct
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.1)},  # false, correct
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.9)},  # true, correct
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.1)},  # false, correct
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.9)},  # true, correct
]

_ANSWERS_WRONG = [
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 1)},  # label=0, pred=1 WRONG
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 0)},  # label=1, pred=0 WRONG
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 0)},  # label=2, pred=0 WRONG
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 1)},  # label=0, pred=1 WRONG
    {"id": "q", "type": "choice", "probabilities": _one_hot(3, 0)},  # label=1, pred=0 WRONG
    {
        "id": "q",
        "type": "score",
        "probabilities": _one_hot(5, 4),
        "expected": 5.0,
    },  # label=0, WRONG
    {
        "id": "q",
        "type": "score",
        "probabilities": _one_hot(5, 0),
        "expected": 1.0,
    },  # label=2, WRONG
    {
        "id": "q",
        "type": "score",
        "probabilities": _one_hot(5, 0),
        "expected": 1.0,
    },  # label=4, WRONG
    {
        "id": "q",
        "type": "score",
        "probabilities": _one_hot(5, 4),
        "expected": 5.0,
    },  # label=1, WRONG
    {
        "id": "q",
        "type": "score",
        "probabilities": _one_hot(5, 0),
        "expected": 1.0,
    },  # label=3, WRONG
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.1)},  # label=True,  pred=False WRONG
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.9)},  # label=False, pred=True  WRONG
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.1)},  # label=True,  pred=False WRONG
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.9)},  # label=False, pred=True  WRONG
    {"id": "q", "type": "noul", "probabilities": _noul_probs(0.1)},  # label=True,  pred=False WRONG
]


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


def test_accuracy_all_correct():
    assert accuracy([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_accuracy_all_wrong():
    assert accuracy([1, 2, 0], [0, 1, 2]) == pytest.approx(0.0)


def test_accuracy_half():
    assert accuracy([0, 1, 0, 1], [0, 1, 1, 0]) == pytest.approx(0.5)


def test_accuracy_empty():
    assert math.isnan(accuracy([], []))


# ---------------------------------------------------------------------------
# macro_f1
# ---------------------------------------------------------------------------


def test_macro_f1_balanced_perfect():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_macro_f1_all_wrong_binary():
    assert macro_f1([1, 1], [0, 0]) == pytest.approx(0.0)


def test_macro_f1_imbalanced():
    # 100 class-0 correct, 1 class-1 correct — macro averages per class equally
    preds = [0] * 100 + [1]
    labels = [0] * 100 + [1]
    # Class 0: P=1, R=1, F1=1. Class 1: P=1, R=1, F1=1. Macro=1.0
    assert macro_f1(preds, labels) == pytest.approx(1.0)


def test_macro_f1_imbalanced_wrong():
    # Class 0 all predicted wrong (predicted as 1)
    preds = [1] * 10 + [1]
    labels = [0] * 10 + [1]
    # Class 0: TP=0, FP=0, FN=10 → F1=0
    # Class 1: TP=1, FP=10, FN=0 → P=1/11, R=1, F1=2/12
    f1_1 = 2 * (1 / 11) * 1 / (1 / 11 + 1)
    assert macro_f1(preds, labels) == pytest.approx((0 + f1_1) / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# nll
# ---------------------------------------------------------------------------


def test_nll_correct_confident():
    probs = [[0.9, 0.1], [0.8, 0.2]]
    labels = [0, 0]
    result = nll(probs, labels)
    assert result == pytest.approx(-math.log(0.9) / 2 + -math.log(0.8) / 2, abs=1e-6)


def test_nll_wrong_confident():
    probs = [[0.05, 0.95]]
    labels = [0]
    result = nll(probs, labels)
    assert result == pytest.approx(-math.log(0.05), abs=1e-6)


def test_nll_clip_zero_prob():
    probs = [[0.0, 1.0]]
    labels = [0]
    result = nll(probs, labels)
    assert math.isfinite(result)
    assert result == pytest.approx(-math.log(1e-7), abs=1e-6)


def test_nll_empty():
    assert math.isnan(nll([], []))


# ---------------------------------------------------------------------------
# brier
# ---------------------------------------------------------------------------


def test_brier_perfect():
    probs = [_one_hot(3, 0), _one_hot(3, 1)]
    labels = [0, 1]
    assert brier(probs, labels) == pytest.approx(0.0)


def test_brier_uniform_binary():
    # uniform over K=2, label is always 0 → brier = (0.5-1)^2 + (0.5-0)^2 = 0.25+0.25 = 0.5
    probs = [_uniform(2)]
    labels = [0]
    assert brier(probs, labels) == pytest.approx(0.5)


def test_brier_worst_case_binary():
    # p=[1,0], label=1 → (1-0)^2 + (0-1)^2 = 2.0
    probs = [[1.0, 0.0]]
    labels = [1]
    assert brier(probs, labels) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ece
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration():
    # Each example: confidence = accuracy within its bin → ECE = 0
    # 10 examples: 5 correct at confidence 0.8, 5 wrong at confidence 0.2
    # But this doesn't give ECE=0. Instead, use exactly calibrated examples:
    # All in a single bin with confidence=0.7 and accuracy=0.7
    # Perfect calibration: p_max = fraction correct in every bin
    # Construct: 5 items with p_max=0.6 (3 correct, 2 wrong) → conf=0.6, acc=0.6
    probs2 = [[0.6, 0.4]] * 3 + [[0.4, 0.6]] * 2
    labels2 = [0] * 3 + [0] * 2  # 3 correct (p=0.6), 2 wrong (p=0.6)
    result = ece(probs2, labels2)
    # conf=0.6, acc=3/5=0.6 → ECE=0
    assert result == pytest.approx(0.0, abs=1e-6)


def test_ece_15_bins():
    # Smoke test: all high-confidence correct → ECE near 0
    probs = [_one_hot(3, 0)] * 100
    labels = [0] * 100
    result = ece(probs, labels)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_ece_perfectly_wrong():
    # All confident and wrong: confidence=1.0, accuracy=0.0 → ECE=1.0
    probs = [[1.0, 0.0]] * 10
    labels = [1] * 10
    assert ece(probs, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mae_expected_level
# ---------------------------------------------------------------------------


def test_mae_expected_level_perfect():
    # predicted mean = true level (1-indexed): label=0 → level 1, expected=1.0
    assert mae_expected_level([1.0, 2.0, 3.0], [0, 1, 2]) == pytest.approx(0.0)


def test_mae_expected_level_off_by_one():
    assert mae_expected_level([2.0, 3.0], [0, 1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# automatable_share
# ---------------------------------------------------------------------------


def test_automatable_share_all_correct():
    # All correct at confidence 0.9 → threshold=0.0 gives share=1.0
    confs = [0.9] * 10
    preds = [0] * 10
    labels = [0] * 10
    share, _ = automatable_share(confs, preds, labels)
    assert share == pytest.approx(1.0)


def test_automatable_share_all_wrong():
    # All wrong → no threshold achieves ≤5% error rate → share=0.0
    confs = [0.9] * 10
    preds = [1] * 10
    labels = [0] * 10
    share, threshold = automatable_share(confs, preds, labels)
    assert share == pytest.approx(0.0)
    assert threshold == pytest.approx(1.0)


def test_automatable_share_threshold_logic():
    # Low-confidence items are wrong, high-confidence items are correct.
    # T should cut below the high-confidence ones.
    confs = [0.3, 0.3, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9]
    preds = [1, 1, 1, 0, 0, 0, 0, 0]
    labels = [0, 0, 0, 0, 0, 0, 0, 0]
    # Only items with conf>0.3 (the 5 high-conf correct ones) meet ≤5% error
    share, threshold = automatable_share(confs, preds, labels)
    assert share == pytest.approx(5 / 8)
    assert threshold < 0.9


def test_automatable_share_empty():
    share, _ = automatable_share([], [], [])
    assert math.isnan(share)


# ---------------------------------------------------------------------------
# noul label casting
# ---------------------------------------------------------------------------


def test_noul_label_bool_cast():
    """bool True/False labels must be cast to int 1/0 inside compute_all."""
    examples = [
        {
            "id": "e0",
            "lang": "hi-Latn",
            "source": "s",
            "questions": [{"qid": "q", "type": "noul", "label": True}],
        },
        {
            "id": "e1",
            "lang": "hi-Latn",
            "source": "s",
            "questions": [{"qid": "q", "type": "noul", "label": False}],
        },
    ]
    answers = [
        {"id": "q", "type": "noul", "probabilities": [0.1, 0.9]},  # pred=True=1, label=True=1
        {"id": "q", "type": "noul", "probabilities": [0.9, 0.1]},  # pred=False=0, label=False=0
    ]
    result = compute_all(answers, examples)
    assert result["noul"]["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


def test_compute_all_perfect():
    result = compute_all(_ANSWERS_PERFECT, _EXAMPLES)
    assert result["choice"]["accuracy"] == pytest.approx(1.0)
    assert result["score"]["accuracy"] == pytest.approx(1.0)
    assert result["noul"]["accuracy"] == pytest.approx(1.0)
    assert result["all"]["accuracy"] == pytest.approx(1.0)
    assert result["score"]["mae_expected_level"] == pytest.approx(0.0)


def test_compute_all_all_wrong():
    result = compute_all(_ANSWERS_WRONG, _EXAMPLES)
    assert result["choice"]["accuracy"] == pytest.approx(0.0)
    assert result["score"]["accuracy"] == pytest.approx(0.0)
    assert result["noul"]["accuracy"] == pytest.approx(0.0)


def test_compute_all_has_all_keys():
    result = compute_all(_ANSWERS_PERFECT, _EXAMPLES)
    for q_type in ("choice", "score", "noul", "all"):
        assert q_type in result
    for key in (
        "accuracy",
        "macro_f1",
        "nll",
        "brier",
        "ece",
        "automatable_share",
        "automatable_threshold",
    ):
        assert key in result["choice"], f"missing {key} in choice"
        assert key in result["noul"], f"missing {key} in noul"
    assert "mae_expected_level" in result["score"]


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------


def test_breakdown_by_lang():
    result = breakdown(_ANSWERS_PERFECT, _EXAMPLES, dim="lang")
    assert "hi-Deva" in result
    assert "bn-Beng" in result
    assert result["hi-Deva"]["all"]["accuracy"] == pytest.approx(1.0)


def test_breakdown_by_source():
    result = breakdown(_ANSWERS_PERFECT, _EXAMPLES, dim="source")
    assert "massive" in result
    assert "synthetic" in result
    assert "comilingua" in result


# ---------------------------------------------------------------------------
# leaderboard
# ---------------------------------------------------------------------------


def test_append_leaderboard_creates_file(tmp_path):
    lb_path = tmp_path / "leaderboard.json"
    results = {"tasks": {"intent_massive": {"metrics": {"all": {"accuracy": 0.82}}}}}
    append_to_leaderboard(results, lb_path, "test-model", "cmul8", "run_001")
    data = json.loads(lb_path.read_text())
    assert data["schema_version"] == 1
    assert len(data["entries"]) == 1
    assert data["entries"][0]["model"] == "test-model"


def test_append_leaderboard_accumulates(tmp_path):
    lb_path = tmp_path / "leaderboard.json"
    results = {"tasks": {}}
    append_to_leaderboard(results, lb_path, "model-a", "cmul8", "run_001")
    append_to_leaderboard(results, lb_path, "model-b", "cmul8", "run_002")
    data = json.loads(lb_path.read_text())
    assert len(data["entries"]) == 2


def test_append_leaderboard_atomic_write(tmp_path):
    lb_path = tmp_path / "leaderboard.json"
    results = {"tasks": {}}
    append_to_leaderboard(results, lb_path, "model-a", "cmul8", "run_001")
    # Tmp file should not remain
    assert not (tmp_path / "leaderboard.tmp").exists()
