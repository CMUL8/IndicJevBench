"""Metric computations for IndicJevBench.

All functions are pure (no I/O). They take Python lists and return dicts.
Importable by the benchmark runner, tests, and notebook analysis.

Confidence formulas
-------------------
choice/score : (p_max - 1/K) / (1 - 1/K)   — normalised margin above chance
noul         : |2*p_true - 1|              — distance from the 0.5 decision boundary

Calibration
-----------
Expected Calibration Error (ECE) uses equal-width bins of max-class
confidence with a 15-bin default, matching standard practice for
K-class calibration reports.

The composite benchmark score (see ``indicjevbench.scoring``) combines
accuracy, calibration, speed, and cost with weights 0.35/0.25/0.20/0.20.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def accuracy(predictions: list[int], labels: list[int]) -> float:
    """Fraction of predictions equal to their labels.

    Args:
        predictions: Predicted class indices (bool allowed for noul).
        labels: Gold class indices, aligned with ``predictions``.

    Returns:
        Accuracy in ``[0, 1]``, or NaN when ``predictions`` is empty.
    """
    if not predictions:
        return float("nan")
    return sum(p == l for p, l in zip(predictions, labels)) / len(predictions)


def macro_f1(predictions: list[int], labels: list[int]) -> float:
    """Sklearn-style macro F1: unweighted mean of per-class F1.

    Args:
        predictions: Predicted class indices.
        labels: Gold class indices, aligned with ``predictions``.

    Returns:
        Macro F1 in ``[0, 1]``, or NaN when ``predictions`` is empty.
    """
    if not predictions:
        return float("nan")
    classes = sorted(set(labels) | set(predictions))
    f1s: list[float] = []
    for c in classes:
        tp = sum(p == c and l == c for p, l in zip(predictions, labels))
        fp = sum(p == c and l != c for p, l in zip(predictions, labels))
        fn = sum(p != c and l == c for p, l in zip(predictions, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / len(f1s)


def nll(probs_list: list[list[float]], labels: list[int]) -> float:
    """Mean negative log-likelihood of the true label under each distribution.

    Args:
        probs_list: One probability distribution per example.
        labels: Gold class indices, aligned with ``probs_list``.

    Returns:
        Mean NLL, or NaN when ``probs_list`` is empty. Probabilities are
        clipped to ``1e-7`` so a zero probability on the gold label yields a
        finite penalty instead of ``inf``.
    """
    if not probs_list:
        return float("nan")
    total = 0.0
    for probs, label in zip(probs_list, labels):
        p = max(probs[label], 1e-7)
        total += -math.log(p)
    return total / len(probs_list)


def brier(probs_list: list[list[float]], labels: list[int]) -> float:
    """Mean squared error of each full distribution vs the one-hot true label.

    Args:
        probs_list: One probability distribution per example.
        labels: Gold class indices, aligned with ``probs_list``.

    Returns:
        Mean Brier score (lower is better; 0 is perfect), or NaN when
        ``probs_list`` is empty.
    """
    if not probs_list:
        return float("nan")
    total = 0.0
    for probs, label in zip(probs_list, labels):
        one_hot = [0.0] * len(probs)
        one_hot[label] = 1.0
        total += sum((p - h) ** 2 for p, h in zip(probs, one_hot))
    return total / len(probs_list)


def ece(probs_list: list[list[float]], labels: list[int], n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width bins of max-class confidence.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|. For noul, the reported
    ``p_max`` is max(p_true, 1-p_true). Predictions are correct when
    ``argmax(probs) == label``.

    Args:
        probs_list: One probability distribution per example.
        labels: Gold class indices, aligned with ``probs_list``.
        n_bins: Number of equal-width confidence bins (default 15).

    Returns:
        ECE in ``[0, 1]``, or NaN when ``probs_list`` is empty.
    """
    if not probs_list:
        return float("nan")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for probs, label in zip(probs_list, labels):
        p_max = max(probs)
        correct = int(probs.index(p_max)) == label
        b = min(int(p_max * n_bins), n_bins - 1)
        bins[b].append((p_max, correct))
    total_ece = 0.0
    n = len(probs_list)
    for b in bins:
        if not b:
            continue
        acc_b = sum(correct for _, correct in b) / len(b)
        conf_b = sum(conf for conf, _ in b) / len(b)
        total_ece += (len(b) / n) * abs(acc_b - conf_b)
    return total_ece


def mae_expected_level(expected_levels: list[float], labels: list[int]) -> float:
    """Mean |predicted_mean_level - true_level| for score questions (1-indexed).

    Args:
        expected_levels: Predicted mean levels (1-indexed), one per example.
        labels: Gold option indices (0-indexed), aligned with
            ``expected_levels``.

    Returns:
        Mean absolute level error, or NaN when ``expected_levels`` is empty.
    """
    if not expected_levels:
        return float("nan")
    return sum(abs(e - (l + 1)) for e, l in zip(expected_levels, labels)) / len(expected_levels)


def automatable_share(
    confidences: list[float],
    predictions: list[int],
    labels: list[int],
    max_error_rate: float = 0.05,
) -> tuple[float, float]:
    """Largest share of items automatable under an error-rate cap.

    The threshold is the highest ``T`` such that items with
    ``confidence > T`` have ``error_rate <= max_error_rate``. A linear scan
    over candidate thresholds (just below each unique confidence, plus an
    include-all candidate) replaces an explicit binary search but keeps the
    same result.

    Args:
        confidences: Calibrated confidence per item.
        predictions: Predicted class indices, aligned with ``confidences``.
        labels: Gold class indices, aligned with ``confidences``.
        max_error_rate: Maximum tolerable error rate above the threshold.

    Returns:
        ``(share, threshold)`` where ``share`` is the fraction of all items
        above the threshold and ``threshold`` is the chosen confidence cut.
        Returns ``(NaN, NaN)`` when ``confidences`` is empty and
        ``(0.0, 1.0)`` when no threshold meets the error cap.
    """
    if not confidences:
        return (float("nan"), float("nan"))

    paired = sorted(zip(confidences, predictions, labels), key=lambda x: x[0])

    # Candidate thresholds: just below each unique confidence value, plus -1 (include all)
    unique_confs = sorted({c for c, _, _ in paired})
    candidates = [-1.0] + [c - 1e-9 for c in unique_confs]

    best_share = 0.0
    best_threshold = 1.0

    for T in candidates:
        above = [(p, l) for c, p, l in paired if c > T]
        if not above:
            continue
        err = sum(p != l for p, l in above) / len(above)
        if err <= max_error_rate:
            share = len(above) / len(confidences)
            if share > best_share:
                best_share = share
                best_threshold = T

    return (best_share, best_threshold)


# ---------------------------------------------------------------------------
# Confidence helpers
# ---------------------------------------------------------------------------


def _confidence_choice_score(probs: list[float]) -> float:
    """Normalised margin above the uniform-chance probability for K options."""
    k = len(probs)
    p_max = max(probs)
    if k <= 1:
        return 1.0
    return (p_max - 1.0 / k) / (1.0 - 1.0 / k)


def _confidence_noul(p_true: float) -> float:
    """Distance of ``p_true`` from the 0.5 decision boundary, in ``[0, 1]``."""
    return abs(2 * p_true - 1)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------


def compute_all(answers: list[dict[str, Any]], examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute all metrics for a list of (answer, example) pairs.

    ``answers`` are adapter answer dicts, each containing at least
    ``{"id", "type", "probabilities"}`` plus an optional ``expected`` mean
    level for score questions. ``examples`` are dataset example dicts, each
    with ``{"id", "lang", "source", "questions": [{"qid", "type", "label"}]}``.
    The two lists must be aligned 1-to-1.

    Confidence per item uses the formulas documented in the module docstring:
    choice/score take the normalised argmax margin; noul takes
    ``|2*p_true - 1|`` with ``p_true = probs[1]``.

    Args:
        answers: List of answer dicts from an adapter.
        examples: List of example dicts from the dataset, aligned with
            ``answers``.

    Returns:
        Nested dict ``{"choice": {...}, "score": {...}, "noul": {...},
        "all": {...}}`` with one entry per question type present. Each
        sub-dict has keys ``n``, ``accuracy``, ``macro_f1``, ``nll``,
        ``brier``, ``ece``, ``automatable_share``, ``automatable_threshold``,
        plus ``mae_expected_level`` for score items only. The ``all`` entry
        pools every answered item.
    """
    by_type: dict[str, list[Any]] = {"choice": [], "score": [], "noul": []}

    for ans, ex in zip(answers, examples):
        q_type = ans["type"]
        if q_type not in by_type:
            continue
        probs = ans["probabilities"]
        # Get the label from the matching question in the example
        label_raw = _get_label(ans["id"], ex)
        if label_raw is None:
            continue
        label = int(label_raw)  # cast bool→int for noul

        if q_type == "noul":
            # probs = [P(false), P(true)]
            p_true = probs[1] if len(probs) >= 2 else probs[0]
            conf = _confidence_noul(p_true)
            pred = int(p_true > 0.5)
        else:
            pred = int(probs.index(max(probs)))
            conf = _confidence_choice_score(probs)

        expected = ans.get("expected")  # only present for score
        by_type[q_type].append((probs, label, pred, conf, expected))

    result: dict[str, Any] = {}
    all_items: list[Any] = []
    for q_type, items in by_type.items():
        if not items:
            continue
        probs_l, labels, preds, confs, expecteds = zip(*items)
        m: dict[str, Any] = {
            "n": len(items),
            "accuracy": accuracy(list(preds), list(labels)),
            "macro_f1": macro_f1(list(preds), list(labels)),
            "nll": nll(list(probs_l), list(labels)),
            "brier": brier(list(probs_l), list(labels)),
            "ece": ece(list(probs_l), list(labels)),
        }
        if q_type == "score":
            valid = [(e, l) for e, l in zip(expecteds, labels) if e is not None]
            if valid:
                exp_l, lab_l = zip(*valid)
                m["mae_expected_level"] = mae_expected_level(list(exp_l), list(lab_l))
        share, threshold = automatable_share(list(confs), list(preds), list(labels))
        m["automatable_share"] = share
        m["automatable_threshold"] = threshold
        result[q_type] = m
        all_items.extend(items)

    if all_items:
        probs_l, labels, preds, confs, _ = zip(*all_items)
        share, threshold = automatable_share(list(confs), list(preds), list(labels))
        result["all"] = {
            "n": len(all_items),
            "accuracy": accuracy(list(preds), list(labels)),
            "macro_f1": macro_f1(list(preds), list(labels)),
            "nll": nll(list(probs_l), list(labels)),
            "brier": brier(list(probs_l), list(labels)),
            "ece": ece(list(probs_l), list(labels)),
            "automatable_share": share,
            "automatable_threshold": threshold,
        }

    return result


def breakdown(
    answers: list[dict[str, Any]], examples: list[dict[str, Any]], dim: str
) -> dict[str, Any]:
    """Group by a dimension field and compute metrics per group.

    Args:
        answers: List of answer dicts (see :func:`compute_all`).
        examples: List of example dicts aligned with ``answers``.
        dim: Grouping dimension — ``"lang"``, ``"source"``, or ``"type"``
            (question type). Missing values fall back to ``"unknown"``.

    Returns:
        Mapping of group key to the :func:`compute_all` result for that group.
    """
    groups: dict[str, tuple[list[Any], list[Any]]] = {}
    for ans, ex in zip(answers, examples):
        if dim == "type":
            key = ans["type"]
        else:
            key = ex.get(dim, "unknown")
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(ans)
        groups[key][1].append(ex)

    return {k: compute_all(a, e) for k, (a, e) in groups.items()}


def _get_label(qid: str, example: dict[str, Any]) -> int | bool | None:
    """Return the gold label for question ``qid`` in an example, if present.

    Args:
        qid: Question identifier to look up.
        example: Example dict with a ``questions`` list of
            ``{"qid", "type", "label"}`` mappings.

    Returns:
        The label (int or bool), or ``None`` when no question matches.
    """
    for q in example.get("questions", []):
        if q.get("qid") == qid:
            return cast(int | bool | None, q.get("label"))
    return None


# ---------------------------------------------------------------------------
# Leaderboard helper
# ---------------------------------------------------------------------------


def append_to_leaderboard(
    results: dict[str, Any],
    leaderboard_path: Path,
    model_name: str,
    submitted_by: str,
    run_id: str,
    notes: str = "",
) -> None:
    """Append a results entry to ``leaderboard.json``, writing atomically.

    Creates the leaderboard file with ``schema_version: 1`` when absent.

    Args:
        results: Run results dict with a ``tasks`` mapping of task name to
            ``{"metrics": {...}}`` data.
        leaderboard_path: Destination leaderboard JSON path.
        model_name: Display name of the evaluated model.
        submitted_by: Submitter identifier.
        run_id: Unique run identifier for provenance.
        notes: Free-form notes attached to the entry.

    Raises:
        OSError: If the leaderboard cannot be read or written.
        ValueError: If an existing leaderboard file is not valid JSON.
    """
    board: dict[str, Any]
    if leaderboard_path.exists():
        board = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    else:
        board = {"schema_version": 1, "entries": []}

    entry = {
        "model": model_name,
        "submitted_by": submitted_by,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "run_id": run_id,
        "tasks": {task: data.get("metrics", {}) for task, data in results.get("tasks", {}).items()},
        "notes": notes,
    }
    board["entries"].append(entry)

    tmp = leaderboard_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(leaderboard_path)
