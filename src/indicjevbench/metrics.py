"""Metric computations for IndicJevBench.

All functions are pure (no I/O). They take Python lists and return dicts.
Importable by runner.py, tests, and notebook analysis.

Confidence formulas
-------------------
choice/score : (p_max - 1/K) / (1 - 1/K)   — normalised margin above chance
noul         : |2*p_true - 1|               — distance from the 0.5 decision boundary
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def accuracy(predictions: list[int], labels: list[int]) -> float:
    if not predictions:
        return float("nan")
    return sum(p == l for p, l in zip(predictions, labels)) / len(predictions)


def macro_f1(predictions: list[int], labels: list[int]) -> float:
    """Sklearn-style macro F1: unweighted mean of per-class F1."""
    if not predictions:
        return float("nan")
    classes = sorted(set(labels) | set(predictions))
    f1s = []
    for c in classes:
        tp = sum(p == c and l == c for p, l in zip(predictions, labels))
        fp = sum(p == c and l != c for p, l in zip(predictions, labels))
        fn = sum(p != c and l == c for p, l in zip(predictions, labels))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / len(f1s)


def nll(probs_list: list[list[float]], labels: list[int]) -> float:
    """Mean negative log-likelihood. Clips P to 1e-7 to avoid log(0)."""
    if not probs_list:
        return float("nan")
    total = 0.0
    for probs, label in zip(probs_list, labels):
        p = max(probs[label], 1e-7)
        total += -math.log(p)
    return total / len(probs_list)


def brier(probs_list: list[list[float]], labels: list[int]) -> float:
    """Mean squared error of full distribution vs one-hot true label."""
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

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    For noul, p_max = max(p_true, 1-p_true). Predictions are correct when argmax==label.
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
    """Mean |predicted_mean_level - true_level| for score questions (1-indexed)."""
    if not expected_levels:
        return float("nan")
    return sum(abs(e - (l + 1)) for e, l in zip(expected_levels, labels)) / len(expected_levels)


def automatable_share(
    confidences: list[float],
    predictions: list[int],
    labels: list[int],
    max_error_rate: float = 0.05,
) -> tuple[float, float]:
    """Return (share, threshold) where threshold is the highest T such that
    items with confidence > T have error_rate <= max_error_rate.

    Binary search over sorted confidence values. Returns (0.0, 1.0) if no
    threshold achieves the target error rate.
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
    k = len(probs)
    p_max = max(probs)
    if k <= 1:
        return 1.0
    return (p_max - 1.0 / k) / (1.0 - 1.0 / k)


def _confidence_noul(p_true: float) -> float:
    return abs(2 * p_true - 1)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def compute_all(answers: list[dict], examples: list[dict]) -> dict:
    """Compute all metrics for a list of (answer, example) pairs.

    answers  : list of answer dicts from the API/local adapter, each containing
               {"id", "type", "probabilities", ...}
    examples : list of example dicts from data.jsonl, matching answers 1-to-1.

    Returns nested dict: {"choice": {...}, "score": {...}, "noul": {...}, "all": {...}}
    Each sub-dict has keys: accuracy, macro_f1, nll, brier, ece,
    plus mae_expected_level (score only), automatable_share, automatable_threshold.
    """
    by_type: dict[str, list] = {"choice": [], "score": [], "noul": []}

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

    result = {}
    all_items = []
    for q_type, items in by_type.items():
        if not items:
            continue
        probs_l, labels, preds, confs, expecteds = zip(*items)
        m = {
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


def breakdown(answers: list[dict], examples: list[dict], dim: str) -> dict:
    """Group by a dimension field and compute metrics per group.

    dim: "lang" | "source" | "type" (question type)
    Returns {group_key: metrics_dict}.
    """
    groups: dict[str, tuple[list, list]] = {}
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


def _get_label(qid: str, example: dict) -> int | bool | None:
    for q in example.get("questions", []):
        if q.get("qid") == qid:
            return q.get("label")
    return None


# ---------------------------------------------------------------------------
# Leaderboard helper
# ---------------------------------------------------------------------------

def append_to_leaderboard(
    results: dict,
    leaderboard_path: Path,
    model_name: str,
    submitted_by: str,
    run_id: str,
    notes: str = "",
) -> None:
    """Append a results entry to leaderboard.json, writing atomically."""
    if leaderboard_path.exists():
        board = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    else:
        board = {"schema_version": 1, "entries": []}

    entry = {
        "model": model_name,
        "submitted_by": submitted_by,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "run_id": run_id,
        "tasks": {
            task: data.get("metrics", {})
            for task, data in results.get("tasks", {}).items()
        },
        "notes": notes,
    }
    board["entries"].append(entry)

    tmp = leaderboard_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(leaderboard_path)
