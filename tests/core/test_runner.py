"""Tests for indicjevbench/core/runner.py — BenchmarkRunner with a FakeAdapter."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from indicjevbench.adapters.base import BenchAdapter
from indicjevbench.core.runner import BenchmarkRunner, percentile
from indicjevbench.schemas.contracts import DecisionResult, Task


def _make_task(i: int, family: str = "intent", q_type: str = "choice") -> Task:
    n_options = 3
    if q_type == "noul":
        question: dict[str, Any] = {
            "type": "noul",
            "instructions": "Decide yes or no.",
            "options": None,
            "criteria": {},
        }
    else:
        question = {
            "type": q_type,
            "instructions": "Pick one.",
            "options": ["a", "b", "c"],
            "criteria": {},
        }
    return Task.from_dict(
        {
            "id": f"{family}-{i:03d}",
            "family": family,
            "lang": "hi-Deva" if i % 2 == 0 else "bn-Beng",
            "state": f"message {i}",
            "question": question,
            "expected": i % 2 if q_type == "noul" else i % n_options,
            "split": "v1",
            "source": "testsrc",
            "license": "CC BY 4.0",
            "provenance": {"origin": "test"},
        }
    )


class FakeAdapter(BenchAdapter):
    """Deterministic adapter: perfect answers, or raises on demand.

    Args:
        fail_ids: Task ids for which ``decide`` raises ``RuntimeError``.
        latency_ms: Fixed latency stamped on every decision.
    """

    def __init__(self, fail_ids: set[str] | None = None, latency_ms: float = 10.0) -> None:
        self._fail_ids = fail_ids or set()
        self._latency_ms = latency_ms

    def decide(self, task: Task) -> DecisionResult:
        if task.id in self._fail_ids:
            raise RuntimeError(f"adapter exploded on {task.id}")
        if task.q_type == "noul":
            p_true = 0.9 if task.expected == 1 else 0.1
            return DecisionResult(
                task_id=task.id,
                probabilities=[1.0 - p_true, p_true],
                answer=bool(task.expected),
                confidence=0.8,
                latency_ms=self._latency_ms,
            )
        probs = [0.05, 0.05, 0.05]
        probs[int(task.expected)] = 0.9
        return DecisionResult(
            task_id=task.id,
            probabilities=probs,
            answer=int(task.expected),
            confidence=0.85,
            latency_ms=self._latency_ms,
        )


# ---------------------------------------------------------------------------
# percentile
# ---------------------------------------------------------------------------


def test_percentile_known_values():
    values = [float(i) for i in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 100) == 100.0


def test_percentile_empty_is_nan():
    assert math.isnan(percentile([], 50))


def test_percentile_single():
    assert percentile([42.0], 95) == 42.0


def test_percentile_does_not_mutate_input():
    values = [3.0, 1.0, 2.0]
    percentile(values, 50)
    assert values == [3.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


def _mixed_tasks(n: int = 6) -> list[Task]:
    tasks = [_make_task(i, family="intent") for i in range(n)]
    tasks += [_make_task(i, family="urgency", q_type="noul") for i in range(n)]
    return tasks


def test_run_result_dict_shape(tmp_path: Path):
    tasks = _mixed_tasks()
    runner = BenchmarkRunner(
        FakeAdapter(), raw_log_path=tmp_path / "raw.jsonl", task_name="test"
    )
    result = runner.run(tasks)

    assert set(result.keys()) == {"n_tasks", "n_answered", "metrics", "latency", "score"}
    assert result["n_tasks"] == len(tasks)
    assert result["n_answered"] == len(tasks)

    metrics = result["metrics"]
    for key in ("all", "choice", "noul", "by_lang", "by_source"):
        assert key in metrics
    assert metrics["choice"]["accuracy"] == 1.0
    assert metrics["noul"]["accuracy"] == 1.0
    assert metrics["all"]["accuracy"] == 1.0

    assert set(metrics["by_lang"].keys()) == {"hi-Deva", "bn-Beng"}
    assert set(metrics["by_source"].keys()) == {"testsrc"}

    assert set(result["latency"].keys()) == {"p50_ms", "p95_ms"}
    assert result["latency"]["p50_ms"] == 10.0

    assert set(result["score"].keys()) == {"indicjev_score", "axes"}
    assert set(result["score"]["axes"].keys()) == {
        "intelligence",
        "calibration",
        "speed",
        "cost",
    }
    # Perfect accuracy + confident calibration + fast latency → high score.
    assert result["score"]["indicjev_score"] > 80.0


def test_run_raw_log_jsonl(tmp_path: Path):
    tasks = _mixed_tasks(3)
    log_path = tmp_path / "raw.jsonl"
    runner = BenchmarkRunner(FakeAdapter(), raw_log_path=log_path)
    runner.run(tasks)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(tasks)
    seen_ids = set()
    for line in lines:
        record = json.loads(line)
        assert set(record.keys()) == {
            "task_id",
            "probabilities",
            "answer",
            "confidence",
            "latency_ms",
            "expected",
            "error",
        }
        assert record["error"] is None
        seen_ids.add(record["task_id"])
    assert seen_ids == {t.id for t in tasks}


def test_run_erroring_tasks_logged_and_skipped(tmp_path: Path, caplog):
    tasks = _mixed_tasks(3)
    fail_ids = {tasks[0].id, tasks[-1].id}
    log_path = tmp_path / "raw.jsonl"
    runner = BenchmarkRunner(FakeAdapter(fail_ids=fail_ids), raw_log_path=log_path)
    with caplog.at_level(logging.WARNING):
        result = runner.run(tasks)

    # Failed tasks are excluded from metrics but counted in n_tasks.
    assert result["n_tasks"] == len(tasks)
    assert result["n_answered"] == len(tasks) - len(fail_ids)

    # Each failure logged at WARNING with the task id.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == len(fail_ids)
    for task_id in fail_ids:
        assert any(task_id in r.getMessage() for r in warnings)

    # Raw log still holds one line per task; failures carry the error.
    records = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(tasks)
    by_id = {r["task_id"]: r for r in records}
    for task_id in fail_ids:
        assert by_id[task_id]["error"] is not None
        assert by_id[task_id]["answer"] == -1
        assert by_id[task_id]["probabilities"] == []


def test_run_without_raw_log(tmp_path: Path):
    runner = BenchmarkRunner(FakeAdapter())
    result = runner.run(_mixed_tasks(2))
    assert result["n_answered"] == 4
    assert not list(tmp_path.iterdir())


def test_run_empty_task_list():
    runner = BenchmarkRunner(FakeAdapter())
    result = runner.run([])
    assert result["n_tasks"] == 0
    assert result["n_answered"] == 0
    assert math.isnan(result["latency"]["p50_ms"])
