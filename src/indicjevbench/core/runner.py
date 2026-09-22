"""Benchmark runner: executes an adapter over tasks and scores results."""

import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from indicjevbench.adapters.base import BenchAdapter, DecisionResult
from indicjevbench.metrics import breakdown, compute_all
from indicjevbench.schemas.contracts import AnswerRecord, Task
from indicjevbench.scoring import indicjev_score
from indicjevbench.utils.atomic import atomic_append_line


def percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile of ``values`` using the legacy index rule.

    Args:
        values: Latency (or other) measurements.
        p: Percentile in ``(0, 100]``.

    Returns:
        The percentile value, or NaN if ``values`` is empty.
    """
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]


class BenchmarkRunner:
    """Run a BenchAdapter over a list of tasks and compute metrics.

    Attributes are injected at construction; ``run`` is the only mutating
    operation and may be called multiple times with the same adapter.

    Args:
        adapter: The BenchAdapter to evaluate.
        raw_log_path: Optional path for the raw per-task JSONL log.
        task_name: Display name used for the progress bar.
        logger: Optional logger; defaults to the module logger.
    """

    def __init__(
        self,
        adapter: BenchAdapter,
        raw_log_path: Path | None = None,
        task_name: str = "",
        logger: logging.Logger | None = None,
    ) -> None:
        self._adapter = adapter
        self._raw_log_path = Path(raw_log_path) if raw_log_path is not None else None
        self._task_name = task_name
        self._logger = logger if logger is not None else logging.getLogger(__name__)

    def run(self, tasks: list[Task]) -> dict[str, Any]:
        """Run the adapter over all tasks and return the results dict.

        Adapter exceptions are caught per task and recorded as failed
        decisions (empty probabilities, ``answer=-1``) so one bad task does
        not abort the run. Tasks that fail or return no probabilities are
        excluded from metric computation.

        Args:
            tasks: List of tasks to evaluate.

        Returns:
            Dict with keys ``n_tasks``, ``n_answered``, ``metrics``
            (``all``/``choice``/``score``/``noul``/``by_lang``/``by_source``),
            ``latency`` (``p50_ms``/``p95_ms``) and ``score``
            (``indicjev_score`` + ``axes``).
        """
        results_raw: list[tuple[DecisionResult, Task]] = []
        latencies: list[float] = []

        for task in tqdm(tasks, desc=self._task_name[:28] or "running", unit="task"):
            try:
                result = self._adapter.decide(task)
            except Exception as e:  # noqa: BLE001 — per-task fault isolation is intentional
                self._logger.warning("task %s failed: %s", task.id, e)
                result = DecisionResult(
                    task_id=task.id,
                    probabilities=[],
                    answer=-1,
                    confidence=0.0,
                    latency_ms=0.0,
                    error=str(e),
                )
            latencies.append(result.latency_ms)

            if self._raw_log_path is not None:
                record = AnswerRecord.from_result(result)
                atomic_append_line(
                    self._raw_log_path, json.dumps(record.to_dict(), ensure_ascii=False)
                )

            results_raw.append((result, task))

        # Build answer/example lists for metrics
        answers: list[dict[str, Any]] = []
        examples: list[dict[str, Any]] = []
        for result, task in results_raw:
            if result.error or not result.probabilities:
                continue
            answers.append(
                {
                    "id": "q0",
                    "type": task.q_type,
                    "probabilities": result.probabilities,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    **({"expected": result.expected} if result.expected is not None else {}),
                }
            )
            examples.append(
                {
                    "id": task.id,
                    "lang": task.lang,
                    "source": task.source,
                    "questions": [{"qid": "q0", "type": task.q_type, "label": task.expected}],
                }
            )

        metrics = compute_all(answers, examples)
        metrics["by_lang"] = breakdown(answers, examples, "lang")
        metrics["by_source"] = breakdown(answers, examples, "source")

        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)
        overall: dict[str, Any] = metrics.get("all", {})
        score = indicjev_score(
            accuracy=overall.get("accuracy", 0.0),
            ece=overall.get("ece", 0.5),
            brier=overall.get("brier", 1.0),
            p50_ms=p50,
        )

        return {
            "n_tasks": len(tasks),
            "n_answered": len(answers),
            "metrics": metrics,
            "latency": {"p50_ms": round(p50, 1), "p95_ms": round(p95, 1)},
            "score": score,
        }
