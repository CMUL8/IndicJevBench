from __future__ import annotations
import json
import time
from pathlib import Path
from tqdm import tqdm
from .tasks import Task, load_tasks
from .adapters.base import BenchAdapter, DecisionResult
from .metrics import compute_all, breakdown
from .scoring import indicjev_score

def run(
    tasks: list[Task],
    adapter: BenchAdapter,
    raw_log_path: Path | None = None,
    task_name: str = "",
) -> dict:
    """Run adapter over all tasks. Returns results dict with metrics."""
    results_raw = []
    latencies = []

    for task in tqdm(tasks, desc=task_name[:28] or "running", unit="task"):
        try:
            result = adapter.decide(task)
        except Exception as e:
            result = DecisionResult(task_id=task.id, probabilities=[], answer=-1,
                                    confidence=0.0, latency_ms=0.0, error=str(e))
        latencies.append(result.latency_ms)

        if raw_log_path:
            raw_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "task_id": result.task_id,
                    "probabilities": result.probabilities,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                    "expected": result.expected,
                    "error": result.error,
                }) + "\n")

        results_raw.append((result, task))

    # Build answer/example lists for metrics
    answers = []
    examples = []
    for result, task in results_raw:
        if result.error or not result.probabilities:
            continue
        answers.append({
            "id": "q0",
            "type": task.q_type,
            "probabilities": result.probabilities,
            "answer": result.answer,
            "confidence": result.confidence,
            **({"expected": result.expected} if result.expected is not None else {}),
        })
        examples.append({
            "id": task.id,
            "lang": task.lang,
            "source": task.source,
            "questions": [{"qid": "q0", "type": task.q_type,
                           "label": task.expected}],
        })

    metrics = compute_all(answers, examples)
    metrics["by_lang"] = breakdown(answers, examples, "lang")
    metrics["by_source"] = breakdown(answers, examples, "source")

    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    overall = metrics.get("all", {})
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

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, int(len(s) * p / 100) - 1)
    return s[idx]
