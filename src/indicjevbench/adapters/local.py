from __future__ import annotations

import time

from indicjevbench.adapters.base import BenchAdapter, DecisionResult
from indicjevbench.schemas.contracts import Task


class LocalAdapter(BenchAdapter):
    """Runs inference against a local Nirṇaya checkpoint."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        try:
            from nirnaya import infer as _infer
            from nirnaya.model.model import NirnayaModel
        except ImportError:
            raise ImportError("LocalAdapter requires the model package to be installed: pip install -e . from project root")
        self._infer = _infer
        self._model = NirnayaModel.load(checkpoint_dir, device=device)
        self._device = device

    def decide(self, task: Task) -> DecisionResult:
        q = task.question
        questions = [{"id": "q0", "type": q.type, "instructions": q.instructions,
                      **({"options": list(q.options)} if q.options else {})}]
        t0 = time.perf_counter()
        answers = self._infer.predict(self._model, task.state, questions, device=self._device)
        latency_ms = (time.perf_counter() - t0) * 1000
        a = answers[0]
        return DecisionResult(task_id=task.id, probabilities=a.probabilities,
                              answer=a.answer, confidence=a.confidence,
                              latency_ms=latency_ms, expected=getattr(a, "expected", None))
