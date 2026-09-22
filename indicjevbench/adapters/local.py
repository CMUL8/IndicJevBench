from __future__ import annotations
import time
from .base import BenchAdapter, DecisionResult

class LocalAdapter(BenchAdapter):
    """Runs inference against a local Nirṇaya checkpoint."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        try:
            from nirnaya.model.model import NirnayaModel
            from nirnaya import infer as _infer
        except ImportError:
            raise ImportError("LocalAdapter requires the model package to be installed: pip install -e . from project root")
        self._infer = _infer
        self._model = NirnayaModel.load(checkpoint_dir, device=device)
        self._device = device

    def decide(self, task) -> DecisionResult:
        q = task.question
        questions = [{"id": "q0", "type": q["type"], "instructions": q["instructions"],
                      **({"options": q["options"]} if q.get("options") else {})}]
        t0 = time.perf_counter()
        answers = self._infer.predict(self._model, task.state, questions, device=self._device)
        latency_ms = (time.perf_counter() - t0) * 1000
        a = answers[0]
        return DecisionResult(task_id=task.id, probabilities=a.probabilities,
                              answer=a.answer, confidence=a.confidence,
                              latency_ms=latency_ms, expected=getattr(a, "expected", None))
