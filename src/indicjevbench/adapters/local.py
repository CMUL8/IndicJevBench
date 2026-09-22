"""Local Nirṇaya checkpoint adapter.

Runs inference against a locally trained Nirṇaya checkpoint via the
``nirnaya`` model package (private dependency; install it from the model
project root with ``pip install -e .``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from indicjevbench.adapters.base import BenchAdapter
from indicjevbench.schemas.contracts import DecisionResult, Task

logger = logging.getLogger(__name__)


class LocalAdapter(BenchAdapter):
    """Runs inference against a local Nirṇaya checkpoint.

    Args:
        checkpoint_dir: Path to the trained checkpoint directory.
        device: Device to run inference on.

    Raises:
        ImportError: If the ``nirnaya`` model package is not installed.
        RuntimeError: If checkpoint loading fails.
    """

    def __init__(self, checkpoint_dir: str, device: str = "cuda") -> None:
        try:
            from nirnaya import infer as _infer
            from nirnaya.model.model import NirnayaModel
        except ImportError as exc:
            raise ImportError(
                "LocalAdapter requires the nirnaya model package. "
                "Install it from the model project root: pip install -e ."
            ) from exc
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be a non-empty str")
        self._infer = _infer
        self._model = NirnayaModel.load(checkpoint_dir, device=device)
        self._device = device
        logger.info("LocalAdapter loaded checkpoint=%s device=%s", checkpoint_dir, device)

    def decide(self, task: Task) -> DecisionResult:
        """Run the checkpoint's predictor on a single task.

        Args:
            task: The benchmark task (state + typed question).

        Returns:
            The predictor's first answer mapped to a DecisionResult,
            including the expected level when the answer carries one.
        """
        q = task.question
        questions: list[dict[str, Any]] = [
            {
                "id": "q0",
                "type": q.type,
                "instructions": q.instructions,
                **({"options": list(q.options)} if q.options else {}),
            }
        ]
        t0 = time.perf_counter()
        answers = self._infer.predict(self._model, task.state, questions, device=self._device)
        latency_ms = (time.perf_counter() - t0) * 1000
        a = answers[0]
        return DecisionResult(task_id=task.id, probabilities=a.probabilities,
                              answer=a.answer, confidence=a.confidence,
                              latency_ms=latency_ms, expected=getattr(a, "expected", None))
