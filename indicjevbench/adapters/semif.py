"""SemIf (Qwen3.5-4B) adapter for IndicJevBench.

SemIf is an open-source Jev-style decision model by Theodore Lee.
Source: https://github.com/TheoLeeCJ/openjev
Model:  Qwen/Qwen3.5-4B (Apache-2.0)

Install:
    pip install git+https://github.com/TheoLeeCJ/openjev.git

Usage:
    indicjevbench run --adapter semif --device cuda
    indicjevbench run --adapter semif --model Qwen/Qwen3.5-4B --device cuda
"""

from __future__ import annotations

import time
from .base import BenchAdapter, DecisionResult

_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


class SemIfAdapter(BenchAdapter):
    """Wrap semif_phase1 (openjev) for IndicJevBench evaluation.

    Maps IndicJevBench Task format → SemIf row format → DecisionResult.

    SemIf row format:
        {
            "id": str,
            "state": str,
            "question": str,           # instructions text
            "options": [{"id": str, "description": str}, ...]
        }

    SemIf assigns option IDs as "0", "1", ... so probabilities[i] = P(option i).
    For noul: options = [{"id": "false", ...}, {"id": "true", ...}].
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        revision: str = _MODEL_REVISION,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_tokens: int = 4096,
    ):
        try:
            from semif_phase1.core import load_causal_model
            from semif_phase1.direct import score as semif_score
        except ImportError:
            raise ImportError(
                "SemIfAdapter requires the semif package.\n"
                "Install: pip install git+https://github.com/TheoLeeCJ/openjev.git"
            )

        self._score = semif_score
        self._max_tokens = max_tokens
        self._model, self._tokenizer, self._metadata = load_causal_model(
            source=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
        )

    def _build_row(self, task) -> tuple[dict, list[str]]:
        """Convert Task to SemIf row. Returns (row_dict, option_ids_in_order)."""
        q = task.question
        q_type = q["type"]
        instructions = q["instructions"]
        raw_options = q.get("options") or []

        if q_type == "noul":
            options = [
                {"id": "false", "description": "No"},
                {"id": "true",  "description": "Yes"},
            ]
            option_order = ["false", "true"]
        else:
            options = [
                {"id": str(i), "description": opt}
                for i, opt in enumerate(raw_options)
            ]
            option_order = [str(i) for i in range(len(raw_options))]

        row = {
            "id": task.id,
            "state": task.state,
            "question": instructions,
            "options": options,
        }
        return row, option_order

    def decide(self, task) -> DecisionResult:
        row, option_order = self._build_row(task)

        t0 = time.perf_counter()
        result = self._score(
            self._model,
            self._tokenizer,
            row,
            self._metadata,
            max_tokens=self._max_tokens,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # SemIf returns option_ids in the order it scored them
        # and probabilities aligned to that order.
        scored_ids = result.get("option_ids", option_order)
        raw_probs = result.get("probabilities", [])

        # Re-align to our expected option order
        id_to_prob = dict(zip(scored_ids, raw_probs))
        probs = [id_to_prob.get(oid, 0.0) for oid in option_order]

        # Normalise in case of floating point drift
        total = sum(probs) or 1.0
        probs = [p / total for p in probs]

        q_type = task.q_type
        if q_type == "noul":
            p_true = probs[1]  # option_order = [false, true]
            answer = p_true > 0.5
            confidence = abs(2 * p_true - 1)
            expected = None
        else:
            argmax = int(max(range(len(probs)), key=lambda i: probs[i]))
            answer = argmax
            k = len(probs)
            confidence = (probs[argmax] - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
            expected = (
                sum((i + 1) * p for i, p in enumerate(probs))
                if q_type == "score" else None
            )

        return DecisionResult(
            task_id=task.id,
            probabilities=probs,
            answer=answer,
            confidence=confidence,
            latency_ms=latency_ms,
            expected=expected,
            tokens_used=result.get("input_tokens"),
        )
