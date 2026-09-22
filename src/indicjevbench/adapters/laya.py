"""Baseline 2: Laya-multilingual intent classifier.

Laya is an encoder-based intent model. It only supports intent_massive
(choice, fixed 60-label head). For score/noul questions it raises
NotImplementedError — the harness skips Laya for those tasks.

Usage:
    from indicjevbench.adapters.laya import LayaAdapter
    adapter = LayaAdapter()                     # tries to load locally
    # or with published numbers fallback:
    adapter = LayaAdapter(published_numbers_path="laya_published.json")
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from indicjevbench.adapters.base import BenchAdapter, DecisionResult
from indicjevbench.schemas.contracts import Task

# Laya's 60 MASSIVE intent labels in their canonical order.
# Used to map predicted class index → option index in our question format.
LAYA_INTENTS: list[str] = []  # populated on first load from model config


class LayaAdapter(BenchAdapter):
    """Wrap convaiinnovations/laya-multilingual as a BenchAdapter.

    Only supports 'choice' questions with the MASSIVE 60-intent label set.
    Raises NotImplementedError for score/noul questions.

    If published_numbers_path is provided, decide() raises NotImplementedError
    always — use the path to report results labeled "as published".
    """

    def __init__(
        self,
        model_id: str = "convaiinnovations/laya-multilingual",
        published_numbers_path: str | None = None,
        device: str = "cuda",
        local_files_only: bool = True,
    ):
        self._published = None
        if published_numbers_path:
            self._published = json.loads(
                Path(published_numbers_path).read_text(encoding="utf-8")
            )
            return  # skip model loading

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, local_files_only=local_files_only
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
            self.model.eval()
            self.model = self.model.to(device)
            self._device = device
            self._torch = torch

            # Build intent→index map from model config
            id2label = self.model.config.id2label  # {int: str}
            global LAYA_INTENTS
            LAYA_INTENTS = [id2label[i] for i in range(len(id2label))]

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Laya ({model_id}): {e}\n"
                "Timebox: 90 minutes. If loading fails, use published_numbers_path "
                "to report Laya numbers labeled 'as published by convaiinnovations'."
            ) from e

    def decide(self, task: Task) -> DecisionResult:
        if self._published is not None:
            raise NotImplementedError(
                "LayaAdapter is in published-numbers mode. "
                "Results are reported as 'as published by convaiinnovations'."
            )

        q = task.question
        if q.type != "choice":
            raise NotImplementedError(
                f"LayaAdapter only supports 'choice' questions, got '{q.type}'. "
                "Laya has a fixed intent classification head and cannot answer "
                "score or noul questions."
            )

        t0 = time.perf_counter()
        inputs = self.tokenizer(task.state, return_tensors="pt",
                                truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs_all = self._torch.softmax(logits, dim=-1).cpu().tolist()

        # Map Laya's label order to this question's option order
        options = q.options or []
        option_lower = [o.lower().strip() for o in options]
        probs = []
        for opt in option_lower:
            # Find matching Laya label (exact or first partial match)
            p = 0.0
            for j, laya_label in enumerate(LAYA_INTENTS):
                if laya_label.lower() == opt or opt in laya_label.lower():
                    p = probs_all[j]
                    break
            probs.append(p)

        # Renormalise (some options may not exist in Laya's label set)
        total = sum(probs) or 1.0
        probs = [p / total for p in probs]
        argmax = int(max(range(len(probs)), key=lambda i: probs[i]))
        k = len(probs)
        conf = (probs[argmax] - 1/k) / (1 - 1/k) if k > 1 else 1.0
        latency_ms = (time.perf_counter() - t0) * 1000

        return DecisionResult(
            task_id=task.id,
            probabilities=probs,
            answer=argmax,
            confidence=conf,
            latency_ms=latency_ms,
        )
