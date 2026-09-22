"""Baseline 1: zero-shot Qwen3-4B-Instruct option log-prob scoring.

Uses the same [STATE]/[QUESTION]/[OPTIONS]/[ANSWER] prompt template as the
trained model so the comparison is fair (same prompt, no fine-tuning).

Usage:
    from indicjevbench.adapters.qwen3_logprob import Qwen3LogprobAdapter
    adapter = Qwen3LogprobAdapter(device="cuda")
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch

from .base import BenchAdapter, DecisionResult

if TYPE_CHECKING:
    pass

# Mirrors format.py template markers
_TEMPLATE = "[STATE]\n{state}\n[QUESTION]\n{instructions}\n[OPTIONS]\n{options}[ANSWER]"
_TEMPLATE_NOUL = "[STATE]\n{state}\n[QUESTION]\n{instructions}\n[ANSWER]"


class Qwen3LogprobAdapter(BenchAdapter):
    """Score each option by the sum of token log-probs from Qwen3-Instruct.

    For choice/score: build the full prompt up to [ANSWER], then score each
    option string by its token-level log-prob via a single forward pass that
    caches the shared prefix.

    For noul: score 'Yes' vs 'No' tokens directly.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct",
        device: str = "cuda",
        local_files_only: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.device = device

    def _build_prefix(self, state: str, question: dict) -> str:
        options = question.get("options") or []
        if question["type"] == "noul":
            return _TEMPLATE_NOUL.format(
                state=state,
                instructions=question["instructions"],
            )
        opts_str = "".join(f"({i+1}) {opt}\n" for i, opt in enumerate(options))
        return _TEMPLATE.format(
            state=state,
            instructions=question["instructions"],
            options=opts_str,
        )

    @torch.inference_mode()
    def _score_continuations(self, prefix: str, continuations: list[str]) -> list[float]:
        """Return log-prob of each continuation given the prefix."""
        enc = self.tokenizer
        prefix_ids = enc.encode(prefix, add_special_tokens=True, return_tensors="pt").to(self.device)

        log_probs = []
        for cont in continuations:
            cont_ids = enc.encode(cont, add_special_tokens=False, return_tensors="pt").to(self.device)
            full_ids = torch.cat([prefix_ids, cont_ids], dim=1)
            out = self.model(full_ids)
            logits = out.logits[0]  # (seq_len, vocab)
            # Tokens to score: cont_ids positions in full sequence
            start = prefix_ids.shape[1] - 1
            lp = 0.0
            for i, tok_id in enumerate(cont_ids[0]):
                lp += torch.log_softmax(logits[start + i], dim=-1)[tok_id].item()
            log_probs.append(lp)

        return log_probs

    def _softmax(self, log_probs: list[float]) -> list[float]:
        max_lp = max(log_probs)
        exp = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(exp)
        return [e / total for e in exp]

    def decide(self, task) -> DecisionResult:
        q = task.question
        prefix = self._build_prefix(task.state, q)
        t0 = time.perf_counter()
        if q["type"] == "noul":
            log_probs = self._score_continuations(prefix, [" No", " Yes"])
            probs = self._softmax(log_probs)  # [P(false), P(true)]
            p_true = probs[1]
            answer_bool = p_true > 0.5
            conf = abs(2 * p_true - 1)
            latency_ms = (time.perf_counter() - t0) * 1000
            return DecisionResult(
                task_id=task.id,
                probabilities=probs,
                answer=answer_bool,
                confidence=conf,
                latency_ms=latency_ms,
            )
        else:
            options = q.get("options") or []
            continuations = [f" {opt}" for opt in options]
            log_probs = self._score_continuations(prefix, continuations)
            probs = self._softmax(log_probs)
            argmax = int(max(range(len(probs)), key=lambda i: probs[i]))
            k = len(probs)
            conf = (probs[argmax] - 1/k) / (1 - 1/k) if k > 1 else 1.0
            expected = None
            if q["type"] == "score":
                expected = sum((i + 1) * p for i, p in enumerate(probs))
            latency_ms = (time.perf_counter() - t0) * 1000
            return DecisionResult(
                task_id=task.id,
                probabilities=probs,
                answer=argmax,
                confidence=conf,
                latency_ms=latency_ms,
                expected=expected,
            )
