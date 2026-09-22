"""Baseline 1: zero-shot Qwen3-4B-Instruct option log-prob scoring.

Uses the same [STATE]/[QUESTION]/[OPTIONS]/[ANSWER] prompt template as the
trained model so the comparison is fair (same prompt, no fine-tuning).

Usage:
    from indicjevbench.adapters.qwen3_logprob import Qwen3LogprobAdapter
    adapter = Qwen3LogprobAdapter(device="cuda")

Requires the ``baselines`` extra (torch, transformers):
``pip install 'indicjevbench[baselines]'``.
"""
# The optional baseline packages this adapter lazy-imports (heavyweight
# torch/transformers, semif_phase1 from git, or the private nirnaya
# checkpoint package) are not installed in the dev environment, so the
# unknown-type diagnostics for their runtime objects cannot be resolved
# here; they are relaxed for this file only, not package-wide.
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false

import logging
import math
import time
from typing import Any

from indicjevbench.adapters.base import BenchAdapter
from indicjevbench.schemas.contracts import DecisionResult, Question, Task

logger = logging.getLogger(__name__)

# Mirrors format.py template markers
_TEMPLATE = "[STATE]\n{state}\n[QUESTION]\n{instructions}\n[OPTIONS]\n{options}[ANSWER]"
_TEMPLATE_NOUL = "[STATE]\n{state}\n[QUESTION]\n{instructions}\n[ANSWER]"


class Qwen3LogprobAdapter(BenchAdapter):
    """Score each option by the sum of token log-probs from Qwen3-Instruct.

    For choice/score: build the full prompt up to [ANSWER], then score each
    option string by its token-level log-prob via a single forward pass that
    caches the shared prefix.

    For noul: score 'Yes' vs 'No' tokens directly.

    Args:
        model_id: Hugging Face model id of a Qwen3-Instruct checkpoint.
        device: Device map for model loading (e.g. ``"cuda"``, ``"cpu"``).
        local_files_only: Only use locally cached weights (no download).

    Raises:
        ImportError: If torch/transformers are missing; install with
            ``pip install 'indicjevbench[baselines]'``.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct",
        device: str = "cuda",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Qwen3LogprobAdapter requires torch and transformers. "
                "Install them with: pip install 'indicjevbench[baselines]'"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
        self._torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.device = device
        logger.info("Qwen3LogprobAdapter loaded model=%s device=%s", model_id, device)

    def _build_prefix(self, state: str, question: Question) -> str:
        """Build the shared prompt prefix up to the [ANSWER] marker.

        Args:
            state: Raw customer/user message.
            question: The typed question.

        Returns:
            The prompt prefix (noul questions use the no-options template).
        """
        options = question.options or []
        if question.type == "noul":
            return _TEMPLATE_NOUL.format(
                state=state,
                instructions=question.instructions,
            )
        opts_str = "".join(f"({i + 1}) {opt}\n" for i, opt in enumerate(options))
        return _TEMPLATE.format(
            state=state,
            instructions=question.instructions,
            options=opts_str,
        )

    def _score_continuations(self, prefix: str, continuations: list[str]) -> list[float]:
        """Return the log-prob of each continuation given the prefix.

        Args:
            prefix: Shared prompt prefix.
            continuations: Candidate continuation strings to score.

        Returns:
            Per-continuation summed token log-probabilities.
        """
        torch = self._torch
        enc = self.tokenizer
        with torch.inference_mode():
            return self._score_continuations_inner(torch, enc, prefix, continuations)

    def _score_continuations_inner(
        self,
        torch: Any,
        enc: Any,
        prefix: str,
        continuations: list[str],
    ) -> list[float]:
        """Token-level scoring of each continuation after a shared prefix.

        Args:
            torch: The torch module (injected for inference-mode scoping).
            enc: The tokenizer.
            prefix: Shared prompt prefix.
            continuations: Candidate continuation strings.

        Returns:
            Per-continuation summed token log-probabilities. Each
            continuation is scored by summing, over its tokens, the
            log-softmax probability assigned to that token at the
            corresponding position of a single forward pass.
        """
        prefix_ids = enc.encode(prefix, add_special_tokens=True, return_tensors="pt").to(
            self.device
        )

        log_probs = []
        for cont in continuations:
            cont_ids = enc.encode(cont, add_special_tokens=False, return_tensors="pt").to(
                self.device
            )
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
        """Numerically stable softmax over a list of log-probabilities.

        Args:
            log_probs: Candidate log-probabilities.

        Returns:
            The normalised probability distribution.
        """
        max_lp = max(log_probs)
        exp = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(exp)
        return [e / total for e in exp]

    def decide(self, task: Task) -> DecisionResult:
        """Score the question's options with token log-probs.

        Args:
            task: The benchmark task (state + typed question).

        Returns:
            For ``noul``: probabilities ``[P(false), P(true)]`` and a bool
            answer (``P(true) > 0.5``). For ``choice``/``score``: a softmax
            over option log-probs, the argmax index as answer, confidence
            ``(p_max - 1/K) / (1 - 1/K)``, and — for ``score`` — the
            expected level ``sum((i + 1) * p_i)``.
        """
        q = task.question
        prefix = self._build_prefix(task.state, q)
        t0 = time.perf_counter()
        if q.type == "noul":
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
            options = q.options or []
            continuations = [f" {opt}" for opt in options]
            log_probs = self._score_continuations(prefix, continuations)
            probs = self._softmax(log_probs)
            argmax = int(max(range(len(probs)), key=lambda i: probs[i]))
            k = len(probs)
            conf = (probs[argmax] - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
            expected = None
            if q.type == "score":
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
