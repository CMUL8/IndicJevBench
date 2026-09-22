"""Baseline 3: Generative API LLM in JSON output mode.

Sends each (state, question) to an OpenAI-compatible endpoint and parses
JSON answer probabilities. Budget-capped at ``max_budget_usd``.

Usage:
    from indicjevbench.adapters.api_llm import APILLMAdapter
    adapter = APILLMAdapter(model="gpt-4o", max_budget_usd=20.0)
    # Pass --max-examples 500 to the CLI for the stratified sample.

Requires the ``openai`` package — install it with the ``baselines`` extra:
``pip install 'indicjevbench[baselines]'``.
"""

import json
import logging
import os
import time
from typing import Any

from indicjevbench.adapters.base import BenchAdapter
from indicjevbench.schemas.contracts import DecisionResult, Task

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3

_SYSTEM_PROMPT = """You are a structured decision assistant. Given a customer message and typed questions, return calibrated probability distributions.

For each question respond with:
- "choice" or "score": a "probabilities" array (one float per option, summing to 1), and "answer" (integer index of the most likely option)
- "noul": "probabilities" as [P(false), P(true)], and "answer" as true or false

Return ONLY a valid JSON object: {"answers": [{"id": "...", "type": "...", "probabilities": [...], "answer": ...}, ...]}"""


class APILLMAdapter(BenchAdapter):
    """OpenAI-compatible chat-completions adapter with a budget cap.

    The adapter asks the model for a single JSON object describing the
    probability distribution over each question's options, parses and
    normalises it (see :meth:`_parse_answers` for the exact semantics), and
    tracks cumulative token cost against ``max_budget_usd``. When the budget
    is exhausted, :meth:`decide` raises ``RuntimeError``; per-task API/parse
    failures are returned as ``DecisionResult`` records with ``error`` set
    after the configured retries are exhausted.

    Args:
        model: OpenAI model name (also used for cost estimation).
        base_url: Optional OpenAI-compatible endpoint override.
        api_key: API key; falls back to ``OPENAI_API_KEY`` then
            ``OPENROUTER_API_KEY`` from the environment. Never logged.
        max_budget_usd: Maximum cumulative estimated spend in USD.

    Raises:
        ImportError: If the ``openai`` package is missing; install with
            ``pip install 'indicjevbench[baselines]'``.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        max_budget_usd: float = 20.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "APILLMAdapter requires the 'openai' package. "
                "Install it with: pip install 'indicjevbench[baselines]'"
            ) from exc

        if max_budget_usd <= 0:
            raise ValueError(f"max_budget_usd must be positive, got {max_budget_usd!r}")

        self._model = model
        self._max_budget = max_budget_usd
        self._spent = 0.0
        # Cost per 1M tokens — rough estimates for budget tracking
        self._cost_per_1m = {
            "gpt-4o": 5.0,
            "gpt-4o-mini": 0.60,
            "gpt-4-turbo": 10.0,
        }
        self._client: Any = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
        )
        logger.info("APILLMAdapter created (model=%s, max_budget_usd=%.2f)", model, max_budget_usd)

    @property
    def budget_spent(self) -> float:
        """Estimated cumulative spend in USD tracked so far."""
        return self._spent

    def decide(self, task: Task) -> DecisionResult:
        """Request a JSON-mode decision for a single task.

        Args:
            task: The benchmark task (state + typed question).

        Returns:
            The parsed decision result. On API/parse failure after all
            retries (or an empty parse) the result carries ``answer=-1``,
            empty probabilities, ``confidence=0.0`` and an ``error`` message.

        Raises:
            RuntimeError: If the configured budget is already exhausted.
        """
        self._check_budget()
        questions = self._build_questions(task)
        user_msg = self._build_user_message(task.state, questions)

        t0 = time.perf_counter()
        parsed, error = self._request_with_retries(user_msg, questions)
        latency_ms = (time.perf_counter() - t0) * 1000

        if error is not None:
            return DecisionResult(
                task_id=task.id,
                probabilities=[],
                answer=-1,
                confidence=0.0,
                latency_ms=latency_ms,
                error=error,
            )
        if not parsed:
            return DecisionResult(
                task_id=task.id,
                probabilities=[],
                answer=-1,
                confidence=0.0,
                latency_ms=latency_ms,
                error="empty parse",
            )
        a = parsed[0]
        return DecisionResult(
            task_id=task.id,
            probabilities=a["probabilities"],
            answer=a["answer"],
            confidence=a["confidence"],
            latency_ms=latency_ms,
            expected=a.get("expected"),
            tokens_used=None,
        )

    def _check_budget(self) -> None:
        """Raise ``RuntimeError`` if the configured budget is exhausted.

        Raises:
            RuntimeError: If estimated spend has reached ``max_budget_usd``.
        """
        if self._spent >= self._max_budget:
            raise RuntimeError(
                f"APILLMAdapter: budget ${self._max_budget} exhausted (spent ${self._spent:.3f})"
            )

    def _build_questions(self, task: Task) -> list[dict[str, Any]]:
        """Convert the task's typed question into the wire payload shape.

        Args:
            task: The benchmark task.

        Returns:
            A single-question list in the shape used by both the user
            message and the parser (``id``/``type``/``instructions`` plus
            optional ``options``).
        """
        q = task.question
        return [
            {
                "id": "q0",
                "type": q.type,
                "instructions": q.instructions,
                **({"options": list(q.options)} if q.options else {}),
            }
        ]

    def _build_user_message(self, state: str, questions: list[dict[str, Any]]) -> str:
        """Render the user message asking for the JSON answer object.

        Args:
            state: Raw customer/user message.
            questions: Question dicts as produced by :meth:`_build_questions`.

        Returns:
            The user-message string sent to the chat-completions endpoint.
        """
        q_parts: list[str] = []
        for q in questions:
            opts = q.get("options")
            opts_str = f"\n  Options: {opts}" if opts else ""
            q_parts.append(
                f'  id="{q["id"]}", type="{q["type"]}", '
                f'instructions="{q["instructions"]}"{opts_str}'
            )
        qs = "\n".join(q_parts)
        return f'Customer message: "{state}"\n\nQuestions:\n{qs}'

    def _estimate_cost(self, n_tokens: int) -> float:
        """Estimate USD cost for ``n_tokens`` at the model's rough rate.

        Args:
            n_tokens: Number of tokens billed.

        Returns:
            Estimated cost in USD (rate per 1M tokens from the internal cost
            table, defaulting to the ``gpt-4o`` rate for unknown models).
        """
        rate = self._cost_per_1m.get(self._model, 5.0)
        return rate * n_tokens / 1_000_000

    def _create_completion(self, user_msg: str) -> Any:
        """Issue a single chat-completions request with fixed sampling params.

        Args:
            user_msg: The rendered user message.

        Returns:
            The raw ``openai`` chat-completions response object.
        """
        return self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
        )

    def _request_with_retries(
        self, user_msg: str, questions: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Call the API and parse the response, retrying failures.

        Args:
            user_msg: The rendered user message.
            questions: Question dicts used to validate/parse the answers.

        Returns:
            ``(parsed, None)`` on success, ``(None, error)`` after the final
            failed attempt. API spend is accumulated as usage is reported.
        """
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = self._create_completion(user_msg)
                text = resp.choices[0].message.content or "{}"
                if resp.usage:
                    self._spent += self._estimate_cost(resp.usage.total_tokens)
                data = json.loads(text)
                return self._parse_answers(data.get("answers", []), questions), None
            except Exception as e:  # noqa: BLE001 - retry semantics: any API/parse failure
                if attempt < _MAX_ATTEMPTS - 1:
                    delay = 2**attempt
                    logger.warning(
                        "API LLM request/parse failed (attempt %d/%d): %s; retrying in %ds",
                        attempt + 1,
                        _MAX_ATTEMPTS,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "API LLM request/parse failed after %d attempts: %s", _MAX_ATTEMPTS, e
                    )
                    return None, str(e)
        return None, "unreachable"  # pragma: no cover - loop always returns

    def _parse_answers(
        self, raw: list[dict[str, Any]], questions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Parse and normalise raw answer entries against known questions.

        Semantics (preserved exactly):

        * Entries whose ``id`` does not match a known question are skipped.
        * ``probabilities`` is normalised to sum to 1; an empty/missing list
          falls back to a uniform distribution over the question's options
          (2-way for ``noul``).
        * ``noul``: ``answer`` may be a bool or a string (``"true"`` /
          ``"yes"`` → ``True``); confidence is ``|2 * P(true) - 1|``.
        * ``choice``/``score``: ``answer`` may be an int or bool (coerced to
          ``int``); confidence is ``(p_max - 1/K) / (1 - 1/K)`` for ``K > 1``
          options, else ``1.0``; ``score`` additionally gets the expected
          level ``sum((i + 1) * p_i)``.

        Args:
            raw: The ``answers`` list decoded from the model's JSON.
            questions: Question dicts (with ``id``) the answers refer to.

        Returns:
            The parsed answer dicts, in input order minus skipped entries.
        """
        q_by_id = {q["id"]: q for q in questions}
        answers: list[dict[str, Any]] = []
        for item in raw:
            parsed = self._parse_one(item, q_by_id)
            if parsed is not None:
                answers.append(parsed)
        return answers

    def _parse_one(
        self, item: dict[str, Any], q_by_id: dict[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Parse a single raw answer entry, or ``None`` if its id is unknown.

        Args:
            item: One raw answer entry from the model JSON.
            q_by_id: Known questions keyed by id.

        Returns:
            The parsed answer dict, or ``None`` when skipped.
        """
        qid = item.get("id", "")
        q = q_by_id.get(qid)
        if q is None:
            return None
        probs = self._normalise_probs(item.get("probabilities", []), q)
        if q["type"] == "noul":
            return self._parse_noul(qid, item, probs)
        return self._parse_categorical(qid, q["type"], item, probs)

    def _normalise_probs(self, probs: list[float], q: dict[str, Any]) -> list[float]:
        """Normalise a raw probability list to sum to 1.

        Args:
            probs: Raw probabilities from the model (may be empty).
            q: The question dict (``options`` used for the uniform fallback).

        Returns:
            The normalised distribution; uniform over options when ``probs``
            is empty (2-way when the question has no options, i.e. ``noul``).
        """
        if not probs:
            k = len(q.get("options") or []) or 2
            probs = [1.0 / k] * k
        total = sum(probs) or 1.0
        return [p / total for p in probs]

    def _parse_noul(self, qid: str, item: dict[str, Any], probs: list[float]) -> dict[str, Any]:
        """Build the parsed answer for a ``noul`` question.

        Args:
            qid: The question id.
            item: The raw answer entry (``answer`` may be bool or string).
            probs: Normalised ``[P(false), P(true)]`` (or a single entry).

        Returns:
            Answer dict with bool ``answer`` and confidence ``|2p - 1|``.
        """
        p_true = probs[1] if len(probs) >= 2 else probs[0]
        answer: Any = item.get("answer", p_true > 0.5)
        if isinstance(answer, str):
            answer = answer.lower() in ("true", "yes")
        conf = abs(2 * p_true - 1)
        return {
            "id": qid,
            "type": "noul",
            "probabilities": probs,
            "answer": bool(answer),
            "confidence": conf,
        }

    def _parse_categorical(
        self, qid: str, q_type: str, item: dict[str, Any], probs: list[float]
    ) -> dict[str, Any]:
        """Build the parsed answer for a ``choice``/``score`` question.

        Args:
            qid: The question id.
            q_type: ``"choice"`` or ``"score"``.
            item: The raw answer entry (``answer`` may be int or bool).
            probs: Normalised option probabilities.

        Returns:
            Answer dict with int ``answer`` and confidence
            ``(p_max - 1/K) / (1 - 1/K)``; ``score`` answers additionally
            carry the expected level ``sum((i + 1) * p_i)``.
        """
        argmax = int(max(range(len(probs)), key=lambda i: probs[i]))
        answer: Any = item.get("answer", argmax)
        if isinstance(answer, bool):
            answer = int(answer)
        k = len(probs)
        conf = (probs[argmax] - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
        out: dict[str, Any] = {
            "id": qid,
            "type": q_type,
            "probabilities": probs,
            "answer": int(answer),
            "confidence": conf,
        }
        if q_type == "score":
            out["expected"] = sum((i + 1) * p for i, p in enumerate(probs))
        return out
