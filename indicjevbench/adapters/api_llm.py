"""Baseline 3: Generative API LLM in JSON output mode.

Sends each (state, question) to an OpenAI-compatible endpoint and parses
JSON answer probabilities. Budget-capped at max_budget_usd.

Usage:
    from indicjevbench.adapters.api_llm import APILLMAdapter
    adapter = APILLMAdapter(model="gpt-4o", max_budget_usd=20.0)
    # Pass --max-examples 500 to the CLI for the stratified sample.
"""

from __future__ import annotations

import json
import math
import os
import time

from .base import BenchAdapter, DecisionResult

_SYSTEM_PROMPT = """You are a structured decision assistant. Given a customer message and typed questions, return calibrated probability distributions.

For each question respond with:
- "choice" or "score": a "probabilities" array (one float per option, summing to 1), and "answer" (integer index of the most likely option)
- "noul": "probabilities" as [P(false), P(true)], and "answer" as true or false

Return ONLY a valid JSON object: {"answers": [{"id": "...", "type": "...", "probabilities": [...], "answer": ...}, ...]}"""


class APILLMAdapter(BenchAdapter):
    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        max_budget_usd: float = 20.0,
    ):
        from openai import OpenAI

        self._model = model
        self._max_budget = max_budget_usd
        self._spent = 0.0
        # Cost per 1M tokens — rough estimates for budget tracking
        self._cost_per_1m = {
            "gpt-4o": 5.0,
            "gpt-4o-mini": 0.60,
            "gpt-4-turbo": 10.0,
        }
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"),
        )

    def _build_user_message(self, state: str, questions: list[dict]) -> str:
        q_parts = []
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
        rate = self._cost_per_1m.get(self._model, 5.0)
        return rate * n_tokens / 1_000_000

    def decide(self, task) -> DecisionResult:
        if self._spent >= self._max_budget:
            raise RuntimeError(
                f"APILLMAdapter: budget ${self._max_budget} exhausted "
                f"(spent ${self._spent:.3f})"
            )

        q = task.question
        questions = [{"id": "q0", "type": q["type"], "instructions": q["instructions"],
                      **({"options": q["options"]} if q.get("options") else {})}]
        user_msg = self._build_user_message(task.state, questions)

        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=1024,
                )
                text = resp.choices[0].message.content or "{}"
                if resp.usage:
                    self._spent += self._estimate_cost(resp.usage.total_tokens)

                data = json.loads(text)
                raw_answers = data.get("answers", [])
                parsed = self._parse_answers(raw_answers, questions)
                break

            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return DecisionResult(task_id=task.id, probabilities=[], answer=-1,
                                          confidence=0.0, latency_ms=latency_ms,
                                          error=str(e))

        latency_ms = (time.perf_counter() - t0) * 1000
        if not parsed:
            return DecisionResult(task_id=task.id, probabilities=[], answer=-1,
                                  confidence=0.0, latency_ms=latency_ms,
                                  error="empty parse")
        a = parsed[0]
        tokens = None
        return DecisionResult(
            task_id=task.id,
            probabilities=a["probabilities"],
            answer=a["answer"],
            confidence=a["confidence"],
            latency_ms=latency_ms,
            expected=a.get("expected"),
            tokens_used=tokens,
        )

    def _parse_answers(self, raw: list[dict], questions: list[dict]) -> list[dict]:
        q_by_id = {q["id"]: q for q in questions}
        answers = []
        for item in raw:
            qid = item.get("id", "")
            q = q_by_id.get(qid)
            if q is None:
                continue
            probs = item.get("probabilities", [])
            if not probs:
                k = len(q.get("options") or []) or 2
                probs = [1.0 / k] * k
            # Normalise
            total = sum(probs) or 1.0
            probs = [p / total for p in probs]

            q_type = q["type"]
            if q_type == "noul":
                p_true = probs[1] if len(probs) >= 2 else probs[0]
                answer = item.get("answer", p_true > 0.5)
                if isinstance(answer, str):
                    answer = answer.lower() in ("true", "yes")
                conf = abs(2 * p_true - 1)
                out: dict = {"id": qid, "type": "noul", "probabilities": probs,
                             "answer": bool(answer), "confidence": conf}
            else:
                argmax = int(max(range(len(probs)), key=lambda i: probs[i]))
                answer = item.get("answer", argmax)
                if isinstance(answer, bool):
                    answer = int(answer)
                k = len(probs)
                conf = (probs[argmax] - 1/k) / (1 - 1/k) if k > 1 else 1.0
                out = {"id": qid, "type": q_type, "probabilities": probs,
                       "answer": int(answer), "confidence": conf}
                if q_type == "score":
                    out["expected"] = sum((i + 1) * p for i, p in enumerate(probs))
            answers.append(out)

        return answers

    @property
    def budget_spent(self) -> float:
        return self._spent
