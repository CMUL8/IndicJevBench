"""Tests for HTTPAdapter against an httpx MockTransport (no network)."""

import json

import httpx
import pytest

from indicjevbench.adapters.http import HTTPAdapter
from indicjevbench.schemas.contracts import Question, Task

_BASE_URL = "http://testserver"


def _task(q_type: str = "choice", options: tuple[str, ...] | None = ("alpha", "beta")) -> Task:
    """Build a minimal Task for adapter calls."""
    return Task(
        id="task-1",
        family="intent",
        lang="hi-Deva",
        state="customer message text",
        question=Question(type=q_type, instructions="Pick one", options=options),
        expected=0,
        split="test",
        source="unit",
        license="MIT",
        provenance={},
    )


def _answer_body(
    probabilities: list[float] | None = None,
    answer: int | bool = 1,
    confidence: float = 0.8,
    expected: float | None = None,
    input_tokens: int | None = 17,
) -> dict:
    """Build a /v1/systemone-style response body."""
    ans: dict = {"probabilities": probabilities or [0.3, 0.7], "answer": answer}
    if confidence is not None:
        ans["confidence"] = confidence
    if expected is not None:
        ans["expected"] = expected
    body: dict = {"answers": [ans]}
    if input_tokens is not None:
        body["usage"] = {"input_tokens": input_tokens}
    return body


def test_decision_round_trip() -> None:
    """A 200 response maps onto DecisionResult fields verbatim."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_answer_body(expected=1.7))

    adapter = HTTPAdapter(_BASE_URL, model="m1", transport=httpx.MockTransport(handler))
    try:
        result = adapter.decide(_task())
    finally:
        adapter.close()

    assert result.task_id == "task-1"
    assert result.probabilities == [0.3, 0.7]
    assert result.answer == 1
    assert result.confidence == 0.8
    assert result.expected == 1.7
    assert result.tokens_used == 17
    assert result.latency_ms >= 0.0

    # Endpoint contract: POST /v1/systemone with model/state/questions.
    assert seen["path"] == "/v1/systemone"
    payload = seen["payload"]
    assert payload["model"] == "m1"
    assert payload["state"] == "customer message text"
    assert payload["questions"] == [
        {"id": "q0", "type": "choice", "instructions": "Pick one", "options": ["alpha", "beta"]}
    ]


def test_noul_payload_omits_options() -> None:
    """Noul questions are sent without an options key."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_answer_body(probabilities=[0.4, 0.6], answer=True))

    adapter = HTTPAdapter(_BASE_URL, transport=httpx.MockTransport(handler))
    try:
        result = adapter.decide(_task(q_type="noul", options=None))
    finally:
        adapter.close()

    assert "options" not in seen["payload"]["questions"][0]
    assert seen["payload"]["questions"][0]["type"] == "noul"
    assert result.answer is True


def test_retry_on_500_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 500 response is retried; a later 200 wins."""
    monkeypatch.setattr("indicjevbench.adapters.http.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=_answer_body())

    adapter = HTTPAdapter(_BASE_URL, transport=httpx.MockTransport(handler))
    try:
        result = adapter.decide(_task())
    finally:
        adapter.close()

    assert calls["n"] == 2
    assert result.answer == 1
    assert result.error is None


def test_error_path_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistent 500s raise HTTPStatusError after exactly 3 attempts."""
    monkeypatch.setattr("indicjevbench.adapters.http.time.sleep", lambda _s: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    adapter = HTTPAdapter(_BASE_URL, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            adapter.decide(_task())
    finally:
        adapter.close()

    assert calls["n"] == 3


def test_missing_answers_raises() -> None:
    """A 200 body without 'answers' is a clear validation error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {"input_tokens": 1}})

    adapter = HTTPAdapter(_BASE_URL, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError, match="answers"):
            adapter.decide(_task())
    finally:
        adapter.close()


def test_constructor_validation() -> None:
    """Empty base_url and non-positive timeout are rejected."""
    with pytest.raises(ValueError):
        HTTPAdapter("")
    with pytest.raises(ValueError):
        HTTPAdapter(_BASE_URL, timeout=0.0)


def test_optional_fields_default() -> None:
    """confidence defaults to 0.0 and usage is optional in the response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answers": [{"probabilities": [1.0], "answer": 0}],
            },
        )

    adapter = HTTPAdapter(_BASE_URL, transport=httpx.MockTransport(handler))
    try:
        result = adapter.decide(_task())
    finally:
        adapter.close()

    assert result.confidence == 0.0
    assert result.tokens_used is None
    assert result.expected is None
