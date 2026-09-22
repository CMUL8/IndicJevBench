"""Tests for APILLMAdapter parsing/normalisation via a fake client (no network)."""

import json
from typing import Any

import pytest

from indicjevbench.adapters.api_llm import APILLMAdapter
from indicjevbench.schemas.contracts import Question, Task


class _FakeUsage:
    """Minimal stand-in for openai Usage."""

    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    """Minimal stand-in for an openai chat completion response."""

    def __init__(self, content: str, total_tokens: int = 100) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(total_tokens)


class _FakeCompletions:
    """chat.completions endpoint that returns a canned JSON payload."""

    def __init__(self, content: str | None, fail: bool = False,
                 total_tokens: int = 100) -> None:
        self._content = content
        self._fail = fail
        self._total_tokens = total_tokens
        self.calls = 0

    def create(self, **kwargs: Any) -> _FakeCompletionResponse:
        self.calls += 1
        if self._fail:
            raise RuntimeError("api boom")
        assert self._content is not None
        return _FakeCompletionResponse(self._content, total_tokens=self._total_tokens)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class FakeOpenAIClient:
    """Drop-in replacement for openai.OpenAI used by APILLMAdapter."""

    def __init__(self, content: str | None, fail: bool = False,
                 total_tokens: int = 100) -> None:
        self._completions = _FakeCompletions(content, fail=fail, total_tokens=total_tokens)
        self.chat = _FakeChat(self._completions)


def _make_adapter(client: FakeOpenAIClient, max_budget: float = 20.0) -> APILLMAdapter:
    """Build an APILLMAdapter without running its network-heavy constructor."""
    adapter = APILLMAdapter.__new__(APILLMAdapter)
    adapter._model = "gpt-4o"
    adapter._max_budget = max_budget
    adapter._spent = 0.0
    adapter._cost_per_1m = {"gpt-4o": 5.0}
    adapter._client = client  # type: ignore[assignment]
    return adapter


def _task(q_type: str = "choice", options: tuple[str, ...] | None = ("a", "b", "c")) -> Task:
    return Task(
        id="task-1",
        family="intent",
        lang="hi-Deva",
        state="customer message text",
        question=Question(type=q_type, instructions="Decide", options=options),
        expected=0,
        split="test",
        source="unit",
        license="MIT",
        provenance={},
    )


_QUESTIONS = [{"id": "q0", "type": "choice", "instructions": "Decide",
               "options": ["a", "b", "c"]}]


# ---------------------------------------------------------------------------
# _parse_answers semantics
# ---------------------------------------------------------------------------


def test_choice_normalisation_and_argmax() -> None:
    """Unnormalised probabilities are normalised; answer defaults to argmax."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    parsed = adapter._parse_answers(
        [{"id": "q0", "probabilities": [2.0, 6.0, 2.0]}],
        _QUESTIONS,
    )
    assert len(parsed) == 1
    assert parsed[0]["probabilities"] == [0.2, 0.6, 0.2]
    assert parsed[0]["answer"] == 1
    # (0.6 - 1/3) / (1 - 1/3) = 0.4
    assert parsed[0]["confidence"] == pytest.approx(0.4)


def test_choice_bool_answer_coerced_to_int() -> None:
    """A bool answer from the model is coerced to an int index."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    parsed = adapter._parse_answers(
        [{"id": "q0", "probabilities": [0.5, 0.5], "answer": True}],
        _QUESTIONS,
    )
    assert parsed[0]["answer"] == 1
    assert isinstance(parsed[0]["answer"], int)


def test_choice_explicit_answer_kept() -> None:
    """An explicit int answer is preserved even when not the argmax."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    parsed = adapter._parse_answers(
        [{"id": "q0", "probabilities": [0.7, 0.2, 0.1], "answer": 2}],
        _QUESTIONS,
    )
    assert parsed[0]["answer"] == 2


def test_choice_missing_probabilities_uniform_fallback() -> None:
    """Missing probabilities fall back to uniform over the question options."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    parsed = adapter._parse_answers([{"id": "q0"}], _QUESTIONS)
    assert parsed[0]["probabilities"] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert parsed[0]["confidence"] == pytest.approx(0.0)


def test_score_expected_level() -> None:
    """Score questions get expected = sum((i + 1) * p_i) over normalised probs."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    questions = [{"id": "q0", "type": "score", "instructions": "Rate",
                  "options": ["1", "2", "3"]}]
    parsed = adapter._parse_answers(
        [{"id": "q0", "probabilities": [1.0, 2.0, 1.0]}],
        questions,
    )
    # normalised: [0.25, 0.5, 0.25] -> expected 0.25*1 + 0.5*2 + 0.25*3 = 2.0
    assert parsed[0]["expected"] == pytest.approx(2.0)
    assert parsed[0]["confidence"] == pytest.approx(0.25)


def test_noul_bool_answer_string() -> None:
    """Noul string answers 'true'/'yes' become bool True."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    questions = [{"id": "q0", "type": "noul", "instructions": "Is it?"}]
    parsed = adapter._parse_answers(
        [{"id": "q0", "probabilities": [0.2, 0.8], "answer": "true"}],
        questions,
    )
    assert parsed[0]["answer"] is True
    # |2 * 0.8 - 1| = 0.6
    assert parsed[0]["confidence"] == pytest.approx(0.6)


def test_noul_string_no_is_false() -> None:
    """Noul string answers other than true/yes become False."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    questions = [{"id": "q0", "type": "noul", "instructions": "Is it?"}]
    parsed = adapter._parse_answers(
        [{"id": "q0", "probabilities": [0.9, 0.1], "answer": "no"}],
        questions,
    )
    assert parsed[0]["answer"] is False
    assert parsed[0]["confidence"] == pytest.approx(0.8)


def test_noul_answer_defaults_to_threshold() -> None:
    """Missing noul answer defaults to P(true) > 0.5."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    questions = [{"id": "q0", "type": "noul", "instructions": "Is it?"}]
    high = adapter._parse_answers([{"id": "q0", "probabilities": [0.4, 0.6]}], questions)
    low = adapter._parse_answers([{"id": "q0", "probabilities": [0.8, 0.2]}], questions)
    assert high[0]["answer"] is True
    assert low[0]["answer"] is False


def test_noul_normalisation() -> None:
    """Noul probabilities are normalised before use."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    questions = [{"id": "q0", "type": "noul", "instructions": "Is it?"}]
    parsed = adapter._parse_answers(
        [{"id": "q0", "probabilities": [1.0, 3.0]}],
        questions,
    )
    assert parsed[0]["probabilities"] == pytest.approx([0.25, 0.75])
    assert parsed[0]["confidence"] == pytest.approx(0.5)


def test_unknown_question_id_skipped() -> None:
    """Answers whose id is not a known question are dropped."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    parsed = adapter._parse_answers(
        [{"id": "nope", "probabilities": [1.0]}],
        _QUESTIONS,
    )
    assert parsed == []


def test_multiple_answers_parsed_in_order() -> None:
    """Multiple valid answers are returned in input order."""
    adapter = _make_adapter(FakeOpenAIClient("{}"))
    questions = _QUESTIONS + [{"id": "q1", "type": "noul", "instructions": "Is it?"}]
    parsed = adapter._parse_answers(
        [
            {"id": "q1", "probabilities": [0.5, 0.5], "answer": False},
            {"id": "q0", "probabilities": [0.0, 1.0, 0.0], "answer": 1},
        ],
        questions,
    )
    assert [a["id"] for a in parsed] == ["q1", "q0"]


# ---------------------------------------------------------------------------
# decide() via fake client
# ---------------------------------------------------------------------------


def test_decide_happy_path() -> None:
    """decide() parses the model JSON into a DecisionResult and tracks spend."""
    payload = json.dumps({
        "answers": [{"id": "q0", "type": "choice",
                     "probabilities": [1.0, 3.0, 0.0], "answer": 1}],
    })
    client = FakeOpenAIClient(payload, total_tokens=1_000_000)
    adapter = _make_adapter(client)

    result = adapter.decide(_task())

    assert result.task_id == "task-1"
    assert result.probabilities == pytest.approx([0.25, 0.75, 0.0])
    assert result.answer == 1
    assert result.confidence == pytest.approx((0.75 - 1 / 3) / (1 - 1 / 3))
    assert result.error is None
    assert result.latency_ms >= 0.0
    # 5.0 USD per 1M tokens * 1M tokens
    assert adapter.budget_spent == pytest.approx(5.0)


def test_decide_error_path_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """API failures surface as DecisionResult with error set, no raise."""
    monkeypatch.setattr("indicjevbench.adapters.api_llm.time.sleep", lambda _s: None)
    client = FakeOpenAIClient(None, fail=True)
    adapter = _make_adapter(client)

    result = adapter.decide(_task())

    assert result.error == "api boom"
    assert result.answer == -1
    assert result.probabilities == []
    assert result.confidence == 0.0
    assert client._completions.calls == 3


def test_decide_empty_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response with no usable answers yields an 'empty parse' error."""
    monkeypatch.setattr("indicjevbench.adapters.api_llm.time.sleep", lambda _s: None)
    client = FakeOpenAIClient("{}")
    adapter = _make_adapter(client)

    result = adapter.decide(_task())

    assert result.error == "empty parse"
    assert result.answer == -1


def test_decide_budget_exhausted() -> None:
    """decide() raises RuntimeError once the budget is spent."""
    client = FakeOpenAIClient("{}")
    adapter = _make_adapter(client, max_budget=1.0)
    adapter._spent = 1.0

    with pytest.raises(RuntimeError, match="budget"):
        adapter.decide(_task())

    assert client._completions.calls == 0
