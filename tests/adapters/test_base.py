"""Tests for the BenchAdapter abstract base and DecisionResult contract."""

import pytest

from indicjevbench.adapters.base import BenchAdapter, DecisionResult
from indicjevbench.schemas.contracts import Question, Task


def test_abc_not_instantiable() -> None:
    """BenchAdapter has abstract methods and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BenchAdapter()  # type: ignore[abstract]


def test_concrete_subclass_without_decide_not_instantiable() -> None:
    """A subclass that does not implement decide() stays abstract."""

    class Incomplete(BenchAdapter):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_concrete_subclass_instantiable() -> None:
    """A subclass implementing decide() instantiates and decides."""

    class Echo(BenchAdapter):
        def decide(self, task: Task) -> DecisionResult:
            return DecisionResult(
                task_id=task.id,
                probabilities=[1.0],
                answer=0,
                confidence=1.0,
                latency_ms=0.0,
            )

    task = Task(
        id="t1",
        family="intent",
        lang="hi-Deva",
        state="namaste",
        question=Question(type="choice", instructions="Pick one", options=("a",)),
        expected=0,
        split="test",
        source="unit",
        license="MIT",
        provenance={},
    )
    result = Echo().decide(task)
    assert result.task_id == "t1"
    assert result.probabilities == [1.0]


def test_decision_result_defaults() -> None:
    """Optional DecisionResult fields default to None."""
    result = DecisionResult(
        task_id="t1",
        probabilities=[0.5, 0.5],
        answer=0,
        confidence=0.0,
        latency_ms=12.5,
    )
    assert result.expected is None
    assert result.tokens_used is None
    assert result.error is None


def test_decision_result_fields() -> None:
    """All DecisionResult fields round-trip."""
    result = DecisionResult(
        task_id="t2",
        probabilities=[0.25, 0.75],
        answer=True,
        confidence=0.5,
        latency_ms=3.0,
        expected=1.75,
        tokens_used=42,
        error="boom",
    )
    assert result.answer is True
    assert result.expected == 1.75
    assert result.tokens_used == 42
    assert result.error == "boom"
