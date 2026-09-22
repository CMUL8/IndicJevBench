"""Tests for indicjevbench/schemas/contracts.py — validation and compat properties."""

from __future__ import annotations

import dataclasses

import pytest

from indicjevbench.schemas.contracts import (
    AnswerRecord,
    DecisionResult,
    Question,
    Task,
)


def _question_data(**overrides: object) -> dict:
    data: dict = {
        "type": "choice",
        "instructions": "Pick the intent.",
        "options": ["a", "b", "c"],
        "criteria": {},
    }
    data.update(overrides)
    return data


def _task_data(**overrides: object) -> dict:
    data: dict = {
        "id": "t-001",
        "family": "intent",
        "lang": "hi-Deva",
        "state": "message text",
        "question": _question_data(),
        "expected": 1,
        "split": "v1",
        "source": "massive",
        "license": "CC BY 4.0",
        "provenance": {"origin": "test"},
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Question validation
# ---------------------------------------------------------------------------


def test_question_valid_choice():
    q = Question.from_dict(_question_data())
    assert q.type == "choice"
    assert q.options == ("a", "b", "c")


def test_question_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown question type"):
        Question.from_dict(_question_data(type="rank"))


def test_question_rejects_empty_instructions():
    with pytest.raises(ValueError, match="instructions"):
        Question.from_dict(_question_data(instructions="   "))


def test_question_choice_requires_options():
    with pytest.raises(ValueError, match="requires non-empty options"):
        Question.from_dict(_question_data(options=None))


def test_question_score_requires_options():
    with pytest.raises(ValueError, match="requires non-empty options"):
        Question.from_dict(_question_data(type="score", options=[]))


def test_question_noul_forbids_options():
    with pytest.raises(ValueError, match="must not carry options"):
        Question.from_dict(_question_data(type="noul", options=["yes", "no"]))


def test_question_noul_ok_without_options():
    q = Question.from_dict(_question_data(type="noul", options=None))
    assert q.options is None


def test_question_missing_key():
    data = _question_data()
    del data["instructions"]
    with pytest.raises(ValueError, match="instructions"):
        Question.from_dict(data)


def test_question_options_normalised_to_tuple():
    q = Question.from_dict(_question_data())
    assert isinstance(q.options, tuple)


# ---------------------------------------------------------------------------
# Task validation
# ---------------------------------------------------------------------------


def test_task_valid_roundtrip():
    task = Task.from_dict(_task_data())
    assert task.id == "t-001"
    assert task.q_type == "choice"


def test_task_compat_properties():
    task = Task.from_dict(_task_data())
    assert task.q_type == task.question.type
    assert task.options == task.question.options


def test_task_rejects_empty_string_fields():
    for field in ("id", "family", "lang", "state", "split", "source", "license"):
        with pytest.raises((TypeError, ValueError), match=field):
            Task.from_dict(_task_data(**{field: ""}))


def test_task_rejects_missing_keys():
    data = _task_data()
    del data["expected"]
    with pytest.raises(ValueError, match="missing required keys"):
        Task.from_dict(data)


def test_task_rejects_non_str_field():
    with pytest.raises(TypeError, match="id"):
        Task.from_dict(_task_data(id=42))


def test_task_expected_out_of_range_choice():
    with pytest.raises(ValueError, match="out of range"):
        Task.from_dict(_task_data(expected=3))  # only 3 options: 0..2


def test_task_expected_in_range_choice():
    task = Task.from_dict(_task_data(expected=2))
    assert task.expected == 2


def test_task_expected_out_of_range_noul():
    question = _question_data(type="noul", options=None)
    with pytest.raises(ValueError, match="0 or 1"):
        Task.from_dict(_task_data(question=question, expected=2))


def test_task_expected_non_int_allowed():
    # Legacy rows may carry non-int expected values; validation leaves them.
    task = Task.from_dict(_task_data(expected="unknown"))
    assert task.expected == "unknown"


def test_task_is_frozen():
    task = Task.from_dict(_task_data())
    with pytest.raises(dataclasses.FrozenInstanceError):
        task.id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DecisionResult / AnswerRecord
# ---------------------------------------------------------------------------


def _decision(**overrides: object) -> DecisionResult:
    data: dict = {
        "task_id": "t-001",
        "probabilities": [0.1, 0.8, 0.1],
        "answer": 1,
        "confidence": 0.7,
        "latency_ms": 12.5,
    }
    data.update(overrides)
    return DecisionResult(**data)  # type: ignore[arg-type]


def test_decision_result_defaults():
    d = _decision()
    assert d.expected is None
    assert d.tokens_used is None
    assert d.error is None


def test_answer_record_from_result_roundtrip():
    d = _decision(expected=3.0, tokens_used=42, error="boom")
    record = AnswerRecord.from_result(d)
    assert record.task_id == "t-001"
    assert record.expected == 3.0
    assert record.error == "boom"
    assert record.to_dict() == {
        "task_id": "t-001",
        "probabilities": [0.1, 0.8, 0.1],
        "answer": 1,
        "confidence": 0.7,
        "latency_ms": 12.5,
        "expected": 3.0,
        "error": "boom",
    }
