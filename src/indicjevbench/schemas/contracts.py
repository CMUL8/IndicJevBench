"""Core data contracts for IndicJevBench.

Frozen dataclasses describing a benchmark task: the typed question
(choice/score/noul), the task row from the JSONL datasets, the adapter's
decision result, and the raw-log answer record.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, cast


def _empty_criteria() -> dict[str, Any]:
    return {}

QuestionType = Literal["choice", "score", "noul"]
"""Supported question types."""

_VALID_QUESTION_TYPES: tuple[str, ...] = ("choice", "score", "noul")


@dataclass(frozen=True)
class Question:
    """A single typed question attached to a task state.

    Attributes:
        type: Question type — one of ``choice``, ``score``, ``noul``.
        instructions: Natural-language instruction shown to the model.
        options: Ordered answer options (``None`` for ``noul`` questions).
        criteria: Free-form rubric/criteria mapping (may be empty).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``type`` is unknown, ``instructions`` is empty, or
            ``options`` are missing (choice/score) or present (noul).
    """

    type: QuestionType
    instructions: str
    options: tuple[str, ...] | None = None
    criteria: dict[str, Any] = field(default_factory=_empty_criteria)

    def __post_init__(self) -> None:
        # Type-level validation happens in from_dict(); here we only enforce
        # invariants that hold regardless of construction path.
        if self.type not in _VALID_QUESTION_TYPES:
            raise ValueError(
                f"unknown question type {self.type!r}; expected one of {_VALID_QUESTION_TYPES}"
            )
        if not self.instructions.strip():
            raise ValueError("question instructions must be a non-empty str")
        if self.type in ("choice", "score") and not self.options:
            raise ValueError(f"question type {self.type!r} requires non-empty options")
        if self.type == "noul" and self.options is not None:
            raise ValueError("question type 'noul' must not carry options")
        if self.options is not None:
            # JSON rows carry lists; normalise to the declared tuple type.
            object.__setattr__(self, "options", tuple(self.options))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        """Build a Question from a raw JSONL question mapping.

        Args:
            data: Mapping with keys ``type``, ``instructions`` and optionally
                ``options`` and ``criteria``.

        Returns:
            The validated Question.

        Raises:
            ValueError: If required keys are missing, values have invalid
                types, or validation fails.
        """
        try:
            q_type = data["type"]
            instructions = data["instructions"]
        except KeyError as exc:
            raise ValueError(f"question missing required key: {exc.args[0]}") from exc
        raw_options: Any = data.get("options")
        if raw_options is not None and (
            not isinstance(raw_options, (list, tuple))
            or not all(isinstance(o, str) for o in cast(Any, raw_options))
        ):
            raise ValueError(
                f"question options must be a list of str or null, got {raw_options!r}"
            )
        raw_criteria: Any = data.get("criteria")
        if raw_criteria is None:
            criteria: dict[str, Any] = {}
        elif isinstance(raw_criteria, dict):
            criteria = cast(dict[str, Any], raw_criteria)
        else:
            raise ValueError(f"question criteria must be an object, got {raw_criteria!r}")
        return cls(
            type=q_type,
            instructions=instructions,
            options=cast(tuple[str, ...] | None, tuple(cast(Any, raw_options)) if raw_options is not None else None),
            criteria=criteria,
        )


@dataclass(frozen=True)
class Task:
    """One benchmark task: a state plus a typed question and gold label.

    Attributes:
        id: Unique task identifier (e.g. ``massive-kn-Knda-0003-q0``).
        family: Task family (e.g. ``intent``, ``lid``, ``urgency``).
        lang: BCP-47-ish language/script tag (e.g. ``hi-Deva``).
        state: The raw customer/user message the question refers to.
        question: The typed Question.
        expected: Gold label — option index for choice/score, 0/1 for noul.
        split: Dataset split name.
        source: Dataset source identifier.
        license: Data license string.
        provenance: Provenance metadata mapping.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If required string fields are empty, the question is
            invalid, or ``expected`` is an int outside the valid label range.
    """

    id: str
    family: str
    lang: str
    state: str
    question: Question
    expected: Any
    split: str
    source: str
    license: str
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        # Type-level validation happens in from_dict(); here we only enforce
        # invariants that hold regardless of construction path.
        for name in ("id", "family", "lang", "state", "split", "source", "license"):
            value = getattr(self, name)
            if not value:
                raise ValueError(f"task field {name!r} must be a non-empty str, got {value!r}")
        self._validate_expected()

    def _validate_expected(self) -> None:
        """Check that an int ``expected`` lies inside the label range.

        Choice/score questions take a gold option index in
        ``[0, len(options))``; noul questions take ``0`` or ``1``. Non-int
        expected values (rare legacy rows) are left untouched.

        Raises:
            ValueError: If ``expected`` is an int outside the valid range.
        """
        expected = self.expected
        if not isinstance(expected, int) or isinstance(expected, bool):
            return
        if self.question.type == "noul":
            if expected not in (0, 1):
                raise ValueError(
                    f"noul expected must be 0 or 1, got {expected!r} (task {self.id!r})"
                )
        elif self.question.options is not None and not 0 <= expected < len(self.question.options):
            raise ValueError(
                f"expected index {expected} out of range for "
                f"{len(self.question.options)} options (task {self.id!r})"
            )

    @property
    def q_type(self) -> str:
        """Question type string (``choice`` | ``score`` | ``noul``)."""
        return self.question.type

    @property
    def options(self) -> tuple[str, ...] | None:
        """Ordered answer options, or ``None`` for noul questions."""
        return self.question.options

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        """Build a Task from a raw JSONL row.

        Args:
            data: Mapping with the task fields (see dataclass attributes).

        Returns:
            The validated Task.

        Raises:
            ValueError: If required keys are missing, values have invalid
                types, or validation fails.
        """
        required = ("id", "family", "lang", "state", "question", "expected",
                    "split", "source", "license", "provenance")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"task row missing required keys: {missing}")
        for name in ("id", "family", "lang", "state", "split", "source", "license"):
            if not isinstance(data[name], str) or not data[name]:
                raise TypeError(
                    f"task field {name!r} must be a non-empty str, got {data[name]!r}"
                )
        if not isinstance(data["provenance"], dict):
            raise TypeError(
                f"task provenance must be an object, got {type(data['provenance']).__name__}"
            )
        return cls(
            id=data["id"],
            family=data["family"],
            lang=data["lang"],
            state=data["state"],
            question=Question.from_dict(data["question"]),
            expected=data["expected"],
            split=data["split"],
            source=data["source"],
            license=data["license"],
            provenance=cast(dict[str, Any], data["provenance"]),
        )


@dataclass
class DecisionResult:
    """Outcome of a single adapter decision on a Task.

    Attributes:
        task_id: Task identifier (mirrors Task.id).
        probabilities: Full probability distribution over options.
        answer: Predicted option index (int) or bool (noul); -1 on failure.
        confidence: Calibrated confidence in ``[0, 1]``.
        latency_ms: Wall-clock decision latency in milliseconds.
        expected: Score-type predicted mean level (1-indexed), if applicable.
        tokens_used: Input tokens consumed, if the adapter reports them.
        error: Error message when the decision failed, else ``None``.
    """

    task_id: str
    probabilities: list[float]
    answer: int | bool
    confidence: float
    latency_ms: float
    expected: float | None = None  # score type: predicted mean level
    tokens_used: int | None = None
    error: str | None = None


@dataclass
class AnswerRecord:
    """One raw-log JSONL line emitted by the benchmark runner.

    Attributes:
        task_id: Task identifier.
        probabilities: Probability distribution returned by the adapter.
        answer: Predicted answer (option index or bool); -1 on failure.
        confidence: Confidence in ``[0, 1]``.
        latency_ms: Decision latency in milliseconds.
        expected: Score-type predicted mean level, if present.
        error: Error message when the decision failed, else ``None``.
    """

    task_id: str
    probabilities: list[float]
    answer: int | bool
    confidence: float
    latency_ms: float
    expected: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable mapping for one raw-log line.

        Returns:
            Dict with the record fields in stable key order.
        """
        return {
            "task_id": self.task_id,
            "probabilities": self.probabilities,
            "answer": self.answer,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "expected": self.expected,
            "error": self.error,
        }

    @classmethod
    def from_result(cls, result: DecisionResult) -> "AnswerRecord":
        """Build an AnswerRecord from a DecisionResult.

        Args:
            result: The adapter decision result.

        Returns:
            The corresponding raw-log record.
        """
        return cls(
            task_id=result.task_id,
            probabilities=result.probabilities,
            answer=result.answer,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            expected=result.expected,
            error=result.error,
        )
