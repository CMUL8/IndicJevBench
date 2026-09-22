from __future__ import annotations
import abc
from dataclasses import dataclass, field

@dataclass
class DecisionResult:
    task_id: str
    probabilities: list[float]
    answer: int | bool
    confidence: float
    latency_ms: float
    expected: float | None = None   # score type: predicted mean level
    tokens_used: int | None = None
    error: str | None = None

class BenchAdapter(abc.ABC):
    """Abstract base for all IndicJevBench model adapters."""

    @abc.abstractmethod
    def decide(self, task) -> DecisionResult:
        """Execute a single decision on a Task. Return DecisionResult."""
