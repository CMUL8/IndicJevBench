"""Abstract base for all IndicJevBench model adapters."""

import abc

from indicjevbench.schemas.contracts import DecisionResult, Task


class BenchAdapter(abc.ABC):
    """Abstract base for all IndicJevBench model adapters."""

    @abc.abstractmethod
    def decide(self, task: Task) -> DecisionResult:
        """Execute a single decision on a Task.

        Args:
            task: The benchmark task (state + typed question).

        Returns:
            The adapter's decision result with probabilities, answer,
            confidence and latency.
        """
