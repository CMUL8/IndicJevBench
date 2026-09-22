"""Abstract base for all IndicJevBench model adapters.

Defines :class:`BenchAdapter`, the interface every model backend must
implement, and re-exports :class:`DecisionResult` for convenience.
"""

import abc

from indicjevbench.schemas.contracts import DecisionResult, Task

__all__ = ["BenchAdapter", "DecisionResult"]


class BenchAdapter(abc.ABC):
    """Abstract base for all IndicJevBench model adapters.

    An adapter turns a single :class:`~indicjevbench.schemas.contracts.Task`
    into a :class:`~indicjevbench.schemas.contracts.DecisionResult`. Concrete
    adapters live in sibling modules (``http``, ``api_llm``, ``local``,
    ``qwen3_logprob``, ``semif``, ``laya``) and load their heavy optional
    dependencies lazily in their constructors.

    Implementations must be pure decision producers: no dataset loading, no
    metric computation, no logging of secrets.
    """

    @abc.abstractmethod
    def decide(self, task: Task) -> DecisionResult:
        """Execute a single decision on a Task.

        Args:
            task: The benchmark task (state + typed question).

        Returns:
            The adapter's decision result with probabilities, answer,
            confidence and latency. On an unrecoverable per-task failure the
            result should carry ``answer=-1`` and a non-``None`` ``error``
            rather than raising, so the runner can record it and continue.

        Raises:
            RuntimeError: Only for unrecoverable, run-level failures (e.g.
                budget exhausted) where continuing makes no sense.
        """
