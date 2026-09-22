"""IndicJevBench — benchmark harness for Indic structured decision AI.

Public API (re-exported here, importable without heavy optional deps):

Data contracts: :class:`Task`, :class:`Question`, :class:`DecisionResult`.
Core: :class:`BenchmarkRunner`, :func:`load_tasks`, :class:`BenchAdapter`.
Metrics: :func:`compute_all`, :func:`breakdown`, :func:`indicjev_score`.
Config: :class:`BenchPaths`, :func:`get_api_key`.

Heavy adapter dependencies (torch/transformers/openai/semif) are imported
lazily inside adapter constructors, never at package import time.
"""

from indicjevbench.adapters.base import BenchAdapter
from indicjevbench.configs.env import get_api_key
from indicjevbench.configs.paths import BenchPaths
from indicjevbench.core.dataset import load_tasks
from indicjevbench.core.runner import BenchmarkRunner
from indicjevbench.metrics import breakdown, compute_all
from indicjevbench.schemas.contracts import DecisionResult, Question, Task
from indicjevbench.scoring import indicjev_score

__version__ = "1.0.0"

__all__ = [
    "BenchAdapter",
    "BenchPaths",
    "BenchmarkRunner",
    "DecisionResult",
    "Question",
    "Task",
    "breakdown",
    "compute_all",
    "get_api_key",
    "indicjev_score",
    "load_tasks",
]
