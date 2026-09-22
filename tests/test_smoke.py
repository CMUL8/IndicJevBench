"""Smoke tests: package imports, version, and public API surface."""

from __future__ import annotations

import importlib

import indicjevbench

_PUBLIC_API = (
    "Task",
    "Question",
    "DecisionResult",
    "BenchAdapter",
    "BenchmarkRunner",
    "load_tasks",
    "compute_all",
    "breakdown",
    "indicjev_score",
    "BenchPaths",
    "get_api_key",
)


def test_version() -> None:
    assert indicjevbench.__version__ == "1.0.0"


def test_public_api_surface() -> None:
    for name in _PUBLIC_API:
        assert hasattr(indicjevbench, name), f"missing public API: {name}"
    assert set(indicjevbench.__all__) == set(_PUBLIC_API)


def test_import_from_new_paths() -> None:
    """Subpackages are importable from their contract locations."""
    for mod in (
        "indicjevbench.configs",
        "indicjevbench.configs.paths",
        "indicjevbench.configs.env",
        "indicjevbench.schemas",
        "indicjevbench.schemas.contracts",
        "indicjevbench.core",
        "indicjevbench.core.dataset",
        "indicjevbench.core.runner",
        "indicjevbench.runner",
        "indicjevbench.runner.cli",
        "indicjevbench.utils",
        "indicjevbench.utils.atomic",
        "indicjevbench.utils.logging",
    ):
        importlib.import_module(mod)

    from indicjevbench.configs import BenchPaths
    from indicjevbench.core import BenchmarkRunner
    from indicjevbench.runner.cli import build_adapter, main
    from indicjevbench.schemas import AnswerRecord, Question, QuestionType, Task
    from indicjevbench.utils import atomic_append_line, atomic_write_text

    assert callable(build_adapter) and callable(main)
    assert BenchPaths is indicjevbench.BenchPaths
    assert BenchmarkRunner is indicjevbench.BenchmarkRunner
    assert Question is indicjevbench.Question and Task is indicjevbench.Task
    assert AnswerRecord is not None and QuestionType is not None
    assert callable(atomic_append_line) and callable(atomic_write_text)


def test_adapters_import_clean_without_heavy_deps() -> None:
    """Adapters package must import without torch/transformers/openai installed."""
    for mod in (
        "indicjevbench.adapters",
        "indicjevbench.adapters.base",
        "indicjevbench.adapters.http",
        "indicjevbench.adapters.api_llm",
        "indicjevbench.adapters.qwen3_logprob",
        "indicjevbench.adapters.semif",
        "indicjevbench.adapters.laya",
        "indicjevbench.adapters.local",
    ):
        importlib.import_module(mod)


def test_bench_paths_default() -> None:
    paths = indicjevbench.BenchPaths.default()
    assert paths.datasets_dir.is_dir()
    assert paths.results_file("run_x").name == "run_x.json"
