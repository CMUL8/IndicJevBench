"""Tests for the indicjevbench CLI: adapter dispatch, --help, unknown adapters."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from typing import Any, cast

import pytest

from indicjevbench.adapters.base import BenchAdapter, DecisionResult
from indicjevbench.runner.cli import build_adapter, main
from indicjevbench.schemas.contracts import Task

_DISPATCH_TARGETS: dict[str, str] = {
    "http": "indicjevbench.adapters.http.HTTPAdapter",
    "local": "indicjevbench.adapters.local.LocalAdapter",
    "qwen3": "indicjevbench.adapters.qwen3_logprob.Qwen3LogprobAdapter",
    "api": "indicjevbench.adapters.api_llm.APILLMAdapter",
    "semif": "indicjevbench.adapters.semif.SemIfAdapter",
}


class _FakeAdapter(BenchAdapter):
    """Test double recording constructor arguments (stands in for heavy adapters)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def decide(self, task: Task) -> DecisionResult:  # pragma: no cover - never called
        raise NotImplementedError


def _namespace(**overrides: Any) -> argparse.Namespace:
    """Build a CLI namespace with the run-subcommand defaults."""
    values: dict[str, Any] = {
        "adapter": "http",
        "endpoint": "http://localhost:8000",
        "checkpoint": "checkpoints/best",
        "model": "my-model",
        "device": "cuda",
        "budget": 20.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture(autouse=True)
def _patch_adapters(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Replace every heavy adapter class with the recording fake."""
    for target in _DISPATCH_TARGETS.values():
        monkeypatch.setattr(target, _FakeAdapter)
    yield


@pytest.mark.parametrize("adapter_name", sorted(_DISPATCH_TARGETS))
def test_build_adapter_dispatch_returns_adapter_type(adapter_name: str) -> None:
    """build_adapter returns the patched adapter class for every --adapter value."""
    result = build_adapter(_namespace(adapter=adapter_name))
    assert isinstance(result, _FakeAdapter)


def test_build_adapter_passes_endpoint_and_model() -> None:
    """http dispatch forwards --endpoint positionally and --model by keyword."""
    fake = cast(
        _FakeAdapter, build_adapter(_namespace(adapter="http", endpoint="http://x:1", model="m1"))
    )
    assert fake.args == ("http://x:1",)
    assert fake.kwargs == {"model": "m1"}


def test_build_adapter_passes_budget_to_api() -> None:
    """api dispatch forwards --model and --budget."""
    fake = cast(_FakeAdapter, build_adapter(_namespace(adapter="api", model="gpt-x", budget=3.5)))
    assert fake.kwargs == {"model": "gpt-x", "max_budget_usd": 3.5}


def test_build_adapter_unknown_raises_value_error() -> None:
    """An adapter outside the known set raises ValueError (defense in depth)."""
    with pytest.raises(ValueError, match="Unknown adapter"):
        build_adapter(_namespace(adapter="bogus"))


def test_run_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`indicjevbench run --help` prints usage and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0
    assert "--adapter" in capsys.readouterr().out


def test_top_level_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """`indicjevbench --help` prints usage and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "run" in capsys.readouterr().out
