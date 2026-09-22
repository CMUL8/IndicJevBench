# Contributing to IndicJevBench

Thanks for helping improve IndicJevBench. This document covers setup, tests,
code style, and the pull-request checklist.

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev          # package + pytest, ruff, pyright
uv sync --extra baselines    # additionally: torch, transformers, openai
```

The package installs from `src/indicjevbench/` in editable mode; `uv run`
commands work from the repo root.

## Running Tests

```bash
uv run pytest -q
```

The full suite runs without GPU or network access (HTTP and API adapters are
tested with `httpx.MockTransport` and fake clients). Tests under `tests/`
mirror the package layout under `src/indicjevbench/`.

Optional checks:

```bash
uv run ruff check src tests
uv run pyright
```

## Adding an Adapter

1. Read [docs/ADAPTERS.md](docs/ADAPTERS.md) — it defines the `BenchAdapter`
   contract, the `/v1/systemone` wire format, and per-adapter notes.
2. Create `src/indicjevbench/adapters/my_adapter.py` subclassing
   `BenchAdapter` and implementing `decide(self, task: Task) -> DecisionResult`.
3. Import heavy optional dependencies lazily inside the constructor, and raise
   `ImportError` naming the extra to install (e.g. `pip install 'indicjevbench[baselines]'`).
4. Add a branch in `build_adapter()` in `src/indicjevbench/runner/cli.py` and
   wire any new CLI flags there.
5. Add tests under `tests/adapters/` — parsing/normalisation logic must be
   tested with fakes/mocks, no real network calls.

## Code Style

- Python >= 3.12 syntax; **full type hints** on every parameter, return value,
  and dataclass field.
- Google-style docstrings (`Args:` / `Returns:` / `Raises:`) on all public
  classes, methods, and functions.
- Frozen dataclasses (`@dataclass(frozen=True)`) for config and data
  contracts; validate input with explicit `ValueError`/`TypeError` — never
  `assert`.
- `logging.getLogger(__name__)` for logging; prints only in CLI entrypoints.
  Never log secrets, API keys, or `.env` values.
- `from __future__ import annotations` is unnecessary on 3.12 — omit it.
- Ruff line length 100; pyright runs in `strict` mode.

## PR Checklist

- [ ] `uv run pytest -q` passes (129 tests).
- [ ] `uv run ruff check src tests` clean; `uv run pyright` clean for touched modules.
- [ ] New public APIs have type hints and Google-style docstrings.
- [ ] New optional dependencies are lazy-imported with clear `ImportError` messages.
- [ ] No secrets, real `.env` files, or API keys in the diff (`.env` is gitignored).
- [ ] `datasets/v1/*.jsonl` untouched — data is frozen for reproducibility.
- [ ] Docs updated if behavior, flags, or the adapter set changed.

## Reporting Issues

Open an issue with the adapter, dataset, model, and a minimal reproduction
(`indicjevbench run` invocation plus the failing task id). Do not paste raw
API keys or raw-log lines containing customer PII.
