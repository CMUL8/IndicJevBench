# Changelog

All notable changes to IndicJevBench are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-09-22

First open-source release.

### Added

- `src/` package layout (`indicjevbench`) installable with `uv sync`; `uv.lock` committed.
- Typed, frozen data contracts in `indicjevbench.schemas.contracts`:
  `Question`, `Task`, `DecisionResult`, `AnswerRecord`, with explicit
  `ValueError`/`TypeError` validation on dataset rows.
- `BenchAdapter` abstract base with lazy-loading concrete adapters:
  `HTTPAdapter` (`/v1/systemone` client), `LocalAdapter` (Nirṇaya checkpoint),
  `Qwen3LogprobAdapter`, `APILLMAdapter` (budget-capped), `SemIfAdapter`,
  `LayaAdapter`.
- `BenchmarkRunner` with per-task fault isolation and atomic raw-log JSONL
  output; `run_evaluation`/`build_adapter`/`close_adapter` CLI plumbing.
- `indicjevbench` console script (`indicjevbench run --adapter ...`).
- Full documentation set, all under `docs/` (benchmark design, datasheet with
  Known Issues, implementation notes, languages, architecture, adapter guide)
  plus README, CHANGELOG, and CONTRIBUTING at the repository root.
- Test suite (129 tests) covering schemas, dataset loading, runner, metrics,
  scoring, adapters (mock-transport, no network/GPU), and CLI dispatch.

### Changed

- Replaced flat package with src layout; `scripts/` reduced to thin argparse
  wrappers (eval_api, eval_qwen3, eval_laya, eval_openjev, package_datasets).
- All public APIs fully type-hinted with Google-style docstrings (Args /
  Returns / Raises); `[tool.pyright] typeCheckingMode = "strict"` and
  `[tool.ruff] line-length = 100` configured.
- Logging via `logging.getLogger(__name__)`; secrets never logged; `.env`
  gitignored with `.env.example` placeholders.
- Heavy optional dependencies (torch, transformers, openai, semif_phase1,
  nirnaya) imported lazily inside adapter constructors.

### Behavior

- **Preserved.** Metric computations (`compute_all`, `breakdown`), confidence
  formulas (choice/score `(p_max−1/K)/(1−1/K)`; noul `|2p−1|`), ECE (15 bins),
  runner result-dict shape, CLI flags, and the IndicJevScore weights
  0.35/0.25/0.20/0.20 are unchanged from the pre-refactor harness.
- `datasets/v1/*.jsonl` content is frozen (bit-for-bit) for reproducibility;
  see docs/DATASHEET.md → Known Issues for the `hinglish_lid` options serialization
  artifact (documented, not fixed; fix planned in v2).

[1.0.0]: https://github.com/cmul8/IndicJevBench/releases/tag/v1.0.0
