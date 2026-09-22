# IndicJevBench

**The first open benchmark for Indic-language structured decision AI.**

IndicJevBench tests whether a model can return calibrated probability distributions for typed questions (intent classification, urgency scoring, escalation detection, agent routing) given a customer message — in Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, and Hinglish.

Models are evaluated through a single interface: the System One-style `/v1/systemone` decision endpoint, which answers **choice**, **score**, and **noul** (yes/no) questions with full probability distributions.

---

## Why IndicJevBench

Most NLP benchmarks test reading comprehension or text generation. IndicJevBench tests **decision quality under uncertainty** for Indian enterprise use cases: payments, e-commerce, customer support, agent routing.

Existing models score poorly on Indic structured decisions because:
- They were never trained to return calibrated probability distributions
- They do not understand Hinglish (code-mixed Roman Hindi)
- Their training data has no UPI/payments or Indian e-commerce context

IndicJevBench makes this gap visible and measurable with a single number: the **IndicJevScore**.

---

## Quick Start

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

### Install

```bash
uv sync                      # core harness
uv sync --extra dev          # + pytest, ruff, pyright
uv sync --extra baselines    # + torch, transformers, openai (local/API baselines)
```

### Run the test suite

```bash
uv run pytest -q
```

### Run against a live HTTP endpoint

Point the harness at any server implementing the `/v1/systemone` contract (see [docs/ADAPTERS.md](docs/ADAPTERS.md)):

```bash
uv run indicjevbench run \
  --adapter http \
  --endpoint http://localhost:8000 \
  --model my-model
```

With no `--tasks` flag, all JSONL files in `datasets/v1/` are evaluated.

### Run the Qwen3 zero-shot baseline (GPU required)

```bash
uv run indicjevbench run \
  --adapter qwen3 \
  --device cuda
```

### Run against a local Nirṇaya checkpoint

```bash
uv run indicjevbench run \
  --adapter local \
  --checkpoint /path/to/checkpoints/best \
  --device cuda
```

### Run against an API LLM (OpenAI-compatible)

```bash
OPENAI_API_KEY=sk-... uv run indicjevbench run \
  --adapter api \
  --model gpt-4o \
  --budget 20.0 \
  --max-examples 500
```

The API key falls back to `OPENROUTER_API_KEY` when `OPENAI_API_KEY` is unset (see `.env.example`). Never commit real keys.

### Convenience scripts

Thin wrappers with baseline-friendly defaults live in `scripts/`:

```bash
python scripts/eval_api.py --model openai/gpt-4o-mini --max-items 200
python scripts/eval_qwen3.py --device cuda
python scripts/eval_laya.py --max-items 1000
python scripts/eval_openjev.py --device auto
```

### Package datasets first

Before running, build the task JSONL files from the frozen upstream test split:

```bash
python scripts/package_datasets.py
```

This reads the upstream pipeline output (`data/final/test.jsonl`, outside this repo) and writes per-task files to `datasets/v1/` plus `datasets/manifest.json`. The `datasets/v1/*.jsonl` content is **frozen** between releases — only re-run this when intentionally rebuilding the datasets.

---

## Scoring Methodology

IndicJevBench reports a composite **IndicJevScore** (0–100) as a weighted geometric mean of four axes:

| Axis | Weight | Formula | Best | Worst |
|------|--------|---------|------|-------|
| Intelligence | 35% | accuracy × 100 | 100 (acc=1.0) | 0 (acc=0.0) |
| Calibration | 25% | mean(1−ECE, 1−Brier/2) × 100 | 100 (ECE=0, Brier=0) | 0 (ECE=1, Brier=2) |
| Speed | 20% | log-scale, 50ms=100, 5000ms=0 | 100 (p50≤50ms) | 0 (p50≥5000ms) |
| Cost | 20% | log-scale, $0.01/1k=100, $10/1k=0; local=100 | 100 (free/local) | 0 ($10+/1k) |

Composite = exp(0.35·ln(intelligence) + 0.25·ln(calibration) + 0.20·ln(speed) + 0.20·ln(cost)); each axis is clipped to [1e-6, 100] before the log so a zero axis drives the composite toward 0. Implemented in `src/indicjevbench/scoring.py`.

### Per-task metrics

Every task also reports: accuracy, macro-F1, NLL, Brier score, ECE (15 bins), MAE of expected level (score tasks), and automatable share (fraction of decisions auto-approvable at <5% error). Metrics are pure functions in `src/indicjevbench/metrics.py` and are additionally broken down by language and source dataset.

---

## Datasets (v1)

Packaged counts from `datasets/manifest.json` (frozen 2026-09-22; 69,402 items total, CC BY 4.0 throughout):

| Dataset name | Family | Languages | Items (v1) | License | Source |
|---|---|---|---|---|---|
| `intent_massive` | intent | hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym, en-Latn | 57,922 | CC BY 4.0 | MASSIVE test split |
| `fintech_banking77` | intent | hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym, en-Latn | 7,816 | CC BY 4.0 | Banking77 + NLLB translation |
| `hinglish_lid` | lid | hi-Latn | 3,220 | CC BY 4.0 | COMI-LINGUA |
| `synthetic_enterprise` | escalation / routing / urgency | hi-Latn, hi-Deva | 444 | CC BY 4.0 | cmul8 synthetic (Qwen3+DeepSeek) |

See [DATASHEET.md](DATASHEET.md) for construction details and known issues, and [docs/LANGUAGES.md](docs/LANGUAGES.md) for the supported-language table.

---

## Adapters

Every model backend implements the `BenchAdapter` interface (one `decide(task) -> DecisionResult` method). See [docs/ADAPTERS.md](docs/ADAPTERS.md) for the contract, per-adapter notes, and how to write your own.

| Adapter | CLI value | Backend | Extra required |
|---|---|---|---|
| `HTTPAdapter` | `http` | Any `/v1/systemone`-compatible HTTP server | — (core) |
| `LocalAdapter` | `local` | Local Nirṇaya checkpoint via the `nirnaya` package | private model package |
| `Qwen3LogprobAdapter` | `qwen3` | Zero-shot option log-prob scoring (Qwen3-4B-Instruct) | `baselines` |
| `APILLMAdapter` | `api` | OpenAI-compatible chat completions in JSON mode | `baselines` |
| `SemIfAdapter` | `semif` | SemIf / OpenJev (Qwen3.5-4B) | `semif_phase1` (openjev) |
| `LayaAdapter` | (via `scripts/eval_laya.py`) | Laya-multilingual intent classifier (choice only) | `baselines` |

---

## Leaderboard

Live at [cmul8-hf/IndicJevBench](https://huggingface.co/datasets/cmul8-hf/IndicJevBench). Measured on 1000 items per dataset (2026-09-22).

| Model | Type | fintech_banking77 | synthetic_enterprise | intent_massive | hinglish_lid |
|-------|------|-------------------|----------------------|----------------|--------------|
| OpenJev (Qwen3.5-4B) | local | acc=0.620, IJScore=78.3 | acc=0.606, IJScore=79.6 | N/A (>16-option limit) | acc=0.998, IJScore=87.2 |
| Laya-multilingual | local | acc=0.455, IJScore=69.3 | acc=0.351, IJScore=58.7 | acc=0.350, IJScore=61.7 | acc=0.493, IJScore=73.7 |
| GPT-4o-mini (OpenRouter) | API | acc=0.775, IJScore=56.1 | acc=0.670, IJScore=46.6 | acc=0.580, IJScore=51.3 | acc=0.175, IJScore=21.2‡ |
| Laya (base) | local | acc=0.256, IJScore=56.2 | acc=0.369, IJScore=64.5 | acc=0.199, IJScore=50.2 | acc=0.569, IJScore=74.0 |
| Qwen3-4B (zero-shot logprob) | local | acc=0.373, IJScore=55.3 | acc=0.525, IJScore=68.5 | acc=0.316, IJScore=49.6 | acc=0.840, IJScore=2.3† |

† Qwen3-4B logprob scoring takes ~16s/item on hinglish_lid (one forward pass per option × many language classes), collapsing the speed axis.
‡ GPT-4o-mini hinglish_lid had 46/200 JSON parse errors (pre-fix run); accuracy and IJScore are underestimates.

Notes: OpenJev has a hard 16-option limit so intent_massive cannot be evaluated. The meaningful signal for general models is in fintech_banking77, intent_massive, and synthetic_enterprise. Laya and Qwen3-4B run locally with ~45ms latency; GPT-4o-mini runs via API at ~2s/call.

---

## Reproducing Results

1. `uv sync --extra baselines` (plus `pip install git+https://github.com/TheoLeeCJ/openjev.git` for OpenJev).
2. Ensure `datasets/v1/*.jsonl` are present (run `python scripts/package_datasets.py` if not).
3. Run the adapter of choice (CLI or `scripts/` wrapper). Results JSON is written to `results/v1/<run_id>.json`; raw per-task decisions stream to `results/v1/<run_id>_<task>_raw.jsonl`.
4. Score breakdowns (by language, source, question type) are inside each results JSON under `metrics.by_lang` / `metrics.by_source`.

Exact hardware, sampling parameters, and sampling budgets affect latency and cost axes; the leaderboard numbers were measured with the defaults in `scripts/`.

---

## Repository Layout

```
├── datasets/v1/            JSONL task files (frozen; built by scripts/package_datasets.py)
├── docs/                   LANGUAGES.md, ARCHITECTURE.md, ADAPTERS.md
├── src/indicjevbench/      Python package (schemas, core, adapters, metrics, scoring, CLI)
├── results/v1/             Raw logs and result JSON files (gitignored)
├── scripts/                Thin CLI wrappers + package_datasets.py
└── tests/                  pytest tests (no GPU/network required)
```

Docs: [BENCHMARK_EXPLAINED.md](BENCHMARK_EXPLAINED.md) · [IMPLEMENTATION.md](IMPLEMENTATION.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/ADAPTERS.md](docs/ADAPTERS.md) · [docs/LANGUAGES.md](docs/LANGUAGES.md) · [DATASHEET.md](DATASHEET.md)

---

## Acknowledgements

IndicJevBench is inspired by and modeled on **[JevBench](https://github.com/fstandhartinger/jevbench)** by Florian Standhartinger, the benchmark for general structured decision AI. We adopted its scoring methodology (4-axis weighted geometric mean), adapter pattern, and benchmark design philosophy. JevBench is MIT-licensed.

---

## Trademark Disclaimer

"Jev", "JevBench", and "System One" are trademarks of TypeSafe. IndicJevBench is an **independent** open-source benchmark: it is not affiliated with, endorsed by, or sponsored by TypeSafe or the JevBench project. The `/v1/systemone` endpoint shape is implemented independently and is used here only as a compatible wire contract.

---

## License

MIT — see [LICENSE](LICENSE). Dataset licenses vary per task; see [DATASHEET.md](DATASHEET.md).

---

## Citation

```bibtex
@misc{indicjevbench2026,
  title  = {IndicJevBench: A Benchmark for Indic-Language Structured Decision AI},
  author = {cmul8},
  year   = {2026},
  url    = {https://github.com/cmul8/IndicJevBench}
}
```
