# IndicJevBench

**The first open benchmark for Indic-language structured decision AI.**

IndicJevBench tests whether a model can return calibrated probability distributions for typed questions (intent classification, urgency scoring, escalation detection, agent routing) given a customer message — in Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, and Hinglish.

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

### Install

```bash
cd bench/indicjevbench
pip install -e .
```

### Run against a live HTTP endpoint

```bash
indicjevbench run \
  --adapter http \
  --endpoint http://localhost:8000 \
  --model my-model
```

### Run the Qwen3 zero-shot baseline (GPU required)

```bash
indicjevbench run \
  --adapter qwen3 \
  --device cuda
```

### Run against a local checkpoint

```bash
indicjevbench run \
  --adapter local \
  --checkpoint /path/to/checkpoints/best \
  --device cuda
```

### Run against an API LLM (OpenAI-compatible)

```bash
OPENAI_API_KEY=sk-... indicjevbench run \
  --adapter api \
  --model gpt-4o \
  --budget 20.0 \
  --max-examples 500
```

### Package datasets first

Before running, build the task JSONL files from the frozen test split:

```bash
python scripts/package_datasets.py
```

This reads `../../data/final/test.jsonl` and writes per-task files to `datasets/v1/`.

---

## Scoring Methodology

IndicJevBench reports a composite **IndicJevScore** (0–100) as the geometric mean of four axes:

| Axis | Weight | Formula | Best | Worst |
|------|--------|---------|------|-------|
| Intelligence | 35% | accuracy × 100 | 100 (acc=1.0) | 0 (acc=0.0) |
| Calibration | 25% | mean(1−ECE, 1−Brier/2) × 100 | 100 (ECE=0, Brier=0) | 0 (ECE=1, Brier=2) |
| Speed | 20% | log-scale, 50ms=100, 5000ms=0 | 100 (p50≤50ms) | 0 (p50≥5000ms) |
| Cost | 20% | log-scale, $0.01/1k=100, $10/1k=0; local=100 | 100 (free/local) | 0 ($10+/1k) |

Composite = exp(Σ wᵢ × ln(max(axisᵢ, 1e-6)))

### Per-task metrics

Every task also reports: accuracy, macro-F1, NLL, Brier score, ECE (15 bins), MAE of expected level (score tasks), and automatable share (fraction of decisions auto-approvable at <5% error).

---

## Tasks

| Dataset name | Family | Languages | Items (v1) | License | Source |
|---|---|---|---|---|---|
| `intent_massive` | intent | hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym | ~6,000 | CC BY 4.0 | MASSIVE test split |
| `fintech_banking77` | intent | hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym | ~2,500 | CC BY 4.0 | Banking77 + NLLB translation |
| `hinglish_lid` | lid | hi-Latn | ~1,000 | CC BY 4.0 | COMI-LINGUA |
| `synthetic_enterprise` | urgency / escalation / routing | hi-Latn, hi-Deva | ~2,500 | CC BY 4.0 | cmul8 synthetic (Qwen3+DeepSeek) |

Exact counts depend on `scripts/package_datasets.py` output after the data pipeline runs.

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

## Adding a New Model / Adapter

1. Create `indicjevbench/adapters/my_model.py` implementing `BenchAdapter`:

```python
from indicjevbench.adapters.base import BenchAdapter, DecisionResult

class MyModelAdapter(BenchAdapter):
    def decide(self, task) -> DecisionResult:
        # Call your model with task.state and task.question
        # Return a DecisionResult with probabilities, answer, confidence, latency_ms
        ...
```

2. Register it in `indicjevbench/cli.py` under `_make_adapter()`.

3. Run:

```bash
indicjevbench run --adapter my_model ...
```

See `IMPLEMENTATION.md` for the full `DecisionResult` schema and `BenchAdapter` interface.

---

## Repository Layout

```
bench/indicjevbench/
├── datasets/v1/          JSONL task files (generated by scripts/package_datasets.py)
├── indicjevbench/        Python package (harness, metrics, adapters)
│   └── adapters/         HTTP, local, Qwen3, API LLM, Laya adapters
├── results/v1/           Raw logs and result JSON files (gitignored)
├── scripts/              package_datasets.py
├── tests/                pytest tests (no GPU/network required)
└── docs/                 LANGUAGES.md
```

---

## Acknowledgements

IndicJevBench is inspired by and modeled on **[JevBench](https://github.com/fstandhartinger/jevbench)** by Florian Standhartinger, the benchmark for general structured decision AI. We adopted its scoring methodology (4-axis geometric mean), adapter pattern, and benchmark design philosophy. JevBench is MIT-licensed.

---

## License

MIT — see [LICENSE](LICENSE). Dataset licenses vary per task; see [DATASHEET.md](DATASHEET.md).
