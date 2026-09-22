# IndicJevBench — Adapters

How to connect a model to IndicJevBench: the `BenchAdapter` contract, the
`/v1/systemone` wire format, per-adapter guides, and testing notes.

## The BenchAdapter Contract

Every backend implements one method (see
`src/indicjevbench/adapters/base.py`):

```python
class BenchAdapter(abc.ABC):
    @abc.abstractmethod
    def decide(self, task: Task) -> DecisionResult:
        """Execute a single decision on a Task."""
```

`Task` carries the raw customer message (`task.state`) and one typed
`Question` (`choice` | `score` | `noul`, with `instructions` and `options`).

`DecisionResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `task_id` | `str` | Mirrors `task.id` |
| `probabilities` | `list[float]` | Full distribution over options (sums to ~1.0); noul uses `[P(false), P(true)]` |
| `answer` | `int \| bool` | Argmax option index (choice/score) or bool (noul); `-1` on failure |
| `confidence` | `float` | In `[0, 1]` — choice/score: `(p_max−1/K)/(1−1/K)`; noul: `|2p_true−1|` |
| `latency_ms` | `float` | Wall-clock time for this decision |
| `expected` | `float \| None` | Score questions: predicted mean level (1-indexed) |
| `tokens_used` | `int \| None` | Input tokens consumed, if the backend reports them |
| `error` | `str \| None` | Set when the decision failed; other fields may be empty |

Contract rules:

- Adapters are **pure decision producers** — no dataset loading, no metric
  computation, no logging of secrets.
- Per-task failures should be returned as a `DecisionResult` with
  `answer=-1` and `error` set, **not** raised, so the runner records the
  failure and continues. Raise only for run-level failures (e.g. budget
  exhausted) where continuing makes no sense.
- Optionally implement `close()` for adapters holding HTTP connections or GPU
  memory; the CLI calls it after the run.

## Per-Adapter Guide

### `HTTPAdapter` (`http`) — any `/v1/systemone` server

The reference integration. `POST {base_url}/v1/systemone` with:

```json
{
  "model": "my-model",
  "state": "raw customer/user message",
  "questions": [
    {
      "id": "q0",
      "type": "choice | score | noul",
      "instructions": "...",
      "options": ["opt1", "opt2"]
    }
  ]
}
```

(`options` is omitted for `noul`. One question per request, sent as `"q0"`.)

Expected 200 response:

```json
{
  "answers": [
    {
      "probabilities": [0.7, 0.2, 0.1],
      "answer": 0,
      "confidence": 0.55,
      "expected": 2.1
    }
  ],
  "usage": {"input_tokens": 123}
}
```

`confidence`, `expected`, and `usage` are optional (defaults: `0.0`, absent,
`None`). Transient failures (HTTP 5xx, connection errors) retry up to 2 times
with a 2 s delay; non-retryable 4xx statuses raise on the first response.
Usage:

```bash
uv run indicjevbench run --adapter http \
  --endpoint http://localhost:8000 --model my-model \
  --max-examples 200
```

### `APILLMAdapter` (`api`) — OpenAI-compatible JSON mode

Sends the (state, question) pair to any OpenAI-compatible chat-completions
endpoint with `response_format={"type": "json_object"}`, `temperature=0.0`,
and a system prompt that requests probability distributions; it then parses
and normalises the JSON (empty/missing probability lists fall back to uniform;
noul answers accept bools or `"true"`/`"yes"` strings; score answers derive
`expected = sum((i+1)·p_i)`).

- **API key:** `api_key=` argument, else `OPENAI_API_KEY`, else
  `OPENROUTER_API_KEY` (never logged). Set `base_url=` for OpenRouter.
- **Budget:** cumulative estimated spend is tracked against
  `max_budget_usd` (rough per-model $/1M-token table; unknown models default
  to the gpt-4o rate). When exhausted, `decide` raises `RuntimeError`.
  Per-task API/parse failures return `error` results after 3 attempts.

```bash
OPENROUTER_API_KEY=sk-or-... uv run indicjevbench run --adapter api \
  --model openai/gpt-4o-mini --budget 20.0 --max-examples 500
```

Requires the `baselines` extra (`uv sync --extra baselines`).

### `Qwen3LogprobAdapter` (`qwen3`) — zero-shot baseline

Scores each option by the summed token log-probability of the option string
given a `[STATE]/[QUESTION]/[OPTIONS]/[ANSWER]` prompt (the same template the
trained Nirṇaya model uses), with a single cached-prefix forward pass per
option. Noul scores `" No"` vs `" Yes"`. Softmax over option scores gives the
distribution; `expected` is derived for score questions.

```bash
uv run indicjevbench run --adapter qwen3 --device cuda
```

Requires the `baselines` extra and a GPU for practical runtimes (logprob
scoring is one forward pass per option — slow on wide option sets).

### `SemIfAdapter` (`semif`) — OpenJev / SemIf

Wraps `semif_phase1` from the
[openjev](https://github.com/TheoLeeCJ/openjev) project (Qwen/Qwen3.5-4B,
Apache-2.0). Maps the IndicJevBench task to SemIf's row format
(`id`/`state`/`question`/`options` with `{"id", "description"}` options),
scores via `semif_phase1.direct.score`, and re-aligns returned probabilities
to the question's option order (noul uses ids `["false", "true"]`).

```bash
pip install git+https://github.com/TheoLeeCJ/openjev.git
uv run indicjevbench run --adapter semif --device auto
```

Requires `semif_phase1` (lazy import; pinned model revision by default).

### `LayaAdapter` — Laya-multilingual (intent-only)

Wraps `convaiinnovations/laya-multilingual` as a **choice-only** baseline:
Laya has a fixed 60-label MASSIVE intent head, so `score`/`noul` questions
raise `NotImplementedError` (skip those datasets). Predicted probabilities are
re-aligned from Laya's label order to each question's option order (matched by
lower-cased label text) and re-normalised.

A **published-numbers mode** (`published_numbers_path=...`) skips model
loading entirely; `decide` raises `NotImplementedError` in that mode and
results are reported "as published by convaiinnovations" instead of re-run.

```bash
uv run python scripts/eval_laya.py --max-items 1000
```

Requires the `baselines` extra (torch + transformers).

### `LocalAdapter` (`local`) — Nirṇaya checkpoint

Runs a locally trained Nirṇaya checkpoint through the private `nirnaya`
model package (install from the model project root: `pip install -e .`).
Calls `nirnaya.infer.predict` with the `/v1/systemone`-shaped question list
and maps the first answer to a `DecisionResult`.

```bash
uv run indicjevbench run --adapter local \
  --checkpoint /path/to/checkpoints/best --device cuda
```

## Adding Your Own Adapter

1. Subclass `BenchAdapter` in `src/indicjevbench/adapters/<name>.py`.
2. Import heavy/optional dependencies **inside the constructor** and raise
   `ImportError` naming the extra to install.
3. Add a branch in `build_adapter()` (`src/indicjevbench/runner/cli.py`) and a
   matching choice in the `--adapter` argument.
4. Add tests under `tests/adapters/` (see below).

## Testing Notes

- **No network in tests.** `HTTPAdapter` accepts an `httpx.MockTransport` —
  `tests/adapters/test_http.py` exercises retries, error statuses, and parse
  failures without sockets.
- **Fake clients for API adapters.** `APILLMAdapter`'s parse/normalise logic
  is tested by injecting a fake `OpenAI` client (`tests/adapters/test_api_llm.py`).
- **Fake adapters for the runner.** `tests/core/test_runner.py` uses a
  `FakeAdapter` to verify metrics shape, raw-log JSONL lines, and error-task
  handling.
- **CLI dispatch.** `tests/runner/test_cli.py` checks `build_adapter` routing
  and `--help` smoke.
