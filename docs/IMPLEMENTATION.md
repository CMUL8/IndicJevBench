# IndicJevBench — Implementation Reference

## Task Dataclass

```python
@dataclass(frozen=True)
class Task:
    id: str          # unique item ID, e.g. "massive-hi-Deva-0001-q0"
    family: str      # task family: "intent" | "urgency" | "escalation" | "routing" | "lid"
    lang: str        # ISO 639-1 + script, e.g. "hi-Deva", "hi-Latn", "bn-Beng"
    state: str       # customer message / input text
    question: Question  # frozen dataclass: type/instructions/options/criteria
    expected: Any    # ground-truth label: int (choice/score) or bool (noul)
    split: str       # "v1"
    source: str      # "massive" | "banking77" | "comilingua" | "synthetic"
    license: str     # "CC BY 4.0" | "MIT"
    provenance: dict # {"origin": ..., "license": ..., "notes": ...}
```

The nested `Question` is itself a frozen dataclass:
`type` (`"choice"` | `"score"` | `"noul"`), `instructions`,
`options: tuple[str, ...] | None` (`None` for noul), and `criteria`.

Properties:
- `task.q_type` → `task.question.type`  (`"choice"` | `"score"` | `"noul"`)
- `task.options` → `task.question.options`

## DecisionResult Schema

```python
@dataclass
class DecisionResult:
    task_id: str              # matches task.id
    probabilities: list[float] # probability per option (sums to ~1.0)
    answer: int | bool        # argmax index (choice/score) or bool (noul)
    confidence: float         # normalised confidence 0.0–1.0
    latency_ms: float         # wall-clock time for the single decision
    expected: float | None    # score type only: predicted mean level (1-indexed)
    tokens_used: int | None   # API adapters: total tokens consumed
    error: str | None         # non-None if the decision failed; other fields may be empty
```

Confidence formulas:
- **choice / score**: `(p_max - 1/K) / (1 - 1/K)` — normalised margin above uniform chance
- **noul**: `|2 * p_true - 1|` — distance from the 0.5 decision boundary

## BenchAdapter Interface

```python
class BenchAdapter(abc.ABC):
    @abc.abstractmethod
    def decide(self, task: Task) -> DecisionResult:
        """Execute a single decision on a Task. Return DecisionResult."""
```

Optional: implement `close()` for adapters that hold HTTP connections or GPU memory.

## Provided Adapters

| Class | File | Description |
|---|---|---|
| `HTTPAdapter` | `adapters/http.py` | POST /v1/systemone to any compatible HTTP server |
| `LocalAdapter` | `adapters/local.py` | Load a local model checkpoint (requires compatible model package) |
| `Qwen3LogprobAdapter` | `adapters/qwen3_logprob.py` | Zero-shot option log-prob scoring via Qwen3-4B-Instruct |
| `APILLMAdapter` | `adapters/api_llm.py` | Generative API LLM in JSON output mode (OpenAI-compatible) |
| `LayaAdapter` | `adapters/laya.py` | Laya-multilingual intent classifier (choice only) |
| `SemIfAdapter` | `adapters/semif.py` | SemIf (Qwen/Qwen3.5-4B) open-source Jev-style model |

## Scoring Math

### IndicJevScore

Four axes, each in [0, 100]:

```
intelligence = accuracy * 100

calibration  = ((1 - ECE) + (1 - min(Brier/2, 1))) / 2 * 100

speed        = 100 - max(0, log10(p50_ms / 50) / log10(100)) * 100
               (50ms → 100, 5000ms → 0, log scale)

cost         = 100 - max(0, log10(cost_per_1k / 0.01) / log10(1000)) * 100
               ($0.01/1k → 100, $10/1k → 0; 0.0 input → 100)
```

Composite (weighted geometric mean):

```
IndicJevScore = exp(0.35*ln(intelligence) + 0.25*ln(calibration)
                  + 0.20*ln(speed)        + 0.20*ln(cost))
```

Axes are clipped to [1e-6, 100] before taking ln to avoid log(0).

### ECE (Expected Calibration Error)

15 equal-width bins of max-class confidence:

```
ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
```

### Brier Score

Mean squared error of the full probability distribution against the one-hot ground-truth label:

```
Brier = (1/N) Σ_i Σ_k (p_ik - y_ik)^2
```

Maximum value for binary (K=2) is 2.0.

### Automatable Share

Highest confidence threshold T such that items with confidence > T have error rate ≤ 5%. Returns (share, threshold) where share = fraction of all items above T.

## How to Add an Adapter

1. Create `src/indicjevbench/adapters/my_adapter.py`.
2. Import `BenchAdapter` and `DecisionResult` from `.base`.
3. Implement `decide(self, task: Task) -> DecisionResult`.
4. Add a branch in `src/indicjevbench/runner/cli.py` under `build_adapter()`.
5. Add tests in `tests/adapters/` if the adapter has non-trivial parsing logic.

## Data Flow

```
data/final/test.jsonl
    → scripts/package_datasets.py
    → datasets/v1/<task_name>.jsonl      (IndicJevBench JSONL format, frozen)
    → indicjevbench.load_tasks()         (or core.dataset.load_tasks)
    → BenchmarkRunner(adapter, raw_log_path=...).run(tasks)
    → results/v1/<run_id>.json + results/v1/<run_id>_<task>_raw.jsonl
```
