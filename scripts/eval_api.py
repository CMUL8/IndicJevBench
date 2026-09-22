"""eval_api.py — run any OpenRouter-compatible API LLM on IndicJevBench.

Sends each item as a structured JSON prompt and parses calibrated probabilities.
Uses OPENROUTER_API_KEY from environment.

Install:
    pip install openai

Run:
    OPENROUTER_API_KEY=sk-or-... python bench/indicjevbench/scripts/eval_api.py
    OPENROUTER_API_KEY=sk-or-... python bench/indicjevbench/scripts/eval_api.py --model openai/gpt-4o
    OPENROUTER_API_KEY=sk-or-... python bench/indicjevbench/scripts/eval_api.py --model anthropic/claude-3-5-haiku --max-items 200
    OPENROUTER_API_KEY=sk-or-... python bench/indicjevbench/scripts/eval_api.py --datasets fintech_banking77 synthetic_enterprise
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "v1"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "v1"

ALL_DATASETS = [
    "intent_massive",
    "fintech_banking77",
    "hinglish_lid",
    "synthetic_enterprise",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_SYSTEM_PROMPT = """You are a structured decision assistant. Given a customer message and typed questions, return calibrated probability distributions as JSON.

For each question:
- "choice" or "score": return "probabilities" (one float per option, sum to 1.0) and "answer" (0-based integer index of best option)
- "noul": return "probabilities" as [P(false), P(true)] and "answer" as true or false

Return ONLY valid JSON: {"answers": [{"id": "q0", "type": "...", "probabilities": [...], "answer": ...}]}"""

# Approximate cost per 1M tokens (input+output) for budget tracking
_COST_PER_1M = {
    "openai/gpt-4o":               5.00,
    "openai/gpt-4o-mini":          0.30,
    "anthropic/claude-3-5-sonnet": 4.50,
    "anthropic/claude-3-5-haiku":  1.00,
    "anthropic/claude-3-haiku":    0.40,
    "google/gemini-flash-1.5":     0.15,
    "google/gemini-pro-1.5":       2.50,
    "meta-llama/llama-3.1-70b-instruct": 0.60,
    "qwen/qwen3-235b-a22b":        0.22,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy(preds: list, labels: list) -> float:
    return sum(p == l for p, l in zip(preds, labels)) / max(len(preds), 1)


def brier_score(probs_list: list[list[float]], labels: list, q_types: list[str]) -> float:
    total = 0.0
    for probs, label, qtype in zip(probs_list, labels, q_types):
        if qtype == "noul":
            p = probs[1] if len(probs) > 1 else probs[0]
            total += (p - (1.0 if label else 0.0)) ** 2
        else:
            for i, p in enumerate(probs):
                total += (p - (1.0 if i == label else 0.0)) ** 2
    return total / max(len(probs_list), 1)


def ece(probs_list: list[list[float]], labels: list, q_types: list[str], n_bins: int = 15) -> float:
    bins = [[] for _ in range(n_bins)]
    for probs, label, qtype in zip(probs_list, labels, q_types):
        if qtype == "noul":
            conf = max(probs)
            pred = probs[1] > 0.5
            correct = pred == bool(label)
        else:
            argmax = max(range(len(probs)), key=lambda i: probs[i])
            conf = probs[argmax]
            correct = argmax == label
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    total = sum(len(b) for b in bins)
    ece_val = 0.0
    for b in bins:
        if b:
            avg_conf = sum(x[0] for x in b) / len(b)
            avg_acc  = sum(x[1] for x in b) / len(b)
            ece_val += (len(b) / total) * abs(avg_acc - avg_conf)
    return ece_val


def indic_jev_score(acc: float, ece_val: float, brier: float, p50_ms: float,
                    cost_per_1k: float) -> dict:
    intelligence = acc * 100
    calibration  = max(0.0, ((1 - ece_val) + (1 - brier / 2)) / 2 * 100)
    if p50_ms <= 50:
        speed = 100.0
    elif p50_ms >= 5000:
        speed = 0.0
    else:
        speed = math.log(5000 / p50_ms) / math.log(5000 / 50) * 100
    # Cost axis: $0.01/1k=100, $10/1k=0 (log scale)
    if cost_per_1k <= 0.01:
        cost = 100.0
    elif cost_per_1k >= 10.0:
        cost = 0.0
    else:
        cost = math.log(10.0 / cost_per_1k) / math.log(10.0 / 0.01) * 100
    axes = {"intelligence": intelligence, "calibration": calibration, "speed": speed, "cost": cost}
    weights = {"intelligence": 0.35, "calibration": 0.25, "speed": 0.20, "cost": 0.20}
    score = math.exp(sum(w * math.log(max(axes[k], 1e-6)) for k, w in weights.items()))
    return {"score": round(score, 2), **{k: round(v, 2) for k, v in axes.items()}}


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def build_user_message(item: dict) -> str:
    q = item["question"]
    opts = q.get("options")
    opts_str = f"\n  Options: {opts}" if opts else ""
    return (f'Customer message: "{item["state"]}"\n\n'
            f'Questions:\n'
            f'  id="q0", type="{q["type"]}", instructions="{q["instructions"]}"{opts_str}')


def parse_response(data: dict, item: dict) -> tuple[list[float], int | bool] | None:
    """Extract (probs, answer) from API JSON response."""
    answers = data.get("answers", [])
    if not answers:
        return None
    ans = answers[0]
    probs = ans.get("probabilities", [])
    q_type = item["question"]["type"]
    n_opts = len(item["question"].get("options") or []) or 2

    if not probs:
        probs = [1.0 / n_opts] * n_opts

    # Pad/truncate to expected length
    if len(probs) < n_opts:
        probs += [0.0] * (n_opts - len(probs))
    probs = probs[:n_opts]

    total = sum(probs) or 1.0
    probs = [p / total for p in probs]

    if q_type == "noul":
        p_true = probs[1] if len(probs) >= 2 else probs[0]
        raw_ans = ans.get("answer", p_true > 0.5)
        if isinstance(raw_ans, str):
            raw_ans = raw_ans.lower() in ("true", "yes")
        return probs, bool(raw_ans)
    else:
        raw_ans = ans.get("answer", int(max(range(len(probs)), key=lambda i: probs[i])))
        if isinstance(raw_ans, bool):
            raw_ans = int(raw_ans)
        return probs, int(raw_ans)


def call_api(client, model: str, item: dict, total_tokens: list) -> tuple[list[float], int | bool, float] | None:
    """Returns (probs, answer, latency_ms) or None on failure."""
    user_msg = build_user_message(item)
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            if resp.usage:
                total_tokens[0] += resp.usage.total_tokens
            text = resp.choices[0].message.content or "{}"
            data = json.loads(text)
            parsed = parse_response(data, item)
            if parsed is None:
                return None
            latency_ms = (time.perf_counter() - t0) * 1000
            return parsed[0], parsed[1], latency_ms
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_items(dataset_name: str) -> list[dict]:
    path = DATASETS_DIR / f"{dataset_name}.jsonl"
    print(f"  Loading {dataset_name} from {path} ...", flush=True)
    if path.exists():
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        print(f"  Loaded {len(items)} items from local file.", flush=True)
        return items

    print(f"  Local file not found, downloading {dataset_name} from HuggingFace ...", flush=True)
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="cmul8-hf/IndicJevBench",
            filename=f"v1/{dataset_name}.jsonl",
            repo_type="dataset",
            etag_timeout=30,
        )
        items = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        print(f"  Downloaded {len(items)} items.", flush=True)
        return items
    except Exception as e:
        print(f"  [skip] Could not load {dataset_name}: {e}", flush=True)
        return []


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(dataset_name: str, client, model: str, max_items: int,
                     budget_usd: float, total_tokens: list, total_spent: list) -> dict:
    items = load_items(dataset_name)
    if not items:
        return {}
    if max_items:
        items = items[:max_items]

    preds, labels, all_probs, q_types, latencies = [], [], [], [], []
    errors = 0

    print(f"\n  {dataset_name}: {len(items)} items", flush=True)
    for i, item in enumerate(items):
        # Budget check
        if total_spent[0] >= budget_usd:
            print(f"    [budget exhausted at ${total_spent[0]:.3f}] stopping.", flush=True)
            break

        q_type = item["question"]["type"]
        expected = item["expected"]

        try:
            result = call_api(client, model, item, total_tokens)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    error on item {i}: {e}", flush=True)
            continue

        if result is None:
            errors += 1
            continue

        probs, answer, latency_ms = result

        # Update spend estimate
        rate = _COST_PER_1M.get(model, 5.0)
        total_spent[0] = rate * total_tokens[0] / 1_000_000

        if q_type == "noul":
            pred = bool(answer)
            label = bool(expected)
        else:
            pred = int(answer)
            label = int(expected)

        preds.append(pred)
        labels.append(label)
        all_probs.append(probs)
        q_types.append(q_type)
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(items)}  acc={accuracy(preds, labels):.3f}"
                  f"  lat={latency_ms:.0f}ms  spent=${total_spent[0]:.3f}", flush=True)

    if not preds:
        return {}

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    acc  = accuracy(preds, labels)
    bri  = brier_score(all_probs, labels, q_types)
    ece_ = ece(all_probs, labels, q_types)

    # Estimate cost per 1k decisions
    cost_per_1k = (total_spent[0] / max(len(preds), 1)) * 1000

    ijs  = indic_jev_score(acc, ece_, bri, p50, cost_per_1k)

    return {
        "dataset":      dataset_name,
        "n_items":      len(preds),
        "errors":       errors,
        "accuracy":     round(acc, 4),
        "brier":        round(bri, 4),
        "ece":          round(ece_, 4),
        "p50_ms":       round(p50, 1),
        "p95_ms":       round(p95, 1),
        "cost_per_1k":  round(cost_per_1k, 4),
        "indic_jev":    ijs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--max-items", type=int, default=200,
                        help="items per dataset (default 200 to control cost), 0=all")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="OpenRouter model ID")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="max spend in USD across entire run")
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY in environment.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("Install: pip install openai")
        sys.exit(1)

    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/CMUL8/IndicJevBench",
            "X-Title": "IndicJevBench",
        },
    )

    print(f"Model: {args.model}  Budget: ${args.budget}  Max items/dataset: {args.max_items or 'all'}", flush=True)
    sys.stdout.flush()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    total_tokens = [0]   # mutable for pass-by-reference across datasets
    total_spent  = [0.0]

    for ds in args.datasets:
        result = evaluate_dataset(ds, client, args.model, args.max_items,
                                  args.budget, total_tokens, total_spent)
        if result:
            all_results.append(result)

    model_short = args.model.replace("/", "_")
    print(f"\n\n=== {args.model} on IndicJevBench ===")
    print(f"{'Dataset':<25} {'Items':>6} {'Acc':>6} {'ECE':>6} {'Brier':>6} {'p50ms':>7} {'$/1k':>7} {'IJScore':>8}")
    print("-" * 82)
    for r in all_results:
        ij = r["indic_jev"]["score"]
        print(f"{r['dataset']:<25} {r['n_items']:>6} {r['accuracy']:>6.3f} "
              f"{r['ece']:>6.3f} {r['brier']:>6.3f} {r['p50_ms']:>7.0f} "
              f"{r['cost_per_1k']:>7.3f} {ij:>8.1f}")

    print(f"\nTotal spent: ${total_spent[0]:.4f} ({total_tokens[0]:,} tokens)")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"api_{model_short}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "total_tokens": total_tokens[0],
            "total_spent_usd": round(total_spent[0], 4),
            "results": all_results,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
