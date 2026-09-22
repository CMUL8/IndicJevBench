"""eval_openjev.py — run OpenJev (semif-phase1) on IndicJevBench datasets.

No harness dependency. Reads packaged JSONL files, runs OpenJev, reports
accuracy, calibration, speed, and IndicJevScore per dataset.

Install:
    pip install git+https://github.com/TheoLeeCJ/openjev.git

Run:
    python bench/indicjevbench/scripts/eval_openjev.py
    python bench/indicjevbench/scripts/eval_openjev.py --max-items 200
    python bench/indicjevbench/scripts/eval_openjev.py --datasets intent_massive fintech_banking77
"""
from __future__ import annotations

import argparse
import json
import math
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
            true_val = 1.0 if label else 0.0
            total += (p - true_val) ** 2
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


def indic_jev_score(acc: float, ece_val: float, brier: float, p50_ms: float) -> dict:
    intelligence  = acc * 100
    calibration   = max(0.0, ((1 - ece_val) + (1 - brier / 2)) / 2 * 100)
    if p50_ms <= 50:
        speed = 100.0
    elif p50_ms >= 5000:
        speed = 0.0
    else:
        speed = math.log(5000 / p50_ms) / math.log(5000 / 50) * 100
    cost  = 100.0  # local model
    axes  = {
        "intelligence": intelligence,
        "calibration":  calibration,
        "speed":        speed,
        "cost":         cost,
    }
    weights = {"intelligence": 0.35, "calibration": 0.25, "speed": 0.20, "cost": 0.20}
    score = math.exp(
        sum(w * math.log(max(axes[k], 1e-6)) for k, w in weights.items())
    )
    return {"score": round(score, 2), **{k: round(v, 2) for k, v in axes.items()}}


# ---------------------------------------------------------------------------
# Build OpenJev row from IndicJevBench item
# ---------------------------------------------------------------------------

def item_to_row(item: dict) -> tuple[dict, list[str]]:
    q = item["question"]
    q_type = q["type"]
    instructions = q["instructions"]
    raw_options = q.get("options") or []

    if q_type == "noul":
        options = [
            {"id": "false", "description": "No"},
            {"id": "true",  "description": "Yes"},
        ]
        option_order = ["false", "true"]
    else:
        options = [{"id": str(i), "description": opt} for i, opt in enumerate(raw_options)]
        option_order = [str(i) for i in range(len(raw_options))]

    row = {
        "id": item["id"],
        "state": item["state"],
        "question": instructions,
        "options": options,
    }
    return row, option_order


def align_probs(result: dict, option_order: list[str]) -> list[float]:
    scored_ids = result.get("option_ids", option_order)
    raw_probs  = result.get("probabilities", [])
    id_to_prob = dict(zip(scored_ids, raw_probs))
    probs = [id_to_prob.get(oid, 0.0) for oid in option_order]
    total = sum(probs) or 1.0
    return [p / total for p in probs]


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------

def _load_items(dataset_name: str) -> list[dict]:
    path = DATASETS_DIR / f"{dataset_name}.jsonl"
    if path.exists():
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    # Fall back to HuggingFace — download raw JSONL via huggingface_hub
    print(f"  Local file not found, downloading {dataset_name} from HuggingFace ...")
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id="cmul8-hf/IndicJevBench",
            filename=f"v1/{dataset_name}.jsonl",
            repo_type="dataset",
        )
        items = []
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        print(f"  Downloaded {len(items)} items.")
        return items
    except Exception as e:
        print(f"  [skip] Could not load {dataset_name}: {e}")
        return []


def evaluate_dataset(
    dataset_name: str,
    model, tokenizer, metadata,
    max_items: int,
    max_tokens: int,
) -> dict:
    from semif_phase1.direct import score as semif_score

    items = _load_items(dataset_name)
    if not items:
        return {}
    if max_items:
        items = items[:max_items]

    # Pre-flight: skip dataset if ANY item has options exceeding OpenJev's 16-option limit
    max_opts = max(len(it["question"].get("options") or []) for it in items[:50])
    if max_opts > 16:
        print(f"\n  {dataset_name}: SKIPPED — up to {max_opts} options, exceeds OpenJev 16-option limit")
        return {"dataset": dataset_name, "skipped": True, "reason": f"up to {max_opts}-option task exceeds OpenJev limit of 16"}

    preds, labels, all_probs, q_types, latencies = [], [], [], [], []
    errors = 0

    print(f"\n  {dataset_name}: {len(items)} items")
    for i, item in enumerate(items):
        row, option_order = item_to_row(item)
        expected = item["expected"]
        q_type = item["question"]["type"]

        t0 = time.perf_counter()
        try:
            result = semif_score(model, tokenizer, row, metadata, max_tokens=max_tokens)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    error on item {i}: {e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        probs = align_probs(result, option_order)

        if q_type == "noul":
            pred = probs[1] > 0.5
        else:
            pred = int(max(range(len(probs)), key=lambda k: probs[k]))

        preds.append(pred)
        labels.append(expected if q_type != "noul" else bool(expected))
        all_probs.append(probs)
        q_types.append(q_type)
        latencies.append(latency_ms)

        if (i + 1) % 50 == 0:
            running_acc = accuracy(preds, labels)
            print(f"    {i+1}/{len(items)}  acc={running_acc:.3f}  lat={latency_ms:.0f}ms")

    if not preds:
        return {}

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]

    acc  = accuracy(preds, labels)
    bri  = brier_score(all_probs, labels, q_types)
    ece_ = ece(all_probs, labels, q_types)
    ijs  = indic_jev_score(acc, ece_, bri, p50)

    return {
        "dataset":    dataset_name,
        "n_items":    len(preds),
        "errors":     errors,
        "accuracy":   round(acc, 4),
        "brier":      round(bri, 4),
        "ece":        round(ece_, 4),
        "p50_ms":     round(p50, 1),
        "p95_ms":     round(p95, 1),
        "indic_jev":  ijs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--max-items", type=int, default=1000, help="items per dataset, 0 = all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    args = parser.parse_args()

    try:
        from semif_phase1.core import load_causal_model
    except ImportError:
        print("Install: pip install git+https://github.com/TheoLeeCJ/openjev.git")
        raise

    print(f"Loading {args.model} ...")
    model, tokenizer, metadata = load_causal_model(
        source=args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
    )
    print("Model loaded.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for ds in args.datasets:
        result = evaluate_dataset(ds, model, tokenizer, metadata, args.max_items, args.max_tokens)
        if result:
            all_results.append(result)

    print("\n\n=== OpenJev on IndicJevBench ===")
    print(f"{'Dataset':<25} {'Items':>6} {'Acc':>6} {'ECE':>6} {'Brier':>6} {'p50ms':>7} {'IJScore':>8}")
    print("-" * 75)
    for r in all_results:
        if r.get("skipped"):
            print(f"{r['dataset']:<25} {'N/A':>6}  -- {r['reason']}")
        else:
            ij = r["indic_jev"]["score"]
            print(f"{r['dataset']:<25} {r['n_items']:>6} {r['accuracy']:>6.3f} "
                  f"{r['ece']:>6.3f} {r['brier']:>6.3f} {r['p50_ms']:>7.0f} {ij:>8.1f}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"openjev_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
