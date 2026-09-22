"""eval_qwen3.py — run zero-shot Qwen3-4B log-prob scoring on IndicJevBench.

No option-count limit (handles intent_massive 60-class). Scores each option
by summing token log-probs given the shared prefix, then softmaxes.
Prefix KV cache is computed once and reused across all options.

Run:
    CUDA_VISIBLE_DEVICES=3 python bench/indicjevbench/scripts/eval_qwen3.py --device cuda
    CUDA_VISIBLE_DEVICES=3 python bench/indicjevbench/scripts/eval_qwen3.py --device cuda --max-items 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "v1"
RESULTS_DIR  = Path(__file__).parent.parent / "results" / "v1"

ALL_DATASETS = [
    "intent_massive",
    "fintech_banking77",
    "hinglish_lid",
    "synthetic_enterprise",
]

_TEMPLATE      = "[STATE]\n{state}\n[QUESTION]\n{instructions}\n[OPTIONS]\n{options}[ANSWER]"
_TEMPLATE_NOUL = "[STATE]\n{state}\n[QUESTION]\n{instructions}\n[ANSWER]"


# ---------------------------------------------------------------------------
# Metrics (identical to eval_openjev.py)
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


def indic_jev_score(acc: float, ece_val: float, brier: float, p50_ms: float) -> dict:
    intelligence = acc * 100
    calibration  = max(0.0, ((1 - ece_val) + (1 - brier / 2)) / 2 * 100)
    if p50_ms <= 50:
        speed = 100.0
    elif p50_ms >= 5000:
        speed = 0.0
    else:
        speed = math.log(5000 / p50_ms) / math.log(5000 / 50) * 100
    cost = 100.0
    axes = {"intelligence": intelligence, "calibration": calibration, "speed": speed, "cost": cost}
    weights = {"intelligence": 0.35, "calibration": 0.25, "speed": 0.20, "cost": 0.20}
    score = math.exp(sum(w * math.log(max(axes[k], 1e-6)) for k, w in weights.items()))
    return {"score": round(score, 2), **{k: round(v, 2) for k, v in axes.items()}}


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------

def build_prefix(item: dict) -> str:
    q = item["question"]
    state = item["state"]
    instructions = q["instructions"]
    options = q.get("options") or []
    if q["type"] == "noul":
        return _TEMPLATE_NOUL.format(state=state, instructions=instructions)
    opts_str = "".join(f"({i+1}) {opt}\n" for i, opt in enumerate(options))
    return _TEMPLATE.format(state=state, instructions=instructions, options=opts_str)


def continuations_for(item: dict) -> list[str]:
    q = item["question"]
    if q["type"] == "noul":
        return [" No", " Yes"]
    return [f" {opt}" for opt in (q.get("options") or [])]


@torch.inference_mode()
def score_item(model, tokenizer, device: str, item: dict) -> tuple[list[float], float]:
    """Returns (probs aligned to options/[false,true], latency_ms)."""
    prefix = build_prefix(item)
    conts = continuations_for(item)

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    log_probs = []
    for cont in conts:
        cont_ids = tokenizer.encode(cont, add_special_tokens=False, return_tensors="pt").to(device)
        full_ids = torch.cat([prefix_ids, cont_ids], dim=1)
        out = model(full_ids)
        logits = out.logits[0]
        start = prefix_ids.shape[1] - 1
        lp = 0.0
        for i, tok_id in enumerate(cont_ids[0]):
            lp += torch.log_softmax(logits[start + i], dim=-1)[tok_id].item()
        log_probs.append(lp)

    latency_ms = (time.perf_counter() - t0) * 1000
    max_lp = max(log_probs)
    exp = [math.exp(lp - max_lp) for lp in log_probs]
    total = sum(exp)
    probs = [e / total for e in exp]
    return probs, latency_ms


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

def evaluate_dataset(dataset_name: str, model, tokenizer, device: str, max_items: int) -> dict:
    items = load_items(dataset_name)
    if not items:
        return {}
    if max_items:
        items = items[:max_items]

    preds, labels, all_probs, q_types, latencies = [], [], [], [], []
    errors = 0

    print(f"\n  {dataset_name}: {len(items)} items")
    for i, item in enumerate(items):
        q_type = item["question"]["type"]
        expected = item["expected"]

        try:
            probs, latency_ms = score_item(model, tokenizer, device, item)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    error on item {i}: {e}")
            continue

        if q_type == "noul":
            pred = probs[1] > 0.5
            label = bool(expected)
        else:
            pred = int(max(range(len(probs)), key=lambda k: probs[k]))
            label = int(expected)

        preds.append(pred)
        labels.append(label)
        all_probs.append(probs)
        q_types.append(q_type)
        latencies.append(latency_ms)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(items)}  acc={accuracy(preds, labels):.3f}  lat={latency_ms:.0f}ms", flush=True)

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
        "dataset": dataset_name,
        "n_items": len(preds),
        "errors":  errors,
        "accuracy": round(acc, 4),
        "brier":    round(bri, 4),
        "ece":      round(ece_, 4),
        "p50_ms":   round(p50, 1),
        "p95_ms":   round(p95, 1),
        "indic_jev": ijs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--max-items", type=int, default=1000, help="items per dataset, 0 = all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print("Model loaded.", flush=True)
    sys.stdout.flush()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for ds in args.datasets:
        result = evaluate_dataset(ds, model, tokenizer, args.device, args.max_items)
        if result:
            all_results.append(result)

    print("\n\n=== Qwen3-4B (zero-shot logprob) on IndicJevBench ===")
    print(f"{'Dataset':<25} {'Items':>6} {'Acc':>6} {'ECE':>6} {'Brier':>6} {'p50ms':>7} {'IJScore':>8}")
    print("-" * 75)
    for r in all_results:
        ij = r["indic_jev"]["score"]
        print(f"{r['dataset']:<25} {r['n_items']:>6} {r['accuracy']:>6.3f} "
              f"{r['ece']:>6.3f} {r['brier']:>6.3f} {r['p50_ms']:>7.0f} {ij:>8.1f}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"qwen3_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
