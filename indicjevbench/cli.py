"""CLI: python -m indicjevbench.cli run --tasks ... --adapter http --endpoint ..."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from . import BENCH_ROOT, DATASETS_DIR, RESULTS_DIR
from .tasks import load_tasks
from .runner import run

def _make_adapter(args):
    if args.adapter == "http":
        from .adapters.http import HTTPAdapter
        return HTTPAdapter(args.endpoint, model=args.model)
    elif args.adapter == "local":
        from .adapters.local import LocalAdapter
        return LocalAdapter(args.checkpoint, device=args.device)
    elif args.adapter == "qwen3":
        from .adapters.qwen3_logprob import Qwen3LogprobAdapter
        return Qwen3LogprobAdapter(device=args.device)
    elif args.adapter == "api":
        from .adapters.api_llm import APILLMAdapter
        return APILLMAdapter(model=args.model, max_budget_usd=args.budget)
    elif args.adapter == "semif":
        from .adapters.semif import SemIfAdapter
        return SemIfAdapter(model_id=args.model or "Qwen/Qwen3.5-4B", device=args.device)
    else:
        raise ValueError(f"Unknown adapter: {args.adapter}")

def cmd_run(args):
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    adapter = _make_adapter(args)

    task_files = [Path(t) for t in args.tasks] if args.tasks else sorted(DATASETS_DIR.glob("*.jsonl"))
    if not task_files:
        print("No task files found. Run scripts/package_datasets.py first.")
        return

    all_results = {"run_id": run_id, "model": args.model, "tasks": {}}

    for task_file in task_files:
        task_name = task_file.stem
        tasks = load_tasks(task_file, max_examples=args.max_examples)
        print(f"\n[indicjevbench] {task_name}: {len(tasks)} tasks")

        raw_log = RESULTS_DIR / f"{run_id}_{task_name}_raw.jsonl"
        result = run(tasks, adapter, raw_log_path=raw_log, task_name=task_name)
        all_results["tasks"][task_name] = result

        s = result["score"]
        m = result["metrics"].get("all", {})
        print(f"  IndicJevScore={s['indicjev_score']}  "
              f"acc={m.get('accuracy', float('nan')):.3f}  "
              f"ece={m.get('ece', float('nan')):.3f}  "
              f"p50={result['latency']['p50_ms']}ms")

    out_path = Path(args.output) if args.output else RESULTS_DIR / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[indicjevbench] Results: {out_path}")

    if hasattr(adapter, "close"):
        adapter.close()

def main():
    parser = argparse.ArgumentParser(prog="indicjevbench")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run benchmark evaluation")
    run_p.add_argument("--tasks", nargs="+", help="JSONL task files (default: all in datasets/v1/)")
    run_p.add_argument("--adapter", required=True, choices=["http", "local", "qwen3", "api", "semif"])
    run_p.add_argument("--endpoint", default="http://localhost:8000")
    run_p.add_argument("--checkpoint", default="checkpoints/best")
    run_p.add_argument("--model", default="my-model")
    run_p.add_argument("--device", default="cuda")
    run_p.add_argument("--budget", type=float, default=20.0)
    run_p.add_argument("--max-examples", type=int, default=None)
    run_p.add_argument("--output", default=None)
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
