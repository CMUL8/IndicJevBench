"""Command-line interface: ``indicjevbench run --tasks ... --adapter http ...``."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indicjevbench.adapters.base import BenchAdapter
from indicjevbench.configs.paths import BenchPaths
from indicjevbench.core.dataset import load_tasks
from indicjevbench.core.runner import BenchmarkRunner
from indicjevbench.utils.atomic import atomic_write_text
from indicjevbench.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def build_adapter(args: argparse.Namespace) -> BenchAdapter:
    """Construct the BenchAdapter selected by ``--adapter``.

    Heavy adapter dependencies (torch, transformers, openai, semif) are
    imported lazily inside each adapter's constructor.

    Args:
        args: Parsed CLI namespace with the ``adapter`` and adapter-specific
            fields (``endpoint``, ``checkpoint``, ``model``, ``device``,
            ``budget``).

    Returns:
        The constructed BenchAdapter.

    Raises:
        ValueError: If ``args.adapter`` is unknown.
        ImportError: If the adapter's optional dependency is not installed.
    """
    if args.adapter == "http":
        from indicjevbench.adapters.http import HTTPAdapter

        return HTTPAdapter(args.endpoint, model=args.model)
    if args.adapter == "local":
        from indicjevbench.adapters.local import LocalAdapter

        return LocalAdapter(args.checkpoint, device=args.device)
    if args.adapter == "qwen3":
        from indicjevbench.adapters.qwen3_logprob import Qwen3LogprobAdapter

        return Qwen3LogprobAdapter(device=args.device)
    if args.adapter == "api":
        from indicjevbench.adapters.api_llm import APILLMAdapter

        return APILLMAdapter(model=args.model, max_budget_usd=args.budget)
    if args.adapter == "semif":
        from indicjevbench.adapters.semif import SemIfAdapter

        return SemIfAdapter(model_id=args.model or "Qwen/Qwen3.5-4B", device=args.device)
    raise ValueError(f"Unknown adapter: {args.adapter}")


def cmd_run(args: argparse.Namespace, paths: BenchPaths | None = None) -> None:
    """Execute the ``run`` subcommand over one or more task files.

    Args:
        args: Parsed CLI namespace.
        paths: Optional BenchPaths override (defaults to ``BenchPaths.default()``).
    """
    paths = paths if paths is not None else BenchPaths.default()
    run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    adapter = build_adapter(args)

    task_files = [Path(t) for t in args.tasks] if args.tasks else paths.dataset_files()
    if not task_files:
        logger.error("No task files found. Run scripts/package_datasets.py first.")
        return

    all_results: dict[str, Any] = {"run_id": run_id, "model": args.model, "tasks": {}}

    for task_file in task_files:
        task_name = task_file.stem
        tasks = load_tasks(task_file, max_examples=args.max_examples)
        logger.info("[%s] %s: %d tasks", run_id, task_name, len(tasks))

        raw_log = paths.results_dir / f"{run_id}_{task_name}_raw.jsonl"
        runner = BenchmarkRunner(adapter, raw_log_path=raw_log, task_name=task_name)
        result = runner.run(tasks)
        all_results["tasks"][task_name] = result

        s = result["score"]
        m = result["metrics"].get("all", {})
        logger.info(
            "[%s] %s: IndicJevScore=%s acc=%.3f ece=%.3f p50=%sms",
            run_id, task_name, s["indicjev_score"],
            m.get("accuracy", float("nan")),
            m.get("ece", float("nan")),
            result["latency"]["p50_ms"],
        )

    out_path = Path(args.output) if args.output else paths.results_file(run_id)
    atomic_write_text(out_path, json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("Results written to %s", out_path)

    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        Parser with the ``run`` subcommand and its flags.
    """
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

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).
    """
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
