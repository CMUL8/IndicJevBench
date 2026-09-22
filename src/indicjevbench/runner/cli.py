"""Command-line interface: ``indicjevbench run --tasks ... --adapter http ...``."""

import argparse
import json
import logging
from collections.abc import Sequence
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


def _default_run_id() -> str:
    """Generate the default timestamped run identifier.

    Returns:
        ``run_YYYYMMDD_HHMMSS`` in UTC.
    """
    return f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}"


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

        return Qwen3LogprobAdapter(model_id=args.model, device=args.device)
    if args.adapter == "api":
        from indicjevbench.adapters.api_llm import APILLMAdapter

        return APILLMAdapter(model=args.model, base_url=args.base_url, max_budget_usd=args.budget)
    if args.adapter == "semif":
        from indicjevbench.adapters.semif import SemIfAdapter

        return SemIfAdapter(model_id=args.model or "Qwen/Qwen3.5-4B", device=args.device)
    raise ValueError(f"Unknown adapter: {args.adapter}")


def run_evaluation(
    adapter: BenchAdapter,
    task_files: Sequence[str | Path],
    *,
    model: str = "unknown",
    max_examples: int | None = None,
    output: str | Path | None = None,
    run_id: str | None = None,
    paths: BenchPaths | None = None,
) -> Path:
    """Run an adapter over task files and write a combined results JSON.

    Prints one summary line per task file (this is the CLI entrypoint, so
    printing is intentional) plus the final results path. Task files that do
    not exist are skipped with a warning. Progress details go to the logger.

    Args:
        adapter: The BenchAdapter to evaluate.
        task_files: JSONL task files to run, in order.
        model: Model label recorded in the results JSON.
        max_examples: Optional per-dataset cap on tasks evaluated.
        output: Optional results JSON path (default: ``results_file(run_id)``).
        run_id: Run identifier (default: timestamped ``run_<ts>``).
        paths: BenchPaths override (default: ``BenchPaths.default()``).

    Returns:
        Path of the written results JSON.

    Raises:
        ValueError: If ``task_files`` is empty.
    """
    paths = paths if paths is not None else BenchPaths.default()
    run_id = run_id if run_id is not None else _default_run_id()
    files = [Path(t) for t in task_files]
    if not files:
        raise ValueError("run_evaluation requires at least one task file")

    all_results: dict[str, Any] = {"run_id": run_id, "model": model, "tasks": {}}

    for task_file in files:
        if not task_file.is_file():
            logger.warning("task file not found, skipping: %s", task_file)
            continue
        task_name = task_file.stem
        tasks = load_tasks(task_file, max_examples=max_examples)
        logger.info("[%s] %s: %d tasks", run_id, task_name, len(tasks))

        raw_log = paths.results_dir / f"{run_id}_{task_name}_raw.jsonl"
        runner = BenchmarkRunner(adapter, raw_log_path=raw_log, task_name=task_name)
        result = runner.run(tasks)
        all_results["tasks"][task_name] = result

        score = result["score"]
        overall = result["metrics"].get("all", {})
        logger.info(
            "[%s] %s: IndicJevScore=%s acc=%.3f ece=%.3f p50=%sms (%d/%d answered)",
            run_id,
            task_name,
            score["indicjev_score"],
            overall.get("accuracy", float("nan")),
            overall.get("ece", float("nan")),
            result["latency"]["p50_ms"],
            result["n_answered"],
            result["n_tasks"],
        )

    out_path = Path(output) if output is not None else paths.results_file(run_id)
    atomic_write_text(out_path, json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("Results written to %s", out_path)
    return out_path


def close_adapter(adapter: BenchAdapter) -> None:
    """Call ``adapter.close()`` if the adapter implements it.

    Most adapters hold no external resources and do not define ``close``;
    this is a no-op for them.

    Args:
        adapter: The adapter to shut down.
    """
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def cmd_run(args: argparse.Namespace, paths: BenchPaths | None = None) -> None:
    """Execute the ``run`` subcommand over one or more task files.

    Args:
        args: Parsed CLI namespace.
        paths: Optional BenchPaths override (defaults to ``BenchPaths.default()``).
    """
    paths = paths if paths is not None else BenchPaths.default()
    adapter = build_adapter(args)

    task_files = [Path(t) for t in args.tasks] if args.tasks else paths.dataset_files()
    if not task_files:
        logger.error("No task files found. Run scripts/package_datasets.py first.")
        return

    try:
        run_evaluation(
            adapter,
            task_files,
            model=args.model,
            max_examples=args.max_examples,
            output=args.output,
            paths=paths,
        )
    finally:
        close_adapter(adapter)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        Parser with the ``run`` subcommand and its flags.
    """
    parser = argparse.ArgumentParser(prog="indicjevbench")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run benchmark evaluation")
    run_p.add_argument("--tasks", nargs="+", help="JSONL task files (default: all in datasets/v1/)")
    run_p.add_argument(
        "--adapter", required=True, choices=["http", "local", "qwen3", "api", "semif"]
    )
    run_p.add_argument("--endpoint", default="http://localhost:8000")
    run_p.add_argument("--base-url", default=None, help="API base URL override (e.g. https://openrouter.ai/api/v1)")
    run_p.add_argument("--checkpoint", default="checkpoints/best")
    run_p.add_argument("--model", default="my-model", help="Model label for results; also used as HF model id for --adapter qwen3/semif")
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
