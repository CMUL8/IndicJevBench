"""Run Laya (convaiinnovations) on IndicJevBench.

Thin wrapper: parses args, builds a LayaAdapter, and calls the shared
package benchmark loop. All evaluation/metric logic lives in indicjevbench.

Install:
    pip install "indicjevbench[baselines]" laya

Run:
    uv run python scripts/eval_laya.py
    uv run python scripts/eval_laya.py --max-items 200
    uv run python scripts/eval_laya.py --checkpoint convaiinnovations/laya-multilingual
    uv run python scripts/eval_laya.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from indicjevbench.configs.paths import BenchPaths
from indicjevbench.runner.cli import close_adapter, run_evaluation

ALL_DATASETS = ["intent_massive", "fintech_banking77", "hinglish_lid", "synthetic_enterprise"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse script CLI flags.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--max-items", type=int, default=1000, help="items per dataset, 0=all")
    parser.add_argument(
        "--checkpoint",
        default="convaiinnovations/laya-multilingual",
        help="HF model ID or local path",
    )
    parser.add_argument("--device", default="cuda", help="cuda / cpu (default: cuda)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code (always 0; failures surface as exceptions).
    """
    args = parse_args(argv)

    from indicjevbench.adapters.laya import LayaAdapter  # lazy: baselines extra + laya pkg

    adapter = LayaAdapter(model_id=args.checkpoint, device=args.device)
    paths = BenchPaths.default()
    task_files = [paths.datasets_dir / f"{name}.jsonl" for name in args.datasets]
    ckpt_short = args.checkpoint.split("/")[-1]
    run_id = f"laya_{ckpt_short}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    try:
        run_evaluation(
            adapter,
            task_files,
            model=args.checkpoint,
            max_examples=args.max_items or None,
            run_id=run_id,
            paths=paths,
        )
    finally:
        close_adapter(adapter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
