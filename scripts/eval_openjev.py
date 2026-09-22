"""Run OpenJev (semif-phase1) on IndicJevBench.

Thin wrapper: parses args, builds a SemIfAdapter, and calls the shared
package benchmark loop. All evaluation/metric logic lives in indicjevbench.

Install:
    pip install git+https://github.com/TheoLeeCJ/openjev.git

Run:
    uv run python scripts/eval_openjev.py
    uv run python scripts/eval_openjev.py --max-items 200
    uv run python scripts/eval_openjev.py --datasets intent_massive fintech_banking77
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
    parser.add_argument("--max-items", type=int, default=1000, help="items per dataset, 0 = all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code (always 0; failures surface as exceptions).
    """
    args = parse_args(argv)

    from indicjevbench.adapters.semif import SemIfAdapter  # lazy: semif_phase1

    adapter = SemIfAdapter(
        model_id=args.model, device=args.device, dtype=args.dtype, max_tokens=args.max_tokens
    )
    paths = BenchPaths.default()
    task_files = [paths.datasets_dir / f"{name}.jsonl" for name in args.datasets]
    run_id = f"openjev_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    try:
        run_evaluation(
            adapter,
            task_files,
            model=args.model,
            max_examples=args.max_items or None,
            run_id=run_id,
            paths=paths,
        )
    finally:
        close_adapter(adapter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
