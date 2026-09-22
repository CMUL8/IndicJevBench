"""Run any OpenAI-compatible API LLM (e.g. OpenRouter) on IndicJevBench.

Thin wrapper: parses args, builds an APILLMAdapter, and calls the shared
package benchmark loop. All evaluation/metric logic lives in indicjevbench.

Install:
    pip install "indicjevbench[baselines]"

Run:
    OPENROUTER_API_KEY=sk-or-... python scripts/eval_api.py
    OPENROUTER_API_KEY=sk-or-... python scripts/eval_api.py --model openai/gpt-4o
    OPENROUTER_API_KEY=sk-or-... python scripts/eval_api.py --datasets fintech_banking77 --max-items 200
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from indicjevbench.configs.env import get_api_key
from indicjevbench.configs.paths import BenchPaths
from indicjevbench.runner.cli import close_adapter, run_evaluation

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
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
    parser.add_argument("--max-items", type=int, default=200,
                        help="items per dataset (default 200 to control cost), 0=all")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="OpenRouter model ID")
    parser.add_argument("--budget", type=float, default=5.0, help="max spend in USD")
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code (1 if no API key is configured).
    """
    args = parse_args(argv)
    api_key = get_api_key("OPENROUTER_API_KEY", "OPENAI_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY in environment.", file=sys.stderr)
        return 1

    from indicjevbench.adapters.api_llm import APILLMAdapter  # lazy: openai extra

    adapter = APILLMAdapter(
        model=args.model, base_url=args.base_url, api_key=api_key, max_budget_usd=args.budget
    )
    paths = BenchPaths.default()
    task_files = [paths.datasets_dir / f"{name}.jsonl" for name in args.datasets]
    run_id = f"api_{args.model.replace('/', '_')}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    try:
        run_evaluation(adapter, task_files, model=args.model,
                       max_examples=args.max_items or None, run_id=run_id, paths=paths)
    finally:
        close_adapter(adapter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
