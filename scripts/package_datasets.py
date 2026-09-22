"""Package raw pipeline output into IndicJevBench ``datasets/v1`` JSONL files.

Thin wrapper around :class:`indicjevbench.core.packaging.DatasetPackager`.
All packaging logic lives in the package; this script only resolves default
paths and parses arguments.

Run:
    uv run python scripts/package_datasets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from indicjevbench.configs.paths import BenchPaths
from indicjevbench.core.packaging import DatasetPackager
from indicjevbench.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code (1 on packaging failure).
    """
    configure_logging()
    default_paths = BenchPaths.default()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-final",
        type=Path,
        default=None,
        help=(
            "raw pipeline JSONL input (default: <repo>/../../data/final/test.jsonl if it exists)"
        ),
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=default_paths.datasets_dir,
        help="output directory for v1 JSONL files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_paths.manifest_path,
        help="output manifest path",
    )
    args = parser.parse_args(argv)

    data_final = args.data_final
    if data_final is None:
        # Default: the upstream pipeline output living outside this repo
        # (../.. /data/final/test.jsonl relative to the repo root).
        data_final = default_paths.bench_root.parent.parent / "data" / "final" / "test.jsonl"

    packager = DatasetPackager(
        data_final=data_final,
        datasets_dir=args.datasets_dir,
        manifest_path=args.manifest,
    )
    try:
        packager.package()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("packaging failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
