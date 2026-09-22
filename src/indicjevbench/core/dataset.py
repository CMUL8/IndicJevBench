"""Dataset loading for IndicJevBench JSONL task files."""

import json
import logging
from pathlib import Path
from typing import Any, cast

from indicjevbench.schemas.contracts import Task

logger = logging.getLogger(__name__)


def load_tasks(path: str | Path, max_examples: int | None = None) -> list[Task]:
    """Load validated tasks from a JSONL dataset file.

    Blank lines are skipped. Each non-blank line must be a JSON object
    convertible to a Task.

    Args:
        path: Path to the ``.jsonl`` task file.
        max_examples: Optional cap on the number of tasks loaded.

    Returns:
        List of validated Task objects in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON or fails Task validation.
        TypeError: If ``max_examples`` is not a positive int or None.
    """
    if max_examples is not None:
        if type(max_examples) is not int:
            raise TypeError(f"max_examples must be a positive int or None, got {max_examples!r}")
        if max_examples <= 0:
            raise ValueError(f"max_examples must be a positive int or None, got {max_examples!r}")

    path = Path(path)
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                task = Task.from_dict(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: bad task row: {exc}") from exc
            tasks.append(task)
            if max_examples is not None and len(tasks) >= max_examples:
                break
    logger.debug("loaded %d tasks from %s", len(tasks), path)
    return tasks


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a dataset manifest JSON file.

    Args:
        path: Path to the manifest (e.g. ``datasets/manifest.json``).

    Returns:
        The parsed manifest mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not valid JSON or not a JSON object.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{path}: manifest must be a JSON object, got {type(data).__name__}")
    return cast(dict[str, Any], data)
